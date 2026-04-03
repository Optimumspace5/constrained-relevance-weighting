from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from itertools import count, permutations
import time

from grid_adventure.grid import GridState, to_state
from grid_adventure.env import ImageObservation
from grid_adventure.step import Action


MOVE_ACTIONS = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
ALL_ACTIONS = (Action.PICK_UP, Action.USE_KEY, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)
MOVE_DELTAS = {
    Action.UP: (0, -1),
    Action.DOWN: (0, 1),
    Action.LEFT: (-1, 0),
    Action.RIGHT: (1, 0),
}

FLOOR_COST = 3
COIN_REWARD = 5
LAVA_DAMAGE = 2
POWERUP_DURATION = 5
SHIELD_USES = 5

NAME_TO_KIND = {
    "gem": "gem",
    "coin": "coin",
    "key": "key",
    "door": "door",
    "wall": "wall",
    "lava": "lava",
    "box": "box",
    "shield": "shield",
    "ghost": "ghost",
    "boots": "boots",
    "exit": "exit",
}


@dataclass(frozen=True, slots=True)
class SearchState:
    pos: int
    boxes: tuple[int, ...]
    gems_mask: int
    coins_mask: int
    keys_mask: int
    doors_mask: int
    shields_mask: int
    ghosts_mask: int
    boots_mask: int
    keys_held: int
    shield_uses: int
    ghost_times: tuple[int, ...]
    boot_times: tuple[int, ...]
    health: int
    turn: int
    win: bool
    lose: bool


class CompactTask1Solver:
    def __init__(self, state: GridState):
        ts = to_state(state)
        self.width = ts.width
        self.height = ts.height
        self.turn_limit = ts.turn_limit or 150
        self.agent_id = next(iter(ts.agent.keys()))
        self.initial_score = ts.score

        self.exit_idx: int | None = None
        self.wall_bits = 0
        self.lava_bits = 0

        self.gem_positions: list[int] = []
        self.coin_positions: list[int] = []
        self.key_positions: list[int] = []
        self.door_positions: list[int] = []
        self.shield_positions: list[int] = []
        self.ghost_positions: list[int] = []
        self.boot_positions: list[int] = []
        self.box_positions: list[int] = []

        for eid, pos in ts.position.items():
            if eid == self.agent_id or eid not in ts.appearance:
                continue
            name = ts.appearance[eid].name
            kind = NAME_TO_KIND.get(name)
            if kind is None:
                continue
            idx = self.xy_to_idx(pos.x, pos.y)

            if kind == "exit":
                self.exit_idx = idx
            elif kind == "wall":
                self.wall_bits |= 1 << idx
            elif kind == "lava":
                self.lava_bits |= 1 << idx
            elif kind == "gem":
                self.gem_positions.append(idx)
            elif kind == "coin":
                self.coin_positions.append(idx)
            elif kind == "key":
                self.key_positions.append(idx)
            elif kind == "door" and eid in ts.blocking:
                self.door_positions.append(idx)
            elif kind == "shield":
                self.shield_positions.append(idx)
            elif kind == "ghost":
                self.ghost_positions.append(idx)
            elif kind == "boots":
                self.boot_positions.append(idx)
            elif kind == "box":
                self.box_positions.append(idx)

        self.gem_bit_by_pos = {p: 1 << i for i, p in enumerate(self.gem_positions)}
        self.coin_bit_by_pos = {p: 1 << i for i, p in enumerate(self.coin_positions)}
        self.key_bit_by_pos = {p: 1 << i for i, p in enumerate(self.key_positions)}
        self.door_bit_by_pos = {p: 1 << i for i, p in enumerate(self.door_positions)}
        self.shield_bit_by_pos = {p: 1 << i for i, p in enumerate(self.shield_positions)}
        self.ghost_bit_by_pos = {p: 1 << i for i, p in enumerate(self.ghost_positions)}
        self.boot_bit_by_pos = {p: 1 << i for i, p in enumerate(self.boot_positions)}

        self.start_pos = self.xy_to_idx(ts.position[self.agent_id].x, ts.position[self.agent_id].y)
        self.start_health = ts.health[self.agent_id].current_health
        self.walk_neighbors = self._build_walk_neighbors()
        self.wall_dists = self._build_wall_distances()

        status = ts.status.get(self.agent_id)
        effect_ids = tuple(sorted(getattr(status, "effect_ids", []))) if status is not None else ()

        boot_times = []
        ghost_times = []
        shield_uses = 0
        for eid in effect_ids:
            if eid in ts.speed and eid in ts.time_limit and ts.time_limit[eid].amount > 0:
                boot_times.append(ts.time_limit[eid].amount)
            if eid in ts.phasing and eid in ts.time_limit and ts.time_limit[eid].amount > 0:
                ghost_times.append(ts.time_limit[eid].amount)
            if eid in ts.immunity and eid in ts.usage_limit and ts.usage_limit[eid].amount > 0:
                shield_uses += ts.usage_limit[eid].amount

        inventory_item_ids = getattr(ts.inventory.get(self.agent_id), "item_ids", [])
        keys_held = sum(1 for eid in inventory_item_ids if eid in ts.key)

        self.initial_state = SearchState(
            pos=self.start_pos,
            boxes=tuple(sorted(self.box_positions)),
            gems_mask=(1 << len(self.gem_positions)) - 1,
            coins_mask=(1 << len(self.coin_positions)) - 1,
            keys_mask=(1 << len(self.key_positions)) - 1,
            doors_mask=(1 << len(self.door_positions)) - 1,
            shields_mask=(1 << len(self.shield_positions)) - 1,
            ghosts_mask=(1 << len(self.ghost_positions)) - 1,
            boots_mask=(1 << len(self.boot_positions)) - 1,
            keys_held=keys_held,
            shield_uses=shield_uses,
            ghost_times=tuple(sorted(ghost_times)),
            boot_times=tuple(sorted(boot_times)),
            health=self.start_health,
            turn=ts.turn,
            win=ts.win,
            lose=ts.lose,
        )

        self._transition_cache: dict[tuple[SearchState, Action], tuple[SearchState, int]] = {}
        self._lower_bound_cache: dict[tuple, int] = {}

    def search_key(self, state: SearchState) -> tuple:
        return (
            state.pos,
            state.boxes,
            state.gems_mask,
            state.coins_mask,
            state.keys_mask,
            state.doors_mask,
            state.shields_mask,
            state.ghosts_mask,
            state.boots_mask,
            state.keys_held,
            state.shield_uses,
            state.ghost_times,
            state.boot_times,
            state.health,
        )

    def _build_walk_neighbors(self) -> list[tuple[int, ...]]:
        total = self.width * self.height
        neighbors: list[tuple[int, ...]] = []
        for idx in range(total):
            if self.wall_bits & (1 << idx):
                neighbors.append(())
                continue
            x, y = self.idx_to_xy(idx)
            out = []
            for dx, dy in MOVE_DELTAS.values():
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    nidx = self.xy_to_idx(nx, ny)
                    if not (self.wall_bits & (1 << nidx)):
                        out.append(nidx)
            neighbors.append(tuple(out))
        return neighbors

    def _build_wall_distances(self) -> list[list[int]]:
        total = self.width * self.height
        inf = total + 1
        dists = [[inf] * total for _ in range(total)]
        for src in range(total):
            if self.wall_bits & (1 << src):
                continue
            q = deque([src])
            dists[src][src] = 0
            while q:
                cur = q.popleft()
                nd = dists[src][cur] + 1
                for nxt in self.walk_neighbors[cur]:
                    if dists[src][nxt] > nd:
                        dists[src][nxt] = nd
                        q.append(nxt)
        return dists

    def static_dist(self, a: int, b: int | None) -> int:
        if b is None:
            return 0
        total = self.width * self.height
        d = self.wall_dists[a][b]
        if d <= total:
            return d
        ax, ay = self.idx_to_xy(a)
        bx, by = self.idx_to_xy(b)
        return abs(ax - bx) + abs(ay - by)

    def xy_to_idx(self, x: int, y: int) -> int:
        return y * self.width + x

    def idx_to_xy(self, idx: int) -> tuple[int, int]:
        return idx % self.width, idx // self.width

    def adjacent_index(self, idx: int, action: Action) -> int | None:
        x, y = self.idx_to_xy(idx)
        dx, dy = MOVE_DELTAS[action]
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.width and 0 <= ny < self.height:
            return self.xy_to_idx(nx, ny)
        return None

    def active_phasing(self, state: SearchState) -> bool:
        return len(state.ghost_times) > 0

    def active_boots(self, state: SearchState) -> bool:
        return len(state.boot_times) > 0

    def decrement_times(self, times: tuple[int, ...]) -> tuple[int, ...]:
        updated = [t - 1 for t in times if t - 1 > 0]
        updated.sort()
        return tuple(updated)

    def collect_at_pos(
        self,
        state: SearchState,
        old_ghost_times: tuple[int, ...],
        old_boot_times: tuple[int, ...],
    ) -> tuple[SearchState, int]:
        pos = state.pos
        score_delta = 0

        gems_mask = state.gems_mask
        coins_mask = state.coins_mask
        keys_mask = state.keys_mask
        shields_mask = state.shields_mask
        ghosts_mask = state.ghosts_mask
        boots_mask = state.boots_mask

        keys_held = state.keys_held
        shield_uses = state.shield_uses
        ghost_times = list(state.ghost_times)
        boot_times = list(state.boot_times)

        if pos in self.gem_bit_by_pos:
            bit = self.gem_bit_by_pos[pos]
            if gems_mask & bit:
                gems_mask &= ~bit

        if pos in self.coin_bit_by_pos:
            bit = self.coin_bit_by_pos[pos]
            if coins_mask & bit:
                coins_mask &= ~bit
                score_delta += COIN_REWARD

        if pos in self.key_bit_by_pos:
            bit = self.key_bit_by_pos[pos]
            if keys_mask & bit:
                keys_mask &= ~bit
                keys_held += 1

        if pos in self.shield_bit_by_pos:
            bit = self.shield_bit_by_pos[pos]
            if shields_mask & bit:
                shields_mask &= ~bit
                shield_uses += SHIELD_USES

        if pos in self.ghost_bit_by_pos:
            bit = self.ghost_bit_by_pos[pos]
            if ghosts_mask & bit:
                ghosts_mask &= ~bit
                ghost_times.append(POWERUP_DURATION)

        if pos in self.boot_bit_by_pos:
            bit = self.boot_bit_by_pos[pos]
            if boots_mask & bit:
                boots_mask &= ~bit
                boot_times.append(POWERUP_DURATION)

        ghost_times.sort()
        boot_times.sort()

        next_state = SearchState(
            pos=pos,
            boxes=state.boxes,
            gems_mask=gems_mask,
            coins_mask=coins_mask,
            keys_mask=keys_mask,
            doors_mask=state.doors_mask,
            shields_mask=shields_mask,
            ghosts_mask=ghosts_mask,
            boots_mask=boots_mask,
            keys_held=keys_held,
            shield_uses=shield_uses,
            ghost_times=tuple(ghost_times),
            boot_times=tuple(boot_times),
            health=state.health,
            turn=state.turn,
            win=state.win,
            lose=state.lose,
        )
        return next_state, score_delta

    def apply_damage_and_win(
        self,
        state: SearchState,
        encountered_lava: set[int],
    ) -> SearchState:
        health = state.health
        shield_uses = state.shield_uses
        phasing_active = len(state.ghost_times) > 0

        for _lava_idx in encountered_lava:
            if shield_uses > 0:
                shield_uses -= 1
                continue
            if phasing_active:
                continue
            health -= LAVA_DAMAGE

        win = state.win
        if self.exit_idx is not None and state.gems_mask == 0 and self.exit_idx in encountered_lava.union({state.pos}):
            win = True

        if not win and self.exit_idx is not None and state.gems_mask == 0:
            if getattr(self, "_visited_positions", None) and self.exit_idx in self._visited_positions:
                win = True

        lose = state.lose or (health <= 0)

        return SearchState(
            pos=state.pos,
            boxes=state.boxes,
            gems_mask=state.gems_mask,
            coins_mask=state.coins_mask,
            keys_mask=state.keys_mask,
            doors_mask=state.doors_mask,
            shields_mask=state.shields_mask,
            ghosts_mask=state.ghosts_mask,
            boots_mask=state.boots_mask,
            keys_held=state.keys_held,
            shield_uses=shield_uses,
            ghost_times=state.ghost_times,
            boot_times=state.boot_times,
            health=health,
            turn=state.turn,
            win=win,
            lose=lose,
        )

    def finalize_times(
        self,
        old_times: tuple[int, ...],
        current_times: tuple[int, ...],
    ) -> tuple[int, ...]:
        updated = list(self.decrement_times(old_times))
        add_count = current_times.count(POWERUP_DURATION) - old_times.count(POWERUP_DURATION)
        if add_count > 0:
            updated.extend([POWERUP_DURATION] * add_count)
        updated.sort()
        return tuple(updated)

    def after_step_finalize(
        self,
        state: SearchState,
        old_ghost_times: tuple[int, ...],
        old_boot_times: tuple[int, ...],
        score_delta: int,
    ) -> tuple[SearchState, int]:
        ghost_times = self.finalize_times(old_ghost_times, state.ghost_times)
        boot_times = self.finalize_times(old_boot_times, state.boot_times)

        turn = state.turn + 1
        lose = state.lose or (turn >= self.turn_limit and not state.win)

        next_state = SearchState(
            pos=state.pos,
            boxes=state.boxes,
            gems_mask=state.gems_mask,
            coins_mask=state.coins_mask,
            keys_mask=state.keys_mask,
            doors_mask=state.doors_mask,
            shields_mask=state.shields_mask,
            ghosts_mask=state.ghosts_mask,
            boots_mask=state.boots_mask,
            keys_held=state.keys_held,
            shield_uses=state.shield_uses,
            ghost_times=ghost_times,
            boot_times=boot_times,
            health=state.health,
            turn=turn,
            win=state.win,
            lose=lose,
        )

        if not next_state.win and not next_state.lose:
            score_delta -= FLOOR_COST

        return next_state, score_delta

    def step_move(self, state: SearchState, action: Action) -> tuple[SearchState, int]:
        old_ghost_times = state.ghost_times
        old_boot_times = state.boot_times
        phasing_active = len(old_ghost_times) > 0
        move_count = 2 if len(old_boot_times) > 0 else 1
        visited_positions: list[int] = []
        damaged_lava: set[int] = set()

        cur_pos = state.pos
        boxes = list(state.boxes)
        box_set = set(boxes)
        doors_mask = state.doors_mask
        health = state.health
        shield_uses = state.shield_uses
        lose = state.lose
        win = state.win

        for _ in range(move_count):
            nxt = self.adjacent_index(cur_pos, action)
            next_pos = cur_pos if nxt is None else nxt

            prev_snapshot = (cur_pos, tuple(sorted(boxes)), health, shield_uses, win, lose)

            pushed = False
            if next_pos in box_set:
                cx, cy = self.idx_to_xy(cur_pos)
                nx, ny = self.idx_to_xy(next_pos)
                dx = nx - cx
                dy = ny - cy
                bx = nx + dx
                by = ny + dy
                if 0 <= bx < self.width and 0 <= by < self.height:
                    push_to = self.xy_to_idx(bx, by)
                    blocked_for_box = bool(self.wall_bits & (1 << push_to))
                    if push_to in box_set:
                        blocked_for_box = True
                    if push_to in self.door_positions:
                        bit = self.door_bit_by_pos[push_to]
                        if doors_mask & bit:
                            blocked_for_box = True
                    if not blocked_for_box:
                        box_set.remove(next_pos)
                        box_set.add(push_to)
                        boxes = sorted(box_set)
                        cur_pos = next_pos
                        pushed = True

            if not pushed:
                if nxt is None:
                    pass
                elif phasing_active:
                    cur_pos = next_pos
                else:
                    blocked = bool(self.wall_bits & (1 << next_pos))
                    if not blocked and next_pos in box_set:
                        blocked = True
                    if not blocked and next_pos in self.door_positions:
                        bit = self.door_bit_by_pos[next_pos]
                        if doors_mask & bit:
                            blocked = True
                    if not blocked:
                        cur_pos = next_pos

            visited_positions.append(cur_pos)

            if self.exit_idx is not None and state.gems_mask == 0 and cur_pos == self.exit_idx:
                win = True

            if (self.lava_bits & (1 << cur_pos)) and cur_pos not in damaged_lava:
                damaged_lava.add(cur_pos)
                if shield_uses > 0:
                    shield_uses -= 1
                elif not phasing_active:
                    health -= LAVA_DAMAGE

            if health <= 0:
                lose = True

            if prev_snapshot == (cur_pos, tuple(sorted(boxes)), health, shield_uses, win, lose):
                break

        self._visited_positions = tuple(visited_positions)
        next_state = SearchState(
            pos=cur_pos,
            boxes=tuple(sorted(boxes)),
            gems_mask=state.gems_mask,
            coins_mask=state.coins_mask,
            keys_mask=state.keys_mask,
            doors_mask=doors_mask,
            shields_mask=state.shields_mask,
            ghosts_mask=state.ghosts_mask,
            boots_mask=state.boots_mask,
            keys_held=state.keys_held,
            shield_uses=shield_uses,
            ghost_times=state.ghost_times,
            boot_times=state.boot_times,
            health=health,
            turn=state.turn,
            win=win,
            lose=lose,
        )
        finalized, score_delta = self.after_step_finalize(next_state, old_ghost_times, old_boot_times, 0)
        self._visited_positions = ()
        return finalized, score_delta

    def step_use_key(self, state: SearchState) -> tuple[SearchState, int]:
        old_ghost_times = state.ghost_times
        old_boot_times = state.boot_times
        keys_held = state.keys_held
        doors_mask = state.doors_mask
        x, y = self.idx_to_xy(state.pos)
        for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.width and 0 <= ny < self.height):
                continue
            idx = self.xy_to_idx(nx, ny)
            if idx in self.door_bit_by_pos and keys_held > 0:
                bit = self.door_bit_by_pos[idx]
                if doors_mask & bit:
                    doors_mask &= ~bit
                    keys_held -= 1

        encountered_lava = {state.pos} if (self.lava_bits & (1 << state.pos)) else set()
        self._visited_positions = (state.pos,)
        tmp = SearchState(
            pos=state.pos,
            boxes=state.boxes,
            gems_mask=state.gems_mask,
            coins_mask=state.coins_mask,
            keys_mask=state.keys_mask,
            doors_mask=doors_mask,
            shields_mask=state.shields_mask,
            ghosts_mask=state.ghosts_mask,
            boots_mask=state.boots_mask,
            keys_held=keys_held,
            shield_uses=state.shield_uses,
            ghost_times=state.ghost_times,
            boot_times=state.boot_times,
            health=state.health,
            turn=state.turn,
            win=state.win or (self.exit_idx == state.pos and state.gems_mask == 0),
            lose=state.lose,
        )
        tmp = self.apply_damage_and_win(tmp, encountered_lava)
        finalized, score_delta = self.after_step_finalize(tmp, old_ghost_times, old_boot_times, 0)
        self._visited_positions = ()
        return finalized, score_delta

    def step_pickup(self, state: SearchState) -> tuple[SearchState, int]:
        old_ghost_times = state.ghost_times
        old_boot_times = state.boot_times
        tmp, score_delta = self.collect_at_pos(state, old_ghost_times, old_boot_times)
        encountered_lava = {tmp.pos} if (self.lava_bits & (1 << tmp.pos)) else set()
        self._visited_positions = (tmp.pos,)
        tmp = SearchState(
            pos=tmp.pos,
            boxes=tmp.boxes,
            gems_mask=tmp.gems_mask,
            coins_mask=tmp.coins_mask,
            keys_mask=tmp.keys_mask,
            doors_mask=tmp.doors_mask,
            shields_mask=tmp.shields_mask,
            ghosts_mask=tmp.ghosts_mask,
            boots_mask=tmp.boots_mask,
            keys_held=tmp.keys_held,
            shield_uses=tmp.shield_uses,
            ghost_times=tmp.ghost_times,
            boot_times=tmp.boot_times,
            health=tmp.health,
            turn=tmp.turn,
            win=tmp.win or (self.exit_idx == tmp.pos and tmp.gems_mask == 0),
            lose=tmp.lose,
        )
        tmp = self.apply_damage_and_win(tmp, encountered_lava)
        finalized, score_delta = self.after_step_finalize(tmp, old_ghost_times, old_boot_times, score_delta)
        self._visited_positions = ()
        return finalized, score_delta

    def step_wait(self, state: SearchState) -> tuple[SearchState, int]:
        old_ghost_times = state.ghost_times
        old_boot_times = state.boot_times
        encountered_lava = {state.pos} if (self.lava_bits & (1 << state.pos)) else set()
        self._visited_positions = (state.pos,)
        tmp = SearchState(
            pos=state.pos,
            boxes=state.boxes,
            gems_mask=state.gems_mask,
            coins_mask=state.coins_mask,
            keys_mask=state.keys_mask,
            doors_mask=state.doors_mask,
            shields_mask=state.shields_mask,
            ghosts_mask=state.ghosts_mask,
            boots_mask=state.boots_mask,
            keys_held=state.keys_held,
            shield_uses=state.shield_uses,
            ghost_times=state.ghost_times,
            boot_times=state.boot_times,
            health=state.health,
            turn=state.turn,
            win=state.win or (self.exit_idx == state.pos and state.gems_mask == 0),
            lose=state.lose,
        )
        tmp = self.apply_damage_and_win(tmp, encountered_lava)
        finalized, score_delta = self.after_step_finalize(tmp, old_ghost_times, old_boot_times, 0)
        self._visited_positions = ()
        return finalized, score_delta

    def transition(self, state: SearchState, action: Action) -> tuple[SearchState, int]:
        cached = self._transition_cache.get((state, action))
        if cached is not None:
            return cached

        if state.win or state.lose:
            result = (state, 0)
        elif action in MOVE_ACTIONS:
            result = self.step_move(state, action)
        elif action == Action.USE_KEY:
            result = self.step_use_key(state)
        elif action == Action.PICK_UP:
            result = self.step_pickup(state)
        else:
            result = self.step_wait(state)

        self._transition_cache[(state, action)] = result
        return result

    @lru_cache(maxsize=None)
    def exact_gem_lb(self, pos: int, gems: tuple[int, ...], exit_idx: int | None) -> int:
        if not gems:
            return self.static_dist(pos, exit_idx)

        best = 10**9
        for perm in permutations(gems):
            cur = pos
            total = 0
            for nxt in perm:
                total += self.static_dist(cur, nxt) + 1
                cur = nxt
            total += self.static_dist(cur, exit_idx)
            best = min(best, total)
        return best

    def remaining_positions(self, mask: int, positions: list[int]) -> tuple[int, ...]:
        out = []
        i = 0
        while mask:
            if mask & 1:
                out.append(positions[i])
            mask >>= 1
            i += 1
        return tuple(out)

    def lower_bound_actions(self, state: SearchState) -> int:
        cache_key = (
            state.pos,
            state.gems_mask,
            state.coins_mask,
            state.ghosts_mask,
            bool(state.ghost_times),
        )
        cached = self._lower_bound_cache.get(cache_key)
        if cached is not None:
            return cached

        exit_idx = self.exit_idx
        pos = state.pos
        gems = self.remaining_positions(state.gems_mask, self.gem_positions)
        coins = self.remaining_positions(state.coins_mask, self.coin_positions)
        ghosts = self.remaining_positions(state.ghosts_mask, self.ghost_positions)

        if gems:
            lb = self.exact_gem_lb(pos, gems, exit_idx)
        elif ghosts and not self.active_phasing(state):
            lb = min(self.static_dist(pos, g) + 1 + self.static_dist(g, exit_idx) for g in ghosts)
        elif coins:
            lb = min(self.static_dist(pos, c) + self.static_dist(c, exit_idx) for c in coins) + len(coins)
        else:
            lb = self.static_dist(pos, exit_idx)

        self._lower_bound_cache[cache_key] = lb
        return lb

    def upper_bound(self, state: SearchState, score: int) -> int:
        coin_count = state.coins_mask.bit_count()
        return score + COIN_REWARD * coin_count - FLOOR_COST * self.lower_bound_actions(state)

    def best_coin_target(self, state: SearchState, coins: tuple[int, ...]) -> int | None:
        if not coins:
            return None
        if self.exit_idx is None:
            return min(coins, key=lambda c: self.static_dist(state.pos, c))

        direct = self.static_dist(state.pos, self.exit_idx)
        best_coin = None
        best_gain = 0

        for c in coins:
            extra_actions = self.static_dist(state.pos, c) + 1 + self.static_dist(c, self.exit_idx) - direct
            gain = COIN_REWARD - FLOOR_COST * extra_actions
            if gain > best_gain:
                best_gain = gain
                best_coin = c

        return best_coin

    def candidate_actions(self, state: SearchState) -> tuple[Action, ...]:
        actions: list[Action] = []
        pos = state.pos

        collectible_here = False
        if pos in self.gem_bit_by_pos and (state.gems_mask & self.gem_bit_by_pos[pos]):
            collectible_here = True
        if pos in self.coin_bit_by_pos and (state.coins_mask & self.coin_bit_by_pos[pos]):
            collectible_here = True
        if pos in self.key_bit_by_pos and (state.keys_mask & self.key_bit_by_pos[pos]):
            collectible_here = True
        if pos in self.shield_bit_by_pos and (state.shields_mask & self.shield_bit_by_pos[pos]):
            collectible_here = True
        if pos in self.ghost_bit_by_pos and (state.ghosts_mask & self.ghost_bit_by_pos[pos]):
            collectible_here = True
        if pos in self.boot_bit_by_pos and (state.boots_mask & self.boot_bit_by_pos[pos]):
            collectible_here = True
        if collectible_here:
            actions.append(Action.PICK_UP)

        if state.keys_held > 0:
            x, y = self.idx_to_xy(pos)
            for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    idx = self.xy_to_idx(nx, ny)
                    if idx in self.door_bit_by_pos and (state.doors_mask & self.door_bit_by_pos[idx]):
                        actions.append(Action.USE_KEY)
                        break

        target: int | None = None
        gems = self.remaining_positions(state.gems_mask, self.gem_positions)
        ghosts = self.remaining_positions(state.ghosts_mask, self.ghost_positions)
        coins = self.remaining_positions(state.coins_mask, self.coin_positions)
        keys = self.remaining_positions(state.keys_mask, self.key_positions)
        shields = self.remaining_positions(state.shields_mask, self.shield_positions)
        boots = self.remaining_positions(state.boots_mask, self.boot_positions)

        if gems:
            target = min(gems, key=lambda p: self.static_dist(pos, p))
        elif ghosts and not self.active_phasing(state):
            target = min(ghosts, key=lambda p: self.static_dist(pos, p))
        elif state.keys_held == 0 and state.doors_mask and keys:
            target = min(keys, key=lambda p: self.static_dist(pos, p))
        elif state.shield_uses == 0 and self.lava_bits and shields:
            target = min(shields, key=lambda p: self.static_dist(pos, p))
        elif not self.active_boots(state) and boots:
            target = min(boots, key=lambda p: self.static_dist(pos, p))
        elif coins:
            target = self.best_coin_target(state, coins)
            if target is None and self.exit_idx is not None:
                target = self.exit_idx
        elif self.exit_idx is not None:
            target = self.exit_idx

        if target is None:
            return tuple(actions + list(MOVE_ACTIONS))

        base_moves = (
            [Action.RIGHT, Action.LEFT, Action.UP, Action.DOWN]
            if self.active_phasing(state)
            else [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        )
        ordered = sorted(
            base_moves,
            key=lambda a: self.static_dist(
                pos if self.adjacent_index(pos, a) is None else self.adjacent_index(pos, a),
                target,
            ),
        )

        box_set = set(state.boxes)
        phasing_active = self.active_phasing(state)
        for action in ordered:
            nxt = self.adjacent_index(pos, action)
            if nxt is None:
                continue
            if phasing_active:
                actions.append(action)
                continue
            if self.wall_bits & (1 << nxt):
                continue
            if nxt in self.door_bit_by_pos and (state.doors_mask & self.door_bit_by_pos[nxt]):
                continue
            if nxt in box_set:
                x, y = self.idx_to_xy(pos)
                nx, ny = self.idx_to_xy(nxt)
                dx = nx - x
                dy = ny - y
                bx = nx + dx
                by = ny + dy
                if not (0 <= bx < self.width and 0 <= by < self.height):
                    continue
                push_to = self.xy_to_idx(bx, by)
                if self.wall_bits & (1 << push_to):
                    continue
                if push_to in box_set:
                    continue
                if push_to in self.door_bit_by_pos and (state.doors_mask & self.door_bit_by_pos[push_to]):
                    continue
                actions.append(action)
                continue
            actions.append(action)

        return tuple(actions)

    def search(self, time_budget_sec: float = 8.0, node_budget: int = 200000) -> tuple[list[Action], bool]:
        frontier: list[tuple[int, int, int, SearchState, int]] = []
        tie = count()
        node_ids = count()
        parent: dict[int, tuple[int | None, Action | None]] = {}
        best_score_for_key: dict[tuple, int] = {}

        best_win_score = float("-inf")
        best_win_node: int | None = None
        best_partial_node: int | None = None
        best_partial_rank = (float("-inf"), float("-inf"))

        root = self.initial_state
        root_score = self.initial_score
        root_id = next(node_ids)
        parent[root_id] = (None, None)
        root_key = self.search_key(root)
        best_score_for_key[root_key] = root_score
        heappush(frontier, (-self.upper_bound(root, root_score), -root_score, next(tie), root_id, root, root_score))

        start = time.perf_counter()
        expanded = 0

        while frontier and expanded < node_budget and (time.perf_counter() - start) < time_budget_sec:
            neg_bound, neg_score, _, node_id, state, score = heappop(frontier)
            bound = -neg_bound

            if bound <= best_win_score:
                break

            state_key = self.search_key(state)
            if score < best_score_for_key.get(state_key, float("-inf")):
                continue

            expanded += 1
            rank = (bound, score)
            if node_id != root_id and rank > best_partial_rank:
                best_partial_rank = rank
                best_partial_node = node_id

            # maybe give 3 tries to win states to find better ones, since the first one found might be a fluke
            if state.win:
                if score > best_win_score:
                    best_win_score = score
                    best_win_node = node_id
                continue

            if state.lose:
                continue

            for action in self.candidate_actions(state):
                next_state, delta_score = self.transition(state, action)
                next_score = score + delta_score
                next_key = self.search_key(next_state)

                if next_score <= best_score_for_key.get(next_key, float("-inf")):
                    continue

                next_bound = self.upper_bound(next_state, next_score)
                if next_bound <= best_win_score:
                    continue

                best_score_for_key[next_key] = next_score
                next_id = next(node_ids)
                parent[next_id] = (node_id, action)
                heappush(frontier, (-next_bound, -next_score, next(tie), next_id, next_state, next_score))

        solved = best_win_node is not None
        if best_win_node is None:
            if best_partial_node is None:
                return [], False
            best_win_node = best_partial_node

        path: list[Action] = []
        cur = best_win_node
        while True:
            prev, action = parent[cur]
            if prev is None or action is None:
                break
            path.append(action)
            cur = prev
        path.reverse()
        return path, solved


class Agent:
    def __init__(self):
        self._plan: list[Action] = []
        self._search_calls = 0
        self._solved = False

    def step(self, state: GridState | ImageObservation) -> Action:
        if not isinstance(state, GridState):
            return Action.WAIT

        if state.turn == 0:
            self._plan = []
            self._search_calls = 0
            self._solved = False

        if state.turn == 0 or (not self._plan and not self._solved and self._search_calls < 2):
            solver = CompactTask1Solver(state)
            if self._search_calls == 0:
                self._plan, self._solved = solver.search(time_budget_sec=7.5, node_budget=450000)
            else:
                self._plan, self._solved = solver.search(time_budget_sec=0.8, node_budget=80000)
            self._search_calls += 1

        if not self._plan:
            return Action.WAIT

        return self._plan.pop(0)

    def parse(self, observation: ImageObservation) -> GridState:
        pass

    def info(self) -> dict[str, str]:
        return {"name": "Task 1 Compact Search Prototype"}