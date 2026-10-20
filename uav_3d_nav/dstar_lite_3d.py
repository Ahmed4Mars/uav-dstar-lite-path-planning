# dstar_lite_3d.py
import math, heapq
import numpy as np

INF = float("inf")


class DStarLite3D:
    """
    3D D* Lite on a voxel grid.

    occ: 3D bool array (nx, ny, nz), True = blocked
    start, goal: (i, j, k) voxel indices
    res_xy: horizontal resolution (m per cell in x,y)
    res_z: vertical resolution  (m per cell in z)
    """

    def __init__(self, occ, start, goal, res_xy=0.5, res_z=0.5):
        self.occ = np.array(occ, dtype=bool)
        self.nx, self.ny, self.nz = self.occ.shape
        self.res_xy = float(res_xy)
        self.res_z = float(res_z)

        self.start = tuple(start)
        self.goal = tuple(goal)

        # g / rhs maps: we only allocate states we actually touch
        self.g = {}      # state -> g-value
        self.rhs = {}    # state -> rhs-value
        self.k_m = 0.0   # not used in this static variant, but kept for completeness

        # 26-connected neighbors in 3D
        self._nbrs = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        # priority queue of (k1, k2, state)
        self._pq = []
        self._key_map = {}  # state -> (k1, k2) for the *current* key

        # D* Lite initialization: rhs(goal)=0, everything else = inf by default
        self.rhs[self.goal] = 0.0
        self._insert(self.goal)

    # --------- basic helpers ---------
    def _g(self, s):
        return self.g.get(s, INF)

    def _rhs(self, s):
        return self.rhs.get(s, INF)

    def _in_bounds(self, i, j, k):
        return (0 <= i < self.nx) and (0 <= j < self.ny) and (0 <= k < self.nz)

    def _is_free(self, s):
        i, j, k = s
        return self._in_bounds(i, j, k) and (not self.occ[i, j, k])

    def _cost(self, a, b):
        ai, aj, ak = a
        bi, bj, bk = b
        dx = (bi - ai) * self.res_xy
        dy = (bj - aj) * self.res_xy
        dz = (bk - ak) * self.res_z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _heuristic(self, s):
        """Euclidean distance from s to current start (admissible)."""
        i, j, k = s
        si, sj, sk = self.start
        dx = (i - si) * self.res_xy
        dy = (j - sj) * self.res_xy
        dz = (k - sk) * self.res_z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _calc_key(self, s):
        g = self._g(s)
        rhs = self._rhs(s)
        m = min(g, rhs)
        return (m + self._heuristic(s) + self.k_m, m)

    def _neighbors(self, s):
        ci, cj, ck = s
        for dx, dy, dz in self._nbrs:
            ni, nj, nk = ci + dx, cj + dy, ck + dz
            if not self._in_bounds(ni, nj, nk):
                continue
            if self.occ[ni, nj, nk]:
                continue
            yield (ni, nj, nk)

    # --------- priority queue helpers ---------
    def _insert(self, s):
        key = self._calc_key(s)
        self._key_map[s] = key
        heapq.heappush(self._pq, (key[0], key[1], s))

    def _remove(self, s):
        # lazy removal: we just drop it from key_map; stale entries in the heap are skipped later
        if s in self._key_map:
            del self._key_map[s]

    def _top_key(self):
        while self._pq:
            k1, k2, s = self._pq[0]
            key = self._key_map.get(s)
            if key is None or key[0] != k1 or key[1] != k2:
                # stale entry
                heapq.heappop(self._pq)
                continue
            return key
        return (INF, INF)

    def _pop(self):
        while self._pq:
            k1, k2, s = heapq.heappop(self._pq)
            key = self._key_map.get(s)
            if key is None or key[0] != k1 or key[1] != k2:
                # stale
                continue
            del self._key_map[s]
            return (k1, k2, s)
        return (INF, INF, None)

    @staticmethod
    def _key_lt(a, b):
        # lexicographic < for 2D keys
        return (a[0] < b[0]) or (a[0] == b[0] and a[1] < b[1])

    # --------- core D* Lite routines ---------
    def _update_vertex(self, u):
        if u != self.goal:
            # rhs(u) = min_{v in Succ(u)} [ c(u,v) + g(v) ]
            min_rhs = INF
            for v in self._neighbors(u):
                val = self._cost(u, v) + self._g(v)
                if val < min_rhs:
                    min_rhs = val
            self.rhs[u] = min_rhs

        if self._g(u) == self._rhs(u):
            self._remove(u)
        else:
            self._insert(u)

    def compute_shortest_path(self):
        """
        Static D* Lite: we don't move the start and we don't update the map online,
        but the algorithm structure is the same as the incremental version.
        """
        while True:
            top_key = self._top_key()
            start_key = self._calc_key(self.start)

            # while TopKey < Key(start) or rhs(start) != g(start)
            if not (self._key_lt(top_key, start_key) or self._rhs(self.start) != self._g(self.start)):
                break

            k_old1, k_old2, u = self._pop()
            if u is None:
                break
            k_old = (k_old1, k_old2)
            k_new = self._calc_key(u)

            if self._key_lt(k_old, k_new):
                # outdated key, reinsert with up-to-date key
                self._insert(u)
                continue

            g_u = self._g(u)
            rhs_u = self._rhs(u)

            if g_u > rhs_u:
                # overconsistent: set g(u) downwards and propagate to predecessors
                self.g[u] = rhs_u
                for s in self._neighbors(u):  # Pred == Succ in undirected grid
                    self._update_vertex(s)
            else:
                # underconsistent: reset g(u) and propagate
                self.g[u] = INF
                self._update_vertex(u)
                for s in self._neighbors(u):
                    self._update_vertex(s)

    def plan(self):
        """
        Run D* Lite once and extract a path from start to goal as
        a list of (i, j, k) voxels. Returns [] if no path exists.
        """
        if (not self._is_free(self.start)) or (not self._is_free(self.goal)):
            return []

        self.compute_shortest_path()

        if math.isinf(self._g(self.start)):
            # no reachable path
            return []

        path = [self.start]
        cur = self.start
        max_steps = self.nx * self.ny * self.nz + 5  # safety

        for _ in range(max_steps):
            if cur == self.goal:
                break

            best_next = None
            best_val = INF

            for v in self._neighbors(cur):
                val = self._cost(cur, v) + self._g(v)
                if val < best_val:
                    best_val = val
                    best_next = v

            if best_next is None or math.isinf(best_val):
                # got stuck
                return []

            if best_next == cur:
                break

            path.append(best_next)
            cur = best_next

        if path[-1] != self.goal:
            return []
        return path
