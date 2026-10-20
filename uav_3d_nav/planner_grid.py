# planner_grid.py
import heapq, math
import numpy as np

def astar(occ, start_ij, goal_ij):
    W, H = occ.shape
    di = [-1,-1,-1,0,0,1,1,1]
    dj = [-1,0,1,-1,1,-1,0,1]
    dc = [math.sqrt(2),1,math.sqrt(2),1,1,math.sqrt(2),1,math.sqrt(2)]

    def inb(i,j): return 0 <= i < W and 0 <= j < H and not occ[i,j]
    si, sj = start_ij; gi, gj = goal_ij
    if not inb(si,sj) or not inb(gi,gj): return None

    g = np.full((W,H), np.inf); g[si,sj]=0.0
    parent = {(si,sj): None}
    h = lambda i,j: math.hypot(i-gi, j-gj)
    pq = [(h(si,sj), (si,sj))]

    while pq:
        _, (i,j) = heapq.heappop(pq)
        if (i,j)==(gi,gj):
            path=[(i,j)]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            path.reverse()
            return path
        for k in range(8):
            ni, nj = i+di[k], j+dj[k]
            if not inb(ni,nj): continue
            # (optional) corner-cut prevention:
            if di[k] and dj[k]:
                if occ[i, nj] or occ[ni, j]:  # don't cut diagonals through corners
                    continue
            ng = g[i,j] + dc[k]
            if ng < g[ni,nj]:
                g[ni,nj] = ng
                parent[(ni,nj)] = (i,j)
                f = ng + h(ni,nj)
                heapq.heappush(pq, (f, (ni,nj)))
    return None
