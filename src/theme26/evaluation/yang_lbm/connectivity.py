"""Exact graph winding check for the implemented periodic D3Q19 stencil.

Not ordinary 6-connected face spanning. Internally label 18-connected voxels;
join seam edges with integer lift displacements. A cycle with nonzero lift in
flow direction is a periodic transporting path. All components are retained.
This certifies discrete graph connectivity, not finite-throat physical accuracy.
"""
import numpy as np
from scipy.ndimage import label, generate_binary_structure
from ._vendor.lattice import E


def periodic_winding(pore, axis=2):
    if axis not in (0,1,2): raise ValueError("axis must be 0,1,2")
    labels, n = label(pore, structure=generate_binary_structure(3,2))
    if n == 0: return False
    parent = np.arange(n+1, dtype=np.int64)
    size = np.ones(n+1, dtype=np.int64)
    delta = np.zeros(n+1, dtype=np.int64)  # t_node - t_parent

    def find(x):
        root=x; total=0
        while parent[root] != root:
            total += int(delta[root]); root=int(parent[root])
        offset=0
        while parent[x] != x:
            nxt=int(parent[x]); d=int(delta[x])
            parent[x]=root; delta[x]=total-offset
            offset += d; x=nxt
        return root,total

    dims=pore.shape
    for e in E[1::2]:  # one of each opposite pair
        for crossing in np.flatnonzero(e):
            idx=[np.arange(d) for d in dims]
            idx[crossing]=np.array([dims[crossing]-1 if e[crossing]>0 else 0])
            dest=[(idx[d]+int(e[d]))%dims[d] for d in range(3)]
            a=labels[np.ix_(*idx)]; b=labels[np.ix_(*dest)]
            actual_dz=(dest[axis]-idx[axis])
            lift=int(e[axis])-actual_dz
            sh=[1,1,1]; sh[axis]=lift.size
            lift=np.broadcast_to(lift.reshape(sh), a.shape)
            valid=(a>0)&(b>0)
            edges=np.unique(np.column_stack((a[valid],b[valid],lift[valid])),axis=0)
            for aa,bb,w in edges:
                ra,da=find(int(aa)); rb,db=find(int(bb))
                if ra==rb:
                    if db-da != w: return True
                    continue
                # Constraint t_b - t_a = w.
                weight=int(w)+da-db  # t_rb - t_ra
                if size[ra]>=size[rb]:
                    parent[rb]=ra; delta[rb]=weight; size[ra]+=size[rb]
                else:
                    parent[ra]=rb; delta[ra]=-weight; size[rb]+=size[ra]
    return False
