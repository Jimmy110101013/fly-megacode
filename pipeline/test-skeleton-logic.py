"""Tree splitting and RDP, checked on synthetic skeletons.

A bug in either wastes a full 268-million-row scan, so they get tested first.
    ./.venv/bin/python pipeline/test-skeleton-logic.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_skeletons import rdp, paths_of        # noqa: E402

fails = 0


def chk(name, cond, detail=""):
    global fails
    print(("PASS" if cond else "FAIL"), name, detail)
    fails += not cond


straight = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], float)
chk("rdp collapses a straight run to its endpoints", len(rdp(straight, 0.5)) == 2)
chk("rdp keeps a real corner",
    len(rdp(np.array([[0, 0, 0], [1, 5, 0], [2, 0, 0]], float), 0.5)) == 3)
chk("rdp drops a corner under epsilon",
    len(rdp(np.array([[0, 0, 0], [1, .01, 0], [2, 0, 0]], float), 0.5)) == 2)

# Y-shaped tree: 1 -> 2 -> 3, branching at 3 into 4 and 5.
nid, par = np.array([1, 2, 3, 4, 5]), np.array([-1, 1, 2, 3, 3])
xyz = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [1, 3, 0], [-1, 3, 0]], float)
runs = paths_of(nid, par, xyz)
chk("every edge appears exactly once", sum(len(r) - 1 for r in runs) == 4)
pts = np.vstack(runs)
chk("no node is dropped at a branch point",
    all(any(np.allclose(p, q) for q in pts) for p in xyz))

chk("disconnected fragments become separate runs",
    len(paths_of(np.array([1, 2, 9, 10]), np.array([-1, 1, -1, 9]),
                 np.array([[0, 0, 0], [0, 1, 0], [5, 5, 5], [5, 6, 5]], float))) == 2)

chain = paths_of(np.arange(1, 21), np.array([-1] + list(range(1, 20))),
                 np.stack([np.zeros(20), np.arange(20), np.zeros(20)], 1))
chk("unbranched chain stays one run", len(chain) == 1 and len(chain[0]) == 20)

print(f"\n{fails} check(s) failed" if fails else "\nall checks passed")
sys.exit(1 if fails else 0)
