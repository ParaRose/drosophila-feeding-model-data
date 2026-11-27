#!/usr/bin/env python3
"""Targeted portability hardening experiments for the 3-connectome SEZ->MN9 study (mn9-portability-final).

Implements all 4 pre-registered tests in `MN9_PORTABILITY_PROTOCOL.md` (commit 36ffab2):
  - Test A (MaleCNS A100 GPU): Transmitter sign-rule sensitivity (A1a diatom18+TH_VUM=+1,
      A1b diatom20+TH_VUM=+1, A2 all consensus_nt=='unclear' -> single-cell predicted_nt + SELK sweep)
  - Test B (Local + MaleCNS A100 GPU): Cross-brain matched-neuron incoming-synapse density
      diagnostics (BANC/v783 & MaleCNS/v783), full dimorphism/fruDsx audit of the 6 male-active
      negatives & all 106 SEZ types, and network-density-matched MaleCNS scaling (W_syn_density)
  - Test C (Local 16-core CPU): FlyWire v783 K=10 MN9-protected rewired null graphs (seeds 9201..9210)
      — C1 sugar-anchor feasibility scan & C2 activity-matched W_syn* calibration
  - Test D (Local + MaleCNS static): Full 44-purely-inhibitory-negative-control 3-brain conservation
      null distribution + BH-FDR corrections across secondary/exploratory tests

Phases:
  --phase local_static   : Test B1 (v783/BANC density) + Test D (v783/BANC 44-ctrl sets) + BH-FDR
  --phase local_test_c   : Test C1 & C2 on FlyWire v783 K=10 MN9-protected rewired graphs
  --phase colab_malecns  : Tests A, B2/B3, D (MaleCNS half) on Colab A100 GPU
  --phase merge          : Merge phase outputs into `mn9_portability_results.json`
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import heapq
import json
import math
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.feather as pf
from numba import njit
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("PROJC_CACHE_DIR", str(Path.home() / ".cache/projc-flywire")))

GCS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
MALECNS_FILES = {
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
}

INH_NTS = {"gaba", "glutamate"}
EXC_NTS = {"acetylcholine", "dopamine", "serotonin", "octopamine", "tyramine", "histamine"}


def nt_to_sign(nt) -> int:
    if nt is None:
        return 0
    s = str(nt).strip().lower()
    if not s or s in ("none", "nan", "na", "unclear", "unknown"):
        return 0
    if s in INH_NTS:
        return -1
    if s in EXC_NTS:
        return 1
    return 0


def split_tokens(val) -> set[str]:
    if val is None:
        return set()
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "na"):
        return set()
    return {p.strip() for p in s.replace(";", ",").split(",") if p.strip()}


def pick_col(cols: list[str], candidates: list[str], required: bool = True) -> str | None:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    if required:
        raise KeyError(f"None of {candidates} found in columns {cols}")
    return None


def summarize_ratios(arr: list[float] | np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a) & (a > 0.0)]
    if a.size == 0:
        return {"n": 0}
    q10, q25, q50, q75, q90 = np.percentile(a, [10, 25, 50, 75, 90])
    lo, hi = np.percentile(a, [10, 90])
    trimmed = a[(a >= lo) & (a <= hi)]
    return {
        "n": int(a.size),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "q10": float(q10),
        "q90": float(q90),
        "mean": float(a.mean()),
        "trimmed_mean_10pct": float(trimmed.mean()) if trimmed.size else float(a.mean()),
        "geom_mean": float(np.exp(np.mean(np.log(a)))),
    }


def stepwise_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    uniq_s = np.sort(np.unique(s))[::-1]
    tp = 0
    fp = 0
    prev_rec = 0.0
    auprc = 0.0
    for thr in uniq_s:
        mask = s == thr
        tp += int(y[mask].sum())
        fp += int((~y[mask].astype(bool)).sum())
        rec = tp / n_pos
        prec = tp / (tp + fp)
        auprc += (rec - prev_rec) * prec
        prev_rec = rec
    return float(auprc)


def midrank_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = stats.rankdata(s, method="average")
    sum_pos = float(ranks[y == 1].sum())
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bh_fdr(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    q = np.empty(m, dtype=np.float64)
    running = 1.0
    for rank_rev in range(m - 1, -1, -1):
        idx = order[rank_rev]
        val = pvals[idx] * m / (rank_rev + 1)
        running = min(running, val)
        q[idx] = running
    return [float(min(1.0, x)) for x in q]


def stratified_auprc_ci95(y_true: np.ndarray, scores: np.ndarray, n_boot: int = 2000, seed: int = 42) -> list[float]:
    y = np.asarray(y_true, dtype=np.int32)
    s = np.asarray(scores, dtype=np.float64)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        v = stepwise_auprc(y, s)
        return [v, v]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    yb = np.concatenate([np.ones(pos_idx.size, dtype=np.int32), np.zeros(neg_idx.size, dtype=np.int32)])
    for b in range(n_boot):
        sp = s[rng.choice(pos_idx, size=pos_idx.size, replace=True)]
        sn = s[rng.choice(neg_idx, size=neg_idx.size, replace=True)]
        vals[b] = stepwise_auprc(yb, np.concatenate([sp, sn]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def paired_bootstrap_auprc_p(
    y_true: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict:
    y = np.asarray(y_true, dtype=np.int32)
    sa = np.asarray(scores_a, dtype=np.float64)
    sb = np.asarray(scores_b, dtype=np.float64)
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    yb = np.concatenate([np.ones(pos_idx.size, dtype=np.int32), np.zeros(neg_idx.size, dtype=np.int32)])
    for b in range(n_boot):
        ip = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        in_ = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([ip, in_])
        diffs[b] = stepwise_auprc(yb, sa[idx]) - stepwise_auprc(yb, sb[idx])
    p_lo = float((np.sum(diffs <= 0.0) + 1) / (n_boot + 1))
    p_hi = float((np.sum(diffs >= 0.0) + 1) / (n_boot + 1))
    p_two = float(min(1.0, 2.0 * min(p_lo, p_hi)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "delta_point": float(stepwise_auprc(y, sa) - stepwise_auprc(y, sb)),
        "ci95": [float(lo), float(hi)],
        "p_two_sided": p_two,
    }


def exact_widest_exc_paths_via_vnc(
    indptr: np.ndarray,
    post_s: np.ndarray,
    w_raw_s: np.ndarray,
    node_class: np.ndarray,
    selk_idxs: list[int],
    mn9_set: set[int],
    is_vnc: np.ndarray,
    describe_node,
    max_k: int = 5,
) -> dict:
    """Exact max-bottleneck purely-excitatory SELK -> MN9 paths traversing >=1 VNC neuron:
    (1) strict hop-count DP for k = 2..max_k (both any-hop VNC and first-hop VNC);
    (2) unconstrained-hop widest-path Dijkstra maximizing (bottleneck, -hops)."""
    n_nodes = len(indptr) - 1
    selk_set = set(int(x) for x in selk_idxs)

    # Aggregate positive outgoing edges from bilateral SELK (sum if both L & R SELK hit same target?
    # Keep per-edge single-neuron paths faithful to neuron-level directed graph: max over L/R SELK -> v)
    selk_edge_best: dict[int, tuple[float, int]] = {}
    for s in selk_idxs:
        if node_class[s] != 0:
            continue
        for e in range(int(indptr[s]), int(indptr[s + 1])):
            w = float(w_raw_s[e])
            if w <= 0.0:
                continue
            v = int(post_s[e])
            if v in selk_set:
                continue
            prev = selk_edge_best.get(v)
            if prev is None or w > prev[0]:
                selk_edge_best[v] = (w, int(s))

    def _run_dp(first_hop_vnc_only: bool) -> dict[str, dict]:
        # State: dp[h][state][u] where state=0 (no VNC yet), state=1 (visited >=1 VNC exc node)
        dp0 = [np.zeros(n_nodes, dtype=np.float64) for _ in range(max_k + 1)]
        dp1 = [np.zeros(n_nodes, dtype=np.float64) for _ in range(max_k + 1)]
        par0 = [np.full(n_nodes, -1, dtype=np.int32) for _ in range(max_k + 1)]
        par1 = [np.full(n_nodes, -1, dtype=np.int32) for _ in range(max_k + 1)]
        pstate1 = [np.full(n_nodes, -1, dtype=np.int8) for _ in range(max_k + 1)]

        for v, (w, s) in selk_edge_best.items():
            if v in mn9_set:
                continue
            if node_class[v] != 0:
                continue
            if is_vnc[v]:
                dp1[1][v] = w
                par1[1][v] = s
                pstate1[1][v] = 0
            elif not first_hop_vnc_only:
                dp0[1][v] = w
                par0[1][v] = s

        res_by_k: dict[str, dict] = {}
        for h in range(2, max_k + 1):
            prev0, prev1 = dp0[h - 1], dp1[h - 1]
            cur0, cur1 = dp0[h], dp1[h]
            p0_arr, p1_arr, ps1_arr = par0[h], par1[h], pstate1[h]

            # Expand from state=0 (only if not first_hop_vnc_only)
            if not first_hop_vnc_only:
                for u in np.flatnonzero(prev0 > 0.0):
                    u_i = int(u)
                    if u_i in mn9_set or node_class[u_i] != 0:
                        continue
                    bu = float(prev0[u_i])
                    for e in range(int(indptr[u_i]), int(indptr[u_i + 1])):
                        w = float(w_raw_s[e])
                        if w <= 0.0:
                            continue
                        v = int(post_s[e])
                        cand = bu if bu < w else w
                        if v in mn9_set:
                            continue  # cannot enter MN9 from state=0 (no VNC visited yet)
                        if node_class[v] != 0:
                            continue
                        if is_vnc[v]:
                            if cand > cur1[v]:
                                cur1[v] = cand
                                p1_arr[v] = u_i
                                ps1_arr[v] = 0
                        else:
                            if cand > cur0[v]:
                                cur0[v] = cand
                                p0_arr[v] = u_i

            # Expand from state=1 (already visited VNC)
            for u in np.flatnonzero(prev1 > 0.0):
                u_i = int(u)
                if u_i in mn9_set or node_class[u_i] != 0:
                    continue
                bu = float(prev1[u_i])
                for e in range(int(indptr[u_i]), int(indptr[u_i + 1])):
                    w = float(w_raw_s[e])
                    if w <= 0.0:
                        continue
                    v = int(post_s[e])
                    if v not in mn9_set and node_class[v] != 0:
                        continue
                    cand = bu if bu < w else w
                    if cand > cur1[v]:
                        cur1[v] = cand
                        p1_arr[v] = u_i
                        ps1_arr[v] = 1

            # Extract best MN9 target at exact hop h
            best_t = None
            best_w = 0.0
            for t in sorted(mn9_set):
                if cur1[t] > best_w:
                    best_w = float(cur1[t])
                    best_t = int(t)
            if best_t is None:
                res_by_k[str(h)] = {"reachable": False, "hops": h, "bottleneck_weight": 0.0, "path_nodes": [], "edge_weights": []}
            else:
                nodes = [best_t]
                st = 1
                cur_n = best_t
                for step_h in range(h, 0, -1):
                    if st == 1:
                        nxt = int(par1[step_h][cur_n])
                        st = int(pstate1[step_h][cur_n])
                    else:
                        nxt = int(par0[step_h][cur_n])
                        st = 0
                    nodes.append(nxt)
                    cur_n = nxt
                nodes.reverse()
                ew = []
                for a, b in zip(nodes[:-1], nodes[1:]):
                    w_ab = 0.0
                    for e in range(int(indptr[a]), int(indptr[a + 1])):
                        if int(post_s[e]) == b and float(w_raw_s[e]) > w_ab:
                            w_ab = float(w_raw_s[e])
                    ew.append(w_ab)
                assert abs(min(ew) - best_w) < 1e-6, f"Path bottleneck mismatch {ew} vs {best_w}"
                res_by_k[str(h)] = {
                    "reachable": True,
                    "hops": h,
                    "bottleneck_weight": float(best_w),
                    "edge_weights": ew,
                    "path_nodes": [describe_node(u) for u in nodes],
                }
        return res_by_k

    def _run_global_widest(first_hop_vnc_only: bool) -> dict:
        # Maximize bottleneck, break ties by minimizing hops -> heap key (-bn, hops, node, state)
        best_bn = np.zeros((2, n_nodes), dtype=np.float64)
        best_hp = np.full((2, n_nodes), 1_000_000, dtype=np.int32)
        parent_n = np.full((2, n_nodes), -1, dtype=np.int32)
        parent_s = np.full((2, n_nodes), -1, dtype=np.int8)
        pq: list[tuple[float, int, int, int]] = []

        for v, (w, s) in selk_edge_best.items():
            if v in mn9_set or node_class[v] != 0:
                continue
            if is_vnc[v]:
                best_bn[1, v] = w
                best_hp[1, v] = 1
                parent_n[1, v] = s
                parent_s[1, v] = 0
                heapq.heappush(pq, (-w, 1, v, 1))
            elif not first_hop_vnc_only:
                best_bn[0, v] = w
                best_hp[0, v] = 1
                parent_n[0, v] = s
                parent_s[0, v] = 0
                heapq.heappush(pq, (-w, 1, v, 0))

        while pq:
            neg_b, hp, u, st = heapq.heappop(pq)
            bn = -neg_b
            if bn < best_bn[st, u] or (bn == best_bn[st, u] and hp > best_hp[st, u]):
                continue
            if u in mn9_set:
                continue
            if node_class[u] != 0:
                continue
            nhp = hp + 1
            for e in range(int(indptr[u]), int(indptr[u + 1])):
                w = float(w_raw_s[e])
                if w <= 0.0:
                    continue
                v = int(post_s[e])
                if v in selk_set:
                    continue
                if v in mn9_set:
                    if st == 0:
                        continue
                    nst = 1
                else:
                    if node_class[v] != 0:
                        continue
                    nst = 1 if (st == 1 or is_vnc[v]) else 0
                nbn = bn if bn < w else w
                if nbn > best_bn[nst, v] or (nbn == best_bn[nst, v] and nhp < best_hp[nst, v]):
                    best_bn[nst, v] = nbn
                    best_hp[nst, v] = nhp
                    parent_n[nst, v] = u
                    parent_s[nst, v] = st
                    heapq.heappush(pq, (-nbn, nhp, v, nst))

        best_t = None
        best_key = (-1.0, 1_000_000)
        for t in sorted(mn9_set):
            bn_t = float(best_bn[1, t])
            hp_t = int(best_hp[1, t])
            if bn_t > 0.0:
                key = (bn_t, -hp_t)
                if key > best_key:
                    best_key = key
                    best_t = int(t)
        if best_t is None:
            return {"reachable": False, "hops": None, "bottleneck_weight": 0.0, "path_nodes": [], "edge_weights": []}
        nodes = [best_t]
        cur_n, cur_s = best_t, 1
        seen = 0
        while cur_n not in selk_set and seen < 50:
            nxt_n = int(parent_n[cur_s, cur_n])
            nxt_s = int(parent_s[cur_s, cur_n])
            nodes.append(nxt_n)
            cur_n, cur_s = nxt_n, nxt_s
            seen += 1
        nodes.reverse()
        ew = []
        for a, b in zip(nodes[:-1], nodes[1:]):
            w_ab = 0.0
            for e in range(int(indptr[a]), int(indptr[a + 1])):
                if int(post_s[e]) == b and float(w_raw_s[e]) > w_ab:
                    w_ab = float(w_raw_s[e])
            ew.append(w_ab)
        assert abs(min(ew) - float(best_bn[1, best_t])) < 1e-6
        return {
            "reachable": True,
            "hops": int(best_hp[1, best_t]),
            "bottleneck_weight": float(best_bn[1, best_t]),
            "edge_weights": ew,
            "path_nodes": [describe_node(u) for u in nodes],
        }

    return {
        "via_any_vnc_by_hop": _run_dp(first_hop_vnc_only=False),
        "first_hop_vnc_by_hop": _run_dp(first_hop_vnc_only=True),
        "global_widest_via_any_vnc": _run_global_widest(first_hop_vnc_only=False),
        "global_widest_first_hop_vnc": _run_global_widest(first_hop_vnc_only=True),
    }


def audit_rewire_scatter(pre_s: np.ndarray, post_s: np.ndarray, reg_ids: np.ndarray, reg_names: list[str]) -> dict:
    """Quantify real outgoing regional fractions from central brain vs incoming-slot fractions under global rewiring."""
    n_reg = len(reg_names)
    cb_id = reg_names.index("central_brain")
    pre_r = reg_ids[pre_s]
    post_r = reg_ids[post_s]
    cb_mask = pre_r == cb_id
    cb_tot = int(cb_mask.sum())
    all_tot = int(len(post_r))
    return {
        "n_nodes_by_region": {reg_names[i]: int((reg_ids == i).sum()) for i in range(n_reg)},
        "real_cb_outgoing_edge_frac": {
            reg_names[i]: float(((cb_mask) & (post_r == i)).sum() / max(1, cb_tot)) for i in range(n_reg)
        },
        "global_incoming_slot_frac": {
            reg_names[i]: float((post_r == i).sum() / max(1, all_tot)) for i in range(n_reg)
        },
    }


@njit(cache=False)
def _rewire_block_swaps_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    block_edges: np.ndarray,
    edge_src: np.ndarray,
    n_attempts: int,
    seed: int,
) -> int:
    m = len(block_edges)
    if m < 2:
        return 0
    np.random.seed(seed)
    accepted = 0
    for _ in range(n_attempts):
        i = np.random.randint(0, m)
        j = np.random.randint(0, m)
        if i == j:
            continue
        e1 = block_edges[i]
        e2 = block_edges[j]
        u1 = edge_src[e1]
        u2 = edge_src[e2]
        if u1 == u2:
            continue
        v1 = indices[e1]
        v2 = indices[e2]
        if v1 == v2 or v2 == u1 or v1 == u2:
            continue
        dup = False
        for k in range(indptr[u1], indptr[u1 + 1]):
            if indices[k] == v2:
                dup = True
                break
        if dup:
            continue
        for k in range(indptr[u2], indptr[u2 + 1]):
            if indices[k] == v1:
                dup = True
                break
        if dup:
            continue
        indices[e1] = v2
        indices[e2] = v1
        accepted += 1
    return accepted


def build_compartment_rewired_graph(
    indptr: np.ndarray,
    orig_indices: np.ndarray,
    edge_src: np.ndarray,
    node_class: np.ndarray,
    node_comp: np.ndarray,
    orig_in_pos: np.ndarray,
    orig_in_neg: np.ndarray,
    w_raw_s: np.ndarray,
    protect_targets: set[int],
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Maslov-Sneppen rewiring strictly within (pre_comp, pre_ei, post_comp, post_ei) blocks,
    preserving exact E/I in-degrees, compartment-to-compartment block connectivity, and all direct MN9 edges."""
    n_nodes = len(indptr) - 1
    new_indices = orig_indices.copy()
    movable = np.ones(len(orig_indices), dtype=bool)
    for t in protect_targets:
        movable &= orig_indices != t

    pre_cls = node_class[edge_src]
    post_cls = node_class[orig_indices]
    pre_c = node_comp[edge_src]
    post_c = node_comp[orig_indices]

    # Group movable edges by (pre_c, pre_cls, post_c, post_cls) in one vectorized pass
    n_comp = int(node_comp.max()) + 1
    block_key = ((pre_c.astype(np.int64) * 3 + pre_cls.astype(np.int64)) * n_comp + post_c.astype(np.int64)) * 3 + post_cls.astype(np.int64)
    mov_eids = np.flatnonzero(movable).astype(np.int32)
    mov_keys = block_key[mov_eids]
    order = np.argsort(mov_keys, kind="mergesort")
    mov_eids = mov_eids[order]
    mov_keys = mov_keys[order]
    uniq_k, start_idx, counts = np.unique(mov_keys, return_index=True, return_counts=True)

    tot_att = 0
    tot_acc = 0
    for bi in range(len(uniq_k)):
        cnt = int(counts[bi])
        if cnt < 2:
            continue
        s0 = int(start_idx[bi])
        eids = mov_eids[s0:s0 + cnt]
        n_att = 2 * cnt
        acc = int(_rewire_block_swaps_numba(indptr, new_indices, eids, edge_src, n_att, seed * 131 + int(uniq_k[bi]) * 17))
        tot_att += n_att
        tot_acc += acc

    # Strict invariants verification
    assert not np.any(new_indices == edge_src), "Self-loop created!"
    assert np.array_equal(node_comp[new_indices], post_c), "Compartment block violated!"
    in_pos = np.zeros(n_nodes, dtype=np.int32)
    in_neg = np.zeros(n_nodes, dtype=np.int32)
    np.add.at(in_pos, new_indices[w_raw_s > 0], 1)
    np.add.at(in_neg, new_indices[w_raw_s < 0], 1)
    assert np.array_equal(in_pos, orig_in_pos), "Excitatory in-degree violated!"
    assert np.array_equal(in_neg, orig_in_neg), "Inhibitory in-degree violated!"
    for t in protect_targets:
        assert np.array_equal(orig_indices == t, new_indices == t), f"Protected target {t} mutated!"

    kept_movable = int((new_indices[movable] == orig_indices[movable]).sum())
    n_mov = int(movable.sum())
    return new_indices, {
        "seed": int(seed),
        "n_blocks": int(len(uniq_k)),
        "n_movable": n_mov,
        "n_protected": int(len(orig_indices) - n_mov),
        "total_attempts": int(tot_att),
        "total_accepted": int(tot_acc),
        "replaced_fraction_movable": float((n_mov - kept_movable) / max(1, n_mov)),
    }


def extract_top15_intermediates_all(
    indptr: np.ndarray,
    post_s: np.ndarray,
    pre_s: np.ndarray,
    w_raw: np.ndarray,
    node_class: np.ndarray,
    n_nodes: int,
    mn9_indices: list[int],
    inh_groups: dict[str, list[int]],
    node_tokens: list[set[str]],
) -> dict[str, dict]:
    """Exact match to mn9_disinhibition_state.py & mn9_malecns_benchmark.py:
    enumerates exact (v=I_1, ep=E_pre) 3-hop pairs sorted by (-bn, -prod), excluding MN9 targets."""
    mn9_set = set(mn9_indices)
    w_to_mn9 = np.zeros(n_nodes, dtype=np.float64)
    for m in mn9_indices:
        mask_m = post_s == m
        np.add.at(w_to_mn9, pre_s[mask_m], w_raw[mask_m].astype(np.float64))
    inh_to_mn9 = np.maximum(0.0, -w_to_mn9)
    exc_to_mn9 = np.maximum(0.0, w_to_mn9)

    inh_to_epre_has = np.zeros(n_nodes, dtype=np.bool_)
    for u in np.flatnonzero(node_class == 1):
        p0, p1 = int(indptr[u]), int(indptr[u + 1])
        for e in range(p0, p1):
            if w_raw[e] < 0 and exc_to_mn9[post_s[e]] > 0.0:
                inh_to_epre_has[u] = True
                break

    out: dict[str, dict] = {}
    for gname, gidxs in inh_groups.items():
        agg_out = defaultdict(float)
        for u in gidxs:
            for e in range(int(indptr[u]), int(indptr[u + 1])):
                agg_out[int(post_s[e])] += float(w_raw[e])
        p2_dis, p2_ant, p3_dis = [], [], []
        for v, w_xv in agg_out.items():
            if w_xv >= 0.0 or v in mn9_set:
                continue
            w_abs = -w_xv
            if inh_to_mn9[v] > 0.0:
                wi = float(inh_to_mn9[v])
                p2_dis.append((min(w_abs, wi), w_abs * wi, v))
            if exc_to_mn9[v] > 0.0:
                we = float(exc_to_mn9[v])
                p2_ant.append((min(w_abs, we), w_abs * we, v))
            if node_class[v] == 1 and inh_to_epre_has[v]:
                for e in range(int(indptr[v]), int(indptr[v + 1])):
                    ep = int(post_s[e])
                    we = float(exc_to_mn9[ep])
                    w_ve = float(-w_raw[e])
                    if we > 0.0 and w_ve > 0.0:
                        bn3 = min(w_abs, w_ve, we)
                        p3_dis.append((bn3, w_abs * w_ve * we, v))
        p2_dis.sort(key=lambda x: (-x[0], -x[1]))
        p2_ant.sort(key=lambda x: (-x[0], -x[1]))
        p3_dis.sort(key=lambda x: (-x[0], -x[1]))

        def uniq_types(lst, k=15):
            ts = set()
            for x in lst[:k]:
                ts.update(node_tokens[x[2]])
            return sorted(ts)

        out[gname] = {
            "n_cells": len(gidxs),
            "hop2_disinh_prod_sum": float(sum(x[1] for x in p2_dis)),
            "hop2_antag_prod_sum": float(sum(x[1] for x in p2_ant)),
            "hop2_net_prod": float(sum(x[1] for x in p2_dis) - sum(x[1] for x in p2_ant)),
            "hop3_disinh_prod_sum": float(sum(x[1] for x in p3_dis)),
            "top15_hop2_ipre_types": uniq_types(p2_dis, 15),
            "top15_hop2_antag_types": uniq_types(p2_ant, 15),
            "top15_hop3_i1_types": uniq_types(p3_dis, 15),
        }
    return out


# ===========================================================================
# PHASE 1: LOCAL STATIC (Test B1 density + Test D v783/BANC 44-ctrl + BH-FDR)
# ===========================================================================
def run_phase_local_static(out_static_path: Path, out_v783_ref_path: Path) -> None:
    sys.path.insert(0, str(BASE_DIR))
    import mn9_transfer_benchmark as mtb
    import mn9_banc_benchmark as mbb

    t0 = time.time()
    print("[local_static] Loading FlyWire v783 graph & annotations...", flush=True)
    comp783 = CACHE_DIR / "data/Completeness_783.csv"
    conn783 = CACHE_DIR / "data/Connectivity_783.parquet"
    ann783 = CACHE_DIR / "data/Supplemental_file1_neuron_annotations.tsv"
    g783 = mtb.load_graph_csr(comp783, conn783)
    fids783 = g783["flyids"]
    n783 = int(len(fids783))
    idx783 = g783["flyid2idx"]

    # Valid-NT incoming & outgoing absolute synapse sums per v783 neuron
    s_in_783 = np.zeros(n783, dtype=np.float64)
    s_out_783 = np.zeros(n783, dtype=np.float64)
    abs_w783 = np.abs(g783["w_raw_s"].astype(np.float64))
    np.add.at(s_in_783, g783["post_s"], abs_w783)
    np.add.at(s_out_783, g783["pre_s"], abs_w783)

    # Parse v783 annotations
    v783_ct_by_fid: dict[int, str] = {}
    v783_sc_by_fid: dict[int, str] = {}
    v783_dim_by_fid: dict[int, str] = {}
    v783_fru_by_fid: dict[int, str] = {}
    with ann783.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rid_s = row.get("root_id", "")
            if not rid_s or not rid_s.isdigit():
                continue
            rid = int(rid_s)
            if rid in idx783:
                ct = (row.get("cell_type") or "").strip() or (row.get("hemibrain_type") or "").strip()
                v783_ct_by_fid[rid] = ct
                v783_sc_by_fid[rid] = (row.get("super_class") or "").strip()
                v783_dim_by_fid[rid] = (row.get("dimorphism") or "").strip()
                v783_fru_by_fid[rid] = (row.get("fru_dsx") or "").strip()

    def v783_region(sc: str) -> str:
        if sc in ("optic", "visual_projection", "visual_centrifugal"):
            return "optic_lobe" if sc == "optic" else "central_brain"
        if "ascending" in sc:
            return "ventral_nerve_cord"
        return "central_brain"

    # Build v783 cell_type stats for MaleCNS & BANC comparison
    ct_to_idxs_783: dict[str, list[int]] = defaultdict(list)
    node_fw_type_783 = [""] * n783
    for i, fid in enumerate(fids783):
        ct = v783_ct_by_fid.get(int(fid), "")
        node_fw_type_783[i] = ct
        if ct:
            ct_to_idxs_783[ct].append(i)

    v783_ct_stats = {}
    for ct, idxs in ct_to_idxs_783.items():
        scs = Counter(v783_sc_by_fid.get(int(fids783[i]), "") for i in idxs)
        top_sc = scs.most_common(1)[0][0]
        reg = v783_region(top_sc)
        sin_vals = [float(s_in_783[i]) for i in idxs]
        sout_vals = [float(s_out_783[i]) for i in idxs]
        v783_ct_stats[ct] = {
            "n_cells": len(idxs),
            "region": reg,
            "super_class": top_sc,
            "mean_s_in": float(np.mean(sin_vals)),
            "total_s_in": float(np.sum(sin_vals)),
            "mean_s_out": float(np.mean(sout_vals)),
            "total_s_out": float(np.sum(sout_vals)),
        }

    # Load locked JSONs for SEZ 106, 44 neg inh controls, and BANC/v783 IDs
    v7_res = json.loads((BASE_DIR / "mn9_transfer_results.json").read_text())
    ba_res = json.loads((BASE_DIR / "mn9_banc_results.json").read_text())
    di_res = json.loads((BASE_DIR / "mn9_disinhibition_results.json").read_text())
    mc_res = json.loads((BASE_DIR / "mn9_malecns_results.json").read_text())
    manifest = json.loads(Path("/tmp/mn9_malecns_input_manifest.json").read_text())

    sez_v783_fids = {r["name"]: [int(x) for x in r["v783_ids"]] for r in v7_res["mapping_audit"]["per_type_sez106"]}
    neg_inh_44 = list(manifest["neg_inh_ctrl44"])
    assert len(neg_inh_44) == 44
    cands4 = ["tentacular", "phantom", "marge", "usnea"]
    all48_names = cands4 + neg_inh_44

    # Extract v783 top15 intermediates for all 48 inhibitory types
    mn9_783_fids = [
        int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_l_v783"]),
        int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_r_v783"]),
    ]
    mn9_783_idxs = [idx783[f] for f in mn9_783_fids]
    inh_groups_783 = {nm: [idx783[f] for f in sez_v783_fids[nm] if f in idx783] for nm in all48_names}
    print("[local_static] Extracting v783 top-15 intermediates for all 48 inhibitory types...", flush=True)
    v783_node_cls = g783["node_class"]
    node_tok_783 = [{ct} if ct else set() for ct in node_fw_type_783]
    top15_v783 = extract_top15_intermediates_all(
        g783["indptr"], g783["post_s"], g783["pre_s"], g783["w_raw_s"],
        v783_node_cls, n783, mn9_783_idxs, inh_groups_783, node_tok_783
    )

    # Now load BANC v888 graph & compute BANC/v783 incoming-synapse ratios + BANC 48-type top-15 intermediates
    print("[local_static] Loading BANC v888 graph for density & 44-control audit...", flush=True)
    bdir = CACHE_DIR / "data/banc"
    meta_tbl = pf.read_table(bdir / "banc_888_meta.feather")
    meta_cols = ["banc_888_id", "root_id", "supervoxel_id", "root_626", "root_850", "root_888", "root_890",
                 "proofread", "roughly_proofread", "super_class", "region", "cell_type", "fafb_cell_type"]
    m_rows = {}
    sv_to_bid, old_to_bid = {}, {}
    for vals in zip(*(meta_tbl[c].to_pylist() for c in meta_cols)):
        d = dict(zip(meta_cols, vals))
        bid = int(d["banc_888_id"])
        m_rows[bid] = d
        if d["supervoxel_id"] and str(d["supervoxel_id"]).isdigit():
            sv_to_bid[int(d["supervoxel_id"])] = bid
        for rc in ("root_id", "root_626", "root_850", "root_888", "root_890"):
            if d[rc] and str(d[rc]).isdigit():
                old_to_bid[int(d[rc])] = bid

    nt_csv = {}
    with (bdir / "banc_888_neurotransmitter_prediction_v2.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = int(row["root_id"])
            nv = row["neurotransmitter_predicted"]
            if nv not in ("", "NA"):
                nt_csv[rid] = nv

    # Reviewed FAFB -> BANC single-cell matches
    rev_fafb_to_banc: dict[int, set[int]] = defaultdict(set)
    with (bdir / "banc_fafb_reviewed_matches.csv.gz").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("valid") != "t":
                continue
            bs, svs, ms = row["pt_root_id"], row["pt_supervoxel_id"], row["match_id"]
            if not (bs.isdigit() and ms.isdigit()):
                continue
            b_raw, fid = int(bs), int(ms)
            rb = b_raw if b_raw in m_rows else (sv_to_bid.get(int(svs)) if svs.isdigit() else old_to_bid.get(b_raw))
            if rb is not None:
                md = m_rows[rb]
                if (md["proofread"] == "TRUE" or md["roughly_proofread"] == "TRUE") and md["super_class"] != "glia":
                    rev_fafb_to_banc[fid].add(rb)

    # Build main BANC graph (identical to mn9_banc_benchmark.py)
    main_bids = np.array(
        sorted(b for b, d in m_rows.items() if (d["proofread"] == "TRUE" or d["roughly_proofread"] == "TRUE") and d["super_class"] != "glia"),
        dtype=np.int64,
    )
    n_banc = int(len(main_bids))
    bid_to_idx = {int(b): i for i, b in enumerate(main_bids)}

    el = pf.read_table(bdir / "banc_888_edgelist_simple_v2.feather")
    r_pre = el["pre"].to_numpy().astype(np.int64)
    r_post = el["post"].to_numpy().astype(np.int64)
    r_cnt = el["count"].to_numpy().astype(np.int32)
    ns = r_pre != r_post
    r_pre, r_post, r_cnt = r_pre[ns], r_post[ns], r_cnt[ns]

    in_main_pre = np.isin(r_pre, main_bids)
    in_main_post = np.isin(r_post, main_bids)
    keep = in_main_pre & in_main_post
    r_pre, r_post, r_cnt = r_pre[keep], r_post[keep], r_cnt[keep]

    nt_ids = np.array(sorted(nt_csv.keys()), dtype=np.int64)
    nt_signs = np.array([nt_to_sign(nt_csv[int(b)]) for b in nt_ids], dtype=np.int8)
    pos_nt = np.minimum(np.searchsorted(nt_ids, r_pre), len(nt_ids) - 1)
    sgn = np.where(nt_ids[pos_nt] == r_pre, nt_signs[pos_nt], 0).astype(np.int8)
    valid = sgn != 0
    r_pre, r_post, r_cnt, sgn = r_pre[valid], r_post[valid], r_cnt[valid], sgn[valid]

    pre_b = np.searchsorted(main_bids, r_pre).astype(np.int32)
    post_b = np.searchsorted(main_bids, r_post).astype(np.int32)
    w_b = (r_cnt.astype(np.int32) * sgn.astype(np.int32)).astype(np.int32)
    order_b = np.lexsort((post_b, pre_b))
    pre_b, post_b, w_b = pre_b[order_b], post_b[order_b], w_b[order_b]
    indptr_b = np.zeros(n_banc + 1, dtype=np.int64)
    np.add.at(indptr_b[1:], pre_b, 1)
    np.cumsum(indptr_b, out=indptr_b)

    s_in_banc = np.zeros(n_banc, dtype=np.float64)
    s_out_banc = np.zeros(n_banc, dtype=np.float64)
    np.add.at(s_in_banc, post_b, np.abs(w_b).astype(np.float64))
    np.add.at(s_out_banc, pre_b, np.abs(w_b).astype(np.float64))

    # BANC / v783 density ratios
    # 1. Single-cell 1:1 reviewed matches
    sc_all, sc_cb, sc_ol = [], [], []
    for fid, bset in rev_fafb_to_banc.items():
        if fid not in idx783 or len(bset) != 1:
            continue
        bid = next(iter(bset))
        if bid not in bid_to_idx:
            continue
        si7 = float(s_in_783[idx783[fid]])
        sib = float(s_in_banc[bid_to_idx[bid]])
        if si7 > 0.0 and sib > 0.0:
            ratio = sib / si7
            sc_all.append(ratio)
            reg = v783_region(v783_sc_by_fid.get(fid, ""))
            if reg == "central_brain":
                sc_cb.append(ratio)
            elif reg == "optic_lobe":
                sc_ol.append(ratio)

    # 2. Homologous cell_type ratios (BANC vs v783)
    banc_ct_to_idxs: dict[str, list[int]] = defaultdict(list)
    node_fw_type_banc = [""] * n_banc
    for i, bid in enumerate(main_bids):
        md = m_rows[int(bid)]
        ft = (md["fafb_cell_type"] or "").strip() or (md["cell_type"] or "").strip()
        node_fw_type_banc[i] = ft
        for tok in split_tokens(md["fafb_cell_type"]) | split_tokens(md["cell_type"]):
            banc_ct_to_idxs[tok].append(i)

    ct_all_b, ct_cb_b, ct_ol_b = [], [], []
    for ct, st7 in v783_ct_stats.items():
        b_idxs = banc_ct_to_idxs.get(ct)
        if not b_idxs or st7["mean_s_in"] <= 0:
            continue
        mb_sin = float(np.mean([s_in_banc[i] for i in b_idxs]))
        if mb_sin > 0:
            r = mb_sin / st7["mean_s_in"]
            ct_all_b.append(r)
            if st7["region"] == "central_brain":
                ct_cb_b.append(r)
            elif st7["region"] == "optic_lobe":
                ct_ol_b.append(r)

    # 3. 105 mapped SEZ types in BANC
    banc_sez_ratios_in, banc_sez_ratios_out = [], []
    b_map_by_name = {r["name"]: r for r in ba_res["mapping_audit"]["per_type_sez106"]}
    for nm, fids in sez_v783_fids.items():
        b_ids = [int(x) for x in b_map_by_name[nm]["m1_banc_ids"]]
        v_idxs = [idx783[f] for f in fids if f in idx783]
        b_idxs = [bid_to_idx[b] for b in b_ids if b in bid_to_idx]
        if v_idxs and b_idxs:
            vi = float(np.mean([s_in_783[i] for i in v_idxs]))
            bi = float(np.mean([s_in_banc[i] for i in b_idxs]))
            vo = float(np.mean([s_out_783[i] for i in v_idxs]))
            bo = float(np.mean([s_out_banc[i] for i in b_idxs]))
            if vi > 0 and bi > 0:
                banc_sez_ratios_in.append(bi / vi)
            if vo > 0 and bo > 0:
                banc_sez_ratios_out.append(bo / vo)

    banc_density = {
        "single_cell_1to1_all": summarize_ratios(sc_all),
        "single_cell_1to1_central_brain": summarize_ratios(sc_cb),
        "single_cell_1to1_optic_lobe": summarize_ratios(sc_ol),
        "cell_type_per_cell_all": summarize_ratios(ct_all_b),
        "cell_type_per_cell_central_brain": summarize_ratios(ct_cb_b),
        "cell_type_per_cell_optic_lobe": summarize_ratios(ct_ol_b),
        "sez105_per_cell_s_in_ratio": summarize_ratios(banc_sez_ratios_in),
        "sez105_per_cell_s_out_ratio": summarize_ratios(banc_sez_ratios_out),
        "mn9_bilateral_mean_ratio": float(2283.5 / 5732.5),
    }
    print(f"  [B1 BANC/v783] single-cell CB median={banc_density['single_cell_1to1_central_brain']['median']:.4f} (IQR {banc_density['single_cell_1to1_central_brain']['q25']:.4f}-{banc_density['single_cell_1to1_central_brain']['q75']:.4f}), SEZ105 median={banc_density['sez105_per_cell_s_in_ratio']['median']:.4f}, MN9={banc_density['mn9_bilateral_mean_ratio']:.4f}", flush=True)

    # Extract BANC top-15 intermediates for all 48 inhibitory types
    mn9_b_idxs = [
        bid_to_idx[int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_l_banc"])],
        bid_to_idx[int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_r_banc"])],
    ]
    banc_node_cls = np.full(n_banc, 2, dtype=np.int8)
    for i, bid in enumerate(main_bids):
        s_nt = nt_to_sign(nt_csv.get(int(bid)))
        if s_nt > 0:
            banc_node_cls[i] = 0
        elif s_nt < 0:
            banc_node_cls[i] = 1
    inh_groups_banc = {nm: [bid_to_idx[int(b)] for b in b_map_by_name[nm]["m1_banc_ids"] if int(b) in bid_to_idx] for nm in all48_names}
    print("[local_static] Extracting BANC top-15 intermediates for all 48 inhibitory types...", flush=True)
    node_tok_banc = [split_tokens(m_rows[int(bid)]["fafb_cell_type"]) or split_tokens(m_rows[int(bid)]["cell_type"]) for bid in main_bids]
    top15_banc = extract_top15_intermediates_all(
        indptr_b, post_b, pre_b, w_b, banc_node_cls, n_banc, mn9_b_idxs, inh_groups_banc, node_tok_banc
    )

    # Compute BH-FDR across all pre-existing secondary/exploratory paired bootstrap tests across 3 brains
    sec_tests = [
        ("v783_all106_vs_2hReLU", float(v7_res["task1_activation_v783"]["all_106"]["bootstrap"]["static_2hop_relu_signed"]["ref_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("v783_all106_vs_rewProtEns", float(v7_res["task1_activation_v783"]["all_106"]["bootstrap"]["rewire_mn9_prot_ensemble_mean"]["ref_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("v783_nd83_vs_2hReLU", float(v7_res["task1_activation_v783"]["nondirect_v783"]["bootstrap"]["static_2hop_relu_signed"]["ref_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("v783_nd83_vs_rewProtEns", float(v7_res["task1_activation_v783"]["nondirect_v783"]["bootstrap"]["rewire_mn9_prot_ensemble_mean"]["ref_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("banc_all106_vs_2hReLU", float(ba_res["task1_activation_banc"]["M1_anat_scale"]["all_106"]["models"]["static_2hop_relu_signed"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("banc_all106_vs_rewProtEns", float(ba_res["task1_activation_banc"]["M1_anat_scale"]["all_106"]["rewire_mn9_protected"]["ensemble_mean_metrics"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("banc_nd90_vs_2hReLU", float(ba_res["task1_activation_banc"]["M1_anat_scale"]["nondirect_banc"]["models"]["static_2hop_relu_signed"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("banc_nd90_vs_rewProtEns", float(ba_res["task1_activation_banc"]["M1_anat_scale"]["nondirect_banc"]["rewire_mn9_protected"]["ensemble_mean_metrics"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("malecns_all106_vs_2hReLU", float(mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["all_106"]["models"]["static_2hop_relu_signed"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("malecns_all106_vs_rewProtEns", float(mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["all_106"]["rewire_mn9_protected"]["ensemble_mean_metrics"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("malecns_nd88_vs_2hReLU", float(mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["nondirect_malecns"]["models"]["static_2hop_relu_signed"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("malecns_nd88_vs_rewProtEns", float(mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["nondirect_malecns"]["rewire_mn9_protected"]["ensemble_mean_metrics"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("malecns_nd75_vs_2hReLU", float(mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["nondirect_common_3brains"]["models"]["static_2hop_relu_signed"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"])),
        ("malecns_nd75_vs_rewProtEns", float(mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["nondirect_common_3brains"]["rewire_mn9_protected"]["ensemble_mean_metrics"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"])),
    ]
    qvals = bh_fdr([p for _, p in sec_tests])
    fdr_table = [{"test": nm, "p_raw": p, "q_bh_fdr": q} for (nm, p), q in zip(sec_tests, qvals)]

    # Co-primary 3-brain summary table (all_106 & novel_95 at 50 Hz)
    co_primary = {
        "v783": {
            "all_106_50hz": float(v7_res["task1_activation_v783"]["all_106"]["bootstrap"]["real_lif_50hz"]["point"]["auprc"]),
            "all_106_ci95": [float(x) for x in v7_res["task1_activation_v783"]["all_106"]["bootstrap"]["real_lif_50hz"]["ci95"]["auprc"]],
            "novel_95_50hz": float(v7_res["task1_activation_v783"]["novel_95"]["bootstrap"]["real_lif_50hz"]["point"]["auprc"]),
            "novel_95_ci95": [float(x) for x in v7_res["task1_activation_v783"]["novel_95"]["bootstrap"]["real_lif_50hz"]["ci95"]["auprc"]],
            "novel_95_rewProt_mean": float(v7_res["task1_activation_v783"]["novel_95"]["rewire_mn9_prot_graphs"]["auprc_mean_sd_min_max"][0]),
            "novel_95_rewProt_wins": int(v7_res["task1_activation_v783"]["novel_95"]["rewire_mn9_prot_graphs"]["real_exceeds_count_auprc"]),
            "novel_95_2hReLU": float(v7_res["task1_activation_v783"]["novel_95"]["bootstrap"]["static_2hop_relu_signed"]["point"]["auprc"]),
        },
        "banc": {
            "all_106_50hz": float(ba_res["task1_activation_banc"]["M1_anat_scale"]["all_106"]["models"]["anat_50hz"]["auprc"]),
            "all_106_ci95": [float(x) for x in ba_res["task1_activation_banc"]["M1_anat_scale"]["all_106"]["models"]["anat_50hz"]["ci95"]["auprc"]],
            "novel_95_50hz": float(ba_res["task1_activation_banc"]["M1_anat_scale"]["novel_95"]["models"]["anat_50hz"]["auprc"]),
            "novel_95_ci95": [float(x) for x in ba_res["task1_activation_banc"]["M1_anat_scale"]["novel_95"]["models"]["anat_50hz"]["ci95"]["auprc"]],
            "novel_95_rewProt_mean": float(ba_res["task1_activation_banc"]["M1_anat_scale"]["novel_95"]["rewire_mn9_protected"]["per_graph_auprc_mean"]),
            "novel_95_rewProt_wins": int(ba_res["task1_activation_banc"]["M1_anat_scale"]["novel_95"]["rewire_mn9_protected"]["graph_level_rank_vs_primary"]["wins"]),
            "novel_95_2hReLU": float(ba_res["task1_activation_banc"]["M1_anat_scale"]["novel_95"]["models"]["static_2hop_relu_signed"]["auprc"]),
        },
        "malecns": {
            "all_106_50hz": float(mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["all_106"]["models"]["50hz"]["auprc"]),
            "all_106_ci95": [float(x) for x in mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["all_106"]["models"]["50hz"]["ci95"]["auprc"]],
            "novel_95_50hz": float(mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["novel_95"]["models"]["50hz"]["auprc"]),
            "novel_95_ci95": [float(x) for x in mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["novel_95"]["models"]["50hz"]["ci95"]["auprc"]],
            "novel_95_rewProt_mean": float(mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["novel_95"]["rewire_mn9_protected"]["per_graph_auprc_mean"]),
            "novel_95_rewProt_wins": int(mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["novel_95"]["rewire_mn9_protected"]["graph_level_rank_vs_primary"]["wins"]),
            "novel_95_2hReLU": float(mc_res["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["novel_95"]["models"]["static_2hop_relu_signed"]["auprc"]),
        },
    }

    # SEZ 106 v783 & BANC per-type S_in / S_out stats for the 6 male FPs & all 106
    sez_v783_banc_stats = {}
    for nm, fids in sez_v783_fids.items():
        vi = [idx783[f] for f in fids if f in idx783]
        bi = [bid_to_idx[int(b)] for b in b_map_by_name[nm]["m1_banc_ids"] if int(b) in bid_to_idx]
        sez_v783_banc_stats[nm] = {
            "v783_n": len(vi),
            "v783_mean_s_in": float(np.mean([s_in_783[i] for i in vi])) if vi else 0.0,
            "v783_mean_s_out": float(np.mean([s_out_783[i] for i in vi])) if vi else 0.0,
            "v783_dimorphism": sorted({v783_dim_by_fid.get(int(fids783[i]), "") for i in vi if v783_dim_by_fid.get(int(fids783[i]), "")}),
            "v783_fru_dsx": sorted({v783_fru_by_fid.get(int(fids783[i]), "") for i in vi if v783_fru_by_fid.get(int(fids783[i]), "")}),
            "banc_n": len(bi),
            "banc_mean_s_in": float(np.mean([s_in_banc[i] for i in bi])) if bi else 0.0,
            "banc_mean_s_out": float(np.mean([s_out_banc[i] for i in bi])) if bi else 0.0,
        }

    v783_ref = {
        "v783_celltype_stats": v783_ct_stats,
        "sez_v783_banc_stats": sez_v783_banc_stats,
        "all48_names": all48_names,
        "neg_inh_44": neg_inh_44,
        "top15_v783": top15_v783,
        "top15_banc": top15_banc,
    }
    out_v783_ref_path.write_text(json.dumps(v783_ref), encoding="utf-8")

    local_static = {
        "co_primary_endpoints_50hz": co_primary,
        "secondary_bh_fdr": fdr_table,
        "test_b_banc_vs_v783_density": banc_density,
        "v783_n_typed_cell_types": len(v783_ct_stats),
        "elapsed_sec": float(time.time() - t0),
    }
    out_static_path.write_text(json.dumps(local_static, indent=2), encoding="utf-8")
    print(f"[local_static] DONE in {time.time()-t0:.1f}s -> {out_static_path} & {out_v783_ref_path}", flush=True)


# ===========================================================================
# PHASE 2: LOCAL TEST C (v783 K=10 MN9-protected rewired anchor & activity-matched null)
# ===========================================================================
@njit(cache=False)
def _run_lif_v783_fast_with_cap(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    w_syn: float,
    exc_idx: np.ndarray,
    rate_hz: float,
    watch_idx: np.ndarray,
    n_trials: int,
    seed_base: int,
    max_brain_hz_cap: float,
) -> tuple[np.ndarray, float, int]:
    """Exact Brian2/Shiu LIF schedule (bit-identical to mn9_banc_benchmark.run_lif_lazy_exact)
    with whole-brain spike counting and early-exit cap for runaway seizure regimes."""
    dt = 0.1
    n_steps = 10000
    dly_steps = 18
    rfc_steps = 22
    t_mbr = 20.0
    tau = 5.0
    a_v = math.exp(-dt / t_mbr)
    a_g = math.exp(-dt / tau)
    b_vg = (tau / (t_mbr - tau)) * (a_v - a_g)
    v_th = 7.0
    w_poi = w_syn * 250.0
    p_poi = rate_hz * 1e-3 * dt

    pow_av = np.empty(n_steps + 2, dtype=np.float64)
    pow_ag = np.empty(n_steps + 2, dtype=np.float64)
    pow_bvg = np.empty(n_steps + 2, dtype=np.float64)
    pow_av[0] = 1.0
    pow_ag[0] = 1.0
    pow_bvg[0] = 0.0
    for k in range(1, n_steps + 2):
        pow_av[k] = pow_av[k - 1] * a_v
        pow_bvg[k] = a_v * pow_bvg[k - 1] + b_vg * pow_ag[k - 1]
        pow_ag[k] = pow_ag[k - 1] * a_g

    n_nodes = len(indptr) - 1
    n_watch = len(watch_idx)
    watch_counts = np.zeros((n_trials, n_watch), dtype=np.int32)

    is_exc = np.zeros(n_nodes, dtype=np.uint8)
    for k in range(len(exc_idx)):
        is_exc[exc_idx[k]] = 1

    u = np.zeros(n_nodes, dtype=np.float64)
    g = np.zeros(n_nodes, dtype=np.float64)
    last_spk = np.full(n_nodes, -100000, dtype=np.int32)
    last_upd = np.zeros(n_nodes, dtype=np.int32)
    in_active = np.zeros(n_nodes, dtype=np.uint8)
    active_list = np.empty(n_nodes, dtype=np.int32)
    next_active = np.empty(n_nodes, dtype=np.int32)
    spiked_now = np.empty(n_nodes, dtype=np.int32)
    touched = np.empty(n_nodes, dtype=np.int32)
    is_touched = np.zeros(n_nodes, dtype=np.uint8)

    ring_len = dly_steps + 1
    ring_spk = np.empty((ring_len, n_nodes), dtype=np.int32)
    ring_cnt = np.zeros(ring_len, dtype=np.int32)

    np.random.seed(seed_base)
    peak_gain = 0.160
    safe_th = 6.95
    cap_spikes_per_trial = int(max_brain_hz_cap * n_nodes)
    total_brain_spikes = 0
    capped_trials = 0
    trials_done = 0

    for tr in range(n_trials):
        n_act = 0
        n_touched = 0
        tr_spikes = 0
        hit_cap = False
        for slot in range(ring_len):
            ring_cnt[slot] = 0
        for k in range(len(exc_idx)):
            ii = exc_idx[k]
            if in_active[ii] == 0:
                in_active[ii] = 1
                active_list[n_act] = ii
                n_act += 1
            if is_touched[ii] == 0:
                is_touched[ii] = 1
                touched[n_touched] = ii
                n_touched += 1
                last_upd[ii] = -1

        for step in range(n_steps):
            n_spk_now = 0
            n_next = 0
            for a_i in range(n_act):
                i = active_list[a_i]
                if is_exc[i] != 0 or (step - last_spk[i]) >= rfc_steps:
                    ui = u[i]
                    gi = g[i]
                    ui_new = a_v * ui + b_vg * gi
                    gi_new = a_g * gi
                    u[i] = ui_new
                    g[i] = gi_new
                    last_upd[i] = step
                    if ui_new > v_th:
                        spiked_now[n_spk_now] = i
                        n_spk_now += 1
                        next_active[n_next] = i
                        n_next += 1
                    elif is_exc[i] != 0:
                        next_active[n_next] = i
                        n_next += 1
                    else:
                        if gi_new > 0.0 and ((ui_new if ui_new > 0.0 else 0.0) + peak_gain * gi_new >= safe_th):
                            next_active[n_next] = i
                            n_next += 1
                        else:
                            in_active[i] = 0
                else:
                    in_active[i] = 0

            tr_spikes += n_spk_now
            fut_slot = (step + dly_steps) % ring_len
            for s_i in range(n_spk_now):
                i = spiked_now[s_i]
                c_f = ring_cnt[fut_slot]
                ring_spk[fut_slot, c_f] = i
                ring_cnt[fut_slot] = c_f + 1
                for w_i in range(n_watch):
                    if watch_idx[w_i] == i:
                        watch_counts[tr, w_i] += 1

            if p_poi > 0.0:
                for k in range(len(exc_idx)):
                    if np.random.random() < p_poi:
                        u[exc_idx[k]] += w_poi

            cur_slot = step % ring_len
            cnt_deliv = ring_cnt[cur_slot]
            for s_i in range(cnt_deliv):
                src = ring_spk[cur_slot, s_i]
                p0 = indptr[src]
                p1 = indptr[src + 1]
                for e in range(p0, p1):
                    dst = indices[e]
                    if is_exc[dst] != 0 or (step - last_spk[dst]) >= rfc_steps:
                        if is_touched[dst] == 0:
                            is_touched[dst] = 1
                            touched[n_touched] = dst
                            n_touched += 1
                            last_upd[dst] = step
                        elif in_active[dst] == 0:
                            dk = step - last_upd[dst]
                            if dk > 0:
                                u_old = u[dst]
                                g_old = g[dst]
                                u[dst] = pow_av[dk] * u_old + pow_bvg[dk] * g_old
                                g[dst] = pow_ag[dk] * g_old
                                last_upd[dst] = step
                        g_new = g[dst] + weights[e]
                        g[dst] = g_new
                        if in_active[dst] == 0:
                            u_cur = u[dst]
                            if g_new > 0.0 and ((u_cur if u_cur > 0.0 else 0.0) + peak_gain * g_new >= safe_th):
                                in_active[dst] = 1
                                next_active[n_next] = dst
                                n_next += 1
            ring_cnt[cur_slot] = 0

            for s_i in range(n_spk_now):
                i = spiked_now[s_i]
                u[i] = 0.0
                g[i] = 0.0
                last_spk[i] = step
                last_upd[i] = step

            for a_i in range(n_next):
                active_list[a_i] = next_active[a_i]
            n_act = n_next

            if tr_spikes > cap_spikes_per_trial:
                hit_cap = True
                capped_trials += 1
                break

        for t_i in range(n_touched):
            i = touched[t_i]
            u[i] = 0.0
            g[i] = 0.0
            last_spk[i] = -100000
            in_active[i] = 0
            is_touched[i] = 0

        total_brain_spikes += tr_spikes
        trials_done = tr + 1
        if hit_cap and capped_trials >= 2:
            break

    mean_brain_hz = float(total_brain_spikes / max(1, trials_done * n_nodes))
    return watch_counts, mean_brain_hz, capped_trials


_TEST_C_STATE = {}


def _init_test_c_worker(npz_path: str):
    d = np.load(npz_path)
    _TEST_C_STATE["indptr"] = d["indptr"]
    _TEST_C_STATE["post_s"] = d["post_s"]
    _TEST_C_STATE["pre_s"] = d["pre_s"]
    _TEST_C_STATE["w_raw_s"] = d["w_raw_s"]
    _TEST_C_STATE["node_class"] = d["node_class"]
    _TEST_C_STATE["orig_in_pos"] = d["orig_in_pos"]
    _TEST_C_STATE["orig_in_neg"] = d["orig_in_neg"]
    _TEST_C_STATE["mn9_idx"] = d["mn9_idx"]
    _TEST_C_STATE["sugar_idx"] = d["sugar_idx"]
    # Warm up numba once
    dummy_w = np.zeros(1, dtype=np.float64)
    _run_lif_v783_fast_with_cap(
        np.array([0, 0], dtype=np.int64), np.zeros(0, dtype=np.int32), dummy_w, 0.275,
        np.array([0], dtype=np.int32), 50.0, np.array([0], dtype=np.int32), 1, 1, 15.0
    )


def _eval_one_v783_rewired_graph(job: dict) -> dict:
    sys.path.insert(0, str(BASE_DIR))
    import shiu_mn9_benchmark as smb

    t0 = time.time()
    k = int(job["k"])
    seed_rw = int(job["seed"])
    type_indices = [np.array(x, dtype=np.int32) for x in job["type_indices"]]
    target_mean_hz = float(job["target_mean_hz"])

    st = _TEST_C_STATE
    mn9_idx = st["mn9_idx"]
    prot_set = set(int(x) for x in mn9_idx)
    w_s_base = (st["w_raw_s"] * np.float32(0.275)).astype(np.float64)

    post_rw, rw_meta = smb.build_rewired_graph(
        st["indptr"], st["post_s"], w_s_base, st["pre_s"],
        st["node_class"], st["orig_in_pos"], st["orig_in_neg"], prot_set, seed_rw
    )
    indptr = st["indptr"]
    w_raw = st["w_raw_s"].astype(np.float64)

    # C1: Sugar 100 Hz anchor feasibility scan across W_syn
    sugar_grid = [0.275, 0.40, 0.55, 0.70, 0.90, 1.20, 1.60, 2.00, 2.50]
    sugar_scan = []
    for ws in sugar_grid:
        cnt, b_hz, cap = _run_lif_v783_fast_with_cap(
            indptr, post_rw, w_raw * ws, ws, st["sugar_idx"], 100.0, mn9_idx, 10, 42, 15.0
        )
        sugar_scan.append({
            "w_syn_mv": ws,
            "mn9_mean_hz": float(cnt.mean()),
            "brain_mean_hz": float(b_hz),
            "seizure_capped": bool(cap > 0),
        })

    # C2: Activity-matched null calibration via deterministic bisection on 106-type mean 50 Hz rate
    def eval_106_mean(ws: float, n_tr: int) -> tuple[np.ndarray, float, int]:
        rates = np.zeros(len(type_indices), dtype=np.float64)
        b_rates = []
        n_cap = 0
        for i, idxs in enumerate(type_indices):
            if idxs.size == 0:
                continue
            cnt, b_hz, cap = _run_lif_v783_fast_with_cap(
                indptr, post_rw, w_raw * ws, ws, idxs, 50.0, mn9_idx, n_tr, 50000 + i * 97, 15.0
            )
            rates[i] = float(cnt.mean())
            b_rates.append(b_hz)
            if cap > 0:
                n_cap += 1
        return rates, float(np.mean(b_rates)) if b_rates else 0.0, n_cap

    # Bracket [lo, hi] where lo=0.275 and step up until mean >= target_mean_hz or seizure
    lo_w = 0.275
    r_lo, b_lo, c_lo = eval_106_mean(lo_w, 5)
    m_lo = float(r_lo.mean())
    hi_w = lo_w
    m_hi = m_lo
    b_hi = b_lo
    c_hi = c_lo
    bisect_trace = [{"w_syn": lo_w, "mean_106_hz": m_lo, "brain_hz": b_lo, "n_capped": c_lo}]

    for cand_w in (0.45, 0.65, 0.85, 1.10, 1.40, 1.80, 2.20, 2.50):
        if m_hi >= target_mean_hz or c_hi >= 5:
            break
        r_c, b_c, c_c = eval_106_mean(cand_w, 5)
        m_c = float(r_c.mean())
        bisect_trace.append({"w_syn": cand_w, "mean_106_hz": m_c, "brain_hz": b_c, "n_capped": c_c})
        if m_c < target_mean_hz and c_c < 5:
            lo_w, m_lo = cand_w, m_c
            hi_w, m_hi, b_hi, c_hi = cand_w, m_c, b_c, c_c
        else:
            hi_w, m_hi, b_hi, c_hi = cand_w, m_c, b_c, c_c
            break

    # Refine within [lo_w, hi_w] via 6 bisection iterations (5 trials each); guard against seizure-capped hi_w
    if c_hi < 5 and abs(m_hi - target_mean_hz) < abs(m_lo - target_mean_hz):
        best_w = hi_w
        best_err = abs(m_hi - target_mean_hz)
    else:
        best_w = lo_w
        best_err = abs(m_lo - target_mean_hz)

    if hi_w > lo_w and m_hi >= target_mean_hz and c_hi < 5:
        a_w, b_w = lo_w, hi_w
        for _ in range(6):
            mid = 0.5 * (a_w + b_w)
            r_m, b_m, c_m = eval_106_mean(mid, 5)
            m_m = float(r_m.mean())
            bisect_trace.append({"w_syn": mid, "mean_106_hz": m_m, "brain_hz": b_m, "n_capped": c_m})
            if abs(m_m - target_mean_hz) < best_err and c_m == 0:
                best_err = abs(m_m - target_mean_hz)
                best_w = mid
            if m_m < target_mean_hz and c_m == 0:
                a_w = mid
            else:
                b_w = mid
            if abs(m_m - target_mean_hz) <= 0.04 * target_mean_hz and c_m == 0:
                best_w = mid
                break

    # Final 30-trial exact evaluation at best_w
    final_rates, final_brain_hz, final_capped = eval_106_mean(best_w, 30)
    return {
        "k": k,
        "seed": seed_rw,
        "sugar_anchor_scan": sugar_scan,
        "bisect_trace": bisect_trace,
        "w_syn_matched_mv": float(best_w),
        "mean_106_hz_matched": float(final_rates.mean()),
        "nonzero_106_matched": int((final_rates > 0.0).sum()),
        "brain_mean_hz_matched": float(final_brain_hz),
        "n_capped_types": int(final_capped),
        "rates_106_matched": [float(x) for x in final_rates],
        "elapsed_sec": float(time.time() - t0),
    }


def run_phase_local_test_c(out_path: Path, workers: int) -> None:
    sys.path.insert(0, str(BASE_DIR))
    import shiu_mn9_benchmark as smb
    import mn9_transfer_benchmark as mtb

    t0 = time.time()
    print(f"[local_test_c] Loading FlyWire v783 graph for K=10 MN9-protected activity-matched null (workers={workers})...", flush=True)
    comp783 = CACHE_DIR / "data/Completeness_783.csv"
    conn783 = CACHE_DIR / "data/Connectivity_783.parquet"
    g783 = mtb.load_graph_csr(comp783, conn783)
    fids783 = g783["flyids"]
    n783 = int(len(fids783))
    idx783 = g783["flyid2idx"]

    v7_res = json.loads((BASE_DIR / "mn9_transfer_results.json").read_text())
    ba_res = json.loads((BASE_DIR / "mn9_banc_results.json").read_text())
    mn9_idx = np.array([
        idx783[int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_l_v783"])],
        idx783[int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_r_v783"])],
    ], dtype=np.int32)

    # Exact 21 v783 sugar GRN IDs (20 unchanged + 1 supervoxel-mapped 720575940620900446 -> 720575940639259967)
    sugar_783_idx = np.array(
        [idx783[720575940639259967 if int(f) == 720575940620900446 else int(f)] for f in smb.SUGAR_GRN_FIDS],
        dtype=np.int32,
    )

    node_class = g783["node_class"]
    orig_in_pos = g783["orig_in_pos"]
    orig_in_neg = g783["orig_in_neg"]

    # Sanity-check our fast lazy-exact kernel against real v783 100 Hz sugar GRN (must match 58.30 Hz!)
    cnt_chk, b_chk, c_chk = _run_lif_v783_fast_with_cap(
        g783["indptr"], g783["post_s"], g783["w_raw_s"].astype(np.float64) * 0.275, 0.275,
        sugar_783_idx, 100.0, mn9_idx, 30, 42, 15.0
    )
    print(f"  [real v783 sugar 100 Hz sanity check] MN9_mean={cnt_chk.mean():.2f} Hz (ref 58.30 Hz), brain_hz={b_chk:.3f}", flush=True)
    assert abs(float(cnt_chk.mean()) - 58.30) < 0.01, f"Sanity check mismatch: {cnt_chk.mean():.4f} vs 58.30"

    npz_tmp = Path("/tmp/mn9_port_v783_graph.npz")
    np.savez(
        npz_tmp,
        indptr=g783["indptr"], post_s=g783["post_s"], pre_s=g783["pre_s"],
        w_raw_s=g783["w_raw_s"], node_class=node_class,
        orig_in_pos=orig_in_pos, orig_in_neg=orig_in_neg,
        mn9_idx=mn9_idx, sugar_idx=sugar_783_idx,
    )

    per_type = v7_res["per_type_task1"]
    v783_ids_by_name = {r["name"]: [int(x) for x in r["v783_ids"]] for r in v7_res["mapping_audit"]["per_type_sez106"]}
    type_indices = [[idx783[f] for f in v783_ids_by_name[r["name"]] if f in idx783] for r in per_type]
    y_bin = np.array([1 if r["y_max"] > 0.0 else 0 for r in per_type], dtype=np.int32)
    is_direct_v783 = np.array([bool(r["is_direct_1hop_v783"]) for r in per_type], dtype=bool)
    prior_pub = np.array([bool(r["prior_published"]) for r in per_type], dtype=bool)
    real_v783_50 = np.array([float(r["v783_real_lif_50hz_mean"]) for r in per_type], dtype=np.float64)
    target_mean_hz = float(real_v783_50.mean())

    jobs = [{"k": k, "seed": 9201 + k, "type_indices": type_indices, "target_mean_hz": target_mean_hz} for k in range(10)]
    results = [None] * 10
    with ProcessPoolExecutor(
        max_workers=min(workers, 10),
        mp_context=mp.get_context("spawn"),
        initializer=_init_test_c_worker,
        initargs=(str(npz_tmp),),
    ) as ex:
        futs = [ex.submit(_eval_one_v783_rewired_graph, j) for j in jobs]
        for fut in as_completed(futs):
            res = fut.result()
            results[res["k"]] = res
            r_arr = np.array(res["rates_106_matched"], dtype=np.float64)
            ap_all = stepwise_auprc(y_bin, r_arr)
            ap_nd = stepwise_auprc(y_bin[~is_direct_v783], r_arr[~is_direct_v783])
            ap_nov = stepwise_auprc(y_bin[~prior_pub], r_arr[~prior_pub])
            print(
                f"  [Test C rewProt#{res['k']}] W*={res['w_syn_matched_mv']:.4f} mV, mean={res['mean_106_hz_matched']:.3f} Hz "
                f"(nz={res['nonzero_106_matched']}/106, brain={res['brain_mean_hz_matched']:.3f} Hz) -> "
                f"AUPRC all106={ap_all:.4f}, nd83={ap_nd:.4f}, nov95={ap_nov:.4f} ({res['elapsed_sec']:.1f}s)",
                flush=True,
            )

    npz_tmp.unlink(missing_ok=True)

    # Aggregate Test C metrics
    real_all = stepwise_auprc(y_bin, real_v783_50)
    real_nd = stepwise_auprc(y_bin[~is_direct_v783], real_v783_50[~is_direct_v783])
    real_nov = stepwise_auprc(y_bin[~prior_pub], real_v783_50[~prior_pub])

    per_g_all, per_g_nd, per_g_nov = [], [], []
    mat_rates = []
    for res in results:
        r_arr = np.array(res["rates_106_matched"], dtype=np.float64)
        mat_rates.append(r_arr)
        per_g_all.append(stepwise_auprc(y_bin, r_arr))
        per_g_nd.append(stepwise_auprc(y_bin[~is_direct_v783], r_arr[~is_direct_v783]))
        per_g_nov.append(stepwise_auprc(y_bin[~prior_pub], r_arr[~prior_pub]))
    ens_rates = np.mean(mat_rates, axis=0)

    # Sugar anchor summary across all 10 rewired graphs
    sugar_by_w = defaultdict(list)
    brain_by_w = defaultdict(list)
    cap_by_w = defaultdict(int)
    for res in results:
        for row in res["sugar_anchor_scan"]:
            w = row["w_syn_mv"]
            sugar_by_w[w].append(row["mn9_mean_hz"])
            brain_by_w[w].append(row["brain_mean_hz"])
            if row["seizure_capped"]:
                cap_by_w[w] += 1

    out = {
        "real_v783_reference": {
            "target_mean_106_hz": target_mean_hz,
            "sugar_100hz_sanity_mn9_hz": float(cnt_chk.mean()),
            "all_106_auprc": real_all,
            "nondirect_83_auprc": real_nd,
            "novel_95_auprc": real_nov,
        },
        "c1_sugar_anchor_summary": [
            {
                "w_syn_mv": w,
                "mn9_mean_hz_across_10": float(np.mean(sugar_by_w[w])),
                "mn9_max_hz_across_10": float(np.max(sugar_by_w[w])),
                "brain_mean_hz_across_10": float(np.mean(brain_by_w[w])),
                "n_graphs_seizure_capped": int(cap_by_w[w]),
            }
            for w in sorted(sugar_by_w.keys())
        ],
        "c2_activity_matched_summary": {
            "w_syn_matched_mean_mv": float(np.mean([r["w_syn_matched_mv"] for r in results])),
            "w_syn_matched_min_max_mv": [float(min(r["w_syn_matched_mv"] for r in results)), float(max(r["w_syn_matched_mv"] for r in results))],
            "matched_mean_106_hz": float(np.mean([r["mean_106_hz_matched"] for r in results])),
            "matched_nonzero_mean": float(np.mean([r["nonzero_106_matched"] for r in results])),
            "matched_brain_hz_mean": float(np.mean([r["brain_mean_hz_matched"] for r in results])),
            "all_106": {
                "per_graph_auprc_list": per_g_all,
                "per_graph_auprc_mean": float(np.mean(per_g_all)),
                "per_graph_auprc_min_max": [float(min(per_g_all)), float(max(per_g_all))],
                "real_wins_out_of_10": int(sum(real_all > x for x in per_g_all)),
                "ensemble_mean_auprc": stepwise_auprc(y_bin, ens_rates),
            },
            "nondirect_83": {
                "per_graph_auprc_list": per_g_nd,
                "per_graph_auprc_mean": float(np.mean(per_g_nd)),
                "per_graph_auprc_min_max": [float(min(per_g_nd)), float(max(per_g_nd))],
                "real_wins_out_of_10": int(sum(real_nd > x for x in per_g_nd)),
                "ensemble_mean_auprc": stepwise_auprc(y_bin[~is_direct_v783], ens_rates[~is_direct_v783]),
            },
            "novel_95": {
                "per_graph_auprc_list": per_g_nov,
                "per_graph_auprc_mean": float(np.mean(per_g_nov)),
                "per_graph_auprc_min_max": [float(min(per_g_nov)), float(max(per_g_nov))],
                "real_wins_out_of_10": int(sum(real_nov > x for x in per_g_nov)),
                "ensemble_mean_auprc": stepwise_auprc(y_bin[~prior_pub], ens_rates[~prior_pub]),
            },
        },
        "per_graph_details": results,
        "elapsed_sec": float(time.time() - t0),
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[local_test_c] DONE in {time.time()-t0:.1f}s -> {out_path}", flush=True)


# ===========================================================================
# PHASE 3: COLAB A100 MALECNS (Tests A, B2/B3, D)
# ===========================================================================
_XORSHIFT_C_SRC = r"""
#include <stdint.h>
int gen_xorshift_kicks(
    uint64_t seed_opto, int n_trials, int n_exc, double p_poi, int n_steps,
    int32_t *out_step, int32_t *out_k, int32_t *out_tr, int max_out
) {
    const double inv_2_53 = 1.0 / 9007199254740992.0;
    int cnt = 0;
    for (int tr = 0; tr < n_trials; ++tr) {
        uint64_t s = seed_opto + (uint64_t)tr * 65537ULL + 0x9E3779B97F4A7C15ULL;
        if (s == 0ULL) s = 0x9E3779B97F4A7C15ULL;
        for (int t = 0; t < n_steps; ++t) {
            for (int k = 0; k < n_exc; ++k) {
                s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
                double rnd = (double)((s * 2685821657736338717ULL) >> 11) * inv_2_53;
                if (rnd < p_poi) {
                    if (cnt < max_out) {
                        out_step[cnt] = t; out_k[cnt] = k; out_tr[cnt] = tr;
                    }
                    cnt++;
                }
            }
        }
    }
    return cnt;
}
"""
_XORSHIFT_LIB = None


def _get_xorshift_lib():
    global _XORSHIFT_LIB
    if _XORSHIFT_LIB is None:
        c_dir = Path(tempfile.gettempdir()) / "mn9_port_xorshift_c"
        c_dir.mkdir(parents=True, exist_ok=True)
        so_path = c_dir / "libxorshift.so"
        if not so_path.is_file():
            c_path = c_dir / "xorshift.c"
            c_path.write_text(_XORSHIFT_C_SRC, encoding="utf-8")
            subprocess.check_call(["gcc", "-O3", "-shared", "-fPIC", str(c_path), "-o", str(so_path)])
        lib = ctypes.CDLL(str(so_path))
        lib.gen_xorshift_kicks.argtypes = [
            ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ]
        lib.gen_xorshift_kicks.restype = ctypes.c_int
        _XORSHIFT_LIB = lib
    return _XORSHIFT_LIB


def gen_opto_kicks(seed_opto: int, n_trials: int, exc_indices: list[int], rate_hz: float):
    n_exc = len(exc_indices)
    if n_exc == 0 or rate_hz <= 0.0:
        e = np.zeros(0, dtype=np.int32)
        return e, e, e
    lib = _get_xorshift_lib()
    p_poi = rate_hz * 1e-4
    est = int(max(1024, n_trials * n_exc * rate_hz * 1.5 + 1024))
    out_step = np.empty(est, dtype=np.int32)
    out_k = np.empty(est, dtype=np.int32)
    out_tr = np.empty(est, dtype=np.int32)
    cnt = lib.gen_xorshift_kicks(
        int(seed_opto), int(n_trials), int(n_exc), float(p_poi), 10000,
        out_step.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_k.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        out_tr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        est,
    )
    if cnt > est:
        est = cnt + 1024
        out_step = np.empty(est, dtype=np.int32)
        out_k = np.empty(est, dtype=np.int32)
        out_tr = np.empty(est, dtype=np.int32)
        cnt = lib.gen_xorshift_kicks(
            int(seed_opto), int(n_trials), int(n_exc), float(p_poi), 10000,
            out_step.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_k.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            out_tr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            est,
        )
    exc_arr = np.asarray(exc_indices, dtype=np.int32)
    return out_step[:cnt], exc_arr[out_k[:cnt]], out_tr[:cnt]


def ensure_gcs_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or dest.stat().st_size == 0:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        print(f"[download] {url} -> {dest}", flush=True)
        subprocess.check_call(["curl", "-fsSL", "--retry", "3", "-o", str(tmp), url])
        tmp.replace(dest)


def run_phase_colab_malecns(
    manifest_path: Path, v783_ref_path: Path, out_path: Path, batch_types: int = 12, test_d_only: bool = False, rev_mode: bool = False
) -> None:
    import torch

    t_all = time.time()
    assert torch.cuda.is_available(), "CUDA required for colab_malecns"
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[colab_malecns] GPU={gpu_name} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB), batch_types={batch_types}", flush=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v783_ref = json.loads(v783_ref_path.read_text(encoding="utf-8"))

    cache_dir = Path("/tmp/malecns_port_cache")
    try:
        paths = {}
        for k, fname in MALECNS_FILES.items():
            p = cache_dir / fname
            ensure_gcs_file(f"{GCS_BASE}/{fname}", p)
            paths[k] = p

        ann_tbl = pf.read_table(paths["annotations"])
        nt_tbl = pf.read_table(paths["neurotransmitters"])
        w_tbl = pf.read_table(paths["weights"])

        # Parse annotations (INCLUDING fruDsx, dimorphism, statusLabel, matchingNotes!)
        ann_cols = ann_tbl.column_names
        c_bid = pick_col(ann_cols, ["bodyId", "bodyid", "body"])
        c_type = pick_col(ann_cols, ["type"])
        c_fw = pick_col(ann_cols, ["flywireType", "flywire_type"], required=False)
        c_sc = pick_col(ann_cols, ["superclass", "super_class"], required=False)
        c_cls = pick_col(ann_cols, ["class"], required=False)
        c_side = pick_col(ann_cols, ["somaSide", "rootSide", "side"], required=False)
        c_status = pick_col(ann_cols, ["status"], required=False)
        c_slbl = pick_col(ann_cols, ["statusLabel"], required=False)
        c_dim = pick_col(ann_cols, ["dimorphism"], required=False)
        c_fru = pick_col(ann_cols, ["fruDsx", "fru_dsx"], required=False)
        c_mn = pick_col(ann_cols, ["matchingNotes", "matching_notes"], required=False)

        nt_cols = nt_tbl.column_names
        c_nt_b = pick_col(nt_cols, ["body", "bodyId"])
        c_nt_cons = pick_col(nt_cols, ["consensus_nt"])
        c_nt_pred = pick_col(nt_cols, ["predicted_nt"])
        c_nt_ctpred = pick_col(nt_cols, ["celltype_predicted_nt"], required=False)

        cons_nt_map, pred_nt_map, ctpred_nt_map = {}, {}, {}
        for vals in zip(
            nt_tbl[c_nt_b].to_pylist(),
            nt_tbl[c_nt_cons].to_pylist(),
            nt_tbl[c_nt_pred].to_pylist(),
            nt_tbl[c_nt_ctpred].to_pylist() if c_nt_ctpred else [None] * len(nt_tbl),
        ):
            if vals[0] is None:
                continue
            b = int(vals[0])
            cons_nt_map[b] = str(vals[1]).strip().lower() if vals[1] is not None else ""
            pred_nt_map[b] = str(vals[2]).strip().lower() if vals[2] is not None else ""
            ctpred_nt_map[b] = str(vals[3]).strip().lower() if vals[3] is not None else ""

        def classify_region(sc: str) -> str:
            s = sc.lower()
            if s.startswith("ol_"):
                return "optic_lobe"
            if s.startswith("vnc_") or "ascending" in s:
                return "ventral_nerve_cord"
            return "central_brain"

        def s_str(x):
            return "" if x is None else str(x).strip()

        traced_meta = {}
        for vals in zip(
            ann_tbl[c_bid].to_pylist(),
            ann_tbl[c_type].to_pylist(),
            ann_tbl[c_fw].to_pylist() if c_fw else [None] * len(ann_tbl),
            ann_tbl[c_sc].to_pylist() if c_sc else [None] * len(ann_tbl),
            ann_tbl[c_cls].to_pylist() if c_cls else [None] * len(ann_tbl),
            ann_tbl[c_side].to_pylist() if c_side else [None] * len(ann_tbl),
            ann_tbl[c_status].to_pylist() if c_status else [None] * len(ann_tbl),
            ann_tbl[c_slbl].to_pylist() if c_slbl else [None] * len(ann_tbl),
            ann_tbl[c_dim].to_pylist() if c_dim else [None] * len(ann_tbl),
            ann_tbl[c_fru].to_pylist() if c_fru else [None] * len(ann_tbl),
            ann_tbl[c_mn].to_pylist() if c_mn else [None] * len(ann_tbl),
        ):
            if vals[0] is None:
                continue
            b = int(vals[0])
            st = s_str(vals[6])
            sc = s_str(vals[3])
            cl = s_str(vals[4])
            if st != "Traced" or "glia" in sc.lower() or "glia" in cl.lower():
                continue
            traced_meta[b] = {
                "id": b, "type": s_str(vals[1]), "flywireType": s_str(vals[2]),
                "superclass": sc, "class": cl, "side": s_str(vals[5]),
                "statusLabel": s_str(vals[7]), "dimorphism": s_str(vals[8]),
                "fruDsx": s_str(vals[9]), "matchingNotes": s_str(vals[10]),
                "region": classify_region(sc),
                "cons_nt": cons_nt_map.get(b, ""), "pred_nt": pred_nt_map.get(b, ""),
                "ctpred_nt": ctpred_nt_map.get(b, ""),
            }

        all_bids = np.array(sorted(traced_meta.keys()), dtype=np.int64)
        n_nodes = int(len(all_bids))
        bid_to_idx = {int(b): i for i, b in enumerate(all_bids)}
        node_meta = [traced_meta[int(b)] for b in all_bids]
        mn9_l_idx, mn9_r_idx = bid_to_idx[10331], bid_to_idx[16949]
        mn9_indices = [mn9_l_idx, mn9_r_idx]

        # Build deduplicated unsigned raw non-self edge list ONCE
        w_cols = w_tbl.column_names
        c_pre = pick_col(w_cols, ["body_pre", "bodyId_pre", "pre"])
        c_post = pick_col(w_cols, ["body_post", "bodyId_post", "post"])
        c_w = pick_col(w_cols, ["weight", "count"])
        raw_pre = w_tbl[c_pre].to_numpy().astype(np.int64)
        raw_post = w_tbl[c_post].to_numpy().astype(np.int64)
        raw_w = w_tbl[c_w].to_numpy().astype(np.int32)

        def map_ids_vec(arr):
            pos = np.searchsorted(all_bids, arr)
            pos_c = np.minimum(pos, n_nodes - 1)
            hit = all_bids[pos_c] == arr
            return np.where(hit, pos_c, -1).astype(np.int32)

        p_idx = map_ids_vec(raw_pre)
        q_idx = map_ids_vec(raw_post)
        ok = (p_idx >= 0) & (q_idx >= 0) & (p_idx != q_idx) & (raw_w > 0)
        p_idx, q_idx, raw_w = p_idx[ok], q_idx[ok], raw_w[ok]

        keys = p_idx.astype(np.int64) * np.int64(n_nodes) + q_idx.astype(np.int64)
        ord_k = np.argsort(keys, kind="mergesort")
        keys_s, w_s = keys[ord_k], raw_w[ord_k]
        uniq_k, first_pos = np.unique(keys_s, return_index=True)
        if len(uniq_k) < len(keys_s):
            w_uniq = np.add.reduceat(w_s.astype(np.int64), first_pos).astype(np.int32)
        else:
            w_uniq = w_s.astype(np.int32)
        u_pre = (uniq_k // np.int64(n_nodes)).astype(np.int32)
        u_post = (uniq_k % np.int64(n_nodes)).astype(np.int32)
        print(f"  MaleCNS raw non-self graph: N={n_nodes:,}, E_uniq={len(u_pre):,}", flush=True)

        # Deterministic SEZ 106 & SELK mapping (identical to mn9_malecns_benchmark.py)
        by_fw: dict[str, list[int]] = defaultdict(list)
        by_ty: dict[str, list[int]] = defaultdict(list)
        for i, m in enumerate(node_meta):
            for tok in split_tokens(m["flywireType"]):
                by_fw[tok].append(i)
            for tok in split_tokens(m["type"]):
                by_ty[tok].append(i)

        sez_list = manifest["sez106"]
        type_indices = []
        sez_idx_by_name = {}
        for r in sez_list:
            hits = set()
            for ct in r["v783_cell_types"]:
                hits.update(by_fw.get(ct, []))
                hits.update(by_ty.get(ct, []))
            idxs = sorted(hits)
            type_indices.append(idxs)
            sez_idx_by_name[r["name"]] = idxs
        selk_idxs = sorted(set(by_ty.get("DNg68", [])) | set(by_fw.get("DNg68", [])))
        assert len(selk_idxs) == 2

        def build_signed_csr(node_signs: np.ndarray):
            e_sgn = node_signs[u_pre]
            keep_e = e_sgn != 0
            pre_s = u_pre[keep_e]
            post_s = u_post[keep_e]
            w_raw_s = (w_uniq[keep_e] * e_sgn[keep_e]).astype(np.int32)
            indptr = np.zeros(n_nodes + 1, dtype=np.int64)
            np.add.at(indptr[1:], pre_s, 1)
            np.cumsum(indptr, out=indptr)
            n_cls = np.where(node_signs > 0, 0, np.where(node_signs < 0, 1, 2)).astype(np.int8)
            s_in = np.zeros(n_nodes, dtype=np.float64)
            s_out = np.zeros(n_nodes, dtype=np.float64)
            np.add.at(s_in, post_s, np.abs(w_raw_s).astype(np.float64))
            np.add.at(s_out, pre_s, np.abs(w_raw_s).astype(np.float64))
            return indptr, pre_s, post_s, w_raw_s, n_cls, s_in, s_out

        # Base consensus_nt signs
        base_signs = np.array([nt_to_sign(m["cons_nt"]) for m in node_meta], dtype=np.int8)
        indptr0, pre0, post0, w0, ncls0, sin0, sout0 = build_signed_csr(base_signs)
        assert abs(sin0[mn9_l_idx] - 5876.0) < 1e-6

        # ---------------------------------------------------------------
        # TEST D: Full 44-control 3-brain conservation null distribution!
        # ---------------------------------------------------------------
        test_d_summary = {}
        if not rev_mode:
            print("[colab_malecns] Running Test D: 44-control 3-brain conservation null distribution...", flush=True)
            node_tok_m = [split_tokens(m["flywireType"]) | split_tokens(m["type"]) for m in node_meta]
            all48 = v783_ref["all48_names"]
            neg44 = v783_ref["neg_inh_44"]
            inh_groups_m = {nm: sez_idx_by_name[nm] for nm in all48}
            top15_m = extract_top15_intermediates_all(
                indptr0, post0, pre0, w0, ncls0, n_nodes, mn9_indices, inh_groups_m, node_tok_m
            )

            # Compute percentiles of all 48 against the 44 negative controls in each brain
            t15_v = v783_ref["top15_v783"]
            t15_b = v783_ref["top15_banc"]

            def pct_against_44(brain_dict: dict, metric: str, name: str) -> float:
                val = brain_dict[name][metric]
                ctrl_vals = [brain_dict[c][metric] for c in neg44]
                return float(100.0 * sum(1 for x in ctrl_vals if x < val) / len(ctrl_vals))

            per_type_d = {}
            for nm in all48:
                p2n_v = pct_against_44(t15_v, "hop2_net_prod", nm)
                p2n_b = pct_against_44(t15_b, "hop2_net_prod", nm)
                p2n_m = pct_against_44(top15_m, "hop2_net_prod", nm)
                p3_v = pct_against_44(t15_v, "hop3_disinh_prod_sum", nm)
                p3_b = pct_against_44(t15_b, "hop3_disinh_prod_sum", nm)
                p3_m = pct_against_44(top15_m, "hop3_disinh_prod_sum", nm)

                s2_v, s2_b, s2_m = set(t15_v[nm]["top15_hop2_ipre_types"]), set(t15_b[nm]["top15_hop2_ipre_types"]), set(top15_m[nm]["top15_hop2_ipre_types"])
                s3_v, s3_b, s3_m = set(t15_v[nm]["top15_hop3_i1_types"]), set(t15_b[nm]["top15_hop3_i1_types"]), set(top15_m[nm]["top15_hop3_i1_types"])
                vb2 = s2_v & s2_b
                vbm2 = vb2 & s2_m
                vb3 = s3_v & s3_b
                vbm3 = vb3 & s3_m
                per_type_d[nm] = {
                    "is_candidate": nm not in neg44,
                    "hop2_net_pct_3brains": [p2n_v, p2n_b, p2n_m],
                    "hop3_disinh_pct_3brains": [p3_v, p3_b, p3_m],
                    "n_vb_hop2": len(vb2), "n_all3_hop2": len(vbm2),
                    "ret_3rd_hop2": float(len(vbm2) / len(vb2)) if vb2 else 0.0,
                    "n_vb_hop3": len(vb3), "n_all3_hop3": len(vbm3),
                    "ret_3rd_hop3": float(len(vbm3) / len(vb3)) if vb3 else 0.0,
                    "shared_all3_hop2": sorted(vbm2),
                    "shared_all3_hop3": sorted(vbm3),
                }

            def null_rate(pred_fn):
                passers = [c for c in neg44 if pred_fn(per_type_d[c])]
                k = len(passers)
                return {"k_pass_of_44": k, "rate": float(k / 44.0), "p_emp_k1_over_45": float((k + 1) / 45.0), "passers": passers}

            test_d_summary = {
                "n_controls": 44,
                "candidates": {c: per_type_d[c] for c in ("tentacular", "phantom", "marge", "usnea")},
                "null_criteria_among_44_neg_inh": {
                    "all3_hop3_ge_80pct": null_rate(lambda r: min(r["hop3_disinh_pct_3brains"]) >= 80.0),
                    "all3_hop3_ge_90pct": null_rate(lambda r: min(r["hop3_disinh_pct_3brains"]) >= 90.0),
                    "all3_hop2net_ge_90pct": null_rate(lambda r: min(r["hop2_net_pct_3brains"]) >= 90.0),
                    "shared_all3_hop2_ge_4": null_rate(lambda r: r["n_all3_hop2"] >= 4),
                    "shared_all3_hop2_ge_5": null_rate(lambda r: r["n_all3_hop2"] >= 5),
                    "shared_all3_hop3_ge_4": null_rate(lambda r: r["n_all3_hop3"] >= 4),
                    "shared_all3_hop3_ge_5": null_rate(lambda r: r["n_all3_hop3"] >= 5),
                    "ret100_hop2_with_vb_ge_4": null_rate(lambda r: r["n_vb_hop2"] >= 4 and r["ret_3rd_hop2"] == 1.0),
                    "ret100_hop3_with_vb_ge_4": null_rate(lambda r: r["n_vb_hop3"] >= 4 and r["ret_3rd_hop3"] == 1.0),
                    "joint_phantom_level_h3ge80_shared3h_ge4": null_rate(lambda r: min(r["hop3_disinh_pct_3brains"]) >= 80.0 and r["n_all3_hop3"] >= 4),
                    "joint_tentacular_level_h3ge90_h2nge90_shared2h5_3h5": null_rate(
                        lambda r: min(r["hop3_disinh_pct_3brains"]) >= 90.0
                        and min(r["hop2_net_pct_3brains"]) >= 90.0
                        and r["n_all3_hop2"] >= 5
                        and r["n_all3_hop3"] >= 5
                    ),
                    "joint_tentacular_strict_100pct_ret": null_rate(
                        lambda r: min(r["hop3_disinh_pct_3brains"]) >= 90.0
                        and min(r["hop2_net_pct_3brains"]) >= 90.0
                        and r["n_all3_hop2"] >= 6
                        and r["n_all3_hop3"] >= 5
                        and r["ret_3rd_hop2"] == 1.0
                        and r["ret_3rd_hop3"] == 1.0
                    ),
                },
                "neg44_distributions": {
                    "n_all3_hop2_median_iqr_max": [
                        float(np.median([per_type_d[c]["n_all3_hop2"] for c in neg44])),
                        float(np.percentile([per_type_d[c]["n_all3_hop2"] for c in neg44], 25)),
                        float(np.percentile([per_type_d[c]["n_all3_hop2"] for c in neg44], 75)),
                        int(max(per_type_d[c]["n_all3_hop2"] for c in neg44)),
                    ],
                    "n_all3_hop3_median_iqr_max": [
                        float(np.median([per_type_d[c]["n_all3_hop3"] for c in neg44])),
                        float(np.percentile([per_type_d[c]["n_all3_hop3"] for c in neg44], 25)),
                        float(np.percentile([per_type_d[c]["n_all3_hop3"] for c in neg44], 75)),
                        int(max(per_type_d[c]["n_all3_hop3"] for c in neg44)),
                    ],
                },
                "per_control_table": {c: per_type_d[c] for c in neg44},
            }
            print("  [Test D] 44-ctrl pass counts:", {k: v["k_pass_of_44"] for k, v in test_d_summary["null_criteria_among_44_neg_inh"].items()}, flush=True)
            if test_d_only:
                out_path.write_text(json.dumps({"test_d_44ctrl_three_brain_null": test_d_summary}, indent=2), encoding="utf-8")
                print(f"[colab_malecns] test_d_only DONE in {time.time()-t_all:.1f}s -> {out_path}", flush=True)
                return

        # ---------------------------------------------------------------
        # TEST B2: MaleCNS vs v783 matched-neuron density + 6 FP dimorphism audit
        # ---------------------------------------------------------------
        print("[colab_malecns] Running Test B2: MaleCNS/v783 incoming-synapse density & dimorphism audit...", flush=True)
        v783_ct_stats = v783_ref["v783_celltype_stats"]
        sez_vb_stats = v783_ref["sez_v783_banc_stats"]

        r_all_pc, r_cb_pc, r_ol_pc = [], [], []
        r_all_tot, r_cb_tot = [], []
        r_cb_sout_pc = []
        for ct, st7 in v783_ct_stats.items():
            m_idxs = sorted(set(by_fw.get(ct, [])) | set(by_ty.get(ct, [])))
            if not m_idxs or st7["mean_s_in"] <= 0:
                continue
            ms_in = float(np.mean([sin0[i] for i in m_idxs]))
            mt_in = float(np.sum([sin0[i] for i in m_idxs]))
            ms_out = float(np.mean([sout0[i] for i in m_idxs]))
            if ms_in > 0:
                rpc = ms_in / st7["mean_s_in"]
                rtot = mt_in / st7["total_s_in"]
                r_all_pc.append(rpc)
                r_all_tot.append(rtot)
                if st7["region"] == "central_brain":
                    r_cb_pc.append(rpc)
                    r_cb_tot.append(rtot)
                    if st7["mean_s_out"] > 0 and ms_out > 0:
                        r_cb_sout_pc.append(ms_out / st7["mean_s_out"])
                elif st7["region"] == "optic_lobe":
                    r_ol_pc.append(rpc)

        sez_m_in_r, sez_m_out_r = [], []
        all106_dim_fru = []
        fp6_audit = {}
        fp6_names = ("peacock", "puddle", "spirit", "mime", "peahen", "crab")
        for r in sez_list:
            nm = r["name"]
            idxs = sez_idx_by_name[nm]
            vb = sez_vb_stats[nm]
            ms_in = float(np.mean([sin0[i] for i in idxs])) if idxs else 0.0
            ms_out = float(np.mean([sout0[i] for i in idxs])) if idxs else 0.0
            if vb["v783_mean_s_in"] > 0 and ms_in > 0:
                sez_m_in_r.append(ms_in / vb["v783_mean_s_in"])
            if vb["v783_mean_s_out"] > 0 and ms_out > 0:
                sez_m_out_r.append(ms_out / vb["v783_mean_s_out"])
            dims = sorted({node_meta[i]["dimorphism"] for i in idxs if node_meta[i]["dimorphism"]})
            frus = sorted({node_meta[i]["fruDsx"] for i in idxs if node_meta[i]["fruDsx"]})
            slbls = dict(Counter(node_meta[i]["statusLabel"] for i in idxs))
            mnotes = sorted({node_meta[i]["matchingNotes"] for i in idxs if node_meta[i]["matchingNotes"]})
            if dims or frus:
                all106_dim_fru.append({"name": nm, "y_bin": r["y_bin"], "n_malecns": len(idxs), "dimorphism": dims, "fruDsx": frus})
            if nm in fp6_names:
                fp6_audit[nm] = {
                    "y_bin": r["y_bin"], "n_v783": vb["v783_n"], "n_banc": vb["banc_n"], "n_malecns": len(idxs),
                    "dimorphism": dims, "fruDsx": frus, "statusLabel": slbls, "matchingNotes": mnotes[:5],
                    "v783_mean_s_in": vb["v783_mean_s_in"], "banc_mean_s_in": vb["banc_mean_s_in"], "malecns_mean_s_in": ms_in,
                    "s_in_ratio_malecns_over_v783": float(ms_in / vb["v783_mean_s_in"]) if vb["v783_mean_s_in"] > 0 else None,
                    "v783_mean_s_out": vb["v783_mean_s_out"], "banc_mean_s_out": vb["banc_mean_s_out"], "malecns_mean_s_out": ms_out,
                    "s_out_ratio_malecns_over_v783": float(ms_out / vb["v783_mean_s_out"]) if vb["v783_mean_s_out"] > 0 else None,
                    "v783_dimorphism": vb["v783_dimorphism"], "v783_fru_dsx": vb["v783_fru_dsx"],
                }

        malecns_density = {
            "cell_type_per_cell_all": summarize_ratios(r_all_pc),
            "cell_type_per_cell_central_brain": summarize_ratios(r_cb_pc),
            "cell_type_per_cell_optic_lobe": summarize_ratios(r_ol_pc),
            "cell_type_total_all": summarize_ratios(r_all_tot),
            "cell_type_total_central_brain": summarize_ratios(r_cb_tot),
            "cell_type_per_cell_cb_s_out": summarize_ratios(r_cb_sout_pc),
            "sez104_per_cell_s_in_ratio": summarize_ratios(sez_m_in_r),
            "sez104_per_cell_s_out_ratio": summarize_ratios(sez_m_out_r),
            "mn9_l_vs_v783_l_ratio": float(5876.0 / 5886.0),
            "mn9_l_vs_v783_mean_ratio": float(5876.0 / 5732.5),
            "all106_any_dimorphism_or_fruDsx": all106_dim_fru,
            "male_fp6_audit": fp6_audit,
        }
        r_med_cb = float(malecns_density["cell_type_per_cell_central_brain"]["median"])
        w_syn_density = float(0.275 / r_med_cb)
        malecns_density["r_med_central_brain"] = r_med_cb
        malecns_density["w_syn_density_mv"] = w_syn_density
        print(f"  [B2 MaleCNS/v783] CB median S_in ratio={r_med_cb:.4f} (IQR {malecns_density['cell_type_per_cell_central_brain']['q25']:.4f}-{malecns_density['cell_type_per_cell_central_brain']['q75']:.4f}) -> W_syn_density={w_syn_density:.6f} mV", flush=True)
        print(f"  [B2 dimorphism/fruDsx in all 106]: {all106_dim_fru}", flush=True)

        # ---------------------------------------------------------------
        # GPU SIMULATOR (exact FP64 CSR/SpMM kernel verbatim from mn9_malecns_benchmark.py)
        # ---------------------------------------------------------------
        from scipy import sparse as sp_sparse

        class GPUGraph:
            def __init__(self, indptr, post_s, w_raw_s, device):
                self.n_nodes = len(indptr) - 1
                self.device = device
                indptr_i64 = indptr.astype(np.int64)
                deg_np = indptr_i64[1:] - indptr_i64[:-1]
                self.indptr = torch.from_numpy(indptr_i64).to(device)
                self.deg = torch.from_numpy(deg_np).to(device)
                self.post_s = torch.from_numpy(post_s.astype(np.int64)).to(device)
                self.w_raw_f64 = torch.from_numpy(w_raw_s.astype(np.float64)).to(device)
                self._w_t_csr = None
                self._indptr_np = indptr
                self._post_np = post_s
                self._w_raw_np = w_raw_s

            def get_wt_csr(self):
                if self._w_t_csr is None:
                    m_csc = sp_sparse.csc_matrix(
                        (self._w_raw_np.astype(np.float64), self._post_np, self._indptr_np),
                        shape=(self.n_nodes, self.n_nodes),
                    )
                    wt_csr = m_csc.tocsr()
                    crow = torch.from_numpy(wt_csr.indptr.astype(np.int64)).to(self.device)
                    col = torch.from_numpy(wt_csr.indices.astype(np.int64)).to(self.device)
                    val = torch.from_numpy(wt_csr.data.astype(np.float64)).to(self.device)
                    self._w_t_csr = torch.sparse_csr_tensor(crow, col, val, size=(self.n_nodes, self.n_nodes), device=self.device)
                return self._w_t_csr

        def _lif_step_novb(u, g, unlock_step, step_i32, zero_f64, a_v: float, b_vg: float, a_g: float, v_th: float):
            elig = unlock_step <= step_i32
            u_new = u * a_v + g * b_vg
            u_out = torch.where(elig, u_new, zero_f64)
            g_out = g * a_g
            spiked = u_out > v_th
            return u_out, g_out, elig, spiked

        try:
            step_fn_novb = torch.compile(_lif_step_novb, dynamic=True)
        except Exception:
            step_fn_novb = _lif_step_novb

        @torch.no_grad()
        def run_batched_gpu(gg, w_syn, is_exc_np, ks, kn, kc, watch_idx_list):
            nonlocal step_fn_novb
            N = gg.n_nodes
            B = is_exc_np.shape[1]
            dt = 0.1
            n_steps = 10000
            dly_steps = 18
            rfc_steps = 22
            ring_len = dly_steps + 1
            a_v = math.exp(-dt / 20.0)
            a_g = math.exp(-dt / 5.0)
            b_vg = (5.0 / 15.0) * (a_v - a_g)
            v_th = 7.0
            w_poi = float(w_syn) * 250.0
            w_s = gg.w_raw_f64 * float(w_syn)
            W = len(watch_idx_list)

            u = torch.zeros((N, B), dtype=torch.float64, device=gg.device)
            g = torch.zeros((N, B), dtype=torch.float64, device=gg.device)
            is_exc_flat = torch.from_numpy(np.ascontiguousarray(is_exc_np)).to(device=gg.device, dtype=torch.bool).view(-1)
            unlock_step = torch.full((N, B), -100000, dtype=torch.int32, device=gg.device)
            unlock_flat = unlock_step.view(-1)
            neg_inf_i32 = torch.tensor(-100000, dtype=torch.int32, device=gg.device)
            zero_f64 = torch.tensor(0.0, dtype=torch.float64, device=gg.device)
            step_cur_tbl = torch.arange(0, n_steps, dtype=torch.int32, device=gg.device)
            step_unlock_tbl = torch.arange(rfc_steps, n_steps + rfc_steps, dtype=torch.int32, device=gg.device)

            watch_t = torch.as_tensor(watch_idx_list, dtype=torch.int64, device=gg.device)
            w_counts = torch.zeros((W, B), dtype=torch.int32, device=gg.device)
            total_spikes = torch.zeros((B,), dtype=torch.int64, device=gg.device)

            if ks.size > 0:
                ord_k = np.argsort(ks, kind="mergesort")
                ks_s = ks[ord_k]
                kick_ptr = np.searchsorted(ks_s, np.arange(n_steps + 1, dtype=np.int32))
                lin_k_np = kn[ord_k].astype(np.int64) * B + kc[ord_k].astype(np.int64)
                kick_lin = torch.from_numpy(lin_k_np).to(device=gg.device, dtype=torch.int64)
                max_k_step = int(np.max(np.diff(kick_ptr)))
                kick_w_buf = torch.full((max(1, max_k_step),), w_poi, dtype=torch.float64, device=gg.device)
            else:
                kick_ptr = np.zeros(n_steps + 1, dtype=np.int64)
                kick_lin = None
                kick_w_buf = None

            ring_src = [None] * ring_len
            ring_col = [None] * ring_len
            indptr = gg.indptr
            deg = gg.deg
            post_s = gg.post_s
            spk_dense_buf = None

            for step in range(n_steps):
                try:
                    u, g, elig, spiked = step_fn_novb(u, g, unlock_step, step_cur_tbl[step], zero_f64, a_v, b_vg, a_g, v_th)
                except Exception:
                    step_fn_novb = _lif_step_novb
                    u, g, elig, spiked = step_fn_novb(u, g, unlock_step, step_cur_tbl[step], zero_f64, a_v, b_vg, a_g, v_th)

                u_flat = u.view(-1)
                g_flat = g.view(-1)
                elig_flat = elig.view(-1)

                src_s, col_s = spiked.nonzero(as_tuple=True)
                fut_slot = (step + dly_steps) % ring_len
                ring_src[fut_slot] = src_s
                ring_col[fut_slot] = col_s

                if src_s.numel() > 0:
                    w_counts.add_(spiked[watch_t].to(torch.int32))
                    total_spikes.add_(torch.bincount(col_s, minlength=B))

                k0 = int(kick_ptr[step])
                k1 = int(kick_ptr[step + 1])
                if k0 < k1:
                    u_flat.index_add_(0, kick_lin[k0:k1], kick_w_buf[: k1 - k0])

                cur_slot = step % ring_len
                src_d = ring_src[cur_slot]
                col_d = ring_col[cur_slot]
                ring_src[cur_slot] = None
                ring_col[cur_slot] = None
                if src_d is not None and src_d.numel() > 0:
                    n_sp = int(src_d.numel())
                    if n_sp <= 15000:
                        d = deg[src_d]
                        total_e = int(d.sum().item())
                        if total_e > 0:
                            starts = indptr[src_d]
                            cum = torch.cumsum(d, 0)
                            base = cum - d
                            e_idx = torch.arange(total_e, device=gg.device, dtype=torch.int64) + torch.repeat_interleave(
                                starts - base, d, output_size=total_e
                            )
                            dst = post_s[e_idx]
                            w_e = w_s[e_idx]
                            c_e = torch.repeat_interleave(col_d, d, output_size=total_e)
                            lin = dst * B + c_e
                            ok = elig_flat[lin]
                            g_flat.index_add_(0, lin[ok], w_e[ok])
                    else:
                        try:
                            if spk_dense_buf is None:
                                spk_dense_buf = torch.zeros((N, B), dtype=torch.float64, device=gg.device)
                            else:
                                spk_dense_buf.zero_()
                            spk_dense_buf[src_d, col_d] = float(w_syn)
                            dg = torch.sparse.mm(gg.get_wt_csr(), spk_dense_buf)
                            g.add_(torch.where(elig, dg, zero_f64))
                        except Exception:
                            for s0 in range(0, n_sp, 15000):
                                sd_c = src_d[s0 : s0 + 15000]
                                cd_c = col_d[s0 : s0 + 15000]
                                d_c = deg[sd_c]
                                tot_c = int(d_c.sum().item())
                                if tot_c > 0:
                                    st_c = indptr[sd_c]
                                    cm_c = torch.cumsum(d_c, 0)
                                    ei_c = torch.arange(tot_c, device=gg.device, dtype=torch.int64) + torch.repeat_interleave(
                                        st_c - (cm_c - d_c), d_c, output_size=tot_c
                                    )
                                    lin_c = post_s[ei_c] * B + torch.repeat_interleave(cd_c, d_c, output_size=tot_c)
                                    ok_c = elig_flat[lin_c]
                                    g_flat.index_add_(0, lin_c[ok_c], w_s[ei_c][ok_c])

                if src_s.numel() > 0:
                    lin_s = src_s * B + col_s
                    u_flat[lin_s] = 0.0
                    g_flat[lin_s] = 0.0
                    unlock_flat[lin_s] = torch.where(is_exc_flat[lin_s], neg_inf_i32, step_unlock_tbl[step])

            return w_counts.cpu().numpy(), total_spikes.cpu().numpy()

        def run_106_sweep_gpu(gg, w_syn, freq_hz, seed_base, n_trials=30):
            means = np.zeros(len(type_indices), dtype=np.float64)
            l_rates = np.zeros(len(type_indices), dtype=np.float64)
            r_rates = np.zeros(len(type_indices), dtype=np.float64)
            tot_brain = np.zeros(len(type_indices), dtype=np.float64)
            for b0 in range(0, len(type_indices), batch_types):
                b1 = min(len(type_indices), b0 + batch_types)
                n_sub = b1 - b0
                B = n_sub * n_trials
                is_exc = np.zeros((n_nodes, B), dtype=bool)
                ks_l, kn_l, kc_l = [], [], []
                for li, gi in enumerate(range(b0, b1)):
                    idxs = type_indices[gi]
                    c0 = li * n_trials
                    if idxs:
                        is_exc[idxs, c0:c0 + n_trials] = True
                        ks, kn, kc = gen_opto_kicks(seed_base + gi, n_trials, idxs, freq_hz)
                        if len(ks) > 0:
                            ks_l.append(ks); kn_l.append(kn); kc_l.append(kc + c0)
                ks_all = np.concatenate(ks_l) if ks_l else np.zeros(0, dtype=np.int32)
                kn_all = np.concatenate(kn_l) if kn_l else np.zeros(0, dtype=np.int32)
                kc_all = np.concatenate(kc_l) if kc_l else np.zeros(0, dtype=np.int32)
                wc, tspk = run_batched_gpu(gg, w_syn, is_exc, ks_all, kn_all, kc_all, mn9_indices)
                for li, gi in enumerate(range(b0, b1)):
                    c0 = li * n_trials
                    sl = wc[:, c0:c0 + n_trials]
                    l_rates[gi] = float(sl[0].mean())
                    r_rates[gi] = float(sl[1].mean())
                    means[gi] = float(sl.mean())
                    tot_brain[gi] = float(tspk[c0:c0 + n_trials].mean() / n_nodes)
            return {"means": means, "l_rates": l_rates, "r_rates": r_rates, "mean_brain_hz": float(tot_brain.mean()), "max_brain_hz": float(tot_brain.max())}

        # Labels & subsets for scoring
        y_bin = np.array([int(r["y_bin"]) for r in sez_list], dtype=np.int32)
        prior_pub = np.array([bool(r["prior_published"]) for r in sez_list], dtype=bool)
        v783_50 = np.array([float(r["v783_50hz"]) for r in sez_list], dtype=np.float64)
        no_ru = np.array([r["name"] != "roundup" for r in sez_list], dtype=bool)
        # Recompute is_direct_malecns on base graph
        is_dir_m = np.zeros(len(sez_list), dtype=bool)
        mn9_set = set(mn9_indices)
        for gi, idxs in enumerate(type_indices):
            for u in idxs:
                if any(int(post0[e]) in mn9_set for e in range(int(indptr0[u]), int(indptr0[u + 1]))):
                    is_dir_m[gi] = True
                    break

        def pack_sweep_eval(res, extra_names=(), is_dir_var=None):
            m = res["means"]
            l = res["l_rates"]
            var_mask = is_dir_m if is_dir_var is None else is_dir_var
            return {
                "all_106_auprc": stepwise_auprc(y_bin, m),
                "all_106_ci95": stratified_auprc_ci95(y_bin, m),
                "all_106_auroc": midrank_auroc(y_bin, m),
                "all_106_l_auprc": stepwise_auprc(y_bin, l),
                "novel_95_auprc": stepwise_auprc(y_bin[~prior_pub], m[~prior_pub]),
                "novel_95_ci95": stratified_auprc_ci95(y_bin[~prior_pub], m[~prior_pub]),
                "novel_95_l_auprc": stepwise_auprc(y_bin[~prior_pub], l[~prior_pub]),
                "nondirect_malecns_auprc": stepwise_auprc(y_bin[~is_dir_m], m[~is_dir_m]),
                "nondirect_variant_auprc": stepwise_auprc(y_bin[~var_mask], m[~var_mask]),
                "nondirect_variant_ci95": stratified_auprc_ci95(y_bin[~var_mask], m[~var_mask]),
                "nondirect_variant_n": int((~var_mask).sum()),
                "nonzero_106": int((m > 0).sum()),
                "mean_brain_hz": res["mean_brain_hz"],
                "max_brain_hz": res["max_brain_hz"],
                "v783_spearman_rho": float(stats.spearmanr(v783_50, m)[0]),
                "v783_spearman_rho_excl_roundup": float(stats.spearmanr(v783_50[no_ru], m[no_ru])[0]),
                "v783_pearson_r_excl_roundup": float(stats.pearsonr(v783_50[no_ru], m[no_ru])[0]),
                "key_rates_mean": {r["name"]: float(m[i]) for i, r in enumerate(sez_list) if r["y_bin"] == 1 or r["name"] in fp6_names or r["name"] in extra_names},
                "key_rates_left": {r["name"]: float(l[i]) for i, r in enumerate(sez_list) if r["y_bin"] == 1 or r["name"] in fp6_names or r["name"] in extra_names},
                "all_106_rates_mean": [float(x) for x in m],
            }

        # ---------------------------------------------------------------
        # PARITY CHECK: Verify roundup 50 Hz @ W_intact matches locked 47.9333 Hz (L=89.0, R=6.8667)
        # ---------------------------------------------------------------
        w_intact = float(0.275 * 5732.5 / 5876.0)
        gg0 = GPUGraph(indptr0, post0, w0, device)
        is_exc_chk = np.zeros((n_nodes, 30), dtype=bool)
        is_exc_chk[type_indices[0], :] = True
        ks_c, kn_c, kc_c = gen_opto_kicks(50000, 30, type_indices[0], 50.0)
        wc_c, _ = run_batched_gpu(gg0, w_intact, is_exc_chk, ks_c, kn_c, kc_c, mn9_indices)
        print(f"  [GPU parity check] roundup 50 Hz @ W_intact: mean={wc_c.mean():.4f} (L={wc_c[0].mean():.4f}, R={wc_c[1].mean():.4f}; locked ref=47.9333, L=89.0, R=6.8667)", flush=True)
        assert abs(float(wc_c.mean()) - 47.93333333333333) < 0.05, f"GPU parity mismatch: {wc_c.mean():.4f}"

        if rev_mode:
            # Revision mode: MaleCNS SELK widest paths (primary + A2), scatter audit, exact is_direct_variant, and A2 + B3 combined!
            del gg0
            torch.cuda.empty_cache()
            is_vnc_m = np.array([m["region"] == "ventral_nerve_cord" for m in node_meta], dtype=bool)
            reg_names_m = ["central_brain", "optic_lobe", "ventral_nerve_cord"]
            reg_map_m = {r: i for i, r in enumerate(reg_names_m)}
            reg_ids_m = np.array([reg_map_m[m["region"]] for m in node_meta], dtype=np.int32)
            scatter_m = audit_rewire_scatter(pre0, post0, reg_ids_m, reg_names_m)

            def _desc_m(u: int) -> dict:
                md = node_meta[u]
                return {"body_id": int(md["id"]), "cell_type": md["type"] or md["flywireType"], "region": md["region"], "side": md["side"], "nt": md["cons_nt"] or md["pred_nt"]}

            print("[rev_colab_malecns] Computing exact SELK->MN9 via-VNC widest paths (primary consensus_nt)...", flush=True)
            selk_wp_primary = exact_widest_exc_paths_via_vnc(indptr0, post0, w0, ncls0, selk_idxs, mn9_set, is_vnc_m, _desc_m, max_k=5)

            # Build A2 (all consensus_nt=='unclear' -> single-cell predicted_nt) graph
            sgn_a2 = base_signs.copy()
            for i, m in enumerate(node_meta):
                if sgn_a2[i] == 0:
                    ps = nt_to_sign(m["pred_nt"])
                    if ps != 0:
                        sgn_a2[i] = ps
            ind_a2, pre_a2, post_a2, w_a2, ncls_a2, sin_a2, sout_a2 = build_signed_csr(sgn_a2)
            print("[rev_colab_malecns] Computing exact SELK->MN9 via-VNC widest paths (A2 predicted_nt)...", flush=True)
            selk_wp_a2 = exact_widest_exc_paths_via_vnc(ind_a2, post_a2, w_a2, ncls_a2, selk_idxs, mn9_set, is_vnc_m, _desc_m, max_k=5)

            # Exact direct-to-MN9 edge audit across primary, A1a, and A2
            diatom_18_idxs = [i for i in sez_idx_by_name["diatom"] if nt_to_sign(node_meta[i]["pred_nt"]) > 0]
            sgn_a1a = base_signs.copy()
            sgn_a1a[diatom_18_idxs + sez_idx_by_name["TH_VUM"]] = 1
            ind_a1a, _, post_a1a, w_a1a, _, sin_a1a, _ = build_signed_csr(sgn_a1a)

            def _direct_mask_and_weights(ind_x, post_x, w_x):
                dm = np.zeros(len(sez_list), dtype=bool)
                new_dir = {}
                for gi, idxs in enumerate(type_indices):
                    w_l = w_r = 0.0
                    for u in idxs:
                        for e in range(int(ind_x[u]), int(ind_x[u + 1])):
                            v = int(post_x[e])
                            if v == mn9_l_idx:
                                w_l += float(w_x[e])
                            elif v == mn9_r_idx:
                                w_r += float(w_x[e])
                    if w_l != 0.0 or w_r != 0.0:
                        dm[gi] = True
                        if not is_dir_m[gi]:
                            new_dir[sez_list[gi]["name"]] = {"w_to_mn9_l": w_l, "w_to_mn9_r": w_r}
                return dm, new_dir

            is_dir_a1a, newly_dir_a1a = _direct_mask_and_weights(ind_a1a, post_a1a, w_a1a)
            is_dir_a2, newly_dir_a2 = _direct_mask_and_weights(ind_a2, post_a2, w_a2)
            print(f"  Newly direct in A1a: {newly_dir_a1a} (n_nd={int((~is_dir_a1a).sum())}); in A2: {newly_dir_a2} (n_nd={int((~is_dir_a2).sum())})", flush=True)

            # Recompute matched central-brain median incoming-synapse ratio under A2 predicted_nt
            v783_ct_stats = v783_ref["v783_celltype_stats"]
            r_cb_a2 = []
            for ct, st7 in v783_ct_stats.items():
                if st7["region"] != "central_brain" or st7["mean_s_in"] <= 0:
                    continue
                m_idxs = sorted(set(by_fw.get(ct, [])) | set(by_ty.get(ct, [])))
                if not m_idxs:
                    continue
                ms_in = float(np.mean([sin_a2[i] for i in m_idxs]))
                if ms_in > 0:
                    r_cb_a2.append(ms_in / st7["mean_s_in"])
            a2_cb_summary = summarize_ratios(r_cb_a2)
            r_med_cb_a2 = float(a2_cb_summary["median"])
            w_syn_a2_b3 = float(0.275 / r_med_cb_a2)
            print(f"  [A2 CB density] median={r_med_cb_a2:.6f} (vs primary {r_med_cb:.6f}) -> W_syn_A2_B3={w_syn_a2_b3:.6f} mV (vs primary B3 {w_syn_density:.6f} mV)", flush=True)

            gg_a2 = GPUGraph(ind_a2, post_a2, w_a2, device)
            t_comb = time.time()
            comb_runs = {}
            w_candidates = [("a2_density_matched", w_syn_a2_b3)]
            if abs(w_syn_a2_b3 - w_syn_density) > 5e-4:
                w_candidates.append(("primary_b3_weight", w_syn_density))

            for tag_w, w_val in w_candidates:
                sw50 = run_106_sweep_gpu(gg_a2, w_val, 50.0, 50000, 30)
                sw200 = run_106_sweep_gpu(gg_a2, w_val, 200.0, 200000, 30)
                selk_sw = {}
                for hz in (20, 50, 100, 200):
                    is_exc = np.zeros((n_nodes, 30), dtype=bool)
                    is_exc[selk_idxs, :] = True
                    ks, kn, kc = gen_opto_kicks(80000 + hz, 30, selk_idxs, float(hz))
                    wc, tspk = run_batched_gpu(gg_a2, w_val, is_exc, ks, kn, kc, mn9_indices)
                    selk_sw[f"{hz}hz"] = {
                        "mn9_l_hz": float(wc[0].mean()), "mn9_r_hz": float(wc[1].mean()),
                        "mn9_mean_hz": float(wc.mean()), "brain_mean_hz": float(tspk.mean() / n_nodes),
                    }
                # Also compute nondirect excluding TH_VUM only (N=87)
                thv_i = next(i for i, r in enumerate(sez_list) if r["name"] == "TH_VUM")
                is_dir_excl_thv = is_dir_m.copy()
                is_dir_excl_thv[thv_i] = True
                ev50 = pack_sweep_eval(sw50, extra_names=("seagull", "gallinule", "trogon", "mist"), is_dir_var=is_dir_a2)
                ev50["nondirect_excl_thvum_87_auprc"] = stepwise_auprc(y_bin[~is_dir_excl_thv], sw50["means"][~is_dir_excl_thv])
                ev200 = pack_sweep_eval(sw200, extra_names=("seagull", "gallinule", "trogon", "mist"), is_dir_var=is_dir_a2)
                ev200["nondirect_excl_thvum_87_auprc"] = stepwise_auprc(y_bin[~is_dir_excl_thv], sw200["means"][~is_dir_excl_thv])
                comb_runs[tag_w] = {"w_syn_mv": float(w_val), "50hz": ev50, "200hz": ev200, "selk_dng68_20_to_200hz": selk_sw}
                print(
                    f"  [A2+B3 {tag_w} W={w_val:.6f}] 50Hz all106={ev50['all_106_auprc']:.4f} {ev50['all_106_ci95']}, "
                    f"nov95={ev50['novel_95_auprc']:.4f} {ev50['novel_95_ci95']}, nd_var86={ev50['nondirect_variant_auprc']:.4f}, "
                    f"rho={ev50['v783_spearman_rho']:.4f}, diatom={ev50['key_rates_mean']['diatom']:.2f}, "
                    f"TH_VUM={ev50['key_rates_mean']['TH_VUM']:.2f}, brain={ev50['mean_brain_hz']:.4f}, "
                    f"fp6={{k:round(ev50['key_rates_mean'][k],2) for k in fp6_names}}, SELK50={selk_sw['50hz']['mn9_mean_hz']:.3f}",
                    flush=True,
                )

            payload = {
                "gpu_name": gpu_name,
                "malecns_scatter_audit": scatter_m,
                "selk_vnc_exact_widest_paths_malecns": {"primary_consensus_nt": selk_wp_primary, "a2_predicted_nt": selk_wp_a2},
                "direct_variant_edges_audit": {
                    "primary_n_direct": int(is_dir_m.sum()), "primary_n_nondirect": int((~is_dir_m).sum()),
                    "a1a_n_direct": int(is_dir_a1a.sum()), "a1a_n_nondirect": int((~is_dir_a1a).sum()), "a1a_newly_direct": newly_dir_a1a,
                    "a2_n_direct": int(is_dir_a2.sum()), "a2_n_nondirect": int((~is_dir_a2).sum()), "a2_newly_direct": newly_dir_a2,
                },
                "test_a2_b3_combined": {
                    "a2_cb_density_summary": a2_cb_summary,
                    "r_med_cb_a2": r_med_cb_a2,
                    "r_med_cb_primary": r_med_cb,
                    "w_syn_a2_b3_mv": w_syn_a2_b3,
                    "w_syn_primary_b3_mv": w_syn_density,
                    "runs": comb_runs,
                    "elapsed_sec": float(time.time() - t_comb),
                },
                "elapsed_sec": float(time.time() - t_all),
            }
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[rev_colab_malecns] DONE in {time.time()-t_all:.1f}s -> {out_path}", flush=True)
            return

        # ---------------------------------------------------------------
        # TEST B3: Network-density-matched MaleCNS scaling simulation!
        # ---------------------------------------------------------------
        print(f"[colab_malecns] Running Test B3: MaleCNS density-matched scaling (W_syn_density={w_syn_density:.6f} mV, 50/200 Hz)...", flush=True)
        t_b3 = time.time()
        b3_50 = run_106_sweep_gpu(gg0, w_syn_density, 50.0, 50000, 30)
        b3_200 = run_106_sweep_gpu(gg0, w_syn_density, 200.0, 200000, 30)
        test_b3 = {
            "w_syn_density_mv": w_syn_density,
            "r_med_central_brain": r_med_cb,
            "50hz": pack_sweep_eval(b3_50),
            "200hz": pack_sweep_eval(b3_200),
            "elapsed_sec": float(time.time() - t_b3),
        }
        print(f"  [Test B3 50Hz] all106={test_b3['50hz']['all_106_auprc']:.4f}, nov95={test_b3['50hz']['novel_95_auprc']:.4f}, nd={test_b3['50hz']['nondirect_malecns_auprc']:.4f}, nz={test_b3['50hz']['nonzero_106']}, brain={test_b3['50hz']['mean_brain_hz']:.3f}, fp6={{k:round(test_b3['50hz']['key_rates_mean'][k],2) for k in fp6_names}}", flush=True)
        del gg0
        torch.cuda.empty_cache()

        # ---------------------------------------------------------------
        # TEST A1a & A1b: diatom (18 or 20 cells) + TH_VUM = +1 @ 0.268284 mV
        # ---------------------------------------------------------------
        w_intact = float(0.275 * 5732.5 / 5876.0)
        diatom_idxs = sez_idx_by_name["diatom"]
        thvum_idxs = sez_idx_by_name["TH_VUM"]
        diatom_18_idxs = [i for i in diatom_idxs if nt_to_sign(node_meta[i]["pred_nt"]) > 0]
        print(f"[colab_malecns] Running Test A1a (diatom18={len(diatom_18_idxs)}+TH_VUM={len(thvum_idxs)} -> +1) & A1b (diatom20+TH_VUM -> +1)...", flush=True)

        test_a = {}
        for tag, plus_idxs in (("A1a_diatom18_thvum_plus1", diatom_18_idxs + thvum_idxs), ("A1b_diatom20_thvum_plus1", diatom_idxs + thvum_idxs)):
            t_a = time.time()
            sgn_a = base_signs.copy()
            sgn_a[plus_idxs] = 1
            ind_a, pre_a, post_a, w_a, _, sin_a, _ = build_signed_csr(sgn_a)
            is_dir_a = np.zeros(len(sez_list), dtype=bool)
            for gi, idxs in enumerate(type_indices):
                for u in idxs:
                    if any(int(post_a[e]) in mn9_set for e in range(int(ind_a[u]), int(ind_a[u + 1]))):
                        is_dir_a[gi] = True
                        break
            gg_a = GPUGraph(ind_a, post_a, w_a, device)
            res_tag = {"n_flipped_neurons": len(plus_idxs), "w_syn_mv": w_intact, "s_in_mn9_l": float(sin_a[mn9_l_idx]), "n_direct_variant": int(is_dir_a.sum())}
            for hz, sb in ((50, 50000), (100, 100000), (200, 200000)):
                sw = run_106_sweep_gpu(gg_a, w_intact, float(hz), sb, 30)
                res_tag[f"{hz}hz"] = pack_sweep_eval(sw, is_dir_var=is_dir_a)
            res_tag["elapsed_sec"] = float(time.time() - t_a)
            test_a[tag] = res_tag
            r50 = res_tag["50hz"]
            print(f"  [{tag} 50Hz] diatom={r50['key_rates_mean']['diatom']:.2f} (L:{r50['key_rates_left']['diatom']:.2f}), TH_VUM={r50['key_rates_mean']['TH_VUM']:.2f}, all106={r50['all_106_auprc']:.4f}, nov95={r50['novel_95_auprc']:.4f}, nd={r50['nondirect_malecns_auprc']:.4f}", flush=True)
            del gg_a
            torch.cuda.empty_cache()

        # ---------------------------------------------------------------
        # TEST A2: ALL consensus_nt=='unclear' -> single-cell predicted_nt + SELK 20-200 Hz
        # ---------------------------------------------------------------
        print("[colab_malecns] Running Test A2: ALL consensus_nt=='unclear' -> single-cell predicted_nt + SELK sweep...", flush=True)
        t_a2 = time.time()
        sgn_a2 = base_signs.copy()
        n_a2_exc, n_a2_inh, n_a2_still0 = 0, 0, 0
        for i, m in enumerate(node_meta):
            if sgn_a2[i] == 0:
                ps = nt_to_sign(m["pred_nt"])
                if ps > 0:
                    sgn_a2[i] = 1; n_a2_exc += 1
                elif ps < 0:
                    sgn_a2[i] = -1; n_a2_inh += 1
                else:
                    n_a2_still0 += 1
        ind_a2, pre_a2, post_a2, w_a2, _, sin_a2, _ = build_signed_csr(sgn_a2)
        is_dir_a2 = np.zeros(len(sez_list), dtype=bool)
        for gi, idxs in enumerate(type_indices):
            for u in idxs:
                if any(int(post_a2[e]) in mn9_set for e in range(int(ind_a2[u]), int(ind_a2[u + 1]))):
                    is_dir_a2[gi] = True
                    break
        s_in_l_a2 = float(sin_a2[mn9_l_idx])
        w_syn_a2 = float(0.275 * 5732.5 / s_in_l_a2)
        gg_a2 = GPUGraph(ind_a2, post_a2, w_a2, device)

        # Inspect SELK top-20 targets (especially ANXXX005 single-cell predicted_nt!)
        selk_out_a2 = defaultdict(float)
        for s in selk_idxs:
            for e in range(int(ind_a2[s]), int(ind_a2[s + 1])):
                selk_out_a2[int(post_a2[e])] += abs(float(w_a2[e]))
        anxxx005_info = [
            {
                "body_id": node_meta[u]["id"], "type": node_meta[u]["type"], "side": node_meta[u]["side"],
                "cons_nt": node_meta[u]["cons_nt"], "pred_nt": node_meta[u]["pred_nt"], "ctpred_nt": node_meta[u]["ctpred_nt"],
                "sign_a2": int(sgn_a2[u]), "syn_from_selk": float(syn),
            }
            for u, syn in sorted(selk_out_a2.items(), key=lambda x: -x[1])[:25]
            if node_meta[u]["type"] == "ANXXX005" or node_meta[u]["cons_nt"] in ("", "unclear")
        ]

        a2_50 = run_106_sweep_gpu(gg_a2, w_syn_a2, 50.0, 50000, 30)
        a2_200 = run_106_sweep_gpu(gg_a2, w_syn_a2, 200.0, 200000, 30)

        # SELK 20/50/100/200 Hz under A2
        selk_a2 = {}
        for hz in (20, 50, 100, 200):
            is_exc = np.zeros((n_nodes, 30), dtype=bool)
            is_exc[selk_idxs, :] = True
            ks, kn, kc = gen_opto_kicks(80000 + hz, 30, selk_idxs, float(hz))
            wc, tspk = run_batched_gpu(gg_a2, w_syn_a2, is_exc, ks, kn, kc, mn9_indices)
            selk_a2[f"{hz}hz"] = {
                "mn9_l_hz": float(wc[0].mean()), "mn9_r_hz": float(wc[1].mean()),
                "mn9_mean_hz": float(wc.mean()), "brain_mean_hz": float(tspk.mean() / n_nodes),
            }

        test_a["A2_all_unclear_predicted_nt"] = {
            "n_unclear_assigned_exc": n_a2_exc,
            "n_unclear_assigned_inh": n_a2_inh,
            "n_unclear_still_zero": n_a2_still0,
            "n_edges_valid_a2": int(len(pre_a2)),
            "s_in_mn9_l_a2": s_in_l_a2,
            "w_syn_intact_a2_mv": w_syn_a2,
            "50hz": pack_sweep_eval(a2_50, is_dir_var=is_dir_a2),
            "200hz": pack_sweep_eval(a2_200, is_dir_var=is_dir_a2),
            "selk_dng68_20_to_200hz": selk_a2,
            "selk_top25_previously_unclear_targets": anxxx005_info,
            "elapsed_sec": float(time.time() - t_a2),
        }
        r50_a2 = test_a["A2_all_unclear_predicted_nt"]["50hz"]
        print(f"  [A2 50Hz] W_a2={w_syn_a2:.6f}, diatom={r50_a2['key_rates_mean']['diatom']:.2f}, TH_VUM={r50_a2['key_rates_mean']['TH_VUM']:.2f}, all106={r50_a2['all_106_auprc']:.4f}, nov95={r50_a2['novel_95_auprc']:.4f}, SELK 50/200={selk_a2['50hz']['mn9_mean_hz']:.3f}/{selk_a2['200hz']['mn9_mean_hz']:.3f}", flush=True)

        payload = {
            "gpu_name": gpu_name,
            "test_a_transmitter_sensitivity": test_a,
            "test_b_malecns_density_and_dimorphism": malecns_density,
            "test_b_malecns_density_scaled_sim": test_b3,
            "test_d_44ctrl_three_brain_null": test_d_summary,
            "elapsed_sec": float(time.time() - t_all),
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[colab_malecns] ALL DONE in {time.time()-t_all:.1f}s -> {out_path}", flush=True)
    finally:
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"[colab_malecns] Cleaned temporary cache {cache_dir}", flush=True)


# ===========================================================================
# PHASE 2B: GPU-ACCELERATED TEST C ON COLAB A100 (bit-identical to local_test_c)
# ===========================================================================
@njit(cache=False)
def _gen_np_kicks(seed: int, n_trials: int, exc_idx: np.ndarray, rate_hz: float):
    """Exact np.random.seed kick sequence identical to run_lif_lazy_exact / Brian2."""
    n_steps = 10000
    n_exc = len(exc_idx)
    p_poi = rate_hz * 1e-4
    np.random.seed(seed)
    est = int(max(1024, n_trials * n_exc * rate_hz * 1.6 + 1024))
    out_step = np.empty(est, dtype=np.int32)
    out_node = np.empty(est, dtype=np.int32)
    out_tr = np.empty(est, dtype=np.int32)
    cnt = 0
    for tr in range(n_trials):
        for step in range(n_steps):
            for k in range(n_exc):
                if np.random.random() < p_poi:
                    if cnt >= len(out_step):
                        new_sz = len(out_step) * 2
                        ns = np.empty(new_sz, dtype=np.int32)
                        nn = np.empty(new_sz, dtype=np.int32)
                        nt = np.empty(new_sz, dtype=np.int32)
                        ns[:cnt] = out_step[:cnt]
                        nn[:cnt] = out_node[:cnt]
                        nt[:cnt] = out_tr[:cnt]
                        out_step, out_node, out_tr = ns, nn, nt
                    out_step[cnt] = step
                    out_node[cnt] = exc_idx[k]
                    out_tr[cnt] = tr
                    cnt += 1
    return out_step[:cnt], out_node[:cnt], out_tr[:cnt]


def run_phase_colab_test_c(
    bundle_npz: Path, meta_json: Path, out_path: Path, batch_types: int = 53, compartment_preserving: bool = False
) -> None:
    import torch
    from scipy import sparse as sp_sparse

    if not compartment_preserving:
        sys.path.insert(0, str(bundle_npz.parent))
        sys.path.insert(0, str(BASE_DIR))
        import shiu_mn9_benchmark as smb

    t0 = time.time()
    assert torch.cuda.is_available(), "CUDA required for colab_test_c"
    device = torch.device("cuda:0")
    print(f"[colab_test_c] GPU={torch.cuda.get_device_name(0)}, batch_types={batch_types}, compartment_preserving={compartment_preserving}", flush=True)

    d = np.load(bundle_npz)
    indptr = d["indptr"]
    post_real = d["post_s"]
    pre_s = d["pre_s"]
    w_raw_s = d["w_raw_s"]
    node_class = d["node_class"]
    orig_in_pos = d["orig_in_pos"]
    orig_in_neg = d["orig_in_neg"]
    mn9_idx = d["mn9_idx"].astype(np.int32)
    sugar_idx = d["sugar_idx"].astype(np.int32)
    node_comp = d["node_comp"].astype(np.int32) if compartment_preserving else None
    n_nodes = len(indptr) - 1

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    type_indices = [np.array(x, dtype=np.int32) for x in meta["type_indices"]]
    y_bin = np.array(meta["y_bin"], dtype=np.int32)
    is_direct_v783 = np.array(meta["is_direct_v783"], dtype=bool)
    prior_pub = np.array(meta["prior_pub"], dtype=bool)
    real_v783_50 = np.array(meta["real_v783_50"], dtype=np.float64)
    target_mean_hz = float(real_v783_50.mean())
    mn9_list = [int(x) for x in mn9_idx]

    class GPUGraphV7:
        def __init__(self, indptr_np, post_np, w_raw_np):
            self.n_nodes = len(indptr_np) - 1
            self.device = device
            ind_i64 = indptr_np.astype(np.int64)
            deg_np = ind_i64[1:] - ind_i64[:-1]
            self.indptr = torch.from_numpy(ind_i64).to(device)
            self.deg = torch.from_numpy(deg_np).to(device)
            self.post_s = torch.from_numpy(post_np.astype(np.int64)).to(device)
            self.w_raw_f64 = torch.from_numpy(w_raw_np.astype(np.float64)).to(device)
            self._w_t_csr = None
            self._indptr_np = indptr_np
            self._post_np = post_np
            self._w_raw_np = w_raw_np

        def get_wt_csr(self):
            if self._w_t_csr is None:
                m_csc = sp_sparse.csc_matrix(
                    (self._w_raw_np.astype(np.float64), self._post_np, self._indptr_np),
                    shape=(self.n_nodes, self.n_nodes),
                )
                wt_csr = m_csc.tocsr()
                self._w_t_csr = torch.sparse_csr_tensor(
                    torch.from_numpy(wt_csr.indptr.astype(np.int64)).to(self.device),
                    torch.from_numpy(wt_csr.indices.astype(np.int64)).to(self.device),
                    torch.from_numpy(wt_csr.data.astype(np.float64)).to(self.device),
                    size=(self.n_nodes, self.n_nodes),
                    device=self.device,
                )
            return self._w_t_csr

    def _lif_step_novb(u, g, unlock_step, step_i32, zero_f64, a_v: float, b_vg: float, a_g: float, v_th: float):
        elig = unlock_step <= step_i32
        u_new = u * a_v + g * b_vg
        u_out = torch.where(elig, u_new, zero_f64)
        g_out = g * a_g
        spiked = u_out > v_th
        return u_out, g_out, elig, spiked

    try:
        step_fn = torch.compile(_lif_step_novb, dynamic=True)
    except Exception:
        step_fn = _lif_step_novb

    @torch.no_grad()
    def run_gpu_batch(gg, w_syn, is_exc_np, ks, kn, kc, max_brain_hz_cap=15.0):
        nonlocal step_fn
        N = gg.n_nodes
        B = is_exc_np.shape[1]
        dt = 0.1
        n_steps = 10000
        dly_steps, rfc_steps = 18, 22
        ring_len = dly_steps + 1
        a_v = math.exp(-dt / 20.0)
        a_g = math.exp(-dt / 5.0)
        b_vg = (5.0 / 15.0) * (a_v - a_g)
        v_th = 7.0
        w_poi = float(w_syn) * 250.0
        w_s = gg.w_raw_f64 * float(w_syn)

        u = torch.zeros((N, B), dtype=torch.float64, device=device)
        g = torch.zeros((N, B), dtype=torch.float64, device=device)
        is_exc_flat = torch.from_numpy(np.ascontiguousarray(is_exc_np)).to(device=device, dtype=torch.bool).view(-1)
        unlock_step = torch.full((N, B), -100000, dtype=torch.int32, device=device)
        unlock_flat = unlock_step.view(-1)
        neg_inf_i32 = torch.tensor(-100000, dtype=torch.int32, device=device)
        zero_f64 = torch.tensor(0.0, dtype=torch.float64, device=device)
        step_cur_tbl = torch.arange(0, n_steps, dtype=torch.int32, device=device)
        step_unlock_tbl = torch.arange(rfc_steps, n_steps + rfc_steps, dtype=torch.int32, device=device)
        watch_t = torch.as_tensor(mn9_list, dtype=torch.int64, device=device)
        w_counts = torch.zeros((len(mn9_list), B), dtype=torch.int32, device=device)
        total_spikes = torch.zeros((B,), dtype=torch.int64, device=device)

        if ks.size > 0:
            ord_k = np.argsort(ks, kind="mergesort")
            ks_s = ks[ord_k]
            kick_ptr = np.searchsorted(ks_s, np.arange(n_steps + 1, dtype=np.int32))
            lin_k_np = kn[ord_k].astype(np.int64) * B + kc[ord_k].astype(np.int64)
            kick_lin = torch.from_numpy(lin_k_np).to(device=device, dtype=torch.int64)
            max_k_step = int(np.max(np.diff(kick_ptr)))
            kick_w_buf = torch.full((max(1, max_k_step),), w_poi, dtype=torch.float64, device=device)
        else:
            kick_ptr = np.zeros(n_steps + 1, dtype=np.int64)
            kick_lin = kick_w_buf = None

        ring_src = [None] * ring_len
        ring_col = [None] * ring_len
        ind_t, deg_t, post_t = gg.indptr, gg.deg, gg.post_s
        spk_dense_buf = None
        cap_spk = int(max_brain_hz_cap * N)
        capped_cols = torch.zeros((B,), dtype=torch.bool, device=device)

        for step in range(n_steps):
            try:
                u, g, elig, spiked = step_fn(u, g, unlock_step, step_cur_tbl[step], zero_f64, a_v, b_vg, a_g, v_th)
            except Exception:
                step_fn = _lif_step_novb
                u, g, elig, spiked = step_fn(u, g, unlock_step, step_cur_tbl[step], zero_f64, a_v, b_vg, a_g, v_th)

            # Mask already-capped columns so seizure cols don't keep generating spikes
            if capped_cols.any():
                spiked.masked_fill_(capped_cols.unsqueeze(0), False)

            u_flat = u.view(-1)
            g_flat = g.view(-1)
            elig_flat = elig.view(-1)

            src_s, col_s = spiked.nonzero(as_tuple=True)
            fut_slot = (step + dly_steps) % ring_len
            ring_src[fut_slot] = src_s
            ring_col[fut_slot] = col_s

            if src_s.numel() > 0:
                w_counts.add_(spiked[watch_t].to(torch.int32))
                total_spikes.add_(torch.bincount(col_s, minlength=B))
                if (step & 63) == 63:
                    newly_cap = total_spikes > cap_spk
                    if newly_cap.any():
                        capped_cols |= newly_cap
                        if capped_cols.all():
                            break

            k0, k1 = int(kick_ptr[step]), int(kick_ptr[step + 1])
            if k0 < k1:
                u_flat.index_add_(0, kick_lin[k0:k1], kick_w_buf[: k1 - k0])

            cur_slot = step % ring_len
            src_d, col_d = ring_src[cur_slot], ring_col[cur_slot]
            ring_src[cur_slot] = ring_col[cur_slot] = None
            if src_d is not None and src_d.numel() > 0:
                n_sp = int(src_d.numel())
                if n_sp <= 15000:
                    d_v = deg_t[src_d]
                    tot_e = int(d_v.sum().item())
                    if tot_e > 0:
                        starts = ind_t[src_d]
                        cum = torch.cumsum(d_v, 0)
                        e_idx = torch.arange(tot_e, device=device, dtype=torch.int64) + torch.repeat_interleave(starts - (cum - d_v), d_v, output_size=tot_e)
                        lin = post_t[e_idx] * B + torch.repeat_interleave(col_d, d_v, output_size=tot_e)
                        ok = elig_flat[lin]
                        g_flat.index_add_(0, lin[ok], w_s[e_idx][ok])
                else:
                    if spk_dense_buf is None:
                        spk_dense_buf = torch.zeros((N, B), dtype=torch.float64, device=device)
                    else:
                        spk_dense_buf.zero_()
                    spk_dense_buf[src_d, col_d] = float(w_syn)
                    dg = torch.sparse.mm(gg.get_wt_csr(), spk_dense_buf)
                    g.add_(torch.where(elig, dg, zero_f64))

            if src_s.numel() > 0:
                lin_s = src_s * B + col_s
                u_flat[lin_s] = 0.0
                g_flat[lin_s] = 0.0
                unlock_flat[lin_s] = torch.where(is_exc_flat[lin_s], neg_inf_i32, step_unlock_tbl[step])

        capped_cols |= (total_spikes > cap_spk)
        return w_counts.cpu().numpy(), total_spikes.cpu().numpy(), capped_cols.cpu().numpy()

    # Pre-generate deterministic kick arrays ONCE (shared across all 10 rewired graphs!)
    sugar_ks10, sugar_kn10, sugar_kc10 = _gen_np_kicks(42, 10, sugar_idx, 100.0)
    sugar_ks30, sugar_kn30, sugar_kc30 = _gen_np_kicks(42, 30, sugar_idx, 100.0)

    # Sanity check real v783 sugar 100 Hz (must match 58.30 Hz EXACTLY!)
    gg_real = GPUGraphV7(indptr, post_real, w_raw_s)
    is_exc_s = np.zeros((n_nodes, 30), dtype=bool)
    is_exc_s[sugar_idx, :] = True
    wc_chk, tsp_chk, _ = run_gpu_batch(gg_real, 0.275, is_exc_s, sugar_ks30, sugar_kn30, sugar_kc30)
    print(f"  [GPU v783 sugar 100 Hz parity check] MN9_mean={wc_chk.mean():.4f} Hz (exact ref=58.3000 Hz), brain_hz={tsp_chk.mean()/n_nodes:.4f}", flush=True)
    assert abs(float(wc_chk.mean()) - 58.30) < 0.01, f"GPU sugar parity mismatch: {wc_chk.mean():.4f}"
    del gg_real
    torch.cuda.empty_cache()

    # Precompute 5-trial and 30-trial 106-type batches ONCE!
    def build_106_batches(n_tr: int, bt: int):
        batches = []
        for b0 in range(0, len(type_indices), bt):
            b1 = min(len(type_indices), b0 + bt)
            n_sub = b1 - b0
            B = n_sub * n_tr
            is_exc = np.zeros((n_nodes, B), dtype=bool)
            ks_l, kn_l, kc_l = [], [], []
            for li, gi in enumerate(range(b0, b1)):
                idxs = type_indices[gi]
                if idxs.size > 0:
                    c0 = li * n_tr
                    is_exc[idxs, c0:c0 + n_tr] = True
                    ks, kn, kc = _gen_np_kicks(50000 + gi * 97, n_tr, idxs, 50.0)
                    if ks.size > 0:
                        ks_l.append(ks); kn_l.append(kn); kc_l.append(kc + c0)
            batches.append((
                b0, b1, is_exc,
                np.concatenate(ks_l) if ks_l else np.zeros(0, dtype=np.int32),
                np.concatenate(kn_l) if kn_l else np.zeros(0, dtype=np.int32),
                np.concatenate(kc_l) if kc_l else np.zeros(0, dtype=np.int32),
            ))
        return batches

    batches_5 = build_106_batches(5, min(106, max(18, batch_types * 3)))
    batches_30 = build_106_batches(30, batch_types)

    def eval_106_gpu(gg, ws: float, batches, n_tr: int):
        rates = np.zeros(len(type_indices), dtype=np.float64)
        b_rates = np.zeros(len(type_indices), dtype=np.float64)
        n_cap = 0
        for b0, b1, is_exc, ks, kn, kc in batches:
            wc, tsp, cap_c = run_gpu_batch(gg, ws, is_exc, ks, kn, kc, max_brain_hz_cap=15.0)
            for li, gi in enumerate(range(b0, b1)):
                c0 = li * n_tr
                rates[gi] = float(wc[:, c0:c0 + n_tr].mean())
                b_rates[gi] = float(tsp[c0:c0 + n_tr].mean() / n_nodes)
                if cap_c[c0:c0 + n_tr].any():
                    n_cap += 1
        return rates, float(b_rates.mean()), n_cap

    prot_set = set(int(x) for x in mn9_idx)
    w_s_base = (w_raw_s * np.float32(0.275)).astype(np.float64)
    is_exc_s10 = np.zeros((n_nodes, 10), dtype=bool)
    is_exc_s10[sugar_idx, :] = True

    results = []
    for k in range(10):
        tk = time.time()
        seed_rw = (9301 if compartment_preserving else 9201) + k
        if compartment_preserving:
            post_rw, rw_meta = build_compartment_rewired_graph(
                indptr, post_real, pre_s, node_class, node_comp, orig_in_pos, orig_in_neg, w_raw_s, prot_set, seed_rw
            )
        else:
            post_rw, rw_meta = smb.build_rewired_graph(
                indptr, post_real, w_s_base, pre_s, node_class, orig_in_pos, orig_in_neg, prot_set, seed_rw
            )
        gg = GPUGraphV7(indptr, post_rw, w_raw_s)

        # Fixed 0.275 mV 30-trial evaluation (always run in compartment-preserving mode for direct comparison)
        fixed_rates = None
        fixed_brain_hz = None
        if compartment_preserving:
            fr, fb, _ = eval_106_gpu(gg, 0.275, batches_30, 30)
            fixed_rates = [float(x) for x in fr]
            fixed_brain_hz = float(fb)

        # C1: Sugar 100 Hz anchor scan
        sugar_grid = [0.275, 0.40, 0.55, 0.70, 0.90, 1.20, 1.60, 2.00, 2.50]
        sugar_scan = []
        for ws in sugar_grid:
            wc, tsp, cap_c = run_gpu_batch(gg, ws, is_exc_s10, sugar_ks10, sugar_kn10, sugar_kc10, max_brain_hz_cap=15.0)
            sugar_scan.append({
                "w_syn_mv": ws,
                "mn9_mean_hz": float(wc.mean()),
                "brain_mean_hz": float(tsp.mean() / n_nodes),
                "seizure_capped": bool(cap_c.any()),
            })

        # C2: Activity-matched bisection
        lo_w = 0.275
        r_lo, b_lo, c_lo = eval_106_gpu(gg, lo_w, batches_5, 5)
        m_lo = float(r_lo.mean())
        hi_w, m_hi, b_hi, c_hi = lo_w, m_lo, b_lo, c_lo
        bisect_trace = [{"w_syn": lo_w, "mean_106_hz": m_lo, "brain_hz": b_lo, "n_capped": c_lo}]

        for cand_w in (0.45, 0.65, 0.85, 1.10, 1.40, 1.80, 2.20, 2.50):
            if m_hi >= target_mean_hz or c_hi >= 5:
                break
            r_c, b_c, c_c = eval_106_gpu(gg, cand_w, batches_5, 5)
            m_c = float(r_c.mean())
            bisect_trace.append({"w_syn": cand_w, "mean_106_hz": m_c, "brain_hz": b_c, "n_capped": c_c})
            if m_c < target_mean_hz and c_c < 5:
                lo_w, m_lo = cand_w, m_c
                hi_w, m_hi, b_hi, c_hi = cand_w, m_c, b_c, c_c
            else:
                hi_w, m_hi, b_hi, c_hi = cand_w, m_c, b_c, c_c
                break

        if c_hi < 5 and abs(m_hi - target_mean_hz) < abs(m_lo - target_mean_hz):
            best_w = hi_w
            best_err = abs(m_hi - target_mean_hz)
        else:
            best_w = lo_w
            best_err = abs(m_lo - target_mean_hz)

        if hi_w > lo_w and m_hi >= target_mean_hz and c_hi < 5:
            a_w, b_w = lo_w, hi_w
            for _ in range(6):
                mid = 0.5 * (a_w + b_w)
                r_m, b_m, c_m = eval_106_gpu(gg, mid, batches_5, 5)
                m_m = float(r_m.mean())
                bisect_trace.append({"w_syn": mid, "mean_106_hz": m_m, "brain_hz": b_m, "n_capped": c_m})
                if abs(m_m - target_mean_hz) < best_err and c_m == 0:
                    best_err = abs(m_m - target_mean_hz)
                    best_w = mid
                if m_m < target_mean_hz and c_m == 0:
                    a_w = mid
                else:
                    b_w = mid
                if abs(m_m - target_mean_hz) <= 0.04 * target_mean_hz and c_m == 0:
                    best_w = mid
                    break

        final_rates, final_brain_hz, final_capped = eval_106_gpu(gg, best_w, batches_30, 30)
        res = {
            "k": k,
            "seed": seed_rw,
            "rewire_meta": rw_meta,
            "sugar_anchor_scan": sugar_scan,
            "bisect_trace": bisect_trace,
            "w_syn_matched_mv": float(best_w),
            "mean_106_hz_matched": float(final_rates.mean()),
            "nonzero_106_matched": int((final_rates > 0.0).sum()),
            "brain_mean_hz_matched": float(final_brain_hz),
            "n_capped_types": int(final_capped),
            "rates_106_matched": [float(x) for x in final_rates],
            "elapsed_sec": float(time.time() - tk),
        }
        if fixed_rates is not None:
            res["rates_106_fixed_0275"] = fixed_rates
            res["brain_mean_hz_fixed_0275"] = fixed_brain_hz
        results.append(res)
        ap_all = stepwise_auprc(y_bin, final_rates)
        ap_nd = stepwise_auprc(y_bin[~is_direct_v783], final_rates[~is_direct_v783])
        ap_nov = stepwise_auprc(y_bin[~prior_pub], final_rates[~prior_pub])
        print(
            f"  [GPU Test C rewProt#{k}] W*={best_w:.4f} mV, mean={final_rates.mean():.3f} Hz "
            f"(nz={res['nonzero_106_matched']}/106, brain={final_brain_hz:.3f} Hz, cap={final_capped}) -> "
            f"AUPRC all106={ap_all:.4f}, nd83={ap_nd:.4f}, nov95={ap_nov:.4f} ({res['elapsed_sec']:.1f}s)",
            flush=True,
        )
        del gg
        torch.cuda.empty_cache()

    real_all = stepwise_auprc(y_bin, real_v783_50)
    real_nd = stepwise_auprc(y_bin[~is_direct_v783], real_v783_50[~is_direct_v783])
    real_nov = stepwise_auprc(y_bin[~prior_pub], real_v783_50[~prior_pub])
    per_g_all, per_g_nd, per_g_nov, mat_rates = [], [], [], []
    for res in results:
        r_arr = np.array(res["rates_106_matched"], dtype=np.float64)
        mat_rates.append(r_arr)
        per_g_all.append(stepwise_auprc(y_bin, r_arr))
        per_g_nd.append(stepwise_auprc(y_bin[~is_direct_v783], r_arr[~is_direct_v783]))
        per_g_nov.append(stepwise_auprc(y_bin[~prior_pub], r_arr[~prior_pub]))
    ens_rates = np.mean(mat_rates, axis=0)

    sugar_by_w, brain_by_w, cap_by_w = defaultdict(list), defaultdict(list), defaultdict(int)
    for res in results:
        for row in res["sugar_anchor_scan"]:
            w = row["w_syn_mv"]
            sugar_by_w[w].append(row["mn9_mean_hz"])
            brain_by_w[w].append(row["brain_mean_hz"])
            if row["seizure_capped"]:
                cap_by_w[w] += 1

    out = {
        "compute_backend": f"GPU FP64 ({torch.cuda.get_device_name(0)})",
        "real_v783_reference": {
            "target_mean_106_hz": target_mean_hz,
            "sugar_100hz_sanity_mn9_hz": float(wc_chk.mean()),
            "all_106_auprc": real_all,
            "nondirect_83_auprc": real_nd,
            "novel_95_auprc": real_nov,
        },
        "c1_sugar_anchor_summary": [
            {
                "w_syn_mv": w,
                "mn9_mean_hz_across_k10": float(np.mean(sugar_by_w[w])),
                "mn9_max_hz_across_k10": float(np.max(sugar_by_w[w])),
                "brain_mean_hz_across_k10": float(np.mean(brain_by_w[w])),
                "n_graphs_seizure_capped": int(cap_by_w[w]),
            }
            for w in sorted(sugar_by_w.keys())
        ],
        "c2_activity_matched_summary": {
            "w_syn_matched_mean_sd_min_max": [
                float(np.mean([r["w_syn_matched_mv"] for r in results])),
                float(np.std([r["w_syn_matched_mv"] for r in results])),
                float(np.min([r["w_syn_matched_mv"] for r in results])),
                float(np.max([r["w_syn_matched_mv"] for r in results])),
            ],
            "mean_106_hz_across_k10": float(np.mean([r["mean_106_hz_matched"] for r in results])),
            "nonzero_106_mean": float(np.mean([r["nonzero_106_matched"] for r in results])),
            "brain_mean_hz_across_k10": float(np.mean([r["brain_mean_hz_matched"] for r in results])),
            "all_106": {
                "real_auprc": real_all,
                "matched_per_graph": per_g_all,
                "matched_mean": float(np.mean(per_g_all)),
                "matched_sd": float(np.std(per_g_all)),
                "real_wins_of_10": int(sum(1 for x in per_g_all if real_all > x)),
                "matched_ensemble_auprc": stepwise_auprc(y_bin, ens_rates),
            },
            "nondirect_83": {
                "real_auprc": real_nd,
                "matched_per_graph": per_g_nd,
                "matched_mean": float(np.mean(per_g_nd)),
                "matched_sd": float(np.std(per_g_nd)),
                "real_wins_of_10": int(sum(1 for x in per_g_nd if real_nd > x)),
                "matched_ensemble_auprc": stepwise_auprc(y_bin[~is_direct_v783], ens_rates[~is_direct_v783]),
            },
            "novel_95": {
                "real_auprc": real_nov,
                "matched_per_graph": per_g_nov,
                "matched_mean": float(np.mean(per_g_nov)),
                "matched_sd": float(np.std(per_g_nov)),
                "real_wins_of_10": int(sum(1 for x in per_g_nov if real_nov > x)),
                "matched_ensemble_auprc": stepwise_auprc(y_bin[~prior_pub], ens_rates[~prior_pub]),
            },
        },
        "per_graph_details": results,
        "elapsed_sec": float(time.time() - t0),
    }
    if compartment_preserving:
        fx_all, fx_nd, fx_nov, fx_mat = [], [], [], []
        for res in results:
            fa = np.array(res["rates_106_fixed_0275"], dtype=np.float64)
            fx_mat.append(fa)
            fx_all.append(stepwise_auprc(y_bin, fa))
            fx_nd.append(stepwise_auprc(y_bin[~is_direct_v783], fa[~is_direct_v783]))
            fx_nov.append(stepwise_auprc(y_bin[~prior_pub], fa[~prior_pub]))
        fx_ens = np.mean(fx_mat, axis=0)
        out["fixed_0275_summary"] = {
            "all_106": {
                "per_graph": fx_all, "mean": float(np.mean(fx_all)), "min_max": [float(np.min(fx_all)), float(np.max(fx_all))],
                "real_wins_of_10": int(sum(1 for x in fx_all if real_all > x)),
                "ensemble_auprc": stepwise_auprc(y_bin, fx_ens),
                "paired_boot_vs_real": paired_bootstrap_auprc_p(y_bin, real_v783_50, fx_ens),
            },
            "nondirect_83": {
                "per_graph": fx_nd, "mean": float(np.mean(fx_nd)), "min_max": [float(np.min(fx_nd)), float(np.max(fx_nd))],
                "real_wins_of_10": int(sum(1 for x in fx_nd if real_nd > x)),
                "ensemble_auprc": stepwise_auprc(y_bin[~is_direct_v783], fx_ens[~is_direct_v783]),
                "paired_boot_vs_real": paired_bootstrap_auprc_p(y_bin[~is_direct_v783], real_v783_50[~is_direct_v783], fx_ens[~is_direct_v783]),
            },
            "novel_95": {
                "per_graph": fx_nov, "mean": float(np.mean(fx_nov)), "min_max": [float(np.min(fx_nov)), float(np.max(fx_nov))],
                "real_wins_of_10": int(sum(1 for x in fx_nov if real_nov > x)),
                "ensemble_auprc": stepwise_auprc(y_bin[~prior_pub], fx_ens[~prior_pub]),
                "paired_boot_vs_real": paired_bootstrap_auprc_p(y_bin[~prior_pub], real_v783_50[~prior_pub], fx_ens[~prior_pub]),
            },
        }
        out["super_class_names"] = meta.get("super_class_names", [])
        out["c2_activity_matched_summary"]["all_106"]["per_graph"] = per_g_all
        out["c2_activity_matched_summary"]["all_106"]["mean"] = float(np.mean(per_g_all))
        out["c2_activity_matched_summary"]["all_106"]["min_max"] = [float(np.min(per_g_all)), float(np.max(per_g_all))]
        out["c2_activity_matched_summary"]["all_106"]["ensemble_auprc"] = stepwise_auprc(y_bin, ens_rates)
        out["c2_activity_matched_summary"]["nondirect_83"]["per_graph"] = per_g_nd
        out["c2_activity_matched_summary"]["nondirect_83"]["mean"] = float(np.mean(per_g_nd))
        out["c2_activity_matched_summary"]["nondirect_83"]["min_max"] = [float(np.min(per_g_nd)), float(np.max(per_g_nd))]
        out["c2_activity_matched_summary"]["nondirect_83"]["ensemble_auprc"] = stepwise_auprc(y_bin[~is_direct_v783], ens_rates[~is_direct_v783])
        out["c2_activity_matched_summary"]["novel_95"]["per_graph"] = per_g_nov
        out["c2_activity_matched_summary"]["novel_95"]["mean"] = float(np.mean(per_g_nov))
        out["c2_activity_matched_summary"]["novel_95"]["min_max"] = [float(np.min(per_g_nov)), float(np.max(per_g_nov))]
        out["c2_activity_matched_summary"]["novel_95"]["ensemble_auprc"] = stepwise_auprc(y_bin[~prior_pub], ens_rates[~prior_pub])
        out["c2_activity_matched_summary"]["all_106"]["paired_boot_vs_real"] = paired_bootstrap_auprc_p(y_bin, real_v783_50, ens_rates)
        out["c2_activity_matched_summary"]["nondirect_83"]["paired_boot_vs_real"] = paired_bootstrap_auprc_p(y_bin[~is_direct_v783], real_v783_50[~is_direct_v783], ens_rates[~is_direct_v783])
        out["c2_activity_matched_summary"]["novel_95"]["paired_boot_vs_real"] = paired_bootstrap_auprc_p(y_bin[~prior_pub], real_v783_50[~prior_pub], ens_rates[~prior_pub])
        m_idx = [r["k"] for r in results if r["mean_106_hz_matched"] >= 1.85 and r["n_capped_types"] == 0]
        out["c2_activity_matched_summary"]["exact_matched_indices"] = m_idx
        if m_idx:
            out["c2_activity_matched_summary"]["exact_matched_summary"] = {
                "n_graphs": len(m_idx),
                "all_106_mean": float(np.mean([per_g_all[i] for i in m_idx])),
                "all_106_real_wins": int(sum(real_all > per_g_all[i] for i in m_idx)),
                "nondirect_83_mean": float(np.mean([per_g_nd[i] for i in m_idx])),
                "nondirect_83_real_wins": int(sum(real_nd > per_g_nd[i] for i in m_idx)),
                "novel_95_mean": float(np.mean([per_g_nov[i] for i in m_idx])),
                "novel_95_real_wins": int(sum(real_nov > per_g_nov[i] for i in m_idx)),
            }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[colab_test_c] ALL 10 GRAPHS DONE in {time.time()-t0:.1f}s -> {out_path}", flush=True)


# ===========================================================================
# PHASE 4: MERGE INTO LOCKED `mn9_portability_results.json`
# ===========================================================================
def run_phase_merge(local_static_p: Path, test_c_p: Path, colab_p: Path, out_p: Path) -> None:
    ls = json.loads(local_static_p.read_text(encoding="utf-8"))
    tc = json.loads(test_c_p.read_text(encoding="utf-8"))
    co = json.loads(colab_p.read_text(encoding="utf-8"))
    merged = {
        "protocol_commit": "36ffab2",
        "co_primary_endpoints_50hz": ls["co_primary_endpoints_50hz"],
        "secondary_bh_fdr": ls["secondary_bh_fdr"],
        "test_a_transmitter_sensitivity": co["test_a_transmitter_sensitivity"],
        "test_b_density_and_dimorphism": {
            "banc_vs_v783_density": ls["test_b_banc_vs_v783_density"],
            "malecns_vs_v783_density_and_dimorphism": co["test_b_malecns_density_and_dimorphism"],
            "malecns_density_scaled_sim": co["test_b_malecns_density_scaled_sim"],
        },
        "test_c_v783_activity_matched_rewire": tc,
        "test_d_44ctrl_three_brain_null": co["test_d_44ctrl_three_brain_null"],
        "compute_provenance": {
            "colab_gpu": co["gpu_name"],
            "local_static_sec": ls["elapsed_sec"],
            "local_test_c_sec": tc["elapsed_sec"],
            "colab_malecns_sec": co["elapsed_sec"],
        },
    }
    out_p.write_text(json.dumps(merged, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[merge] Wrote {out_p} ({out_p.stat().st_size / 1e3:.1f} KB)", flush=True)


# ===========================================================================
# REVISION PHASES: SELK EXACT WIDEST PATHS, SCATTER AUDIT, COMPARTMENT NULL, A2+B3
# ===========================================================================
def run_phase_rev_local_static(
    rev_static_out: Path, bundle_npz: Path, meta_json: Path, manifest_out: Path, v783_ref_out: Path
) -> None:
    sys.path.insert(0, str(BASE_DIR))
    import mn9_transfer_benchmark as mtb
    import shiu_mn9_benchmark as smb

    t0 = time.time()
    print("[rev_local_static] Loading FlyWire v783 & BANC v888 graphs...", flush=True)
    comp783 = CACHE_DIR / "data/Completeness_783.csv"
    conn783 = CACHE_DIR / "data/Connectivity_783.parquet"
    ann783 = CACHE_DIR / "data/Supplemental_file1_neuron_annotations.tsv"
    g783 = mtb.load_graph_csr(comp783, conn783)
    fids783 = g783["flyids"]
    n783 = int(len(fids783))
    idx783 = g783["flyid2idx"]

    v783_sc_by_fid: dict[int, str] = {}
    v783_ct_by_fid: dict[int, str] = {}
    with ann783.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rid_s = row.get("root_id", "")
            if rid_s and rid_s.isdigit():
                rid = int(rid_s)
                if rid in idx783:
                    v783_sc_by_fid[rid] = (row.get("super_class") or "").strip() or "unknown"
                    v783_ct_by_fid[rid] = (row.get("cell_type") or "").strip() or (row.get("hemibrain_type") or "").strip()

    sc_list_v7 = [v783_sc_by_fid.get(int(fid), "unknown") for fid in fids783]
    sc_names_v7 = sorted(set(sc_list_v7))
    sc_to_id_v7 = {s: i for i, s in enumerate(sc_names_v7)}
    node_comp_v7 = np.array([sc_to_id_v7[s] for s in sc_list_v7], dtype=np.int32)

    def v7_reg(sc: str) -> str:
        if sc == "optic":
            return "optic_lobe"
        if "ascending" in sc:
            return "ventral_nerve_cord"
        return "central_brain"

    reg_names_3 = ["central_brain", "optic_lobe", "ventral_nerve_cord"]
    reg_map_3 = {r: i for i, r in enumerate(reg_names_3)}
    v7_reg_ids = np.array([reg_map_3[v7_reg(s)] for s in sc_list_v7], dtype=np.int32)
    scatter_v7 = audit_rewire_scatter(g783["pre_s"], g783["post_s"], v7_reg_ids, reg_names_3)
    scatter_v7["super_class_counts"] = dict(Counter(sc_list_v7))

    # Build v783 bundle for Colab A100 super_class-preserving Test C
    v7_res = json.loads((BASE_DIR / "mn9_transfer_results.json").read_text(encoding="utf-8"))
    ba_res = json.loads((BASE_DIR / "mn9_banc_results.json").read_text(encoding="utf-8"))
    mc_res = json.loads((BASE_DIR / "mn9_malecns_results.json").read_text(encoding="utf-8"))
    port = json.loads((BASE_DIR / "mn9_portability_results.json").read_text(encoding="utf-8"))
    per_type_v7 = v7_res["per_type_task1"]
    v783_ids_by_name = {r["name"]: [int(x) for x in r["v783_ids"]] for r in v7_res["mapping_audit"]["per_type_sez106"]}
    type_indices_v7 = [[int(idx783[f]) for f in v783_ids_by_name[r["name"]] if f in idx783] for r in per_type_v7]
    mn9_anat = ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]
    mn9_idx_v7 = np.array([int(idx783[int(mn9_anat["mn9_l_v783"])]), int(idx783[int(mn9_anat["mn9_r_v783"])])], dtype=np.int32)
    sugar_idx_v7 = np.array(
        [idx783[720575940639259967 if int(f) == 720575940620900446 else int(f)] for f in smb.SUGAR_GRN_FIDS],
        dtype=np.int32,
    )
    y_bin = np.array([1 if r["y_max"] > 0.0 else 0 for r in per_type_v7], dtype=np.int32)
    prior_pub = np.array([bool(r["prior_published"]) for r in per_type_v7], dtype=bool)
    is_dir_v7 = np.array([bool(r["is_direct_1hop_v783"]) for r in per_type_v7], dtype=bool)
    real_v7_50 = np.array([float(r["v783_real_lif_50hz_mean"]) for r in per_type_v7], dtype=np.float64)

    np.savez(
        bundle_npz,
        indptr=g783["indptr"],
        post_s=g783["post_s"],
        pre_s=g783["pre_s"],
        w_raw_s=g783["w_raw_s"],
        node_class=g783["node_class"],
        node_comp=node_comp_v7,
        orig_in_pos=g783["orig_in_pos"],
        orig_in_neg=g783["orig_in_neg"],
        mn9_idx=mn9_idx_v7,
        sugar_idx=sugar_idx_v7,
    )
    meta_json.write_text(
        json.dumps(
            {
                "super_class_names": sc_names_v7,
                "type_indices": type_indices_v7,
                "y_bin": y_bin.tolist(),
                "is_direct_v783": is_dir_v7.tolist(),
                "prior_pub": prior_pub.tolist(),
                "real_v783_50": real_v7_50.tolist(),
            }
        ),
        encoding="utf-8",
    )

    # Also build v783_ref & manifest for rev_colab_malecns (self-contained)
    s_in_783 = np.zeros(n783, dtype=np.float64)
    s_out_783 = np.zeros(n783, dtype=np.float64)
    abs_w7 = np.abs(g783["w_raw_s"].astype(np.float64))
    np.add.at(s_in_783, g783["post_s"], abs_w7)
    np.add.at(s_out_783, g783["pre_s"], abs_w7)
    ct_to_idxs_783: dict[str, list[int]] = defaultdict(list)
    for i, fid in enumerate(fids783):
        ct = v783_ct_by_fid.get(int(fid), "")
        if ct:
            ct_to_idxs_783[ct].append(i)
    v783_ct_stats = {}
    for ct, idxs in ct_to_idxs_783.items():
        top_sc = Counter(sc_list_v7[i] for i in idxs).most_common(1)[0][0]
        v783_ct_stats[ct] = {
            "n_cells": len(idxs),
            "region": v7_reg(top_sc),
            "mean_s_in": float(np.mean([s_in_783[i] for i in idxs])),
            "total_s_in": float(np.sum([s_in_783[i] for i in idxs])),
            "mean_s_out": float(np.mean([s_out_783[i] for i in idxs])),
        }

    # Load BANC v888 main graph & metadata for SELK exact widest paths + scatter audit
    bdir = CACHE_DIR / "data/banc"
    meta_tbl = pf.read_table(bdir / "banc_888_meta.feather")
    m_rows: dict[int, dict] = {}
    for vals in zip(
        meta_tbl["banc_888_id"].to_pylist(),
        meta_tbl["region"].to_pylist(),
        meta_tbl["super_class"].to_pylist(),
        meta_tbl["cell_type"].to_pylist(),
        meta_tbl["fafb_cell_type"].to_pylist(),
        meta_tbl["side"].to_pylist(),
        meta_tbl["proofread"].to_pylist(),
        meta_tbl["roughly_proofread"].to_pylist(),
    ):
        if vals[0] is None:
            continue
        rid = int(vals[0])
        m_rows[rid] = {
            "region": str(vals[1] or ""), "super_class": str(vals[2] or ""),
            "cell_type": str(vals[3] or ""), "fafb_cell_type": str(vals[4] or ""),
            "side": str(vals[5] or ""), "proofread": str(vals[6] or ""), "roughly_proofread": str(vals[7] or ""),
        }
    nt_csv: dict[int, str] = {}
    with (bdir / "banc_888_neurotransmitter_prediction_v2.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nt_csv[int(row["root_id"])] = row.get("neurotransmitter_predicted") or ""
    gb = np.load(CACHE_DIR / "banc_ckpt/banc_main_graph.npz")
    main_cand_b = np.array(
        sorted(b for b, d in m_rows.items() if (d["proofread"] == "TRUE" or d["roughly_proofread"] == "TRUE") and d["super_class"] != "glia"),
        dtype=np.int64,
    )
    el_b = pf.read_table(bdir / "banc_888_edgelist_simple_v2.feather")
    ep_b = el_b["pre"].to_numpy().astype(np.int64)
    eq_b = el_b["post"].to_numpy().astype(np.int64)
    ns_b = ep_b != eq_b
    ep_b, eq_b = ep_b[ns_b], eq_b[ns_b]
    nt_ids_b = np.array(sorted(nt_csv.keys()), dtype=np.int64)
    nt_sgn_b = np.array([nt_to_sign(nt_csv[int(b)]) for b in nt_ids_b], dtype=np.int8)
    p_nt = np.minimum(np.searchsorted(nt_ids_b, ep_b), len(nt_ids_b) - 1)
    v_nt = np.where(nt_ids_b[p_nt] == ep_b, nt_sgn_b[p_nt], 0) != 0
    keep_b = np.isin(ep_b, main_cand_b) & np.isin(eq_b, main_cand_b) & v_nt
    bids_b = np.union1d(ep_b[keep_b], eq_b[keep_b])
    assert len(bids_b) == int(gb["n_nodes"]), f"BANC node count mismatch: {len(bids_b)} vs {int(gb['n_nodes'])}"
    bid_to_idx_b = {int(b): i for i, b in enumerate(bids_b)}
    is_vnc_b = np.array([m_rows[int(b)]["region"] == "ventral_nerve_cord" for b in bids_b], dtype=bool)
    reg_b_ids = np.array(
        [reg_map_3[m_rows[int(b)]["region"]] if m_rows[int(b)]["region"] in reg_map_3 else 0 for b in bids_b],
        dtype=np.int32,
    )
    scatter_banc = audit_rewire_scatter(gb["pre_s"], gb["post_s"], reg_b_ids, reg_names_3)

    mn9_ids_b = [int(mn9_anat["mn9_l_banc"]), int(mn9_anat["mn9_r_banc"])]
    selk_ids_b = [int(x) for x in ba_res["selk_full_cns_diagnosis"]["selk_banc_ids"]]
    mn9_set_b = {bid_to_idx_b[x] for x in mn9_ids_b}
    selk_idxs_b = [bid_to_idx_b[x] for x in selk_ids_b]

    def _desc_b(u: int) -> dict:
        bid = int(bids_b[u])
        md = m_rows[bid]
        return {"banc_id": bid, "cell_type": md["cell_type"] or md["fafb_cell_type"], "region": md["region"], "side": md["side"], "nt": nt_csv.get(bid, "")}

    selk_wp_banc = exact_widest_exc_paths_via_vnc(
        gb["indptr"], gb["post_s"], gb["w_raw_s"], gb["node_class"], selk_idxs_b, mn9_set_b, is_vnc_b, _desc_b, max_k=5
    )
    print("  [BANC SELK widest paths via any VNC]:", {k: v["bottleneck_weight"] for k, v in selk_wp_banc["via_any_vnc_by_hop"].items()}, "global:", selk_wp_banc["global_widest_via_any_vnc"]["bottleneck_weight"], "hops:", selk_wp_banc["global_widest_via_any_vnc"]["hops"], flush=True)

    # Recompute exact Test A non-direct variant AUPRCs (N=87 excl TH_VUM, N=86 excl TH_VUM+justice, N=88 fixed) + 95% CIs!
    names_106 = [r["name"] for r in per_type_v7]
    mc_by_name = {r["name"]: r for r in mc_res["per_type_3brains"]}
    is_dir_m_primary = np.array([bool(mc_by_name[nm]["is_direct_malecns"]) for nm in names_106], dtype=bool)
    ct_by_name = {r["name"]: r["v783_cell_types"] for r in mc_res["mapping_audit"]["per_type_sez106"]}
    thv_idx = names_106.index("TH_VUM")
    just_idx = names_106.index("justice")
    mask_nd88 = ~is_dir_m_primary
    mask_nd87 = mask_nd88.copy(); mask_nd87[thv_idx] = False
    mask_nd86 = mask_nd87.copy(); mask_nd86[just_idx] = False

    def _eval_rates_ci(rates_list: list[float]) -> dict:
        r = np.asarray(rates_list, dtype=np.float64)
        return {
            "all_106_auprc": stepwise_auprc(y_bin, r),
            "all_106_ci95": stratified_auprc_ci95(y_bin, r),
            "novel_95_auprc": stepwise_auprc(y_bin[~prior_pub], r[~prior_pub]),
            "novel_95_ci95": stratified_auprc_ci95(y_bin[~prior_pub], r[~prior_pub]),
            "nondirect_fixed88_auprc": stepwise_auprc(y_bin[mask_nd88], r[mask_nd88]),
            "nondirect_fixed88_ci95": stratified_auprc_ci95(y_bin[mask_nd88], r[mask_nd88]),
            "nondirect_excl_thvum_87_auprc": stepwise_auprc(y_bin[mask_nd87], r[mask_nd87]),
            "nondirect_excl_thvum_87_ci95": stratified_auprc_ci95(y_bin[mask_nd87], r[mask_nd87]),
            "nondirect_excl_thvum_justice_86_auprc": stepwise_auprc(y_bin[mask_nd86], r[mask_nd86]),
            "nondirect_excl_thvum_justice_86_ci95": stratified_auprc_ci95(y_bin[mask_nd86], r[mask_nd86]),
        }

    test_a_nd_audit = {}
    for var_key in ("A1a_diatom18_thvum_plus1", "A1b_diatom20_thvum_plus1", "A2_all_unclear_predicted_nt"):
        test_a_nd_audit[var_key] = {}
        for hz in ("50hz", "100hz", "200hz"):
            if hz in port["test_a_transmitter_sensitivity"][var_key]:
                test_a_nd_audit[var_key][hz] = _eval_rates_ci(port["test_a_transmitter_sensitivity"][var_key][hz]["all_106_rates_mean"])
    b3_ci_audit = {
        hz: _eval_rates_ci(port["test_b_density_and_dimorphism"]["malecns_density_scaled_sim"][hz]["all_106_rates_mean"])
        for hz in ("50hz", "200hz")
    }
    print(f"  [Test A1a 50Hz] nd88={test_a_nd_audit['A1a_diatom18_thvum_plus1']['50hz']['nondirect_fixed88_auprc']:.4f}, nd87={test_a_nd_audit['A1a_diatom18_thvum_plus1']['50hz']['nondirect_excl_thvum_87_auprc']:.4f}, nd86={test_a_nd_audit['A1a_diatom18_thvum_plus1']['50hz']['nondirect_excl_thvum_justice_86_auprc']:.4f}", flush=True)
    print(f"  [Test A2  50Hz] nd88={test_a_nd_audit['A2_all_unclear_predicted_nt']['50hz']['nondirect_fixed88_auprc']:.4f}, nd87={test_a_nd_audit['A2_all_unclear_predicted_nt']['50hz']['nondirect_excl_thvum_87_auprc']:.4f}, nd86={test_a_nd_audit['A2_all_unclear_predicted_nt']['50hz']['nondirect_excl_thvum_justice_86_auprc']:.4f}", flush=True)

    # Save minimal manifest & v783_ref for rev_colab_malecns (reusing exact v783 cell types from MaleCNS mapping audit)
    sez_v7_banc = {}
    for i, r in enumerate(per_type_v7):
        nm = r["name"]
        vi = type_indices_v7[i]
        sez_v7_banc[nm] = {
            "v783_n": len(vi), "v783_mean_s_in": float(np.mean([s_in_783[k] for k in vi])) if vi else 0.0,
            "v783_mean_s_out": float(np.mean([s_out_783[k] for k in vi])) if vi else 0.0,
            "v783_dimorphism": [], "v783_fru_dsx": [], "banc_n": 0, "banc_mean_s_in": 0.0, "banc_mean_s_out": 0.0,
        }
    v783_ref_out.write_text(json.dumps({"v783_celltype_stats": v783_ct_stats, "sez_v783_banc_stats": sez_v7_banc, "all48_names": [], "neg_inh_44": [], "top15_v783": {}, "top15_banc": {}}), encoding="utf-8")
    manifest_out.write_text(
        json.dumps(
            {
                "sez106": [
                    {
                        "name": r["name"], "y_bin": int(y_bin[i]), "prior_published": bool(prior_pub[i]),
                        "v783_50hz": float(real_v7_50[i]), "v783_cell_types": ct_by_name[r["name"]],
                    }
                    for i, r in enumerate(per_type_v7)
                ]
            }
        ),
        encoding="utf-8",
    )

    out_payload = {
        "selk_vnc_exact_widest_paths_banc": selk_wp_banc,
        "scatter_v783": scatter_v7,
        "scatter_banc": scatter_banc,
        "test_a_nondirect_variant_audit": test_a_nd_audit,
        "test_b3_bootstrap_ci_audit": b3_ci_audit,
        "elapsed_sec": float(time.time() - t0),
    }
    rev_static_out.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    print(f"[rev_local_static] DONE in {time.time()-t0:.1f}s -> {rev_static_out}", flush=True)


def _init_comp_c_worker(bundle_npz: str):
    d = np.load(bundle_npz)
    _TEST_C_STATE["indptr"] = d["indptr"]
    _TEST_C_STATE["post_s"] = d["post_s"]
    _TEST_C_STATE["pre_s"] = d["pre_s"]
    _TEST_C_STATE["w_raw_s"] = d["w_raw_s"]
    _TEST_C_STATE["node_class"] = d["node_class"]
    _TEST_C_STATE["node_comp"] = d["node_comp"]
    _TEST_C_STATE["orig_in_pos"] = d["orig_in_pos"]
    _TEST_C_STATE["orig_in_neg"] = d["orig_in_neg"]
    _TEST_C_STATE["mn9_idx"] = d["mn9_idx"].astype(np.int32)
    dummy_w = np.zeros(1, dtype=np.float64)
    _run_lif_v783_fast_with_cap(
        np.array([0, 0], dtype=np.int64), np.zeros(0, dtype=np.int32), dummy_w, 0.275,
        np.array([0], dtype=np.int32), 50.0, np.array([0], dtype=np.int32), 1, 1, 15.0
    )


def _eval_one_comp_c_graph(job: dict) -> dict:
    t0 = time.time()
    k = int(job["k"])
    seed_rw = int(job["seed"])
    type_indices = [np.array(x, dtype=np.int32) for x in job["type_indices"]]
    target_mean_hz = float(job["target_mean_hz"])
    st = _TEST_C_STATE
    indptr = st["indptr"]
    w_raw = st["w_raw_s"].astype(np.float64)
    mn9_idx = st["mn9_idx"]
    prot_set = set(int(x) for x in mn9_idx)

    post_rw, rw_meta = build_compartment_rewired_graph(
        indptr, st["post_s"], st["pre_s"], st["node_class"], st["node_comp"],
        st["orig_in_pos"], st["orig_in_neg"], st["w_raw_s"], prot_set, seed_rw
    )

    def eval_106_mean(ws: float, n_tr: int):
        rates = np.zeros(len(type_indices), dtype=np.float64)
        b_rates = []
        n_cap = 0
        w_s = w_raw * ws
        for i, idxs in enumerate(type_indices):
            if idxs.size == 0:
                continue
            cnt, b_hz, cap = _run_lif_v783_fast_with_cap(
                indptr, post_rw, w_s, ws, idxs, 50.0, mn9_idx, n_tr, 50000 + i * 97, 15.0
            )
            rates[i] = float(cnt.mean())
            b_rates.append(b_hz)
            if cap > 0:
                n_cap += 1
        return rates, float(np.mean(b_rates)) if b_rates else 0.0, n_cap

    # Fixed 0.275 mV (30 trials)
    fx_rates, fx_brain, fx_cap = eval_106_mean(0.275, 30)

    # Activity-matched calibration with explicit c_hi < 5 seizure guard
    lo_w = 0.275
    r_lo, b_lo, c_lo = eval_106_mean(lo_w, 5)
    m_lo = float(r_lo.mean())
    hi_w, m_hi, b_hi, c_hi = lo_w, m_lo, b_lo, c_lo
    bisect_trace = [{"w_syn": lo_w, "mean_106_hz": m_lo, "brain_hz": b_lo, "n_capped": c_lo}]

    for cand_w in (0.45, 0.65, 0.85, 1.10, 1.40, 1.80, 2.20, 2.50):
        if m_hi >= target_mean_hz or c_hi >= 5:
            break
        r_c, b_c, c_c = eval_106_mean(cand_w, 5)
        m_c = float(r_c.mean())
        bisect_trace.append({"w_syn": cand_w, "mean_106_hz": m_c, "brain_hz": b_c, "n_capped": c_c})
        if m_c < target_mean_hz and c_c < 5:
            lo_w, m_lo = cand_w, m_c
            hi_w, m_hi, b_hi, c_hi = cand_w, m_c, b_c, c_c
        else:
            hi_w, m_hi, b_hi, c_hi = cand_w, m_c, b_c, c_c
            break

    if c_hi < 5 and abs(m_hi - target_mean_hz) < abs(m_lo - target_mean_hz):
        best_w, best_err = hi_w, abs(m_hi - target_mean_hz)
    else:
        best_w, best_err = lo_w, abs(m_lo - target_mean_hz)

    if hi_w > lo_w and m_hi >= target_mean_hz and c_hi < 5:
        a_w, b_w = lo_w, hi_w
        for _ in range(6):
            mid = 0.5 * (a_w + b_w)
            r_m, b_m, c_m = eval_106_mean(mid, 5)
            m_m = float(r_m.mean())
            bisect_trace.append({"w_syn": mid, "mean_106_hz": m_m, "brain_hz": b_m, "n_capped": c_m})
            if abs(m_m - target_mean_hz) < best_err and c_m == 0:
                best_err = abs(m_m - target_mean_hz)
                best_w = mid
            if m_m < target_mean_hz and c_m == 0:
                a_w = mid
            else:
                b_w = mid
            if abs(m_m - target_mean_hz) <= 0.04 * target_mean_hz and c_m == 0:
                best_w = mid
                break

    final_rates, final_brain_hz, final_capped = eval_106_mean(best_w, 30)
    return {
        "k": k, "seed": seed_rw, "rewire_meta": rw_meta, "bisect_trace": bisect_trace,
        "w_syn_matched_mv": float(best_w), "mean_106_hz_matched": float(final_rates.mean()),
        "nonzero_106_matched": int((final_rates > 0.0).sum()), "brain_mean_hz_matched": float(final_brain_hz),
        "n_capped_types": int(final_capped), "rates_106_matched": [float(x) for x in final_rates],
        "rates_106_fixed_0275": [float(x) for x in fx_rates], "brain_mean_hz_fixed_0275": float(fx_brain),
        "n_capped_fixed_0275": int(fx_cap), "elapsed_sec": float(time.time() - t0),
    }


_COMP_C_FORK: dict[str, Any] = {}


def _eval_one_type_fork(arg: tuple[int, float, int]) -> tuple[int, float, float, int]:
    i, ws, n_tr = arg
    st = _COMP_C_FORK
    idxs = st["type_indices"][i]
    if idxs.size == 0:
        return i, 0.0, 0.0, 0
    cnt, b_hz, cap = _run_lif_v783_fast_with_cap(
        st["indptr"], st["post_rw"], st["w_s"], ws, idxs, 50.0, st["mn9_idx"], n_tr, 50000 + i * 97, 15.0
    )
    return i, float(cnt.mean()), float(b_hz), int(cap)


def run_phase_rev_local_comp_c(bundle_npz: Path, meta_json: Path, out_path: Path, workers: int = 24) -> None:
    t0 = time.time()
    d = np.load(bundle_npz)
    indptr = d["indptr"]
    post_real = d["post_s"]
    pre_s = d["pre_s"]
    w_raw_s = d["w_raw_s"]
    w_raw_f64 = w_raw_s.astype(np.float64)
    node_class = d["node_class"]
    node_comp = d["node_comp"].astype(np.int32)
    orig_in_pos = d["orig_in_pos"]
    orig_in_neg = d["orig_in_neg"]
    mn9_idx = d["mn9_idx"].astype(np.int32)
    prot_set = set(int(x) for x in mn9_idx)

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    type_indices = [np.array(x, dtype=np.int32) for x in meta["type_indices"]]
    y_bin = np.array(meta["y_bin"], dtype=np.int32)
    is_direct_v783 = np.array(meta["is_direct_v783"], dtype=bool)
    prior_pub = np.array(meta["prior_pub"], dtype=bool)
    real_v783_50 = np.array(meta["real_v783_50"], dtype=np.float64)
    target_mean_hz = float(real_v783_50.mean())

    # Warm up Numba JIT once in parent before forking so all workers inherit compiled code
    _run_lif_v783_fast_with_cap(
        indptr, post_real, w_raw_f64 * 0.275, 0.275, type_indices[0][:1], 50.0, mn9_idx, 1, 50000, 15.0
    )
    print(f"[rev_local_comp_c] JIT warmed up, running K=10 compartment-preserving graphs with {workers} fork workers...", flush=True)

    ctx = mp.get_context("fork")

    def eval_106_par(post_rw: np.ndarray, ws: float, n_tr: int, early_stop_cap: int = 0) -> tuple[np.ndarray, float, int]:
        _COMP_C_FORK["indptr"] = indptr
        _COMP_C_FORK["post_rw"] = post_rw
        _COMP_C_FORK["w_s"] = w_raw_f64 * ws
        _COMP_C_FORK["mn9_idx"] = mn9_idx
        _COMP_C_FORK["type_indices"] = type_indices
        rates = np.zeros(len(type_indices), dtype=np.float64)
        b_rates = []
        n_cap = 0
        args_list = [(i, ws, n_tr) for i in range(len(type_indices))]
        pool = ctx.Pool(processes=workers)
        aborted = False
        try:
            for i, r_m, b_hz, cap in pool.imap_unordered(_eval_one_type_fork, args_list, chunksize=1):
                rates[i] = r_m
                if type_indices[i].size > 0:
                    b_rates.append(b_hz)
                if cap > 0:
                    n_cap += 1
                    if early_stop_cap > 0 and n_cap >= early_stop_cap:
                        aborted = True
                        pool.terminate()
                        break
            if not aborted:
                pool.close()
        finally:
            pool.join()
        return rates, (float(np.mean(b_rates)) if b_rates else 0.0), n_cap

    results = []
    for k in range(10):
        tk = time.time()
        seed_rw = 9301 + k
        post_rw, rw_meta = build_compartment_rewired_graph(
            indptr, post_real, pre_s, node_class, node_comp, orig_in_pos, orig_in_neg, w_raw_s, prot_set, seed_rw
        )
        fx_rates, fx_brain, fx_cap = eval_106_par(post_rw, 0.275, 30, early_stop_cap=0)

        lo_w = 0.275
        r_lo, b_lo, c_lo = eval_106_par(post_rw, lo_w, 5, early_stop_cap=5)
        m_lo = float(r_lo.mean())
        hi_w, m_hi, b_hi, c_hi = lo_w, m_lo, b_lo, c_lo
        bisect_trace = [{"w_syn": lo_w, "mean_106_hz": m_lo, "brain_hz": b_lo, "n_capped": c_lo}]

        for cand_w in (0.45, 0.65, 0.85, 1.10, 1.40, 1.80, 2.20, 2.50):
            if m_hi >= target_mean_hz or c_hi >= 5:
                break
            r_c, b_c, c_c = eval_106_par(post_rw, cand_w, 5, early_stop_cap=5)
            m_c = float(r_c.mean())
            bisect_trace.append({"w_syn": cand_w, "mean_106_hz": m_c, "brain_hz": b_c, "n_capped": c_c})
            print(f"    [compC #{k} bracket] W={cand_w:.3f} -> mean={m_c:.3f}Hz, brain={b_c:.3f}Hz, cap={c_c} ({time.time()-tk:.1f}s)", flush=True)
            if m_c < target_mean_hz and c_c < 5:
                lo_w, m_lo = cand_w, m_c
                hi_w, m_hi, b_hi, c_hi = cand_w, m_c, b_c, c_c
            else:
                hi_w, m_hi, b_hi, c_hi = cand_w, m_c, b_c, c_c
                break

        if c_hi < 5 and abs(m_hi - target_mean_hz) < abs(m_lo - target_mean_hz):
            best_w, best_err = hi_w, abs(m_hi - target_mean_hz)
        else:
            best_w, best_err = lo_w, abs(m_lo - target_mean_hz)

        if hi_w > lo_w and m_hi >= target_mean_hz and c_hi < 5:
            a_w, b_w = lo_w, hi_w
            for bstep in range(6):
                mid = 0.5 * (a_w + b_w)
                r_m, b_m, c_m = eval_106_par(post_rw, mid, 5, early_stop_cap=1)
                m_m = float(r_m.mean())
                bisect_trace.append({"w_syn": mid, "mean_106_hz": m_m, "brain_hz": b_m, "n_capped": c_m})
                print(f"    [compC #{k} bisect {bstep+1}/6] W={mid:.4f} -> mean={m_m:.3f}Hz, brain={b_m:.3f}Hz, cap={c_m} ({time.time()-tk:.1f}s)", flush=True)
                if abs(m_m - target_mean_hz) < best_err and c_m == 0:
                    best_err = abs(m_m - target_mean_hz)
                    best_w = mid
                if m_m < target_mean_hz and c_m == 0:
                    a_w = mid
                else:
                    b_w = mid
                if abs(m_m - target_mean_hz) <= 0.04 * target_mean_hz and c_m == 0:
                    best_w = mid
                    break

        final_rates, final_brain_hz, final_capped = eval_106_par(post_rw, best_w, 30, early_stop_cap=0)
        res = {
            "k": k, "seed": seed_rw, "rewire_meta": rw_meta, "bisect_trace": bisect_trace,
            "w_syn_matched_mv": float(best_w), "mean_106_hz_matched": float(final_rates.mean()),
            "nonzero_106_matched": int((final_rates > 0.0).sum()), "brain_mean_hz_matched": float(final_brain_hz),
            "n_capped_types": int(final_capped), "rates_106_matched": [float(x) for x in final_rates],
            "rates_106_fixed_0275": [float(x) for x in fx_rates], "brain_mean_hz_fixed_0275": float(fx_brain),
            "n_capped_fixed_0275": int(fx_cap), "elapsed_sec": float(time.time() - tk),
        }
        results.append(res)
        r_arr = final_rates
        f_arr = fx_rates
        print(
            f"  [compC #{k}] fixed all={stepwise_auprc(y_bin, f_arr):.4f} nd={stepwise_auprc(y_bin[~is_direct_v783], f_arr[~is_direct_v783]):.4f} | "
            f"W*={best_w:.4f} mean={r_arr.mean():.3f}Hz (nz={res['nonzero_106_matched']}, brain={final_brain_hz:.3f}, cap={final_capped}) "
            f"all={stepwise_auprc(y_bin, r_arr):.4f} nd={stepwise_auprc(y_bin[~is_direct_v783], r_arr[~is_direct_v783]):.4f} nov={stepwise_auprc(y_bin[~prior_pub], r_arr[~prior_pub]):.4f} ({res['elapsed_sec']:.1f}s)",
            flush=True,
        )

    real_all = stepwise_auprc(y_bin, real_v783_50)
    real_nd = stepwise_auprc(y_bin[~is_direct_v783], real_v783_50[~is_direct_v783])
    real_nov = stepwise_auprc(y_bin[~prior_pub], real_v783_50[~prior_pub])

    def _pack_group(key_rates: str) -> dict:
        pg_all, pg_nd, pg_nov, mat = [], [], [], []
        for res in results:
            a = np.array(res[key_rates], dtype=np.float64)
            mat.append(a)
            pg_all.append(stepwise_auprc(y_bin, a))
            pg_nd.append(stepwise_auprc(y_bin[~is_direct_v783], a[~is_direct_v783]))
            pg_nov.append(stepwise_auprc(y_bin[~prior_pub], a[~prior_pub]))
        ens = np.mean(mat, axis=0)
        return {
            "all_106": {
                "real_auprc": real_all, "per_graph": pg_all, "mean": float(np.mean(pg_all)), "sd": float(np.std(pg_all)),
                "min_max": [float(np.min(pg_all)), float(np.max(pg_all))], "real_wins_of_10": int(sum(real_all > x for x in pg_all)),
                "ensemble_auprc": stepwise_auprc(y_bin, ens), "paired_boot_vs_real": paired_bootstrap_auprc_p(y_bin, real_v783_50, ens),
            },
            "nondirect_83": {
                "real_auprc": real_nd, "per_graph": pg_nd, "mean": float(np.mean(pg_nd)), "sd": float(np.std(pg_nd)),
                "min_max": [float(np.min(pg_nd)), float(np.max(pg_nd))], "real_wins_of_10": int(sum(real_nd > x for x in pg_nd)),
                "ensemble_auprc": stepwise_auprc(y_bin[~is_direct_v783], ens[~is_direct_v783]),
                "paired_boot_vs_real": paired_bootstrap_auprc_p(y_bin[~is_direct_v783], real_v783_50[~is_direct_v783], ens[~is_direct_v783]),
            },
            "novel_95": {
                "real_auprc": real_nov, "per_graph": pg_nov, "mean": float(np.mean(pg_nov)), "sd": float(np.std(pg_nov)),
                "min_max": [float(np.min(pg_nov)), float(np.max(pg_nov))], "real_wins_of_10": int(sum(real_nov > x for x in pg_nov)),
                "ensemble_auprc": stepwise_auprc(y_bin[~prior_pub], ens[~prior_pub]),
                "paired_boot_vs_real": paired_bootstrap_auprc_p(y_bin[~prior_pub], real_v783_50[~prior_pub], ens[~prior_pub]),
            },
        }

    c2_sum = _pack_group("rates_106_matched")
    c2_sum["w_syn_matched_mean_sd_min_max"] = [
        float(np.mean([r["w_syn_matched_mv"] for r in results])),
        float(np.std([r["w_syn_matched_mv"] for r in results])),
        float(np.min([r["w_syn_matched_mv"] for r in results])),
        float(np.max([r["w_syn_matched_mv"] for r in results])),
    ]
    c2_sum["mean_106_hz_across_k10"] = float(np.mean([r["mean_106_hz_matched"] for r in results]))
    c2_sum["nonzero_106_mean"] = float(np.mean([r["nonzero_106_matched"] for r in results]))
    c2_sum["brain_mean_hz_across_k10"] = float(np.mean([r["brain_mean_hz_matched"] for r in results]))

    # Exact-matched subset (graphs reaching >=1.85 Hz without seizure bracket stop)
    m_idx = [r["k"] for r in results if r["mean_106_hz_matched"] >= 1.85]
    c2_sum["exact_matched_indices"] = m_idx
    if m_idx:
        c2_sum["exact_matched_summary"] = {
            "n_graphs": len(m_idx),
            "all_106_mean": float(np.mean([c2_sum["all_106"]["per_graph"][i] for i in m_idx])),
            "all_106_real_wins": int(sum(real_all > c2_sum["all_106"]["per_graph"][i] for i in m_idx)),
            "nondirect_83_mean": float(np.mean([c2_sum["nondirect_83"]["per_graph"][i] for i in m_idx])),
            "nondirect_83_real_wins": int(sum(real_nd > c2_sum["nondirect_83"]["per_graph"][i] for i in m_idx)),
            "novel_95_mean": float(np.mean([c2_sum["novel_95"]["per_graph"][i] for i in m_idx])),
            "novel_95_real_wins": int(sum(real_nov > c2_sum["novel_95"]["per_graph"][i] for i in m_idx)),
        }

    out = {
        "compute_backend": "Local CPU Numba FP64 lazy-exact (10 workers)",
        "super_class_names": meta["super_class_names"],
        "real_v783_reference": {"target_mean_106_hz": target_mean_hz, "all_106_auprc": real_all, "nondirect_83_auprc": real_nd, "novel_95_auprc": real_nov},
        "fixed_0275_summary": _pack_group("rates_106_fixed_0275"),
        "c2_activity_matched_summary": c2_sum,
        "per_graph_details": results,
        "elapsed_sec": float(time.time() - t0),
    }
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[rev_local_comp_c] ALL 10 COMPARTMENT-PRESERVING GRAPHS DONE in {time.time()-t0:.1f}s -> {out_path}", flush=True)


def run_phase_rev_merge(rev_static_p: Path, rev_comp_p: Path, rev_mc_p: Path, port_json_p: Path) -> None:
    port = json.loads(port_json_p.read_text(encoding="utf-8"))
    rs = json.loads(rev_static_p.read_text(encoding="utf-8"))
    rc = json.loads(rev_comp_p.read_text(encoding="utf-8"))
    rm = json.loads(rev_mc_p.read_text(encoding="utf-8"))

    port["selk_vnc_exact_widest_paths"] = {
        "banc_v888": rs["selk_vnc_exact_widest_paths_banc"],
        "malecns_v1_0": rm["selk_vnc_exact_widest_paths_malecns"],
    }
    port["rewire_compartment_scatter_audit"] = {
        "flywire_v783": rs["scatter_v783"],
        "banc_v888": rs["scatter_banc"],
        "malecns_v1_0": rm["malecns_scatter_audit"],
    }
    port["test_a_nondirect_variant_audit"] = {
        "direct_variant_edges_audit": rm["direct_variant_edges_audit"],
        "variant_auprc_and_ci95": rs["test_a_nondirect_variant_audit"],
        "test_b3_bootstrap_ci95": rs["test_b3_bootstrap_ci_audit"],
    }
    port["test_a2_b3_combined"] = rm["test_a2_b3_combined"]
    port["test_c_v783_compartment_preserving_rewire"] = rc

    # Update secondary BH-FDR table to include the compartment-preserving rewiring comparisons (16 tests total)
    # Deep-copy base rows so the 14-test primary table is never mutated in place.
    base_fdr = [dict(row) for row in port["secondary_bh_fdr"] if "compartment-preserving" not in row["test"]]
    q_base14 = bh_fdr([r["p_raw"] for r in base_fdr])
    for r, q in zip(base_fdr, q_base14):
        r["q_bh_fdr"] = q
    port["secondary_bh_fdr"] = base_fdr
    p_comp_all = float(rc["c2_activity_matched_summary"]["all_106"]["paired_boot_vs_real"]["p_two_sided"])
    p_comp_nd = float(rc["c2_activity_matched_summary"]["nondirect_83"]["paired_boot_vs_real"]["p_two_sided"])
    ext_fdr = [dict(row) for row in base_fdr] + [
        {"test": "v783 all_106 vs compartment-preserving activity-matched rewProt ensemble", "p_raw": p_comp_all},
        {"test": "v783 nondirect_83 vs compartment-preserving activity-matched rewProt ensemble", "p_raw": p_comp_nd},
    ]
    q_ext = bh_fdr([r["p_raw"] for r in ext_fdr])
    for r, q in zip(ext_fdr, q_ext):
        r["q_bh_fdr"] = q
    port["secondary_bh_fdr_16tests_with_compartment"] = ext_fdr
    port["compute_provenance"]["rev_a100_gpu"] = rm["gpu_name"]
    port["compute_provenance"]["rev_comp_test_c_sec"] = rc["elapsed_sec"]
    port["compute_provenance"]["rev_malecns_a2_b3_sec"] = rm["elapsed_sec"]

    port_json_p.write_text(json.dumps(port, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[rev_merge] Updated {port_json_p} ({port_json_p.stat().st_size / 1e3:.1f} KB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--phase",
        required=True,
        choices=[
            "local_static", "local_test_c", "colab_test_c", "colab_malecns", "merge",
            "rev_local_static", "rev_local_comp_c", "rev_colab_comp_c", "rev_colab_malecns", "rev_merge",
        ],
    )
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--batch-types", type=int, default=12)
    ap.add_argument("--test-d-only", action="store_true")
    ap.add_argument("--test-c-bundle", type=Path, default=Path("/tmp/mn9_port_v783_bundle.npz"))
    ap.add_argument("--test-c-meta", type=Path, default=Path("/tmp/mn9_port_v783_meta.json"))
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/mn9_malecns_input_manifest.json"))
    ap.add_argument("--v783-ref", type=Path, default=Path("/tmp/mn9_port_v783_ref.json"))
    ap.add_argument("--local-static-out", type=Path, default=Path("/tmp/mn9_port_local_static.json"))
    ap.add_argument("--test-c-out", type=Path, default=Path("/tmp/mn9_port_test_c.json"))
    ap.add_argument("--colab-out", type=Path, default=Path("/tmp/mn9_port_colab_out.json"))
    ap.add_argument("--output", type=Path, default=BASE_DIR / "mn9_portability_results.json")
    args = ap.parse_args()

    if args.phase == "local_static":
        run_phase_local_static(args.local_static_out, args.v783_ref)
    elif args.phase == "local_test_c":
        run_phase_local_test_c(args.test_c_out, args.workers)
    elif args.phase == "colab_test_c":
        run_phase_colab_test_c(args.test_c_bundle, args.test_c_meta, args.test_c_out, batch_types=args.batch_types)
    elif args.phase == "colab_malecns":
        run_phase_colab_malecns(args.manifest, args.v783_ref, args.colab_out, batch_types=args.batch_types, test_d_only=args.test_d_only)
    elif args.phase == "merge":
        run_phase_merge(args.local_static_out, args.test_c_out, args.colab_out, args.output)
    elif args.phase == "rev_local_static":
        run_phase_rev_local_static(args.local_static_out, args.test_c_bundle, args.test_c_meta, args.manifest, args.v783_ref)
    elif args.phase == "rev_local_comp_c":
        run_phase_rev_local_comp_c(args.test_c_bundle, args.test_c_meta, args.test_c_out, workers=args.workers)
    elif args.phase == "rev_colab_comp_c":
        run_phase_colab_test_c(args.test_c_bundle, args.test_c_meta, args.test_c_out, batch_types=args.batch_types, compartment_preserving=True)
    elif args.phase == "rev_colab_malecns":
        run_phase_colab_malecns(args.manifest, args.v783_ref, args.colab_out, batch_types=args.batch_types, rev_mode=True)
    elif args.phase == "rev_merge":
        run_phase_rev_merge(args.local_static_out, args.test_c_out, args.colab_out, args.output)


if __name__ == "__main__":
    main()
