#!/usr/bin/env python3
"""Cross-individual FlyWire -> BANC v888 full-CNS zero-fitting transfer benchmark (mn9-banc-transfer-01).

Implements the locked protocol in MN9_BANC_PROTOCOL.md (commit 8abf965e349200ed2117caf7fecda4ce767d112c):
1. Build official BANC v888 proofread+roughly_proofread non-glial directed neuron graph
   from banc_888_edgelist_simple_v2.feather (removing self-loops and missing-NT edges;
   never guessing +1) with full NT coverage & cross-brain anatomy audit vs FlyWire v783.
2. Dual-layer zero-guess FAFB v783 -> BANC v888 homology mapping:
   - M1 (reviewed_only): banc_fafb_reviewed_matches.csv.gz resolved deterministically to v888 (105/106).
   - M2 (reviewed_plus_type): M1 + deterministic fafb_match / exact cell_type completion for
     unmatched v783 IDs (106/106).
3. Two zero-behavior-label, zero-peripheral-GRN W_syn primary settings:
   - Setting 1 (frozen): W_syn = 0.275 mV.
   - Setting 2 (anatomical MN9 incoming synapse scale): W_syn_anat = 0.275 * S_in(MN9,v783)/S_in(MN9,BANC).
4. Task 1 (106 SEZ activation) on BANC v888: 6 static baselines, real LIF (20/50/100/200 Hz, M1 & M2),
   and K=10 rewire_full (9301..9310) + K=10 rewire_mn9_protected (9401..9410, freezing all 336 MN9
   1-hop incoming edges) evaluated identically under BOTH Setting 1 and Setting 2, with strict
   graph-level rank (1/11) vs cell-type stratified paired bootstrap (B=2000) separation.
5. Full-CNS SELK (DNg68) zero-parameter mechanistic diagnosis (plus Fdg & Amulet references):
   brain vs VNC postsynaptic target E/I composition, excitatory paths (unrestricted, brain-only,
   and forced-via-VNC to MN9), and 20/50/100/200 Hz MN9 & top-target LIF responses.
6. Bounded Task 2 (11/11 water silencing) check using BANC annotated LB3a (water, ppk28) GRNs
   at locked W_syn settings without retuning, plus auxiliary sugar-GRN diagnostic note.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import multiprocessing as mp
import pickle
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pyarrow.feather as pf
from numba import njit
from scipy import stats

import mn9_transfer_benchmark as mtb
import shiu_mn9_benchmark as smb

PROTOCOL_COMMIT = "8abf965e349200ed2117caf7fecda4ce767d112c"
INH_NTS = {"gaba", "glutamate"}
EXC_NTS = {"acetylcholine", "dopamine", "serotonin", "octopamine", "tyramine", "histamine"}
_WORKER_GRAPH_CACHE: dict[str, dict] = {}
_WORKER_WS_CACHE: dict[tuple[str, float], np.ndarray] = {}


@njit
def run_lif_lazy_exact(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    w_syn: float,
    exc_idx: np.ndarray,
    exc_rate_hz: float,
    slnc_mask: np.ndarray,
    watch_idx: np.ndarray,
    n_trials: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Exact Brian2 LIF schedule with analytical O(1) lazy decay for subthreshold/inhibitory neurons.

    Mathematically identical to mtb.run_lif_brian2_wsyn: verified max_diff == 0 spikes across
    30-trial FlyWire v783 sugar anchor (9.7x faster) and BANC anat_scale cascades (4.4x faster).
    """
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
    p_poi = exc_rate_hz * 1e-3 * dt

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
    max_ring_occ = 0

    np.random.seed(seed)
    peak_gain = 0.160  # strict upper bound > max_k pow_bvg[k] = 0.15749013
    safe_th = 6.95     # conservative margin below v_th = 7.0

    for tr in range(n_trials):
        n_act = 0
        n_touched = 0
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

            fut_slot = (step + dly_steps) % ring_len
            for s_i in range(n_spk_now):
                i = spiked_now[s_i]
                c_f = ring_cnt[fut_slot]
                ring_spk[fut_slot, c_f] = i
                ring_cnt[fut_slot] = c_f + 1
                for w_i in range(n_watch):
                    if watch_idx[w_i] == i:
                        watch_counts[tr, w_i] += 1
            if ring_cnt[fut_slot] > max_ring_occ:
                max_ring_occ = ring_cnt[fut_slot]

            if p_poi > 0.0:
                for k in range(len(exc_idx)):
                    if np.random.random() < p_poi:
                        u[exc_idx[k]] += w_poi

            cur_slot = step % ring_len
            cnt_deliv = ring_cnt[cur_slot]
            for s_i in range(cnt_deliv):
                src = ring_spk[cur_slot, s_i]
                if slnc_mask[src] != 0:
                    continue
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

        for t_i in range(n_touched):
            i = touched[t_i]
            u[i] = 0.0
            g[i] = 0.0
            last_spk[i] = -100000
            in_active[i] = 0
            is_touched[i] = 0

    return watch_counts, max_ring_occ


def nt_to_sign(nt: str | None) -> int:
    if nt is None:
        return 0
    s = str(nt).strip().lower()
    if s in INH_NTS:
        return -1
    if s in EXC_NTS:
        return 1
    return 0


def _save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(path)


def build_csr_from_edges(
    node_ids: np.ndarray,
    pre_ids: np.ndarray,
    post_ids: np.ndarray,
    w_raw: np.ndarray,
) -> dict:
    """Build CSR arrays compatible with smb.build_rewired_graph and mtb.run_lif_brian2_wsyn."""
    node_ids = np.sort(node_ids.astype(np.int64))
    n_nodes = len(node_ids)
    pre_idx = np.searchsorted(node_ids, pre_ids).astype(np.int32)
    post_idx = np.searchsorted(node_ids, post_ids).astype(np.int32)
    assert np.array_equal(node_ids[pre_idx], pre_ids), "pre ID searchsorted mismatch"
    assert np.array_equal(node_ids[post_idx], post_ids), "post ID searchsorted mismatch"
    assert not np.any(pre_idx == post_idx), "Self-loops found during CSR build"

    order = np.lexsort((post_idx, pre_idx))
    pre_s = pre_idx[order]
    post_s = post_idx[order]
    w_raw_s = w_raw[order].astype(np.float32)

    if len(pre_s) > 1:
        dups = int(((pre_s[1:] == pre_s[:-1]) & (post_s[1:] == post_s[:-1])).sum())
        assert dups == 0, f"Duplicate edges found: {dups}"

    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    np.add.at(indptr[1:], pre_s, 1)
    np.cumsum(indptr, out=indptr)

    node_class = np.full(n_nodes, 2, dtype=np.int8)
    has_pos = np.zeros(n_nodes, dtype=bool)
    has_neg = np.zeros(n_nodes, dtype=bool)
    has_pos[pre_s[w_raw_s > 0]] = True
    has_neg[pre_s[w_raw_s < 0]] = True
    assert not np.any(has_pos & has_neg), "Mixed-sign presynaptic node found"
    node_class[has_pos] = 0
    node_class[has_neg] = 1

    orig_in_pos = np.zeros(n_nodes, dtype=np.int32)
    orig_in_neg = np.zeros(n_nodes, dtype=np.int32)
    np.add.at(orig_in_pos, post_s[w_raw_s > 0], 1)
    np.add.at(orig_in_neg, post_s[w_raw_s < 0], 1)

    return {
        "n_nodes": n_nodes,
        "node_ids": node_ids,
        "id2idx": {int(nid): i for i, nid in enumerate(node_ids)},
        "indptr": indptr,
        "pre_s": pre_s,
        "post_s": post_s,
        "w_raw_s": w_raw_s,
        "node_class": node_class,
        "orig_in_pos": orig_in_pos,
        "orig_in_neg": orig_in_neg,
    }


def _load_banc_npz(npz_path: Path) -> dict:
    key = str(npz_path)
    if key not in _WORKER_GRAPH_CACHE:
        _WORKER_GRAPH_CACHE.clear()
        _WORKER_WS_CACHE.clear()
        d = np.load(npz_path)
        _WORKER_GRAPH_CACHE[key] = {
            "n_nodes": int(d["n_nodes"]),
            "indptr": d["indptr"],
            "pre_s": d["pre_s"],
            "post_s": d["post_s"],
            "w_raw_s": d["w_raw_s"],
            "node_class": d["node_class"],
            "orig_in_pos": d["orig_in_pos"],
            "orig_in_neg": d["orig_in_neg"],
        }
    return _WORKER_GRAPH_CACHE[key]


def _get_worker_ws(npz_path: Path, w_syn: float) -> tuple[dict, np.ndarray]:
    g = _load_banc_npz(npz_path)
    wk = (str(npz_path), round(float(w_syn), 12))
    if wk not in _WORKER_WS_CACHE:
        _WORKER_WS_CACHE[wk] = (g["w_raw_s"] * np.float32(w_syn)).astype(np.float64)
    return g, _WORKER_WS_CACHE[wk]


def _single_sim_worker(job: dict) -> dict:
    """Spawn-safe worker running a single arbitrary LIF trial batch (for parallel Step 4)."""
    t0 = time.time()
    g, w_s = _get_worker_ws(Path(job["npz_path"]), float(job["w_syn"]))
    slnc = np.zeros(g["n_nodes"], dtype=np.uint8)
    for si in job.get("sil_idx", []):
        slnc[int(si)] = 1
    exc_idx = np.array(job["exc_idx"], dtype=np.int32)
    watch_idx = np.array(job["watch_idx"], dtype=np.int32)
    cnts, occ = run_lif_lazy_exact(
        g["indptr"], g["post_s"], w_s, float(job["w_syn"]), exc_idx,
        float(job["rate_hz"]), slnc, watch_idx, int(job["n_trials"]), int(job["seed"]),
    )
    return {
        "tag": job["tag"],
        "means": cnts.mean(axis=0).tolist(),
        "max_occ": int(occ),
        "elapsed_sec": float(time.time() - t0),
    }


def _run_m1_m2_task1(
    indptr: np.ndarray,
    post_s: np.ndarray,
    w_s: np.ndarray,
    w_syn: float,
    m1_indices: list[np.ndarray],
    m2_indices: list[np.ndarray],
    mn9_watch: np.ndarray,
    slnc_none: np.ndarray,
    rate_hz: float,
    n_trials: int,
    seed_base: int,
    start_i: int = 0,
    end_i: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Run Task 1 for M1 and M2 over [start_i, end_i), reusing bit-identical M1==M2 types."""
    if end_i is None:
        end_i = len(m1_indices)
    n_slice = end_i - start_i
    m1_means = np.zeros((n_slice, 2), dtype=np.float64)
    m1_stds = np.zeros((n_slice, 2), dtype=np.float64)
    m2_means = np.zeros((n_slice, 2), dtype=np.float64)
    m2_stds = np.zeros((n_slice, 2), dtype=np.float64)
    max_occ = 0

    for off, i in enumerate(range(start_i, end_i)):
        idx1 = m1_indices[i]
        idx2 = m2_indices[i]
        if len(idx1) > 0:
            cnts, occ = run_lif_lazy_exact(
                indptr, post_s, w_s, w_syn, idx1, rate_hz, slnc_none, mn9_watch, n_trials, seed_base + i * 97
            )
            m1_means[off] = cnts.mean(axis=0)
            m1_stds[off] = cnts.std(axis=0)
            if occ > max_occ:
                max_occ = occ
        if len(idx1) == len(idx2) and np.array_equal(idx1, idx2):
            m2_means[off] = m1_means[off]
            m2_stds[off] = m1_stds[off]
        elif len(idx2) > 0:
            cnts2, occ2 = run_lif_lazy_exact(
                indptr, post_s, w_s, w_syn, idx2, rate_hz, slnc_none, mn9_watch, n_trials, seed_base + i * 97
            )
            m2_means[off] = cnts2.mean(axis=0)
            m2_stds[off] = cnts2.std(axis=0)
            if occ2 > max_occ:
                max_occ = occ2
    return m1_means, m1_stds, m2_means, m2_stds, int(max_occ)


def _real_slice_worker(job: dict) -> dict:
    ckpt = Path(job["ckpt_path"])
    if ckpt.exists():
        return json.load(open(ckpt))
    t0 = time.time()
    w_syn = float(job["w_syn"])
    g, w_s = _get_worker_ws(Path(job["npz_path"]), w_syn)
    slnc_none = np.zeros(g["n_nodes"], dtype=np.uint8)
    mn9_watch = np.array(job["mn9_watch"], dtype=np.int32)
    m1_idx = [np.array(x, dtype=np.int32) for x in job["m1_indices"]]
    m2_idx = [np.array(x, dtype=np.int32) for x in job["m2_indices"]]

    m1_m, m1_sd, m2_m, m2_sd, occ = _run_m1_m2_task1(
        g["indptr"], g["post_s"], w_s, w_syn, m1_idx, m2_idx, mn9_watch, slnc_none,
        float(job["rate_hz"]), int(job["n_trials"]), int(job["seed_base"]),
        int(job["start_i"]), int(job["end_i"]),
    )
    out = {
        "kind": "real_slice",
        "label": job["label"],
        "start_i": int(job["start_i"]),
        "end_i": int(job["end_i"]),
        "m1_means": m1_m.tolist(),
        "m1_stds": m1_sd.tolist(),
        "m2_means": m2_m.tolist(),
        "m2_stds": m2_sd.tolist(),
        "max_occ": int(occ),
        "elapsed_sec": float(time.time() - t0),
    }
    _save_json_atomic(ckpt, out)
    return out


def _rewire_banc_worker(job: dict) -> dict:
    """Build one BANC v888 rewired graph and evaluate M1 & M2 under both Setting 1 and Setting 2."""
    ckpt = Path(job["ckpt_path"])
    if ckpt.exists():
        return json.load(open(ckpt))
    t0 = time.time()
    w1 = float(job["w_syn_frozen"])
    w2 = float(job["w_syn_anat"])
    g, w_s1 = _get_worker_ws(Path(job["npz_path"]), w1)
    _, w_s2 = _get_worker_ws(Path(job["npz_path"]), w2)
    n_nodes = g["n_nodes"]
    indptr = g["indptr"]
    pre_s = g["pre_s"]
    post_s = g["post_s"]
    node_class = g["node_class"]
    orig_in_pos = g["orig_in_pos"]
    orig_in_neg = g["orig_in_neg"]

    prot_set = set(job["protect_targets"])
    seed_rw = int(job["seed"])

    rw_indices, meta = smb.build_rewired_graph(
        indptr, post_s, w_s1, pre_s, node_class, orig_in_pos, orig_in_neg, prot_set, seed_rw
    )
    m1_idx = [np.array(x, dtype=np.int32) for x in job["m1_indices"]]
    m2_idx = [np.array(x, dtype=np.int32) for x in job["m2_indices"]]
    mn9_watch = np.array(job["mn9_watch"], dtype=np.int32)
    slnc_none = np.zeros(n_nodes, dtype=np.uint8)
    n_trials = int(job["n_trials"])

    # Setting 1 (frozen 0.275 mV)
    m1_f, _, m2_f, _, occ1 = _run_m1_m2_task1(
        indptr, rw_indices, w_s1, w1, m1_idx, m2_idx, mn9_watch, slnc_none, 50.0, n_trials, 50000
    )
    # Setting 2 (anatomical scale W_syn_anat)
    m1_a, _, m2_a, _, occ2 = _run_m1_m2_task1(
        indptr, rw_indices, w_s2, w2, m1_idx, m2_idx, mn9_watch, slnc_none, 50.0, n_trials, 50000
    )

    # Optional bounded water baseline check on rewired graphs
    water_checks = {}
    for wkey, winfo in job.get("water_baseline_checks", {}).items():
        w_idx = np.array(winfo["idx"], dtype=np.int32)
        ws = float(winfo["w_syn"])
        rate_hz = float(winfo["rate_hz"])
        ws_arr = w_s1 if abs(ws - w1) < 1e-9 else w_s2
        cnts, _ = run_lif_lazy_exact(
            indptr, rw_indices, ws_arr, ws, w_idx, rate_hz, slnc_none, mn9_watch, n_trials, 70000
        )
        water_checks[wkey] = {
            "mn9_l_hz": float(cnts[:, 0].mean()),
            "mn9_r_hz": float(cnts[:, 1].mean()),
            "mn9_mean_hz": float(cnts.mean()),
        }

    out = {
        "family": job["family"],
        "idx": int(job["idx"]),
        "seed": seed_rw,
        "meta": meta,
        "frozen_m1_50hz": m1_f.mean(axis=1).tolist(),
        "frozen_m2_50hz": m2_f.mean(axis=1).tolist(),
        "anat_m1_50hz": m1_a.mean(axis=1).tolist(),
        "anat_m2_50hz": m2_a.mean(axis=1).tolist(),
        "water_baseline_checks": water_checks,
        "max_ring_occ": int(max(occ1, occ2)),
        "elapsed_sec": float(time.time() - t0),
    }
    _save_json_atomic(ckpt, out)
    return out


def shortest_exc_path_with_constraint(
    indptr: np.ndarray,
    post_s: np.ndarray,
    w_raw_s: np.ndarray,
    node_class: np.ndarray,
    sources: list[int],
    targets: set[int],
    allowed_intermediate: np.ndarray | None = None,
    init_bottlenecks: dict[int, float] | None = None,
) -> dict:
    """BFS min-hop purely-excitatory path from sources to targets (maximizing bottleneck within the minimum hop count)."""
    n_nodes = len(indptr) - 1
    dist = np.full(n_nodes, -1, dtype=np.int32)
    best_bottleneck = np.zeros(n_nodes, dtype=np.float64)
    parent = np.full(n_nodes, -1, dtype=np.int32)
    q = deque()
    for s in sources:
        dist[s] = 0
        best_bottleneck[s] = float(init_bottlenecks[s]) if init_bottlenecks is not None else 1e9
        q.append(s)

    found_dist = None
    while q:
        u = q.popleft()
        du = int(dist[u])
        if found_dist is not None and du >= found_dist:
            continue
        if node_class[u] != 0:  # must be excitatory presynaptic
            continue
        p0, p1 = int(indptr[u]), int(indptr[u + 1])
        for e in range(p0, p1):
            w = float(w_raw_s[e])
            if w <= 0:
                continue
            v = int(post_s[e])
            if v not in targets and allowed_intermediate is not None and not allowed_intermediate[v]:
                continue
            cand_bn = min(float(best_bottleneck[u]), w)
            if dist[v] == -1:
                dist[v] = du + 1
                best_bottleneck[v] = cand_bn
                parent[v] = u
                if v in targets:
                    found_dist = du + 1
                else:
                    q.append(v)
            elif dist[v] == du + 1 and cand_bn > best_bottleneck[v]:
                best_bottleneck[v] = cand_bn
                parent[v] = u

    reached = [t for t in sorted(targets) if dist[t] > 0]
    if not reached:
        return {"reachable": False, "min_hops": None, "best_target": None, "bottleneck_weight": 0.0, "path_indices": []}
    best_t = min(reached, key=lambda t: (int(dist[t]), -float(best_bottleneck[t])))
    path = []
    cur = best_t
    while cur != -1:
        path.append(int(cur))
        cur = int(parent[cur])
    path.reverse()
    return {
        "reachable": True,
        "min_hops": int(dist[best_t]),
        "best_target": int(best_t),
        "bottleneck_weight": float(best_bottleneck[best_t]),
        "path_indices": path,
    }


def evaluate_subset_suite(
    y_bin: np.ndarray,
    y_max: np.ndarray,
    y_mean: np.ndarray,
    subset_mask: np.ndarray,
    real_scores: dict[str, np.ndarray],
    primary_key: str,
    static_scores: dict[str, np.ndarray],
    rw_full_list: list[np.ndarray],
    rw_prot_list: list[np.ndarray],
) -> dict:
    """Evaluate real, static, and K=10 rewired graphs on one subset with strict inference separation."""
    idx = np.flatnonzero(subset_mask)
    yb = y_bin[idx].astype(np.int32)
    ymax = y_max[idx]
    ymean = y_mean[idx]
    n_pos = int(yb.sum())
    n_neg = int(len(yb) - n_pos)

    ens_full = np.mean([arr[idx] for arr in rw_full_list], axis=0)
    ens_prot = np.mean([arr[idx] for arr in rw_prot_list], axis=0)

    scores_for_boot: dict[str, np.ndarray] = {}
    for k, arr in real_scores.items():
        scores_for_boot[k] = arr[idx]
    for k, arr in static_scores.items():
        scores_for_boot[k] = arr[idx]
    scores_for_boot["rewire_full_ensemble_mean"] = ens_full
    scores_for_boot["rewire_mn9_prot_ensemble_mean"] = ens_prot

    boot_all = smb.stratified_bootstrap_summary(
        yb, ymax, ymean, scores_for_boot, ref_model=primary_key, n_boot=2000, seed=20260783
    )

    def one_model_metrics(s: np.ndarray, bkey: str | None = None) -> dict:
        ap = smb.average_precision_exact(yb, s)
        auc = smb.auroc_exact(yb, s)
        out = {
            "auprc": float(ap),
            "auroc": float(auc),
            "spearman_max": smb.spearman_safe(ymax, s),
            "spearman_mean": smb.spearman_safe(ymean, s),
            "heldout_cv": smb.evaluate_heldout_cv(yb, s),
            "n_nonzero": int((s > 0.0).sum()),
        }
        if bkey is not None and bkey in boot_all:
            out["ci95"] = boot_all[bkey]["ci95"]
            if bkey != primary_key:
                out["celltype_boot_primary_minus_this"] = boot_all[bkey]["ref_minus_this"]
        return out

    models_out = {}
    for k, arr in real_scores.items():
        models_out[k] = one_model_metrics(arr[idx], bkey=k)
    for k, arr in static_scores.items():
        models_out[k] = one_model_metrics(arr[idx], bkey=k)

    def eval_rewire_family(rw_list: list[np.ndarray], ens: np.ndarray, ens_bkey: str) -> dict:
        per_graph = [one_model_metrics(arr[idx], bkey=None) for arr in rw_list]
        aps = [m["auprc"] for m in per_graph]
        aucs = [m["auroc"] for m in per_graph]
        loocv_f1s = [m["heldout_cv"]["loocv"]["f1"] for m in per_graph]
        real_ap = models_out[primary_key]["auprc"]
        wins = int(sum(real_ap > a + 1e-12 for a in aps))
        ties = int(sum(abs(real_ap - a) <= 1e-12 for a in aps))
        n_g = len(rw_list)
        return {
            "n_graphs": n_g,
            "graph_level_rank_vs_primary": {
                "wins": wins,
                "ties": ties,
                "losses": n_g - wins - ties,
                "empirical_p_resolution": float(1.0 / (n_g + 1)),
                "empirical_p_upper": float((n_g - wins + 1) / (n_g + 1)),
            },
            "per_graph_auprc_mean": float(np.mean(aps)),
            "per_graph_auprc_min": float(np.min(aps)),
            "per_graph_auprc_max": float(np.max(aps)),
            "per_graph_auprc_list": [float(a) for a in aps],
            "per_graph_auroc_mean": float(np.mean(aucs)),
            "per_graph_loocv_f1_mean": float(np.mean(loocv_f1s)),
            "ensemble_mean_metrics": one_model_metrics(ens, bkey=ens_bkey),
        }

    return {
        "n_total": int(len(idx)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "random_auprc_baseline": float(n_pos / len(idx)) if len(idx) else 0.0,
        "primary_model_key": primary_key,
        "models": models_out,
        "rewire_full": eval_rewire_family(rw_full_list, ens_full, "rewire_full_ensemble_mean"),
        "rewire_mn9_protected": eval_rewire_family(rw_prot_list, ens_prot, "rewire_mn9_prot_ensemble_mean"),
    }


def main() -> None:
    import os
    cache_root = Path(os.environ.get("PROJC_CACHE_DIR", str(Path.home() / ".cache/projc-flywire")))
    parser = argparse.ArgumentParser()
    parser.add_argument("--banc-dir", type=Path, default=cache_root / "data/banc")
    parser.add_argument("--comp-783", type=Path, default=cache_root / "data/Completeness_783.csv")
    parser.add_argument("--conn-783", type=Path, default=cache_root / "data/Connectivity_783.parquet")
    parser.add_argument("--ann-783", type=Path, default=cache_root / "data/Supplemental_file1_neuron_annotations.tsv")
    parser.add_argument("--sez-pickle", type=Path, default=cache_root / "source02/sez_neurons.pickle")
    parser.add_argument("--xlsx", type=Path, default=cache_root / "source02/shiu_moesm2.xlsx")
    parser.add_argument("--v783-results", type=Path, default=Path("mn9_transfer_results.json"))
    parser.add_argument("--ckpt-dir", type=Path, default=cache_root / "banc_ckpt")
    parser.add_argument("--output", type=Path, default=Path("mn9_banc_results.json"))
    parser.add_argument("--n-rewires", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    edgelist_path = args.banc_dir / "banc_888_edgelist_simple_v2.feather"
    meta_path = args.banc_dir / "banc_888_meta.feather"
    nt_path = args.banc_dir / "banc_888_neurotransmitter_prediction_v2.csv"
    rev_path = args.banc_dir / "banc_fafb_reviewed_matches.csv.gz"

    hashes = {
        "banc_888_edgelist_simple_v2_feather": smb.sha256_file(edgelist_path),
        "banc_888_meta_feather": smb.sha256_file(meta_path),
        "banc_888_neurotransmitter_prediction_v2_csv": smb.sha256_file(nt_path),
        "banc_fafb_reviewed_matches_csv_gz": smb.sha256_file(rev_path),
        "Completeness_783_csv": smb.sha256_file(args.comp_783),
        "Connectivity_783_parquet": smb.sha256_file(args.conn_783),
        "flywire_annotations_v783_v3_2_0_tsv": smb.sha256_file(args.ann_783),
    }

    print("[1/7] Loading BANC v888 metadata, NT predictions, reviewed matches & edgelist...", flush=True)
    meta_tbl = pf.read_table(meta_path)
    meta_cols = [
        "banc_888_id", "root_id", "supervoxel_id", "root_626", "root_850", "root_888", "root_890",
        "proofread", "roughly_proofread", "side", "region", "super_class", "cell_class", "cell_sub_class",
        "cell_type", "fafb_cell_type", "fafb_match", "neurotransmitter_predicted", "neurotransmitter_score",
        "neurotransmitter_verified", "cell_function", "cell_function_detailed", "nerve",
    ]
    m_rows: dict[int, dict] = {}
    sv_to_bid: dict[int, int] = {}
    old_root_to_bid: dict[int, int] = {}
    by_banc_ct: dict[str, set[int]] = defaultdict(set)
    by_banc_fafb_ct: dict[str, set[int]] = defaultdict(set)
    meta_fafb: dict[int, set[int]] = defaultdict(set)

    for vals in zip(*(meta_tbl[c].to_pylist() for c in meta_cols)):
        d = dict(zip(meta_cols, vals))
        bid = int(d["banc_888_id"])
        m_rows[bid] = d
        sv = d["supervoxel_id"]
        if sv and str(sv).isdigit():
            sv_to_bid[int(sv)] = bid
        for rc in ("root_id", "root_626", "root_850", "root_888", "root_890"):
            rv = d[rc]
            if rv and str(rv).isdigit():
                old_root_to_bid[int(rv)] = bid
        is_main_node = (d["proofread"] == "TRUE" or d["roughly_proofread"] == "TRUE") and (d["super_class"] != "glia")
        if is_main_node:
            if d["cell_type"]:
                by_banc_ct[str(d["cell_type"])].add(bid)
            if d["fafb_cell_type"]:
                by_banc_fafb_ct[str(d["fafb_cell_type"])].add(bid)
            if d["fafb_match"] and str(d["fafb_match"]).isdigit():
                meta_fafb[int(d["fafb_match"])].add(bid)

    nt_csv: dict[int, tuple[str, float, int]] = {}
    with open(nt_path) as f:
        for row in csv.DictReader(f):
            rid = int(row["root_id"])
            nt_val = row["neurotransmitter_predicted"]
            if nt_val not in ("", "NA"):
                nt_csv[rid] = (nt_val, float(row["neurotransmitter_score"]), int(row["count"]))

    # Load reviewed matches (deterministic resolution to banc_888_id)
    rev_resolved: dict[int, set[int]] = defaultdict(set)
    rev_res_stats = Counter()
    with open(rev_path) as f:
        for row in csv.DictReader(f):
            if row.get("valid") != "t":
                rev_res_stats["invalid_flag"] += 1
                continue
            bid_s, sv_s, mid_s = row["pt_root_id"], row["pt_supervoxel_id"], row["match_id"]
            if not (bid_s.isdigit() and mid_s.isdigit()):
                rev_res_stats["non_digit"] += 1
                continue
            bid_raw, fid = int(bid_s), int(mid_s)
            resolved_bid = None
            if bid_raw in m_rows:
                resolved_bid = bid_raw
                rev_res_stats["direct_888"] += 1
            elif sv_s.isdigit() and int(sv_s) in sv_to_bid:
                resolved_bid = sv_to_bid[int(sv_s)]
                rev_res_stats["via_supervoxel"] += 1
            elif bid_raw in old_root_to_bid:
                resolved_bid = old_root_to_bid[bid_raw]
                rev_res_stats["via_old_root"] += 1
            else:
                rev_res_stats["unresolved"] += 1
            if resolved_bid is not None:
                d = m_rows[resolved_bid]
                if (d["proofread"] == "TRUE" or d["roughly_proofread"] == "TRUE") and d["super_class"] != "glia":
                    rev_resolved[fid].add(resolved_bid)
                else:
                    rev_res_stats["filtered_unproofread_or_glia"] += 1

    # Load edgelist and build main BANC neuron graph (pr|rpr & noglia & pre!=post & valid NT)
    el = pf.read_table(edgelist_path)
    raw_pre = el["pre"].to_numpy().astype(np.int64)
    raw_post = el["post"].to_numpy().astype(np.int64)
    raw_cnt = el["count"].to_numpy().astype(np.int32)
    n_raw_edges = int(len(raw_cnt))
    n_self_loops = int((raw_pre == raw_post).sum())
    ns_mask = raw_pre != raw_post
    ns_pre, ns_post, ns_cnt = raw_pre[ns_mask], raw_post[ns_mask], raw_cnt[ns_mask]

    # Map presynaptic NT signs vectorized via sorted lookup
    nt_ids = np.array(sorted(nt_csv.keys()), dtype=np.int64)
    nt_signs = np.array([nt_to_sign(nt_csv[int(b)][0]) for b in nt_ids], dtype=np.int8)
    main_node_set = np.array(
        sorted(b for b, d in m_rows.items() if (d["proofread"] == "TRUE" or d["roughly_proofread"] == "TRUE") and d["super_class"] != "glia"),
        dtype=np.int64,
    )

    def lookup_signs(ids: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(nt_ids, ids)
        pos_c = np.minimum(pos, len(nt_ids) - 1)
        hit = nt_ids[pos_c] == ids
        out = np.zeros(len(ids), dtype=np.int8)
        out[hit] = nt_signs[pos_c[hit]]
        return out

    ns_signs = lookup_signs(ns_pre)
    valid_nt_mask = ns_signs != 0
    in_main_mask = np.isin(ns_pre, main_node_set) & np.isin(ns_post, main_node_set)
    main_edge_mask = in_main_mask & valid_nt_mask

    main_pre = ns_pre[main_edge_mask]
    main_post = ns_post[main_edge_mask]
    main_w_raw = (ns_cnt[main_edge_mask].astype(np.int32) * ns_signs[main_edge_mask].astype(np.int32)).astype(np.float32)

    # Keep all main_node_set nodes present in main edges (plus any target nodes if isolated)
    active_main_nodes = np.union1d(main_pre, main_post)
    g_banc = build_csr_from_edges(active_main_nodes, main_pre, main_post, main_w_raw)
    idx_banc = g_banc["id2idx"]

    # Save compact CSR npz for fast low-RAM spawn workers
    banc_npz = args.ckpt_dir / "banc_main_graph.npz"
    np.savez(
        banc_npz,
        n_nodes=np.int32(g_banc["n_nodes"]),
        indptr=g_banc["indptr"],
        pre_s=g_banc["pre_s"],
        post_s=g_banc["post_s"],
        w_raw_s=g_banc["w_raw_s"],
        node_class=g_banc["node_class"],
        orig_in_pos=g_banc["orig_in_pos"],
        orig_in_neg=g_banc["orig_in_neg"],
    )

    # Also build full-edgelist sensitivity graph (all non-self valid-NT edges incl. fragments/glia)
    all_pre = ns_pre[valid_nt_mask]
    all_post = ns_post[valid_nt_mask]
    all_w_raw = (ns_cnt[valid_nt_mask].astype(np.int32) * ns_signs[valid_nt_mask].astype(np.int32)).astype(np.float32)
    g_banc_all = build_csr_from_edges(np.union1d(all_pre, all_post), all_pre, all_post, all_w_raw)
    idx_banc_all = g_banc_all["id2idx"]
    banc_all_npz = args.ckpt_dir / "banc_all_graph.npz"
    np.savez(
        banc_all_npz,
        n_nodes=np.int32(g_banc_all["n_nodes"]),
        indptr=g_banc_all["indptr"],
        pre_s=g_banc_all["pre_s"],
        post_s=g_banc_all["post_s"],
        w_raw_s=g_banc_all["w_raw_s"],
        node_class=g_banc_all["node_class"],
        orig_in_pos=g_banc_all["orig_in_pos"],
        orig_in_neg=g_banc_all["orig_in_neg"],
    )

    # NT concordance vs neurotransmitter_verified in main graph
    nt_ver_conc = Counter()
    for b in active_main_nodes.tolist():
        pred = nt_csv[b][0] if b in nt_csv else "MISSING"
        ps = nt_to_sign(pred)
        ver = m_rows[b]["neurotransmitter_verified"]
        if ver:
            v_tokens = {t.strip().lower() for t in str(ver).split(",")}
            v_inh = bool(v_tokens & INH_NTS)
            v_exc = bool(v_tokens & EXC_NTS)
            if v_inh and not v_exc:
                nt_ver_conc[f"ver_I->pred_{'I' if ps == -1 else ('E' if ps == 1 else 'M')}"] += 1
            elif v_exc and not v_inh:
                nt_ver_conc[f"ver_E->pred_{'I' if ps == -1 else ('E' if ps == 1 else 'M')}"] += 1
            else:
                nt_ver_conc["ver_mixed_or_other"] += 1

    region_counts = Counter(m_rows[int(b)]["region"] for b in active_main_nodes.tolist())
    print(
        f"  BANC main graph: {g_banc['n_nodes']:,} nodes, {len(g_banc['pre_s']):,} edges "
        f"(E={(g_banc['node_class']==0).sum():,}, I={(g_banc['node_class']==1).sum():,}, sink={(g_banc['node_class']==2).sum():,}), "
        f"sum|w|={float(np.abs(g_banc['w_raw_s']).sum()):,.0f}",
        flush=True,
    )

    print("[2/7] Loading FlyWire v783 reference, behavioral labels & building M1/M2 mappings...", flush=True)
    v783_res = json.load(open(args.v783_results))
    ma783 = v783_res["mapping_audit"]
    g783 = mtb.load_graph_csr(args.comp_783, args.conn_783)
    idx783 = g783["flyid2idx"]

    v783_ct: dict[int, str] = {}
    with open(args.ann_783) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            v783_ct[int(row["root_id"])] = row["cell_type"]

    sez_dict = pickle.load(open(args.sez_pickle, "rb"))
    st3_records = smb.load_sup_table_3(args.xlsx, sez_dict)
    st9b_records = smb.load_supp_table_9b(args.xlsx, sez_dict)
    v783_sez_by_name = {r["name"]: r for r in ma783["per_type_sez106"]}

    # MN9 in v783 and BANC
    mn9_l_783 = int(ma783["mn9_l"]["v783_id"])
    mn9_r_783 = int(ma783["mn9_r"]["v783_id"])
    mn9_l_banc = sorted(rev_resolved[mn9_l_783])[0]
    mn9_r_banc = sorted(rev_resolved[mn9_r_783])[0]
    assert len(rev_resolved[mn9_l_783]) == 1 and len(rev_resolved[mn9_r_783]) == 1
    assert mn9_l_banc == 720575941410682479 and mn9_r_banc == 720575941623285450
    mn9_watch_banc = np.array([idx_banc[mn9_l_banc], idx_banc[mn9_r_banc]], dtype=np.int32)
    mn9_watch_all = np.array([idx_banc_all[mn9_l_banc], idx_banc_all[mn9_r_banc]], dtype=np.int32)

    # Compute anatomical MN9 incoming synapse scale W_syn_anat (zero behavior labels!)
    def mn9_incoming_stats(g: dict, mn9_i: int) -> dict:
        m = g["post_s"] == mn9_i
        w = g["w_raw_s"][m]
        pre_i = g["pre_s"][m]
        return {
            "n_edges": int(m.sum()),
            "n_exc": int((w > 0).sum()),
            "n_inh": int((w < 0).sum()),
            "sum_pos": float(w[w > 0].sum()),
            "sum_neg": float(w[w < 0].sum()),
            "sum_abs": float(np.abs(w).sum()),
            "pre_indices": pre_i,
            "weights": w,
        }

    mn9_l_783_st = mn9_incoming_stats(g783, idx783[mn9_l_783])
    mn9_r_783_st = mn9_incoming_stats(g783, idx783[mn9_r_783])
    mn9_l_banc_st = mn9_incoming_stats(g_banc, idx_banc[mn9_l_banc])
    mn9_r_banc_st = mn9_incoming_stats(g_banc, idx_banc[mn9_r_banc])

    s_in_v783_mean = 0.5 * (mn9_l_783_st["sum_abs"] + mn9_r_783_st["sum_abs"])
    s_in_banc_mean = 0.5 * (mn9_l_banc_st["sum_abs"] + mn9_r_banc_st["sum_abs"])
    w_syn_frozen = float(smb.W_SYN)  # 0.275 mV
    w_syn_anat = float(w_syn_frozen * (s_in_v783_mean / s_in_banc_mean))

    # Cross-brain homologous presynaptic cell-type overlap on MN9_l and MN9_r
    def mn9_presyn_type_overlap(st783: dict, stbanc: dict) -> dict:
        t783_w: dict[str, float] = defaultdict(float)
        for pi, w in zip(st783["pre_indices"].tolist(), st783["weights"].tolist()):
            fid = int(g783["flyids"][pi])
            ct = v783_ct.get(fid, "")
            if ct:
                t783_w[ct] += abs(w)
        tbanc_w: dict[str, float] = defaultdict(float)
        for pi, w in zip(stbanc["pre_indices"].tolist(), stbanc["weights"].tolist()):
            bid = int(g_banc["node_ids"][pi])
            ct = m_rows[bid]["fafb_cell_type"] or m_rows[bid]["cell_type"] or ""
            if ct and not ct.startswith("auto:"):
                tbanc_w[ct] += abs(w)
        s7 = set(t783_w.keys())
        sb = set(tbanc_w.keys())
        shared = s7 & sb
        return {
            "v783_typed_presyn_types": len(s7),
            "banc_typed_presyn_types": len(sb),
            "shared_types": len(shared),
            "jaccard_types": float(len(shared) / max(1, len(s7 | sb))),
            "v783_syn_frac_in_shared_types": float(sum(t783_w[t] for t in shared) / max(1.0, st783["sum_abs"])),
            "banc_syn_frac_in_shared_types": float(sum(tbanc_w[t] for t in shared) / max(1.0, stbanc["sum_abs"])),
            "top_shared_types_by_banc_syn": sorted(
                [(t, float(tbanc_w[t]), float(t783_w[t])) for t in shared], key=lambda x: -x[1]
            )[:15],
        }

    mn9_l_overlap = mn9_presyn_type_overlap(mn9_l_783_st, mn9_l_banc_st)
    mn9_r_overlap = mn9_presyn_type_overlap(mn9_r_783_st, mn9_r_banc_st)
    print(
        f"  MN9 abs_in: v783 mean={s_in_v783_mean:.1f} ({mn9_l_783_st['sum_abs']:.0f}/{mn9_r_783_st['sum_abs']:.0f}), "
        f"BANC mean={s_in_banc_mean:.1f} ({mn9_l_banc_st['sum_abs']:.0f}/{mn9_r_banc_st['sum_abs']:.0f}) "
        f"=> W_syn_anat = {w_syn_anat:.6f} mV (scale {w_syn_anat/w_syn_frozen:.4f}x)",
        flush=True,
    )
    print(
        f"  MN9 shared presyn cell_types: L={mn9_l_overlap['shared_types']} "
        f"(BANC syn share {mn9_l_overlap['banc_syn_frac_in_shared_types']:.3f}), "
        f"R={mn9_r_overlap['shared_types']} (BANC syn share {mn9_r_overlap['banc_syn_frac_in_shared_types']:.3f})",
        flush=True,
    )

    # Build M1, M2, and M1+fafb_match mappings for all 106 SEZ types
    def map_fids_dual(fids_783: list[int]) -> tuple[list[int], list[int], list[int], int]:
        m1_set: set[int] = set()
        m1_mf_set: set[int] = set()
        m2_set: set[int] = set()
        n_rev_matched_fids = 0
        for fid in fids_783:
            hits = rev_resolved.get(fid, set())
            mf_hits = meta_fafb.get(fid, set())
            if hits:
                n_rev_matched_fids += 1
                m1_set.update(hits)
                m1_mf_set.update(hits)
                m2_set.update(hits)
            else:
                if mf_hits:
                    m1_mf_set.update(mf_hits)
                    m2_set.update(mf_hits)
                ct = v783_ct.get(fid, "")
                if ct:
                    m2_set.update(by_banc_fafb_ct.get(ct, set()))
                    m2_set.update(by_banc_ct.get(ct, set()))
        m1_list = sorted(b for b in m1_set if b in idx_banc)
        m2_list = sorted(b for b in m2_set if b in idx_banc)
        m1_mf_list = sorted(b for b in m1_mf_set if b in idx_banc)
        return m1_list, m2_list, m1_mf_list, n_rev_matched_fids

    m1_indices: list[np.ndarray] = []
    m2_indices: list[np.ndarray] = []
    m1mf_indices: list[np.ndarray] = []
    m1_indices_all: list[np.ndarray] = []
    m2_indices_all: list[np.ndarray] = []
    per_type_mapping = []
    ei_concordance_sez = Counter()

    for rec in st3_records:
        name = rec["name"]
        fids_783 = [int(x) for x in v783_sez_by_name[name]["v783_ids"]]
        m1_bids, m2_bids, m1mf_bids, n_rev_f = map_fids_dual(fids_783)
        m1_indices.append(np.array([idx_banc[b] for b in m1_bids], dtype=np.int32))
        m2_indices.append(np.array([idx_banc[b] for b in m2_bids], dtype=np.int32))
        m1mf_indices.append(np.array([idx_banc[b] for b in m1mf_bids], dtype=np.int32))
        m1_indices_all.append(np.array([idx_banc_all[b] for b in m1_bids], dtype=np.int32))
        m2_indices_all.append(np.array([idx_banc_all[b] for b in m2_bids], dtype=np.int32))

        # E/I concordance between v783 and BANC M1
        c7_set = [int(g783["node_class"][idx783[fid]]) for fid in fids_783 if fid in idx783]
        cb_set = [int(g_banc["node_class"][idx_banc[b]]) for b in m1_bids]
        for fid in fids_783:
            c7 = int(g783["node_class"][idx783[fid]])
            for b in sorted(rev_resolved.get(fid, set())):
                if b in idx_banc:
                    cb = int(g_banc["node_class"][idx_banc[b]])
                    ei_concordance_sez[f"v783_{c7}->banc_{cb}"] += 1

        out_syn_m1 = float(sum(np.abs(g_banc["w_raw_s"][g_banc["indptr"][idx_banc[b]] : g_banc["indptr"][idx_banc[b] + 1]]).sum() for b in m1_bids))
        out_syn_v783 = float(sum(np.abs(g783["w_raw_s"][g783["indptr"][idx783[f]] : g783["indptr"][idx783[f] + 1]]).sum() for f in fids_783))
        per_type_mapping.append(
            {
                "name": name,
                "n_v783": len(fids_783),
                "n_v783_rev_matched": n_rev_f,
                "n_m1_banc": len(m1_bids),
                "n_m2_banc": len(m2_bids),
                "n_m1mf_banc": len(m1mf_bids),
                "m1_eq_m2": bool(m1_bids == m2_bids),
                "m1_banc_ids": m1_bids,
                "m2_banc_ids": m2_bids,
                "v783_out_syn": out_syn_v783,
                "banc_m1_out_syn": out_syn_m1,
                "v783_cls_counts": dict(Counter(c7_set)),
                "banc_m1_cls_counts": dict(Counter(cb_set)),
            }
        )

    n_m1_cov = sum(1 for x in m1_indices if len(x) > 0)
    n_m2_cov = sum(1 for x in m2_indices if len(x) > 0)
    n_m1_eq_m2 = sum(1 for r in per_type_mapping if r["m1_eq_m2"])
    print(
        f"  SEZ 106 mapping: M1={n_m1_cov}/106 ({sum(len(x) for x in m1_indices)} cells), "
        f"M2={n_m2_cov}/106 ({sum(len(x) for x in m2_indices)} cells), M1==M2 on {n_m1_eq_m2}/106 types",
        flush=True,
    )

    # Water 11 silencing types mapping
    water11_banc_indices: list[np.ndarray] = []
    water11_mapping = []
    for rec in st9b_records:
        fids_783 = [int(x) for x in rec["fids"]]  # 100% Tier 1 unchanged in v783
        m1_b, m2_b, _, n_rev_f = map_fids_dual(fids_783)
        water11_banc_indices.append(np.array([idx_banc[b] for b in m1_b], dtype=np.int32))
        water11_mapping.append(
            {
                "name": rec["name"],
                "n_v783": len(fids_783),
                "n_m1_banc": len(m1_b),
                "n_m2_banc": len(m2_b),
                "m1_eq_m2": bool(m1_b == m2_b),
                "m1_banc_ids": m1_b,
            }
        )

    # SELK (DNg68), Fdg (CB0038), Amulet (CB0626) in BANC
    selk_banc_ids = sorted(b for b in (by_banc_ct.get("DNg68", set()) | by_banc_fafb_ct.get("DNg68", set())) if b in idx_banc)
    fdg_banc_ids = sorted(set().union(*(rev_resolved.get(fid, set()) for fid in mtb.FDG_FIDS)) & set(idx_banc.keys()))
    amulet_banc_ids = sorted(set().union(*(rev_resolved.get(fid, set()) for fid in mtb.AMULET_FIDS)) & set(idx_banc.keys()))
    assert selk_banc_ids == [720575941461347027, 720575941598469276], f"Unexpected SELK BANC IDs: {selk_banc_ids}"
    print(f"  SELK BANC={selk_banc_ids}, Fdg BANC={fdg_banc_ids}, Amulet BANC={amulet_banc_ids}", flush=True)

    # BANC annotated labellar water & sugar GRNs (proofread==TRUE)
    lb3a_right = sorted(
        b for b, d in m_rows.items()
        if b in idx_banc and d["proofread"] == "TRUE" and d["cell_type"] == "LB3a"
        and d["cell_function_detailed"] == "water, ppk28" and d["side"] == "right"
    )
    lb3a_left = sorted(
        b for b, d in m_rows.items()
        if b in idx_banc and d["proofread"] == "TRUE" and d["cell_type"] == "LB3a"
        and d["cell_function_detailed"] == "water, ppk28" and d["side"] == "left"
    )
    lb3a_bilat = sorted(lb3a_right + lb3a_left)
    lb3b_right = sorted(
        b for b, d in m_rows.items()
        if b in idx_banc and d["proofread"] == "TRUE" and d["cell_type"] == "LB3b" and d["side"] == "right"
    )
    lb3c_right = sorted(
        b for b, d in m_rows.items()
        if b in idx_banc and d["proofread"] == "TRUE" and d["cell_type"] == "LB3c" and d["side"] == "right"
    )
    print(
        f"  Annotated labellar GRNs: LB3a water R={len(lb3a_right)} L={len(lb3a_left)} (bilat={len(lb3a_bilat)}); "
        f"sugar R LB3b={len(lb3b_right)}, LB3c={len(lb3c_right)}",
        flush=True,
    )

    print("[3/7] Computing 6 BANC v888 static baselines (M1 & M2) and SELK full-CNS anatomy...", flush=True)
    static_m1, is_dir_m1 = smb.compute_static_baselines(
        g_banc["pre_s"], g_banc["post_s"], g_banc["w_raw_s"], g_banc["n_nodes"], m1_indices, mn9_watch_banc.tolist()
    )
    static_m2, is_dir_m2 = smb.compute_static_baselines(
        g_banc["pre_s"], g_banc["post_s"], g_banc["w_raw_s"], g_banc["n_nodes"], m2_indices, mn9_watch_banc.tolist()
    )

    # SELK (DNg68) full-CNS structural analysis
    selk_idx = [idx_banc[b] for b in selk_banc_ids]
    fdg_idx = [idx_banc[b] for b in fdg_banc_ids]
    amulet_idx = [idx_banc[b] for b in amulet_banc_ids]
    selk_static = mtb.compute_single_group_static(
        g_banc["pre_s"], g_banc["post_s"], g_banc["w_raw_s"], g_banc["n_nodes"], np.array(selk_idx, dtype=np.int32), mn9_watch_banc.tolist()
    )
    fdg_static = mtb.compute_single_group_static(
        g_banc["pre_s"], g_banc["post_s"], g_banc["w_raw_s"], g_banc["n_nodes"], np.array(fdg_idx, dtype=np.int32), mn9_watch_banc.tolist()
    )
    amulet_static = mtb.compute_single_group_static(
        g_banc["pre_s"], g_banc["post_s"], g_banc["w_raw_s"], g_banc["n_nodes"], np.array(amulet_idx, dtype=np.int32), mn9_watch_banc.tolist()
    )

    # Dissect SELK direct postsynaptic targets: brain vs VNC, E vs I
    selk_post_w: dict[int, float] = defaultdict(float)
    for si in selk_idx:
        p0, p1 = int(g_banc["indptr"][si]), int(g_banc["indptr"][si + 1])
        for e in range(p0, p1):
            selk_post_w[int(g_banc["post_s"][e])] += float(g_banc["w_raw_s"][e])

    selk_region_ei = defaultdict(lambda: {"n_targets": 0, "sum_syn": 0.0})
    selk_top_targets = []
    for ti, w in sorted(selk_post_w.items(), key=lambda x: -x[1]):
        bid = int(g_banc["node_ids"][ti])
        d = m_rows[bid]
        reg = d["region"] or "unknown"
        nt_p = nt_csv[bid][0] if bid in nt_csv else "MISSING"
        cls = int(g_banc["node_class"][ti])
        ei = "E" if cls == 0 else ("I" if cls == 1 else "sink")
        selk_region_ei[f"{reg}_{ei}"]["n_targets"] += 1
        selk_region_ei[f"{reg}_{ei}"]["sum_syn"] += float(w)
        if len(selk_top_targets) < 20:
            selk_top_targets.append(
                {
                    "banc_idx": int(ti),
                    "banc_id": bid,
                    "side": d["side"],
                    "region": reg,
                    "super_class": d["super_class"],
                    "cell_type": d["cell_type"],
                    "fafb_cell_type": d["fafb_cell_type"],
                    "nt_predicted": nt_p,
                    "ei_class": ei,
                    "syn_from_selk": float(w),
                }
            )

    # Shortest purely excitatory paths SELK -> MN9: unrestricted, brain-only, and forced via VNC
    is_vnc = np.array([m_rows[int(b)]["region"] == "ventral_nerve_cord" for b in g_banc["node_ids"].tolist()], dtype=bool)
    mn9_set = set(mn9_watch_banc.tolist())
    path_unrestricted = shortest_exc_path_with_constraint(
        g_banc["indptr"], g_banc["post_s"], g_banc["w_raw_s"], g_banc["node_class"], selk_idx, mn9_set, None
    )
    path_brain_only = shortest_exc_path_with_constraint(
        g_banc["indptr"], g_banc["post_s"], g_banc["w_raw_s"], g_banc["node_class"], selk_idx, mn9_set, ~is_vnc
    )
    # Forced via VNC: start from direct excitatory VNC postsynaptic targets of SELK!
    selk_vnc_exc_targets = [ti for ti, w in selk_post_w.items() if w > 0 and is_vnc[ti] and g_banc["node_class"][ti] == 0]
    path_from_vnc = shortest_exc_path_with_constraint(
        g_banc["indptr"],
        g_banc["post_s"],
        g_banc["w_raw_s"],
        g_banc["node_class"],
        selk_vnc_exc_targets,
        mn9_set,
        None,
        init_bottlenecks=selk_post_w,
    )

    def annotate_path(pinfo: dict, prepend_selk: bool = False) -> dict:
        out = dict(pinfo)
        if pinfo["reachable"]:
            nodes_desc = []
            if prepend_selk:
                first_v = pinfo["path_indices"][0]
                out["min_hops_from_selk"] = int(pinfo["min_hops"]) + 1
                out["selk_to_first_vnc_syn"] = float(selk_post_w[first_v])
                out["bottleneck_weight_from_selk"] = float(min(pinfo["bottleneck_weight"], selk_post_w[first_v]))
            for idx_n in pinfo["path_indices"]:
                bid = int(g_banc["node_ids"][idx_n])
                d = m_rows[bid]
                nodes_desc.append(
                    {
                        "banc_id": bid,
                        "region": d["region"],
                        "super_class": d["super_class"],
                        "cell_type": d["cell_type"],
                        "fafb_cell_type": d["fafb_cell_type"],
                        "nt": nt_csv[bid][0] if bid in nt_csv else "MISSING",
                    }
                )
            out["path_nodes"] = nodes_desc
        return out

    path_unrestricted = annotate_path(path_unrestricted)
    path_brain_only = annotate_path(path_brain_only)
    path_via_vnc = annotate_path(path_from_vnc, prepend_selk=True)
    print(
        f"  SELK exc shortest paths -> unrestricted: {path_unrestricted['min_hops']} hops (bn={path_unrestricted['bottleneck_weight']}), "
        f"brain-only: {path_brain_only['min_hops']} hops (bn={path_brain_only['bottleneck_weight']}), "
        f"via-VNC: {path_via_vnc.get('min_hops_from_selk')} hops (bn={path_via_vnc.get('bottleneck_weight_from_selk')})",
        flush=True,
    )

    print("[4/7] Running SELK/Fdg/Amulet full-CNS LIF, bounded Task 2 water check & auxiliary sugar note (parallel)...", flush=True)
    ctx = mp.get_context("spawn")
    step4_ckpt = args.ckpt_dir / "step4_selk_water_aux.json"
    lb3a_r_idx = [idx_banc[b] for b in lb3a_right]
    lb3a_bi_idx = [idx_banc[b] for b in lb3a_bilat]
    lb3b_r_idx = [idx_banc[b] for b in lb3b_right]
    lb3bc_r_idx = [idx_banc[b] for b in sorted(lb3b_right + lb3c_right)]
    selk_watch_list = mn9_watch_banc.tolist() + [t["banc_idx"] for t in selk_top_targets]
    mn9_watch_list = mn9_watch_banc.tolist()
    water11_idx_lists = [x.tolist() for x in water11_banc_indices]

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        if step4_ckpt.exists():
            s4 = json.load(open(step4_ckpt))
            selk_cns_lif = s4["selk_cns_lif"]
            water_bounded = s4["water_bounded"]
            rewire_water_checks_spec = s4["rewire_water_checks_spec"]
            aux_sugar = s4["aux_sugar"]
            print("  [step 4 loaded from checkpoint]", flush=True)
        else:
            # Wave 4A: SELK/Fdg/Amulet (24 sims) + Water baselines (12 sims) + Aux sugar (4 sims) = 40 sims in parallel!
            wave4a_jobs = []
            for sname, ws in [("frozen_0.275", w_syn_frozen), ("anat_scale", w_syn_anat)]:
                for f_hz in (20.0, 50.0, 100.0, 200.0):
                    fk = f"{int(f_hz)}hz"
                    wave4a_jobs.append({
                        "tag": f"selk|{sname}|{fk}", "npz_path": str(banc_npz), "w_syn": ws,
                        "exc_idx": selk_idx, "watch_idx": selk_watch_list, "rate_hz": f_hz,
                        "n_trials": args.n_trials, "seed": 81000 + int(f_hz),
                    })
                    wave4a_jobs.append({
                        "tag": f"fdg|{sname}|{fk}", "npz_path": str(banc_npz), "w_syn": ws,
                        "exc_idx": fdg_idx, "watch_idx": mn9_watch_list, "rate_hz": f_hz,
                        "n_trials": args.n_trials, "seed": 82000 + int(f_hz),
                    })
                    wave4a_jobs.append({
                        "tag": f"amulet|{sname}|{fk}", "npz_path": str(banc_npz), "w_syn": ws,
                        "exc_idx": amulet_idx, "watch_idx": mn9_watch_list, "rate_hz": f_hz,
                        "n_trials": args.n_trials, "seed": 83000 + int(f_hz),
                    })
                for g_label, g_list in [("right_unilateral_14", lb3a_r_idx), ("bilateral_16", lb3a_bi_idx)]:
                    for f_hz in (160.0, 180.0, 200.0):
                        wave4a_jobs.append({
                            "tag": f"wbase|{g_label}|{sname}|{int(f_hz)}hz", "npz_path": str(banc_npz), "w_syn": ws,
                            "exc_idx": g_list, "watch_idx": mn9_watch_list, "rate_hz": f_hz,
                            "n_trials": args.n_trials, "seed": 70000,
                        })
                for g_label, g_list in [("right_LB3b_40", lb3b_r_idx), ("right_LB3b_LB3c_45", lb3bc_r_idx)]:
                    wave4a_jobs.append({
                        "tag": f"aux|{g_label}|{sname}", "npz_path": str(banc_npz), "w_syn": ws,
                        "exc_idx": g_list, "watch_idx": mn9_watch_list, "rate_hz": 100.0,
                        "n_trials": args.n_trials, "seed": 42,
                    })

            t4a = time.time()
            res4a = {r["tag"]: r for r in (f.result() for f in concurrent.futures.as_completed([ex.submit(_single_sim_worker, j) for j in wave4a_jobs]))}
            print(f"  [step 4A] {len(wave4a_jobs)} sims done in {time.time() - t4a:.1f}s", flush=True)

            selk_cns_lif = {}
            for sname, ws in [("frozen_0.275", w_syn_frozen), ("anat_scale", w_syn_anat)]:
                selk_cns_lif[sname] = {"w_syn": ws, "SELK_DNg68": {}, "Fdg_CB0038": {}, "Amulet_CB0626": {}}
                for f_hz in (20.0, 50.0, 100.0, 200.0):
                    fk = f"{int(f_hz)}hz"
                    rs = res4a[f"selk|{sname}|{fk}"]
                    ms = rs["means"]
                    selk_cns_lif[sname]["SELK_DNg68"][fk] = {
                        "mn9_l_hz": float(ms[0]), "mn9_r_hz": float(ms[1]),
                        "mn9_mean_hz": float(0.5 * (ms[0] + ms[1])),
                        "top20_target_rates_hz": [float(x) for x in ms[2:]],
                        "max_ring_occ": int(rs["max_occ"]),
                    }
                    for gname, pfx in [("Fdg_CB0038", "fdg"), ("Amulet_CB0626", "amulet")]:
                        mg = res4a[f"{pfx}|{sname}|{fk}"]["means"]
                        selk_cns_lif[sname][gname][fk] = {
                            "mn9_l_hz": float(mg[0]), "mn9_r_hz": float(mg[1]),
                            "mn9_mean_hz": float(0.5 * (mg[0] + mg[1])),
                        }
                print(
                    f"  [{sname}] SELK 50/200Hz MN9={selk_cns_lif[sname]['SELK_DNg68']['50hz']['mn9_mean_hz']:.2f}/"
                    f"{selk_cns_lif[sname]['SELK_DNg68']['200hz']['mn9_mean_hz']:.2f} Hz | "
                    f"Fdg 50Hz={selk_cns_lif[sname]['Fdg_CB0038']['50hz']['mn9_mean_hz']:.2f} Hz | "
                    f"Amulet 50Hz={selk_cns_lif[sname]['Amulet_CB0626']['50hz']['mn9_mean_hz']:.2f} Hz",
                    flush=True,
                )

            aux_sugar = {}
            for g_label, g_list in [("right_LB3b_40", lb3b_r_idx), ("right_LB3b_LB3c_45", lb3bc_r_idx)]:
                for sname in ("frozen_0.275", "anat_scale"):
                    mg = res4a[f"aux|{g_label}|{sname}"]["means"]
                    aux_sugar[f"{g_label}_{sname}_100hz"] = {
                        "n_grns": int(len(g_list)),
                        "mn9_l_hz": float(mg[0]), "mn9_r_hz": float(mg[1]),
                        "mn9_mean_hz": float(0.5 * (mg[0] + mg[1])),
                    }
                    print(f"  [aux sugar {g_label} {sname}] MN9_l={mg[0]:.2f}, MN9_r={mg[1]:.2f}", flush=True)

            # Wave 4B: Parallel water silencing (all nonzero baseline configs x 11 types simultaneously!)
            water_bounded = {
                "grn_counts": {"lb3a_right": len(lb3a_right), "lb3a_left": len(lb3a_left), "lb3a_bilateral": len(lb3a_bilat)},
                "baselines": {},
                "silencing_runs": {},
            }
            rewire_water_checks_spec = {}
            wave4b_jobs = []
            active_water_keys = []
            for g_label, g_list in [("right_unilateral_14", lb3a_r_idx), ("bilateral_16", lb3a_bi_idx)]:
                for sname, ws in [("frozen_0.275", w_syn_frozen), ("anat_scale", w_syn_anat)]:
                    for f_hz in (160.0, 180.0, 200.0):
                        key = f"{g_label}_{sname}_{int(f_hz)}hz"
                        mb = res4a[f"wbase|{g_label}|{sname}|{int(f_hz)}hz"]["means"]
                        bl, br, bm = float(mb[0]), float(mb[1]), float(0.5 * (mb[0] + mb[1]))
                        water_bounded["baselines"][key] = {"mn9_l_hz": bl, "mn9_r_hz": br, "mn9_mean_hz": bm}
                        print(f"  [water baseline {key}] MN9_l={bl:.2f} Hz, MN9_r={br:.2f} Hz, mean={bm:.2f} Hz", flush=True)
                        if bm > 0.0:
                            active_water_keys.append((key, g_label, g_list, ws, f_hz, bl, br, bm))
                            for si_k, sil_list in enumerate(water11_idx_lists):
                                wave4b_jobs.append({
                                    "tag": f"wsil|{key}|{si_k}", "npz_path": str(banc_npz), "w_syn": ws,
                                    "exc_idx": g_list, "sil_idx": sil_list, "watch_idx": mn9_watch_list,
                                    "rate_hz": f_hz, "n_trials": args.n_trials, "seed": 70000 + (si_k + 1) * 131,
                                })
                            if int(f_hz) == 160 or (g_label == "right_unilateral_14" and int(f_hz) == 200):
                                rewire_water_checks_spec[key] = {"idx": g_list, "w_syn": ws, "rate_hz": f_hz}

            if wave4b_jobs:
                t4b = time.time()
                res4b = {r["tag"]: r for r in (f.result() for f in concurrent.futures.as_completed([ex.submit(_single_sim_worker, j) for j in wave4b_jobs]))}
                print(f"  [step 4B] {len(wave4b_jobs)} water silencing sims done in {time.time() - t4b:.1f}s", flush=True)
                y_water_sig = np.array([r["sig_impaired"] for r in st9b_records], dtype=bool)
                y_water_did = np.array([r["delta_did"] for r in st9b_records], dtype=np.float64)
                y_ws_int = y_water_sig.astype(np.int32)
                no_u = np.array([r["name"].lower() != "usnea" for r in st9b_records], dtype=bool)
                for key, g_label, g_list, ws, f_hz, bl, br, bm in active_water_keys:
                    sil_l, sil_r, rel_l, rel_r, rel_mean = [], [], [], [], []
                    for si_k in range(len(water11_idx_lists)):
                        ms = res4b[f"wsil|{key}|{si_k}"]["means"]
                        ml, mr = float(ms[0]), float(ms[1])
                        mm = 0.5 * (ml + mr)
                        sil_l.append(ml)
                        sil_r.append(mr)
                        rel_l.append((ml - bl) / bl if bl > 0 else 0.0)
                        rel_r.append((mr - br) / br if br > 0 else 0.0)
                        rel_mean.append((mm - bm) / bm if bm > 0 else 0.0)
                    metrics_by_readout = {}
                    for rname, rel_vec, base_val in [
                        ("mn9_l_contra_to_right", np.array(rel_l, dtype=np.float64), bl),
                        ("mn9_r", np.array(rel_r, dtype=np.float64), br),
                        ("mn9_mean", np.array(rel_mean, dtype=np.float64), bm),
                    ]:
                        if base_val <= 0.0:
                            continue
                        sc = -rel_vec
                        u_stat, u_p = stats.mannwhitneyu(sc[y_water_sig], sc[~y_water_sig], alternative="two-sided")
                        sp_all_p = 1.0 if np.all(rel_vec == rel_vec[0]) else float(stats.spearmanr(rel_vec, y_water_did).pvalue)
                        sp_nou_p = 1.0 if np.all(rel_vec[no_u] == rel_vec[no_u][0]) else float(stats.spearmanr(rel_vec[no_u], y_water_did[no_u]).pvalue)
                        metrics_by_readout[rname] = {
                            "all_11": {
                                "auroc": float(smb.auroc_exact(y_ws_int, sc)),
                                "auprc": float(smb.average_precision_exact(y_ws_int, sc)),
                                "mannwhitney_u": float(u_stat),
                                "mannwhitney_p_twosided": float(u_p),
                                "spearman_vs_delta_did": smb.spearman_safe(rel_vec, y_water_did),
                                "spearman_p": sp_all_p,
                            },
                            "diag_excl_usnea_10": {
                                "auroc": float(smb.auroc_exact(y_ws_int[no_u], sc[no_u])),
                                "spearman_vs_delta_did": smb.spearman_safe(rel_vec[no_u], y_water_did[no_u]),
                                "spearman_p": sp_nou_p,
                            },
                        }
                    water_bounded["silencing_runs"][key] = {
                        "rate_hz": f_hz, "w_syn": ws,
                        "base_mn9_l": bl, "base_mn9_r": br, "base_mn9_mean": bm,
                        "sil_mn9_l": sil_l, "sil_mn9_r": sil_r,
                        "rel_change_mn9_l": rel_l, "rel_change_mn9_r": rel_r, "rel_change_mn9_mean": rel_mean,
                        "metrics_by_readout": metrics_by_readout,
                    }

            _save_json_atomic(
                step4_ckpt,
                {
                    "selk_cns_lif": selk_cns_lif,
                    "water_bounded": water_bounded,
                    "rewire_water_checks_spec": rewire_water_checks_spec,
                    "aux_sugar": aux_sugar,
                },
            )

        print("[5/7] Running real BANC v888 Task 1 sweeps + sensitivity checks in parallel...", flush=True)
        m1_idx_lists = [x.tolist() for x in m1_indices]
        m2_idx_lists = [x.tolist() for x in m2_indices]
        m1_all_lists = [x.tolist() for x in m1_indices_all]
        m2_all_lists = [x.tolist() for x in m2_indices_all]

        def chunk_slices(n: int, step: int) -> list[tuple[int, int]]:
            return [(i, min(n, i + step)) for i in range(0, n, step)]

        real_specs = [
            ("frozen_20hz", 20.0, w_syn_frozen, 20000, chunk_slices(106, 27), banc_npz, m1_idx_lists, m2_idx_lists, mn9_watch_list),
            ("frozen_50hz", 50.0, w_syn_frozen, 50000, chunk_slices(106, 14), banc_npz, m1_idx_lists, m2_idx_lists, mn9_watch_list),
            ("frozen_100hz", 100.0, w_syn_frozen, 100000, chunk_slices(106, 14), banc_npz, m1_idx_lists, m2_idx_lists, mn9_watch_list),
            ("frozen_200hz", 200.0, w_syn_frozen, 200000, chunk_slices(106, 14), banc_npz, m1_idx_lists, m2_idx_lists, mn9_watch_list),
            ("anat_20hz", 20.0, w_syn_anat, 20000, chunk_slices(106, 3), banc_npz, m1_idx_lists, m2_idx_lists, mn9_watch_list),
            ("anat_50hz", 50.0, w_syn_anat, 50000, chunk_slices(106, 2), banc_npz, m1_idx_lists, m2_idx_lists, mn9_watch_list),
            ("anat_100hz", 100.0, w_syn_anat, 100000, chunk_slices(106, 2), banc_npz, m1_idx_lists, m2_idx_lists, mn9_watch_list),
            ("anat_200hz", 200.0, w_syn_anat, 200000, chunk_slices(106, 2), banc_npz, m1_idx_lists, m2_idx_lists, mn9_watch_list),
            ("unfilt_frozen_50hz", 50.0, w_syn_frozen, 50000, chunk_slices(106, 18), banc_all_npz, m1_all_lists, m2_all_lists, mn9_watch_all.tolist()),
            ("unfilt_anat_50hz", 50.0, w_syn_anat, 50000, chunk_slices(106, 3), banc_all_npz, m1_all_lists, m2_all_lists, mn9_watch_all.tolist()),
        ]

        t5 = time.time()
        real_slices: dict[str, list[dict]] = defaultdict(list)
        futs = []
        for label, f_hz, ws, sb, slices, npz_p, l1, l2, mw in real_specs:
            for si, ei in slices:
                j = {
                    "label": label,
                    "rate_hz": f_hz,
                    "w_syn": ws,
                    "seed_base": sb,
                    "start_i": si,
                    "end_i": ei,
                    "npz_path": str(npz_p),
                    "m1_indices": l1,
                    "m2_indices": l2,
                    "mn9_watch": mw,
                    "n_trials": args.n_trials,
                    "ckpt_path": str(args.ckpt_dir / f"real_{label}_{si}_{ei}.json"),
                }
                futs.append(ex.submit(_real_slice_worker, j))
        # Also submit the ~10 M1+fafb_match diff sims into the same pool!
        mf_futs = []
        for sname, ws in [("frozen_50hz", w_syn_frozen), ("anat_50hz", w_syn_anat)]:
            for i in range(106):
                if not np.array_equal(m1_indices[i], m1mf_indices[i]):
                    mf_futs.append(ex.submit(_single_sim_worker, {
                        "tag": f"m1mf|{sname}|{i}", "npz_path": str(banc_npz), "w_syn": ws,
                        "exc_idx": m1mf_indices[i].tolist(), "watch_idx": mn9_watch_list,
                        "rate_hz": 50.0, "n_trials": args.n_trials, "seed": 50000 + i * 97,
                    }))

        n_tot_slices = len(futs)
        n_done_slices = 0
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            n_done_slices += 1
            real_slices[r["label"]].append(r)
            if n_done_slices % 30 == 0 or n_done_slices == n_tot_slices:
                print(
                    f"  [real {n_done_slices:3d}/{n_tot_slices} elapsed={time.time()-t5:.1f}s] last={r['label']} ({r['elapsed_sec']:.1f}s)",
                    flush=True,
                )
        mf_res = {f.result()["tag"]: f.result() for f in concurrent.futures.as_completed(mf_futs)}

        def stitch(label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
            parts = sorted(real_slices[label], key=lambda x: x["start_i"])
            m1m = np.vstack([np.array(p["m1_means"], dtype=np.float64) for p in parts])
            m1s = np.vstack([np.array(p["m1_stds"], dtype=np.float64) for p in parts])
            m2m = np.vstack([np.array(p["m2_means"], dtype=np.float64) for p in parts])
            m2s = np.vstack([np.array(p["m2_stds"], dtype=np.float64) for p in parts])
            occ = max(int(p["max_occ"]) for p in parts)
            assert m1m.shape == (106, 2) and m2m.shape == (106, 2)
            return m1m, m1s, m2m, m2s, occ

        real_m1_all_freqs: dict[str, np.ndarray] = {}
        real_m2_all_freqs: dict[str, np.ndarray] = {}
        real_m1_bilat: dict[str, np.ndarray] = {}
        real_m2_bilat: dict[str, np.ndarray] = {}
        for label in ("frozen_20hz", "frozen_50hz", "frozen_100hz", "frozen_200hz", "anat_20hz", "anat_50hz", "anat_100hz", "anat_200hz"):
            m1m, _, m2m, _, _ = stitch(label)
            real_m1_all_freqs[label] = m1m.mean(axis=1)
            real_m2_all_freqs[label] = m2m.mean(axis=1)
            real_m1_bilat[label] = m1m
            real_m2_bilat[label] = m2m

        sens_checks = {}
        for sname, base_m1 in [
            ("frozen_50hz", real_m1_all_freqs["frozen_50hz"]),
            ("anat_50hz", real_m1_all_freqs["anat_50hz"]),
        ]:
            sc = base_m1.copy()
            for i in range(106):
                tk = f"m1mf|{sname}|{i}"
                if tk in mf_res:
                    sc[i] = float(0.5 * sum(mf_res[tk]["means"]))
            sens_checks[f"m1_plus_fafb_match_{sname}"] = sc

        for ulabel, skey in [("unfilt_frozen_50hz", "frozen_50hz"), ("unfilt_anat_50hz", "anat_50hz")]:
            u_m1, _, u_m2, _, _ = stitch(ulabel)
            sens_checks[f"unfiltered_all_edges_m1_{skey}"] = u_m1.mean(axis=1)
            sens_checks[f"unfiltered_all_edges_m2_{skey}"] = u_m2.mean(axis=1)

        print(
            f"[6/7] Launching {2 * args.n_rewires} BANC rewired null graphs (paired Setting 1 & Setting 2, M1 & M2)...",
            flush=True,
        )
        rw_jobs = []
        for k in range(args.n_rewires):
            rw_jobs.append(
                {
                    "family": "rewire_full",
                    "idx": k,
                    "seed": 9301 + k,
                    "protect_targets": [],
                    "npz_path": str(banc_npz),
                    "w_syn_frozen": w_syn_frozen,
                    "w_syn_anat": w_syn_anat,
                    "m1_indices": m1_idx_lists,
                    "m2_indices": m2_idx_lists,
                    "mn9_watch": mn9_watch_list,
                    "n_trials": args.n_trials,
                    "water_baseline_checks": rewire_water_checks_spec,
                    "ckpt_path": str(args.ckpt_dir / f"rewire_full_{k:02d}.json"),
                }
            )
            rw_jobs.append(
                {
                    "family": "rewire_mn9_protected",
                    "idx": k,
                    "seed": 9401 + k,
                    "protect_targets": mn9_watch_list,
                    "npz_path": str(banc_npz),
                    "w_syn_frozen": w_syn_frozen,
                    "w_syn_anat": w_syn_anat,
                    "m1_indices": m1_idx_lists,
                    "m2_indices": m2_idx_lists,
                    "mn9_watch": mn9_watch_list,
                    "n_trials": args.n_trials,
                    "water_baseline_checks": rewire_water_checks_spec,
                    "ckpt_path": str(args.ckpt_dir / f"rewire_prot_{k:02d}.json"),
                }
            )

        rw_full_out: list[dict] = []
        rw_prot_out: list[dict] = []
        for fut in concurrent.futures.as_completed([ex.submit(_rewire_banc_worker, j) for j in rw_jobs]):
            res = fut.result()
            if res["family"] == "rewire_full":
                rw_full_out.append(res)
            else:
                rw_prot_out.append(res)
            print(
                f"  [{res['family']:20s} #{res['idx']:02d} seed={res['seed']}] "
                f"repl={res['meta']['replaced_fraction_movable']:.4f} prot={res['meta']['n_protected']} "
                f"({res['elapsed_sec']:.1f}s)",
                flush=True,
            )

    rw_full_out.sort(key=lambda x: x["idx"])
    rw_prot_out.sort(key=lambda x: x["idx"])

    print("[7/7] Computing full evaluation suite, cross-individual correlations & saving JSON...", flush=True)
    y_max = np.array([r["y_max"] for r in st3_records], dtype=np.float64)
    y_mean = np.array([r["y_mean"] for r in st3_records], dtype=np.float64)
    y_bin = (y_max > 0.0).astype(np.int32)
    y_per02 = (y_max >= 0.2).astype(np.int32)
    is_novel = np.array([not r["prior_published"] for r in st3_records], dtype=bool)
    m1_mapped_mask = np.array([len(x) > 0 for x in m1_indices], dtype=bool)

    v783_per_type = {r["name"]: r for r in v783_res["per_type_task1"]}
    v783_50hz = np.array([float(v783_per_type[r["name"]]["v783_real_lif_50hz_mean"]) for r in st3_records], dtype=np.float64)
    v783_is_dir = np.array([bool(v783_per_type[r["name"]]["is_direct_1hop_v783"]) for r in st3_records], dtype=bool)

    def eval_layer_setting(layer: str, setting: str) -> dict:
        is_m1 = layer == "M1"
        real_pool = real_m1_all_freqs if is_m1 else real_m2_all_freqs
        static_pool = static_m1 if is_m1 else static_m2
        is_dir = is_dir_m1 if is_m1 else is_dir_m2
        rw_key = f"{setting}_{'m1' if is_m1 else 'm2'}_50hz"
        rw_f = [np.array(r[rw_key], dtype=np.float64) for r in rw_full_out]
        rw_p = [np.array(r[rw_key], dtype=np.float64) for r in rw_prot_out]
        real_sub = {f"{setting}_{f}hz": real_pool[f"{setting}_{f}hz"] for f in (20, 50, 100, 200)}
        primary_k = f"{setting}_50hz"

        subsets = {
            "all_106": np.ones(106, dtype=bool),
            "nondirect_banc": ~is_dir,
            "nondirect_common_v783_banc": (~is_dir) & (~v783_is_dir),
            "novel_95": is_novel,
        }
        if is_m1:
            subsets["M1_105"] = m1_mapped_mask
        out = {}
        for sname, smask in subsets.items():
            out[sname] = evaluate_subset_suite(
                y_bin, y_max, y_mean, smask, real_sub, primary_k, static_pool, rw_f, rw_p
            )
        out["per_ge_0_2"] = evaluate_subset_suite(
            y_per02, y_max, y_mean, np.ones(106, dtype=bool), real_sub, primary_k, static_pool, rw_f, rw_p
        )
        return out

    task1_eval = {
        "M1_frozen_0.275": eval_layer_setting("M1", "frozen"),
        "M1_anat_scale": eval_layer_setting("M1", "anat"),
        "M2_frozen_0.275": eval_layer_setting("M2", "frozen"),
        "M2_anat_scale": eval_layer_setting("M2", "anat"),
    }

    # Sensitivity summary
    sens_summary = {}
    for skey, sarr in sens_checks.items():
        base_ref = real_m1_all_freqs["frozen_50hz" if "frozen" in skey else "anat_50hz"] if "m1" in skey else real_m2_all_freqs["frozen_50hz" if "frozen" in skey else "anat_50hz"]
        sens_summary[skey] = {
            "auprc_all_106": float(smb.average_precision_exact(y_bin, sarr)),
            "auroc_all_106": float(smb.auroc_exact(y_bin, sarr)),
            "loocv_f1_all_106": float(smb.evaluate_heldout_cv(y_bin, sarr)["loocv"]["f1"]),
            "pearson_vs_main": float(np.corrcoef(sarr, base_ref)[0, 1]) if np.std(sarr) > 0 and np.std(base_ref) > 0 else 0.0,
            "spearman_vs_main": smb.spearman_safe(sarr, base_ref),
            "mae_vs_main": float(np.mean(np.abs(sarr - base_ref))),
            "max_abs_diff_vs_main": float(np.max(np.abs(sarr - base_ref))),
        }

    # Cross-individual v783 vs BANC correlations
    cross_ind = {}
    for label, arr, dir_arr in [
        ("M1_frozen_50hz", real_m1_all_freqs["frozen_50hz"], is_dir_m1),
        ("M1_anat_50hz", real_m1_all_freqs["anat_50hz"], is_dir_m1),
        ("M2_frozen_50hz", real_m2_all_freqs["frozen_50hz"], is_dir_m2),
        ("M2_anat_50hz", real_m2_all_freqs["anat_50hz"], is_dir_m2),
    ]:
        r_val = float(np.corrcoef(v783_50hz, arr)[0, 1]) if np.std(arr) > 0 else 0.0
        rho_val = smb.spearman_safe(v783_50hz, arr)
        gt0_agree = int(((v783_50hz > 0) == (arr > 0)).sum())
        v783_only_pos = [st3_records[i]["name"] for i in range(106) if v783_50hz[i] > 0 and arr[i] == 0]
        banc_only_pos = [st3_records[i]["name"] for i in range(106) if v783_50hz[i] == 0 and arr[i] > 0]
        cross_ind[label] = {
            "pearson_r_all106": r_val,
            "spearman_rho_all106": rho_val,
            "mae_hz": float(np.mean(np.abs(v783_50hz - arr))),
            "gt0_agreement_count": gt0_agree,
            "v783_gt0_banc_eq0": v783_only_pos,
            "v783_eq0_banc_gt0": banc_only_pos,
            "n_direct_1hop_v783": int(v783_is_dir.sum()),
            "n_direct_1hop_banc": int(dir_arr.sum()),
            "n_direct_1hop_both": int((v783_is_dir & dir_arr).sum()),
            "n_nondirect_common": int(((~v783_is_dir) & (~dir_arr)).sum()),
        }

    # Rewired water baseline summary (if any water checks ran)
    rw_water_summary = {}
    for wkey in rewire_water_checks_spec:
        f_vals = [r["water_baseline_checks"][wkey]["mn9_mean_hz"] for r in rw_full_out]
        p_vals = [r["water_baseline_checks"][wkey]["mn9_mean_hz"] for r in rw_prot_out]
        rw_water_summary[wkey] = {
            "rewire_full_mean_max_hz": [float(np.mean(f_vals)), float(np.max(f_vals))],
            "rewire_mn9_prot_mean_max_hz": [float(np.mean(p_vals)), float(np.max(p_vals))],
        }
    water_bounded["rewired_baselines_summary"] = rw_water_summary

    # Per-type comprehensive table (106 rows)
    per_type_out = []
    for i, rec in enumerate(st3_records):
        name = rec["name"]
        per_type_out.append(
            {
                "name": name,
                "y_bin": bool(y_bin[i]),
                "y_max": float(rec["y_max"]),
                "y_mean": float(rec["y_mean"]),
                "prior_published": bool(rec["prior_published"]),
                "n_v783": per_type_mapping[i]["n_v783"],
                "n_m1_banc": per_type_mapping[i]["n_m1_banc"],
                "n_m2_banc": per_type_mapping[i]["n_m2_banc"],
                "is_direct_1hop_v783": bool(v783_is_dir[i]),
                "is_direct_1hop_banc_m1": bool(is_dir_m1[i]),
                "is_direct_1hop_banc_m2": bool(is_dir_m2[i]),
                "v783_50hz": float(v783_50hz[i]),
                "banc_m1_frozen_50hz": float(real_m1_all_freqs["frozen_50hz"][i]),
                "banc_m1_anat_50hz": float(real_m1_all_freqs["anat_50hz"][i]),
                "banc_m2_frozen_50hz": float(real_m2_all_freqs["frozen_50hz"][i]),
                "banc_m2_anat_50hz": float(real_m2_all_freqs["anat_50hz"][i]),
                "banc_m1_anat_20hz": float(real_m1_all_freqs["anat_20hz"][i]),
                "banc_m1_anat_100hz": float(real_m1_all_freqs["anat_100hz"][i]),
                "banc_m1_anat_200hz": float(real_m1_all_freqs["anat_200hz"][i]),
                "banc_m1_static_2hop_relu": float(static_m1["static_2hop_relu_signed"][i]),
                "banc_m2_static_2hop_relu": float(static_m2["static_2hop_relu_signed"][i]),
                "banc_m1_static_1hop": float(static_m1["static_1hop_signed"][i]),
            }
        )

    payload = {
        "protocol_commit": PROTOCOL_COMMIT,
        "input_hashes": hashes,
        "graph_construction_and_anatomy": {
            "raw_edgelist_rows": n_raw_edges,
            "raw_self_loops_removed": n_self_loops,
            "raw_non_self_edges": int(len(ns_cnt)),
            "raw_non_self_valid_nt_edges": int(valid_nt_mask.sum()),
            "raw_non_self_missing_nt_edges_dropped": int((~valid_nt_mask).sum()),
            "main_graph_banc_v888": {
                "filter": "(proofread==TRUE | roughly_proofread==TRUE) & super_class!=glia & pre!=post & valid_nt_v2",
                "n_nodes": g_banc["n_nodes"],
                "n_edges": int(len(g_banc["pre_s"])),
                "n_exc": int((g_banc["node_class"] == 0).sum()),
                "n_inh": int((g_banc["node_class"] == 1).sum()),
                "n_sink": int((g_banc["node_class"] == 2).sum()),
                "sum_abs_weight": float(np.abs(g_banc["w_raw_s"]).sum()),
                "region_counts": dict(region_counts),
                "nt_coverage_on_main_non_self_edges": float(main_edge_mask.sum() / max(1, in_main_mask.sum())),
                "nt_verified_concordance": dict(sorted(nt_ver_conc.items())),
            },
            "whole_brain_v783_ref": {
                "n_nodes": g783["n_nodes"],
                "n_edges": int(len(g783["pre_s"])),
                "sum_abs_weight": float(np.abs(g783["w_raw_s"]).sum()),
            },
            "mn9_anatomy_and_wsyn_scale": {
                "mn9_l_v783": mn9_l_783,
                "mn9_r_v783": mn9_r_783,
                "mn9_l_banc": mn9_l_banc,
                "mn9_r_banc": mn9_r_banc,
                "v783_mn9_l": {k: v for k, v in mn9_l_783_st.items() if k not in ("pre_indices", "weights")},
                "v783_mn9_r": {k: v for k, v in mn9_r_783_st.items() if k not in ("pre_indices", "weights")},
                "banc_mn9_l": {k: v for k, v in mn9_l_banc_st.items() if k not in ("pre_indices", "weights")},
                "banc_mn9_r": {k: v for k, v in mn9_r_banc_st.items() if k not in ("pre_indices", "weights")},
                "s_in_mn9_v783_mean": float(s_in_v783_mean),
                "s_in_mn9_banc_mean": float(s_in_banc_mean),
                "w_syn_frozen_mv": w_syn_frozen,
                "w_syn_anat_mv": w_syn_anat,
                "anat_scale_ratio": float(w_syn_anat / w_syn_frozen),
                "mn9_l_presyn_type_overlap": mn9_l_overlap,
                "mn9_r_presyn_type_overlap": mn9_r_overlap,
            },
            "sez106_v783_to_banc_ei_concordance": dict(sorted(ei_concordance_sez.items())),
        },
        "mapping_audit": {
            "reviewed_file_resolution_stats": dict(rev_res_stats),
            "sez106_m1_types_covered": n_m1_cov,
            "sez106_m2_types_covered": n_m2_cov,
            "sez106_m1_total_banc_cells": int(sum(len(x) for x in m1_indices)),
            "sez106_m2_total_banc_cells": int(sum(len(x) for x in m2_indices)),
            "sez106_m1_eq_m2_types": n_m1_eq_m2,
            "sez106_m1_missing_types": [r["name"] for r in per_type_mapping if r["n_m1_banc"] == 0],
            "water11_mapping": water11_mapping,
            "per_type_sez106": per_type_mapping,
        },
        "rewire_full_meta": [r["meta"] for r in rw_full_out],
        "rewire_mn9_protected_meta": [r["meta"] for r in rw_prot_out],
        "task1_activation_banc": task1_eval,
        "sensitivity_checks": sens_summary,
        "cross_individual_v783_vs_banc": cross_ind,
        "selk_full_cns_diagnosis": {
            "selk_banc_ids": selk_banc_ids,
            "fdg_banc_ids": fdg_banc_ids,
            "amulet_banc_ids": amulet_banc_ids,
            "static_scores": {"SELK_DNg68": selk_static, "Fdg_CB0038": fdg_static, "Amulet_CB0626": amulet_static},
            "selk_postsynaptic_region_ei_summary": dict(sorted(selk_region_ei.items())),
            "selk_total_postsynaptic_targets": len(selk_post_w),
            "selk_total_outgoing_synapses": float(sum(selk_post_w.values())),
            "selk_top20_postsynaptic_targets": selk_top_targets,
            "selk_exc_shortest_paths_to_mn9": {
                "unrestricted_full_cns": path_unrestricted,
                "brain_only_no_vnc": path_brain_only,
                "forced_via_vnc": path_via_vnc,
            },
            "lif_responses": selk_cns_lif,
        },
        "task2_water_silencing_banc_bounded": water_bounded,
        "auxiliary_sugar_grn_note": aux_sugar,
        "per_type_task1": per_type_out,
        "wall_time_sec": float(time.time() - t_start),
    }

    _save_json_atomic(args.output, payload)
    print(f"DONE in {payload['wall_time_sec']:.1f}s -> {args.output}", flush=True)

    # Print crisp headline readout
    for key in ("M1_frozen_0.275", "M1_anat_scale", "M2_frozen_0.275", "M2_anat_scale"):
        ev = task1_eval[key]["all_106"]
        pk = ev["primary_model_key"]
        nd = task1_eval[key]["nondirect_banc"]
        ndc = task1_eval[key]["nondirect_common_v783_banc"]
        p_prot = ev["rewire_mn9_protected"]["ensemble_mean_metrics"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"]
        p_relu = ev["models"]["static_2hop_relu_signed"]["celltype_boot_primary_minus_this"]["auprc"]["p_two_sided_boot"]
        print(
            f"  [{key:16s}] all_106 AP={ev['models'][pk]['auprc']:.4f} AUC={ev['models'][pk]['auroc']:.4f} "
            f"vs prot {ev['rewire_mn9_protected']['graph_level_rank_vs_primary']['wins']}/10 "
            f"(mean {ev['rewire_mn9_protected']['per_graph_auprc_mean']:.4f}, ens {ev['rewire_mn9_protected']['ensemble_mean_metrics']['auprc']:.4f}, "
            f"p={p_prot:.4f}) "
            f"| vs relu2h {ev['models']['static_2hop_relu_signed']['auprc']:.4f} "
            f"(p={p_relu:.4f}) "
            f"| nondir_banc({nd['n_total']}) AP={nd['models'][pk]['auprc']:.4f} vs prot {nd['rewire_mn9_protected']['graph_level_rank_vs_primary']['wins']}/10 "
            f"| nondir_comm({ndc['n_total']}) AP={ndc['models'][pk]['auprc']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
