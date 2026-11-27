#!/usr/bin/env python3
"""FlyWire v783 zero-fitting transfer & cross-lab descriptive readout benchmark (mn9-transfer-01).

Implements the locked protocol in MN9_TRANSFER_PROTOCOL.md:
1. Official v630 -> v783 3-tier ID mapping audit (Tier 1 unchanged 1:1, Tier 2 exact
   supervoxel_id lineage match via flyconnectome/flywire_annotations v1.1.0 -> v3.2.0,
   Tier 3 unmappable) for MN9_l/r, 21 sugar GRNs, 18 water GRNs, Zorro, and 106 SEZ types.
2. Quantitative v630 -> v783 graph delta across whole brain, GRNs, SEZ 106/11, MN9 1-hop
   incoming edges, and Connectivity_783 vs flywire_annotations v3.2.0 NT concordance.
3. Zero-behavior-label W_syn assessment:
   (i) Fully frozen W_syn = 0.275 mV (100 Hz sugar GRN anchor check vs ST 1D).
   (ii) Deterministic single-point sugar-anchor 1D check (0.265..0.285 mV, 0.001 mV step)
        without accessing any 106/11 behavioral labels; full recalibrated null duel if
        protocol trigger (|diff| > 3.0 Hz or |z| >= 3.0) fires, plus sensitivity check.
4. Full v783 evaluation on Task 1 (106 SEZ activation, 20/50/100/200 Hz) and Task 2
   (11 water silencing, 160/180/200 Hz) against 6 static graph baselines and
   K=10 rewire_full (seeds 9101..9110) + K=10 rewire_mn9_protected (seeds 9201..9210).
5. Strict inference-level separation (graph-level empirical rank 1/11 vs cell-type-level
   stratified paired bootstrap B=2000) and Module A descriptive frozen readouts for
   independent-lab qualified SEZ types (SELK/DNg68, Fdg, plus Amulet/CB0626 reference).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import multiprocessing as mp
import pickle
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numba import njit
from scipy import stats

import shiu_mn9_benchmark as smb

# Independent-lab Module A qualified/reference neurons
# SELK (DNg68): Savas, Barnea et al. (2026) Nat Commun, PMC13096221
SELK_V783_FIDS = [720575940629543409, 720575940630808827]
SELK_V630_FIDS = [720575940620668609, 720575940630808827]  # Left mapped via exact supervoxel_id 78959297648207291
# Amulet (CB0626): postsynaptic partner of SELK in Savas 2026 & SS32423 in Shiu 2024 Sup Table 3
AMULET_FIDS = [720575940604470240, 720575940635211768]
# Fdg (CB0038): Flood, Yoshihara et al. (2013) Nature, PMC3727048 & Shiu 2024 Sup Table 3
FDG_FIDS = [720575940647030324, 720575940631997032]


@njit
def run_lif_brian2_wsyn(
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
    """Exact Brian2 discrete-time LIF schedule parameterized by scalar w_syn (bit-identical to smb at 0.275)."""
    dt = 0.1  # ms
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

    n_nodes = len(indptr) - 1
    n_watch = len(watch_idx)
    watch_counts = np.zeros((n_trials, n_watch), dtype=np.int32)

    is_exc = np.zeros(n_nodes, dtype=np.uint8)
    for k in range(len(exc_idx)):
        is_exc[exc_idx[k]] = 1

    u = np.zeros(n_nodes, dtype=np.float64)
    g = np.zeros(n_nodes, dtype=np.float64)
    last_spk = np.full(n_nodes, -100000, dtype=np.int32)
    in_active = np.zeros(n_nodes, dtype=np.uint8)
    active_list = np.empty(n_nodes, dtype=np.int32)
    next_active = np.empty(n_nodes, dtype=np.int32)
    spiked_now = np.empty(n_nodes, dtype=np.int32)

    ring_len = dly_steps + 1
    ring_spk = np.empty((ring_len, n_nodes), dtype=np.int32)
    ring_cnt = np.zeros(ring_len, dtype=np.int32)
    max_ring_occ = 0

    np.random.seed(seed)

    for tr in range(n_trials):
        n_act = 0
        for i in range(n_nodes):
            last_spk[i] = -100000
        for slot in range(ring_len):
            ring_cnt[slot] = 0
        for k in range(len(exc_idx)):
            idx = exc_idx[k]
            if in_active[idx] == 0:
                in_active[idx] = 1
                active_list[n_act] = idx
                n_act += 1

        for step in range(n_steps):
            n_spk_now = 0
            n_next = 0
            for a_i in range(n_act):
                i = active_list[a_i]
                nr = 1
                if is_exc[i] == 0 and (step - last_spk[i]) < rfc_steps:
                    nr = 0
                if nr == 1:
                    ui = u[i]
                    gi = g[i]
                    ui_new = a_v * ui + b_vg * gi
                    gi_new = a_g * gi
                    u[i] = ui_new
                    g[i] = gi_new
                    if ui_new > v_th:
                        spiked_now[n_spk_now] = i
                        n_spk_now += 1
                        next_active[n_next] = i
                        n_next += 1
                    elif is_exc[i] != 0 or abs(ui_new) > 1e-6 or abs(gi_new) > 1e-6:
                        next_active[n_next] = i
                        n_next += 1
                    else:
                        u[i] = 0.0
                        g[i] = 0.0
                        in_active[i] = 0
                else:
                    next_active[n_next] = i
                    n_next += 1

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
                        idx = exc_idx[k]
                        u[idx] += w_poi

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
                        g[dst] += weights[e]
                        if in_active[dst] == 0:
                            in_active[dst] = 1
                            next_active[n_next] = dst
                            n_next += 1
            ring_cnt[cur_slot] = 0

            for s_i in range(n_spk_now):
                i = spiked_now[s_i]
                u[i] = 0.0
                g[i] = 0.0
                last_spk[i] = step

            for a_i in range(n_next):
                active_list[a_i] = next_active[a_i]
            n_act = n_next

        for a_i in range(n_act):
            i = active_list[a_i]
            u[i] = 0.0
            g[i] = 0.0
            in_active[i] = 0

    return watch_counts, max_ring_occ


def load_graph_csr(comp_path: Path, conn_path: Path) -> dict:
    with open(comp_path) as f:
        r = csv.reader(f)
        next(r)
        flyids = np.array([int(x[0]) for x in r], dtype=np.int64)
    flyid2idx = {int(fid): i for i, fid in enumerate(flyids)}
    n_nodes = len(flyids)

    tbl = pq.read_table(
        conn_path,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = tbl["Presynaptic_Index"].to_numpy().astype(np.int32)
    post = tbl["Postsynaptic_Index"].to_numpy().astype(np.int32)
    w_raw = tbl["Excitatory x Connectivity"].to_numpy().astype(np.float32)

    order = np.argsort(pre, kind="mergesort")
    pre_s = pre[order]
    post_s = post[order]
    w_raw_s = w_raw[order]
    indptr = np.empty(n_nodes + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(np.bincount(pre_s, minlength=n_nodes), out=indptr[1:])

    node_class = np.full(n_nodes, 2, dtype=np.int8)
    has_pos = np.zeros(n_nodes, dtype=bool)
    has_neg = np.zeros(n_nodes, dtype=bool)
    has_pos[pre_s[w_raw_s > 0]] = True
    has_neg[pre_s[w_raw_s < 0]] = True
    assert not np.any(has_pos & has_neg), f"Mixed-sign presynaptic node found in {conn_path}"
    node_class[has_pos] = 0
    node_class[has_neg] = 1

    orig_in_pos = np.bincount(post_s[w_raw_s > 0], minlength=n_nodes).astype(np.int32)
    orig_in_neg = np.bincount(post_s[w_raw_s < 0], minlength=n_nodes).astype(np.int32)
    return {
        "flyids": flyids,
        "flyid2idx": flyid2idx,
        "n_nodes": n_nodes,
        "indptr": indptr,
        "pre_s": pre_s,
        "post_s": post_s,
        "w_raw_s": w_raw_s,
        "node_class": node_class,
        "orig_in_pos": orig_in_pos,
        "orig_in_neg": orig_in_neg,
    }


def run_all_sez_wsyn(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    w_syn: float,
    type_indices: list[np.ndarray],
    mn9_watch: np.ndarray,
    slnc_none: np.ndarray,
    rate_hz: float,
    n_trials: int,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_types = len(type_indices)
    means = np.zeros((n_types, 2), dtype=np.float64)
    stds = np.zeros((n_types, 2), dtype=np.float64)
    max_occ = 0
    for i, idxs in enumerate(type_indices):
        if len(idxs) == 0:
            continue
        cnts, occ = run_lif_brian2_wsyn(
            indptr, indices, weights, w_syn, idxs, rate_hz, slnc_none, mn9_watch, n_trials, seed_base + i * 97
        )
        means[i] = cnts.mean(axis=0)
        stds[i] = cnts.std(axis=0)
        if occ > max_occ:
            max_occ = occ
    return means, stds, max_occ


def run_water_silencing_wsyn(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    w_syn: float,
    water_grn_idx: np.ndarray,
    silencing_type_indices: list[np.ndarray],
    mn9_watch: np.ndarray,
    rate_hz: float,
    n_trials: int,
    seed_base: int,
) -> dict:
    n_nodes = len(indptr) - 1
    slnc_mask = np.zeros(n_nodes, dtype=np.uint8)
    base_cnts, _ = run_lif_brian2_wsyn(
        indptr, indices, weights, w_syn, water_grn_idx, rate_hz, slnc_mask, mn9_watch, n_trials, seed_base
    )
    base_l = float(base_cnts[:, 0].mean())
    base_r = float(base_cnts[:, 1].mean())
    base_mean = 0.5 * (base_l + base_r)

    sil_l, sil_r, rel_r, rel_mean = [], [], [], []
    for i, idxs in enumerate(silencing_type_indices):
        slnc_mask[:] = 0
        for idx in idxs:
            slnc_mask[idx] = 1
        cnts, _ = run_lif_brian2_wsyn(
            indptr, indices, weights, w_syn, water_grn_idx, rate_hz, slnc_mask, mn9_watch, n_trials, seed_base + (i + 1) * 131
        )
        ml = float(cnts[:, 0].mean())
        mr = float(cnts[:, 1].mean())
        mm = 0.5 * (ml + mr)
        sil_l.append(ml)
        sil_r.append(mr)
        rel_r.append((mr - base_r) / base_r if base_r > 0 else 0.0)
        rel_mean.append((mm - base_mean) / base_mean if base_mean > 0 else 0.0)
    return {
        "rate_hz": rate_hz,
        "w_syn": w_syn,
        "base_mn9_l": base_l,
        "base_mn9_r": base_r,
        "base_mn9_mean": base_mean,
        "sil_mn9_l": sil_l,
        "sil_mn9_r": sil_r,
        "rel_change_mn9_r": rel_r,
        "rel_change_mn9_mean": rel_mean,
    }


def _save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(path)


def _real_sez_slice_worker(job: dict) -> dict:
    """Spawn-safe worker evaluating a slice of the 106 SEZ types with exact per-type seeds."""
    ckpt = Path(job["ckpt_path"])
    if ckpt.exists():
        return json.load(open(ckpt))
    t0 = time.time()
    g = load_graph_csr(Path(job["comp_path"]), Path(job["conn_path"]))
    w_syn = float(job["w_syn"])
    w_s = (g["w_raw_s"] * np.float32(w_syn)).astype(np.float64)
    slnc_none = np.zeros(g["n_nodes"], dtype=np.uint8)
    mn9_watch = np.array(job["mn9_watch"], dtype=np.int32)
    rate_hz = float(job["rate_hz"])
    n_trials = int(job["n_trials"])
    seed_base = int(job["seed_base"])
    start_i = int(job["start_i"])
    end_i = int(job["end_i"])
    type_indices = [np.array(x, dtype=np.int32) for x in job["type_indices"]]

    # Warmup JIT
    _ = run_lif_brian2_wsyn(g["indptr"], g["post_s"], w_s, w_syn, type_indices[start_i][:1], 50.0, slnc_none, mn9_watch, 1, 0)

    means = []
    stds = []
    max_occ = 0
    for i in range(start_i, end_i):
        idxs = type_indices[i]
        if len(idxs) == 0:
            means.append([0.0, 0.0])
            stds.append([0.0, 0.0])
            continue
        cnts, occ = run_lif_brian2_wsyn(
            g["indptr"], g["post_s"], w_s, w_syn, idxs, rate_hz, slnc_none, mn9_watch, n_trials, seed_base + i * 97
        )
        means.append(cnts.mean(axis=0).tolist())
        stds.append(cnts.std(axis=0).tolist())
        if occ > max_occ:
            max_occ = occ
    out = {
        "kind": "sez_slice",
        "label": job["label"],
        "start_i": start_i,
        "end_i": end_i,
        "means": means,
        "stds": stds,
        "max_occ": int(max_occ),
        "elapsed_sec": float(time.time() - t0),
    }
    _save_json_atomic(ckpt, out)
    return out


def _real_water_worker(job: dict) -> dict:
    """Spawn-safe worker evaluating one real v783 water silencing sweep."""
    ckpt = Path(job["ckpt_path"])
    if ckpt.exists():
        return json.load(open(ckpt))
    t0 = time.time()
    g = load_graph_csr(Path(job["comp_path"]), Path(job["conn_path"]))
    w_syn = float(job["w_syn"])
    w_s = (g["w_raw_s"] * np.float32(w_syn)).astype(np.float64)
    water_idx = np.array(job["water_idx"], dtype=np.int32)
    water_sil_indices = [np.array(x, dtype=np.int32) for x in job["water_sil_indices"]]
    mn9_watch = np.array(job["mn9_watch"], dtype=np.int32)
    slnc_none = np.zeros(g["n_nodes"], dtype=np.uint8)
    _ = run_lif_brian2_wsyn(g["indptr"], g["post_s"], w_s, w_syn, water_idx[:1], 50.0, slnc_none, mn9_watch, 1, 0)
    res = run_water_silencing_wsyn(
        g["indptr"], g["post_s"], w_s, w_syn, water_idx, water_sil_indices, mn9_watch,
        float(job["rate_hz"]), int(job["n_trials"]), int(job["seed_base"])
    )
    out = {"kind": "water", "label": job["label"], "result": res, "elapsed_sec": float(time.time() - t0)}
    _save_json_atomic(ckpt, out)
    return out


def _rewire_worker(job: dict) -> dict:
    """Spawn-safe worker building and evaluating one v783 rewired null graph."""
    ckpt = Path(job["ckpt_path"])
    if ckpt.exists():
        return json.load(open(ckpt))
    g = load_graph_csr(Path(job["comp_path"]), Path(job["conn_path"]))
    n_nodes = g["n_nodes"]
    indptr = g["indptr"]
    pre_s = g["pre_s"]
    post_s = g["post_s"]
    w_raw_s = g["w_raw_s"]
    node_class = g["node_class"]
    orig_in_pos = g["orig_in_pos"]
    orig_in_neg = g["orig_in_neg"]

    w_syn = float(job["w_syn"])
    w_s = (w_raw_s * np.float32(w_syn)).astype(np.float64)
    prot_set = set(job["protect_targets"])
    seed_rw = int(job["seed"])

    t0 = time.time()
    rw_indices, meta = smb.build_rewired_graph(
        indptr, post_s, w_s, pre_s, node_class, orig_in_pos, orig_in_neg, prot_set, seed_rw
    )
    type_indices = [np.array(x, dtype=np.int32) for x in job["type_indices"]]
    water_idx = np.array(job["water_idx"], dtype=np.int32)
    water_sil_indices = [np.array(x, dtype=np.int32) for x in job["water_sil_indices"]]
    mn9_watch = np.array(job["mn9_watch"], dtype=np.int32)
    slnc_none = np.zeros(n_nodes, dtype=np.uint8)

    # Warmup JIT
    _ = run_lif_brian2_wsyn(indptr, rw_indices, w_s, w_syn, water_idx[:1], 50.0, slnc_none, mn9_watch, 1, 0)

    rw_m, _, rw_occ = run_all_sez_wsyn(
        indptr, rw_indices, w_s, w_syn, type_indices, mn9_watch, slnc_none, 50.0, int(job["n_trials"]), 50000
    )
    rw_w = run_water_silencing_wsyn(
        indptr, rw_indices, w_s, w_syn, water_idx, water_sil_indices, mn9_watch, 160.0, int(job["n_trials"]), 70000
    )

    out = {
        "family": job["family"],
        "idx": job["idx"],
        "seed": seed_rw,
        "w_syn": w_syn,
        "meta": meta,
        "scores_50hz": rw_m.mean(axis=1).tolist(),
        "water_160hz": rw_w,
        "max_ring_occ": int(rw_occ),
        "elapsed_sec": float(time.time() - t0),
    }

    # Optional second W_syn evaluation on exact same rewired topology if triggered
    if job.get("w_syn_alt") is not None:
        w_alt = float(job["w_syn_alt"])
        w_s_alt = (w_raw_s * np.float32(w_alt)).astype(np.float64)
        rw_m_alt, _, _ = run_all_sez_wsyn(
            indptr, rw_indices, w_s_alt, w_alt, type_indices, mn9_watch, slnc_none, 50.0, int(job["n_trials"]), 50000
        )
        rw_w_alt = run_water_silencing_wsyn(
            indptr, rw_indices, w_s_alt, w_alt, water_idx, water_sil_indices, mn9_watch, 160.0, int(job["n_trials"]), 70000
        )
        out["alt_scores_50hz"] = rw_m_alt.mean(axis=1).tolist()
        out["alt_water_160hz"] = rw_w_alt

    _save_json_atomic(ckpt, out)
    return out


def compute_single_group_static(
    pre_s: np.ndarray,
    post_s: np.ndarray,
    w_raw_s: np.ndarray,
    n_nodes: int,
    idxs: np.ndarray,
    mn9_indices: list[int],
) -> dict[str, float]:
    scores, is_dir = smb.compute_static_baselines(pre_s, post_s, w_raw_s, n_nodes, [idxs], mn9_indices)
    return {
        **{k: float(v[0]) for k, v in scores.items()},
        "is_direct_1hop": bool(is_dir[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comp-630", type=Path, default=Path("/tmp/projc-flywire-cache/completeness_630.csv"))
    parser.add_argument("--conn-630", type=Path, default=Path("/tmp/projc-flywire-cache/connectivity_630.parquet"))
    parser.add_argument("--comp-783", type=Path, default=Path("/tmp/projc-flywire-cache/Completeness_783.csv"))
    parser.add_argument("--conn-783", type=Path, default=Path("/tmp/projc-flywire-cache/Connectivity_783.parquet"))
    parser.add_argument("--ann-630", type=Path, default=Path("/tmp/projc-flywire-cache/Supplemental_file1_annotations_v630_v1.1.0.tsv"))
    parser.add_argument("--ann-783", type=Path, default=Path("/tmp/projc-flywire-cache/Supplemental_file1_neuron_annotations.tsv"))
    parser.add_argument("--sez-pickle", type=Path, default=Path("/tmp/projc-source02-probe/sez_neurons.pickle"))
    parser.add_argument("--xlsx", type=Path, default=Path("/tmp/projc-source02-probe/shiu_moesm2.xlsx"))
    parser.add_argument("--v630-results", type=Path, default=Path("shiu_mn9_results.json"))
    parser.add_argument("--output", type=Path, default=Path("mn9_transfer_results.json"))
    parser.add_argument("--n-rewires", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--ckpt-dir", type=Path, default=Path("/tmp/projc-flywire-cache/v783_ckpt"))
    args = parser.parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    hashes = {
        "completeness_630_csv": smb.sha256_file(args.comp_630),
        "connectivity_630_parquet": smb.sha256_file(args.conn_630),
        "Completeness_783_csv": smb.sha256_file(args.comp_783),
        "Connectivity_783_parquet": smb.sha256_file(args.conn_783),
        "flywire_annotations_v630_v1_1_0_tsv": smb.sha256_file(args.ann_630),
        "flywire_annotations_v783_v3_2_0_tsv": smb.sha256_file(args.ann_783),
        "sez_neurons_pickle": smb.sha256_file(args.sez_pickle),
        "shiu_moesm2_xlsx": smb.sha256_file(args.xlsx),
    }

    print("[1/7] Loading FlyWire v630 & v783 graphs and annotations...", flush=True)
    g630 = load_graph_csr(args.comp_630, args.conn_630)
    g783 = load_graph_csr(args.comp_783, args.conn_783)

    v630_ann: dict[int, dict[str, str]] = {}
    with open(args.ann_630) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["root_id"].isdigit():
                v630_ann[int(row["root_id"])] = row

    v783_ann: dict[int, dict[str, str]] = {}
    v783_by_sv: dict[str, list[int]] = {}
    with open(args.ann_783) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rid = int(row["root_id"])
            v783_ann[rid] = row
            sv = row.get("supervoxel_id", "")
            if sv and sv != "NA":
                v783_by_sv.setdefault(sv, []).append(rid)

    # Check NT concordance between Connectivity_783 and flywire_annotations v3.2.0
    nt_concordance = Counter()
    for i, fid in enumerate(g783["flyids"]):
        c = int(g783["node_class"][i])
        if c == 2:
            continue
        row = v783_ann.get(int(fid))
        nt = row.get("top_nt", "MISSING") if row else "MISSING"
        nt_concordance[f"{nt}->{'E' if c == 0 else 'I'}"] += 1

    # Load behavioral inputs
    sez_dict = pickle.load(open(args.sez_pickle, "rb"))
    st3_records = smb.load_sup_table_3(args.xlsx, sez_dict)
    st9b_records = smb.load_supp_table_9b(args.xlsx, sez_dict)
    v630_prev = json.load(open(args.v630_results))

    # Strict 3-tier v630 -> v783 mapper
    idx783 = g783["flyid2idx"]
    idx630 = g630["flyid2idx"]

    def map_one(fid630: int) -> dict:
        r6 = v630_ann.get(fid630, {})
        sv6 = r6.get("supervoxel_id", "")
        if fid630 in idx783:
            r7 = v783_ann.get(fid630, {})
            sv7 = r7.get("supervoxel_id", "")
            return {
                "v630_id": fid630,
                "v783_id": fid630,
                "tier": "tier1_unchanged",
                "supervoxel_id": sv6,
                "supervoxel_consistent": bool(sv6 == sv7) if (sv6 and sv7) else None,
                "cell_type_v783": r7.get("cell_type", ""),
                "side_v783": r7.get("side", ""),
                "top_nt_v783": r7.get("top_nt", ""),
            }
        cands = [x for x in v783_by_sv.get(sv6, []) if x in idx783] if sv6 else []
        if len(cands) == 1:
            fid7 = cands[0]
            r7 = v783_ann.get(fid7, {})
            return {
                "v630_id": fid630,
                "v783_id": fid7,
                "tier": "tier2_supervoxel",
                "supervoxel_id": sv6,
                "supervoxel_consistent": True,
                "cell_type_v783": r7.get("cell_type", ""),
                "side_v783": r7.get("side", ""),
                "top_nt_v783": r7.get("top_nt", ""),
            }
        return {
            "v630_id": fid630,
            "v783_id": None,
            "tier": "tier3_unmappable",
            "supervoxel_id": sv6,
            "supervoxel_consistent": False,
            "cell_type_v783": "",
            "side_v783": "",
            "top_nt_v783": "",
        }

    # Audit all target groups
    all_unique_v630 = sorted(
        set(
            [smb.MN9_L_FID, smb.MN9_R_FID]
            + smb.SUGAR_GRN_FIDS
            + smb.WATER_GRN_FIDS
            + smb.ZORRO_FIDS
            + [fid for rec in st3_records for fid in rec["fids"]]
        )
    )
    id_map_table = {fid: map_one(fid) for fid in all_unique_v630}

    # Check collision-freedom among mapped v783 IDs
    rev_check: dict[int, list[int]] = {}
    for fid6, rec in id_map_table.items():
        if rec["v783_id"] is not None:
            rev_check.setdefault(rec["v783_id"], []).append(fid6)
    collisions = {k: v for k, v in rev_check.items() if len(v) > 1}
    assert len(collisions) == 0, f"Many-to-one v630->v783 collision detected: {collisions}"

    # Build v783 indices for every group
    mn9_l_783 = id_map_table[smb.MN9_L_FID]["v783_id"]
    mn9_r_783 = id_map_table[smb.MN9_R_FID]["v783_id"]
    assert mn9_l_783 is not None and mn9_r_783 is not None
    mn9_l_idx783 = idx783[mn9_l_783]
    mn9_r_idx783 = idx783[mn9_r_783]
    mn9_watch_783 = np.array([mn9_l_idx783, mn9_r_idx783], dtype=np.int32)

    sugar_783_fids = [id_map_table[x]["v783_id"] for x in smb.SUGAR_GRN_FIDS]
    water_783_fids = [id_map_table[x]["v783_id"] for x in smb.WATER_GRN_FIDS]
    assert all(x is not None for x in sugar_783_fids)
    assert all(x is not None for x in water_783_fids)
    sugar_idx783 = np.array([idx783[x] for x in sugar_783_fids], dtype=np.int32)
    water_idx783 = np.array([idx783[x] for x in water_783_fids], dtype=np.int32)

    type_indices_783: list[np.ndarray] = []
    type_mapping_audit: list[dict] = []
    type_complete_mapped = np.zeros(len(st3_records), dtype=bool)
    for i, rec in enumerate(st3_records):
        mapped = [id_map_table[x] for x in rec["fids"]]
        f7 = [m["v783_id"] for m in mapped if m["v783_id"] is not None]
        n_t1 = sum(1 for m in mapped if m["tier"] == "tier1_unchanged")
        n_t2 = sum(1 for m in mapped if m["tier"] == "tier2_supervoxel")
        n_t3 = sum(1 for m in mapped if m["tier"] == "tier3_unmappable")
        type_complete_mapped[i] = n_t3 == 0
        type_indices_783.append(np.array([idx783[x] for x in f7], dtype=np.int32))
        type_mapping_audit.append(
            {
                "name": rec["name"],
                "n_v630": len(rec["fids"]),
                "n_v783": len(f7),
                "n_tier1": n_t1,
                "n_tier2": n_t2,
                "n_tier3": n_t3,
                "v630_ids": rec["fids"],
                "v783_ids": f7,
                "unmapped_v630_ids": [m["v630_id"] for m in mapped if m["tier"] == "tier3_unmappable"],
            }
        )

    water_sil_indices_783: list[np.ndarray] = []
    water_mapping_audit: list[dict] = []
    for rec in st9b_records:
        mapped = [id_map_table[x] for x in rec["fids"]]
        f7 = [m["v783_id"] for m in mapped if m["v783_id"] is not None]
        water_sil_indices_783.append(np.array([idx783[x] for x in f7], dtype=np.int32))
        water_mapping_audit.append(
            {
                "name": rec["name"],
                "n_v630": len(rec["fids"]),
                "n_v783": len(f7),
                "n_tier1": sum(1 for m in mapped if m["tier"] == "tier1_unchanged"),
                "n_tier2": sum(1 for m in mapped if m["tier"] == "tier2_supervoxel"),
                "n_tier3": sum(1 for m in mapped if m["tier"] == "tier3_unmappable"),
            }
        )

    # [2/7] Quantitative v630 -> v783 Graph Delta Audit
    print("[2/7] Computing quantitative v630 -> v783 graph delta...", flush=True)

    def outgoing_stats(g: dict, fids: list[int]) -> dict:
        tot_edges = 0
        tot_abs_w = 0.0
        tot_pos_w = 0.0
        tot_neg_w = 0.0
        n_e = n_i = n_s = 0
        for fid in fids:
            u = g["flyid2idx"][fid]
            p0, p1 = int(g["indptr"][u]), int(g["indptr"][u + 1])
            ws = g["w_raw_s"][p0:p1]
            tot_edges += p1 - p0
            tot_abs_w += float(np.abs(ws).sum())
            tot_pos_w += float(ws[ws > 0].sum())
            tot_neg_w += float(ws[ws < 0].sum())
            c = int(g["node_class"][u])
            if c == 0:
                n_e += 1
            elif c == 1:
                n_i += 1
            else:
                n_s += 1
        return {
            "n_nodes": len(fids),
            "n_edges": tot_edges,
            "sum_abs_weight": tot_abs_w,
            "sum_pos_weight": tot_pos_w,
            "sum_neg_weight": tot_neg_w,
            "n_exc": n_e,
            "n_inh": n_i,
            "n_sink": n_s,
        }

    # Build global v630->v783 map via supervoxel_id to match MN9 incoming presynaptic partners across versions
    all_v630_to_v783: dict[int, int] = {}
    for fid6 in g630["flyids"]:
        fid6_i = int(fid6)
        if fid6_i in idx783:
            all_v630_to_v783[fid6_i] = fid6_i
        else:
            sv = v630_ann.get(fid6_i, {}).get("supervoxel_id", "")
            c = [x for x in v783_by_sv.get(sv, []) if x in idx783] if sv else []
            if len(c) == 1:
                all_v630_to_v783[fid6_i] = c[0]

    def mn9_edge_delta(mn9_idx6: int, mn9_idx7: int) -> dict:
        m6 = g630["post_s"] == mn9_idx6
        m7 = g783["post_s"] == mn9_idx7
        e6 = {int(g630["flyids"][p]): float(w) for p, w in zip(g630["pre_s"][m6], g630["w_raw_s"][m6])}
        e7 = {int(g783["flyids"][p]): float(w) for p, w in zip(g783["pre_s"][m7], g783["w_raw_s"][m7])}
        # Project v630 presynaptic IDs into v783 ID space where deterministically mappable
        e6_proj = {}
        unmapped_pre6 = 0
        for p6, w6 in e6.items():
            p7 = all_v630_to_v783.get(p6)
            if p7 is None:
                unmapped_pre6 += 1
            else:
                e6_proj[p7] = e6_proj.get(p7, 0.0) + w6
        common = set(e6_proj) & set(e7)
        added = set(e7) - set(e6_proj)
        removed = set(e6_proj) - set(e7)
        exact_same_w = sum(1 for k in common if e6_proj[k] == e7[k])
        sign_flipped = sum(1 for k in common if (e6_proj[k] > 0) != (e7[k] > 0))
        abs_diffs = [abs(e7[k] - e6_proj[k]) for k in common]
        return {
            "v630_edges": len(e6),
            "v630_exc_inh": [sum(1 for w in e6.values() if w > 0), sum(1 for w in e6.values() if w < 0)],
            "v630_sum_pos_neg": [sum(w for w in e6.values() if w > 0), sum(w for w in e6.values() if w < 0)],
            "v783_edges": len(e7),
            "v783_exc_inh": [sum(1 for w in e7.values() if w > 0), sum(1 for w in e7.values() if w < 0)],
            "v783_sum_pos_neg": [sum(w for w in e7.values() if w > 0), sum(w for w in e7.values() if w < 0)],
            "v630_pre_unmapped": unmapped_pre6,
            "common_partners": len(common),
            "added_partners_v783": len(added),
            "removed_partners_v630": len(removed),
            "common_exact_same_weight": exact_same_w,
            "common_sign_flipped": sign_flipped,
            "common_mean_abs_weight_diff": float(np.mean(abs_diffs)) if abs_diffs else 0.0,
            "common_max_abs_weight_diff": float(np.max(abs_diffs)) if abs_diffs else 0.0,
        }

    sign_flips_target = []
    for fid6, rec in id_map_table.items():
        fid7 = rec["v783_id"]
        if fid7 is None:
            continue
        c6 = int(g630["node_class"][idx630[fid6]])
        c7 = int(g783["node_class"][idx783[fid7]])
        if c6 != c7:
            owners = [r["name"] for r in st3_records if fid6 in r["fids"]]
            sign_flips_target.append(
                {
                    "owners": owners,
                    "v630_id": fid6,
                    "v783_id": fid7,
                    "v630_cls": c6,
                    "v783_cls": c7,
                    "cell_type_v783": rec["cell_type_v783"],
                    "top_nt_v783": rec["top_nt_v783"],
                    "known_nt_v783": v783_ann.get(fid7, {}).get("known_nt", ""),
                }
            )

    sez_mapped_v630 = [fid for fid in sorted(set(x for r in st3_records for x in r["fids"])) if id_map_table[fid]["v783_id"] is not None]
    sez_mapped_v783 = [id_map_table[fid]["v783_id"] for fid in sez_mapped_v630]
    graph_delta = {
        "whole_brain_v630": {
            "n_nodes": g630["n_nodes"],
            "n_edges": int(len(g630["pre_s"])),
            "n_exc": int((g630["node_class"] == 0).sum()),
            "n_inh": int((g630["node_class"] == 1).sum()),
            "n_sink": int((g630["node_class"] == 2).sum()),
            "sum_abs_weight": float(np.abs(g630["w_raw_s"]).sum()),
        },
        "whole_brain_v783": {
            "n_nodes": g783["n_nodes"],
            "n_edges": int(len(g783["pre_s"])),
            "n_exc": int((g783["node_class"] == 0).sum()),
            "n_inh": int((g783["node_class"] == 1).sum()),
            "n_sink": int((g783["node_class"] == 2).sum()),
            "sum_abs_weight": float(np.abs(g783["w_raw_s"]).sum()),
        },
        "v783_nt_concordance_vs_v3_2_0": dict(sorted(nt_concordance.items())),
        "sugar_grn_21_v630": outgoing_stats(g630, smb.SUGAR_GRN_FIDS),
        "sugar_grn_21_v783": outgoing_stats(g783, sugar_783_fids),
        "water_grn_18_v630": outgoing_stats(g630, smb.WATER_GRN_FIDS),
        "water_grn_18_v783": outgoing_stats(g783, water_783_fids),
        "sez_target_mapped_368_v630": outgoing_stats(g630, sez_mapped_v630),
        "sez_target_mapped_368_v783": outgoing_stats(g783, sez_mapped_v783),
        "mn9_l_1hop_delta": mn9_edge_delta(idx630[smb.MN9_L_FID], mn9_l_idx783),
        "mn9_r_1hop_delta": mn9_edge_delta(idx630[smb.MN9_R_FID], mn9_r_idx783),
        "target_sign_flips": sign_flips_target,
    }

    # [3/7] Zero-behavior-label W_syn Sugar Anchor Check (Setting i = 0.275 mV & Setting ii = 1D Sugar Anchor)
    print("[3/7] Running v783 sugar 100 Hz anchor check (frozen 0.275 mV + 1D single-point check)...", flush=True)
    slnc_none_783 = np.zeros(g783["n_nodes"], dtype=np.uint8)
    w783_0275 = (g783["w_raw_s"] * np.float32(smb.W_SYN)).astype(np.float64)
    # Warmup JIT
    _ = run_lif_brian2_wsyn(g783["indptr"], g783["post_s"], w783_0275, smb.W_SYN, sugar_idx783[:1], 50.0, slnc_none_783, mn9_watch_783, 1, 0)

    st1d_watch_783 = np.array([idx783[id_map_table[fid]["v783_id"]] for _, fid, _ in smb.ST1D_REF], dtype=np.int32)
    s_cnts, s_occ = run_lif_brian2_wsyn(
        g783["indptr"], g783["post_s"], w783_0275, smb.W_SYN, sugar_idx783, 100.0, slnc_none_783, st1d_watch_783, args.n_trials, 42
    )
    anchor_rows = []
    mn9_z_max = 0.0
    for j, (nm, fid, pub) in enumerate(smb.ST1D_REF):
        m = float(s_cnts[:, j].mean())
        sd = float(s_cnts[:, j].std())
        z = (m - pub) / (sd / math.sqrt(args.n_trials) + 1e-9)
        if nm in ("MN9_l", "MN9_r"):
            mn9_z_max = max(mn9_z_max, abs(z))
        anchor_rows.append(
            {
                "neuron": nm,
                "v630_id": fid,
                "v783_id": id_map_table[fid]["v783_id"],
                "v783_mean_hz": m,
                "v783_sd_hz": sd,
                "pub_st1d_hz": pub,
                "diff_hz": m - pub,
                "z": z,
            }
        )
    st1d_mn9_target = 0.5 * (49.0333333333333 + 68.0)
    mn9_mean_0275 = 0.5 * (anchor_rows[0]["v783_mean_hz"] + anchor_rows[1]["v783_mean_hz"])
    recal_triggered = bool((abs(mn9_mean_0275 - st1d_mn9_target) > 3.0) or (mn9_z_max >= 3.0))
    print(
        f"  [Sugar Anchor 0.275 mV] MN9_mean={mn9_mean_0275:.3f} vs ST1D={st1d_mn9_target:.3f} "
        f"(diff={mn9_mean_0275 - st1d_mn9_target:+.3f} Hz, max|z_MN9|={mn9_z_max:.2f}) -> recal_triggered={recal_triggered}",
        flush=True,
    )

    # Single-point 1D sweep around 0.275 mV (0.265..0.285 step 0.001 mV) using ONLY 100 Hz sugar GRN -> MN9_mean
    sweep_ckpt = args.ckpt_dir / "sugar_1d_sweep.json"
    if sweep_ckpt.exists():
        sweep_rows = json.load(open(sweep_ckpt))["rows"]
    else:
        sweep_grid = [round(0.265 + 0.001 * k, 3) for k in range(21)]
        sweep_rows = []
        for w_cand in sweep_grid:
            w_arr = (g783["w_raw_s"] * np.float32(w_cand)).astype(np.float64)
            c_sw, _ = run_lif_brian2_wsyn(
                g783["indptr"], g783["post_s"], w_arr, w_cand, sugar_idx783, 100.0, slnc_none_783, mn9_watch_783, args.n_trials, 42
            )
            ml = float(c_sw[:, 0].mean())
            mr = float(c_sw[:, 1].mean())
            mm = 0.5 * (ml + mr)
            sweep_rows.append({"w_syn": w_cand, "mn9_l_hz": ml, "mn9_r_hz": mr, "mn9_mean_hz": mm, "abs_err_hz": abs(mm - st1d_mn9_target)})
        _save_json_atomic(sweep_ckpt, {"rows": sweep_rows})
    best_cal = min(sweep_rows, key=lambda r: (r["abs_err_hz"], abs(r["w_syn"] - smb.W_SYN)))
    w_syn_cal = float(best_cal["w_syn"])
    print(
        f"  [Single-point 1D Sugar Anchor] best W_syn_cal={w_syn_cal:.3f} mV "
        f"(MN9_mean={best_cal['mn9_mean_hz']:.3f} Hz, err={best_cal['abs_err_hz']:.3f} Hz)",
        flush=True,
    )

    # [4/7] Real v783 LIF runs (Task 1: 20/50/100/200 Hz + wcal + Task 2: 160/180/200 Hz + wcal) in parallel + Static Baselines
    print(f"[4/7] Running real v783 LIF (Task 1 & Task 2) across {args.workers} workers + static baselines...", flush=True)
    t_real = time.time()
    static_scores_783, is_direct_1hop_783 = smb.compute_static_baselines(
        g783["pre_s"], g783["post_s"], g783["w_raw_s"], g783["n_nodes"], type_indices_783, [mn9_l_idx783, mn9_r_idx783]
    )

    type_idx_lists = [x.tolist() for x in type_indices_783]
    water_sil_lists = [x.tolist() for x in water_sil_indices_783]

    real_sez_specs = [
        ("real_lif_50hz", 50.0, smb.W_SYN, 50000, [(0, 53), (53, 106)]),
        ("real_lif_20hz", 20.0, smb.W_SYN, 20000, [(0, 106)]),
        ("real_lif_100hz", 100.0, smb.W_SYN, 100000, [(0, 35), (35, 70), (70, 106)]),
        ("real_lif_200hz", 200.0, smb.W_SYN, 200000, [(0, 27), (27, 54), (54, 80), (80, 106)]),
    ]
    wcal_key = f"real_lif_50hz_wcal_{w_syn_cal:.3f}"
    if abs(w_syn_cal - smb.W_SYN) > 1e-6:
        real_sez_specs.append((wcal_key, 50.0, w_syn_cal, 50000, [(0, 53), (53, 106)]))

    real_water_specs = [
        ("water_160", 160.0, smb.W_SYN, 70000),
        ("water_180", 180.0, smb.W_SYN, 72000),
        ("water_200", 200.0, smb.W_SYN, 74000),
    ]
    if abs(w_syn_cal - smb.W_SYN) > 1e-6:
        real_water_specs.append(("water_160_wcal", 160.0, w_syn_cal, 70000))

    ctx = mp.get_context("spawn")
    real_sez_slices: dict[str, list[dict]] = {}
    real_water_out: dict[str, dict] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = []
        for label, f_hz, ws, sb, slices in real_sez_specs:
            for si, ei in slices:
                j = {
                    "label": label,
                    "rate_hz": f_hz,
                    "w_syn": ws,
                    "seed_base": sb,
                    "start_i": si,
                    "end_i": ei,
                    "comp_path": str(args.comp_783),
                    "conn_path": str(args.conn_783),
                    "type_indices": type_idx_lists,
                    "mn9_watch": mn9_watch_783.tolist(),
                    "n_trials": args.n_trials,
                    "ckpt_path": str(args.ckpt_dir / f"real_{label}_{si}_{ei}.json"),
                }
                futs.append(ex.submit(_real_sez_slice_worker, j))
        for label, f_hz, ws, sb in real_water_specs:
            j = {
                "label": label,
                "rate_hz": f_hz,
                "w_syn": ws,
                "seed_base": sb,
                "comp_path": str(args.comp_783),
                "conn_path": str(args.conn_783),
                "water_idx": water_idx783.tolist(),
                "water_sil_indices": water_sil_lists,
                "mn9_watch": mn9_watch_783.tolist(),
                "n_trials": args.n_trials,
                "ckpt_path": str(args.ckpt_dir / f"real_{label}.json"),
            }
            futs.append(ex.submit(_real_water_worker, j))
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            if res["kind"] == "sez_slice":
                real_sez_slices.setdefault(res["label"], []).append(res)
                print(
                    f"  [real {res['label']:24s} {res['start_i']:3d}:{res['end_i']:3d}] done ({res['elapsed_sec']:.1f}s)",
                    flush=True,
                )
            else:
                real_water_out[res["label"]] = res["result"]
                print(
                    f"  [real {res['label']:24s}] base MN9_r={res['result']['base_mn9_r']:.2f}Hz ({res['elapsed_sec']:.1f}s)",
                    flush=True,
                )

    def stitch_sez(label: str) -> tuple[np.ndarray, np.ndarray, int]:
        parts = sorted(real_sez_slices[label], key=lambda x: x["start_i"])
        m = np.vstack([np.array(p["means"], dtype=np.float64) for p in parts])
        sd = np.vstack([np.array(p["stds"], dtype=np.float64) for p in parts])
        occ = max(int(p["max_occ"]) for p in parts)
        assert m.shape == (106, 2)
        return m, sd, occ

    real_50_m, real_50_sd, real_50_occ = stitch_sez("real_lif_50hz")
    real_50_score = real_50_m.mean(axis=1)
    real_freq_scores = {"real_lif_50hz": real_50_score}
    for f_hz in (20, 100, 200):
        m_f, _, _ = stitch_sez(f"real_lif_{f_hz}hz")
        real_freq_scores[f"real_lif_{f_hz}hz"] = m_f.mean(axis=1)

    cal_real_50_score = None
    cal_water_160 = None
    if abs(w_syn_cal - smb.W_SYN) > 1e-6:
        cal_m, _, _ = stitch_sez(wcal_key)
        cal_real_50_score = cal_m.mean(axis=1)
        real_freq_scores[wcal_key] = cal_real_50_score
        cal_water_160 = real_water_out["water_160_wcal"]

    water_real_160 = real_water_out["water_160"]
    water_real_180 = real_water_out["water_180"]
    water_real_200 = real_water_out["water_200"]
    print(
        f"  Real v783 LIF done in {time.time() - t_real:.1f}s | Water 160Hz base MN9_r={water_real_160['base_mn9_r']:.2f} Hz, "
        f"MN9_l={water_real_160['base_mn9_l']:.2f} Hz",
        flush=True,
    )

    # [5/7] Module A descriptive frozen readouts (SELK/DNg68 [novel out-of-106], Fdg [overlap], Amulet/CB0626 [reference])
    print("[5/7] Running Module A descriptive frozen readouts on v630 & v783...", flush=True)
    slnc_none_630 = np.zeros(g630["n_nodes"], dtype=np.uint8)
    w630_0275 = (g630["w_raw_s"] * np.float32(smb.W_SYN)).astype(np.float64)
    mn9_watch_630 = np.array([idx630[smb.MN9_L_FID], idx630[smb.MN9_R_FID]], dtype=np.int32)

    mod_a_specs = [
        ("SELK_DNg68", "layer_A2_novel_out_of_106", SELK_V630_FIDS, SELK_V783_FIDS),
        ("Fdg_CB0038", "layer_A1_overlap_cross_lab", FDG_FIDS, [id_map_table[x]["v783_id"] for x in FDG_FIDS]),
        ("Amulet_CB0626_ref", "reference_downstream_of_SELK", AMULET_FIDS, [id_map_table[x]["v783_id"] for x in AMULET_FIDS]),
    ]
    module_a_readouts = {}
    for label, layer, fids6, fids7 in mod_a_specs:
        i6 = np.array([idx630[x] for x in fids6], dtype=np.int32)
        i7 = np.array([idx783[x] for x in fids7], dtype=np.int32)
        r6, r7 = {}, {}
        for f_hz, sb in [(20.0, 21000), (50.0, 51000), (100.0, 101000), (200.0, 201000)]:
            c6, _ = run_lif_brian2_wsyn(g630["indptr"], g630["post_s"], w630_0275, smb.W_SYN, i6, f_hz, slnc_none_630, mn9_watch_630, args.n_trials, sb)
            c7, _ = run_lif_brian2_wsyn(g783["indptr"], g783["post_s"], w783_0275, smb.W_SYN, i7, f_hz, slnc_none_783, mn9_watch_783, args.n_trials, sb)
            r6[f"{int(f_hz)}hz"] = {"mn9_l": float(c6[:, 0].mean()), "mn9_r": float(c6[:, 1].mean()), "mn9_mean": float(c6.mean())}
            r7[f"{int(f_hz)}hz"] = {"mn9_l": float(c7[:, 0].mean()), "mn9_r": float(c7[:, 1].mean()), "mn9_mean": float(c7.mean())}
        st6 = compute_single_group_static(g630["pre_s"], g630["post_s"], g630["w_raw_s"], g630["n_nodes"], i6, [int(mn9_watch_630[0]), int(mn9_watch_630[1])])
        st7 = compute_single_group_static(g783["pre_s"], g783["post_s"], g783["w_raw_s"], g783["n_nodes"], i7, [mn9_l_idx783, mn9_r_idx783])
        module_a_readouts[label] = {
            "layer": layer,
            "v630_ids": fids6,
            "v783_ids": fids7,
            "v630_lif": r6,
            "v783_lif": r7,
            "v630_static": st6,
            "v783_static": st7,
        }
        print(
            f"  [Module A {label}] v630 50Hz={r6['50hz']['mn9_mean']:.2f} Hz (200Hz={r6['200hz']['mn9_mean']:.2f}, 2hopReLU={st6['static_2hop_relu_signed']:.1f}) | "
            f"v783 50Hz={r7['50hz']['mn9_mean']:.2f} Hz (200Hz={r7['200hz']['mn9_mean']:.2f}, 2hopReLU={st7['static_2hop_relu_signed']:.1f})",
            flush=True,
        )

    # [6/7] K=10 rewire_full & K=10 rewire_mn9_protected on v783 (spawn-safe parallel workers)
    print(f"[6/7] Running {2 * args.n_rewires} v783 rewired null graphs across {args.workers} workers...", flush=True)
    t_rw = time.time()
    jobs = []
    type_idx_lists = [x.tolist() for x in type_indices_783]
    water_sil_lists = [x.tolist() for x in water_sil_indices_783]
    w_alt_for_nulls = w_syn_cal if recal_triggered else None

    for k in range(args.n_rewires):
        jobs.append(
            {
                "family": "rewire_full",
                "idx": k,
                "seed": 9101 + k,
                "protect_targets": [],
                "comp_path": str(args.comp_783),
                "conn_path": str(args.conn_783),
                "w_syn": smb.W_SYN,
                "w_syn_alt": w_alt_for_nulls,
                "type_indices": type_idx_lists,
                "water_idx": water_idx783.tolist(),
                "water_sil_indices": water_sil_lists,
                "mn9_watch": mn9_watch_783.tolist(),
                "n_trials": args.n_trials,
                "ckpt_path": str(args.ckpt_dir / f"rewire_full_{9101 + k}.json"),
            }
        )
    for k in range(args.n_rewires):
        jobs.append(
            {
                "family": "rewire_mn9_prot",
                "idx": k,
                "seed": 9201 + k,
                "protect_targets": [int(mn9_l_idx783), int(mn9_r_idx783)],
                "comp_path": str(args.comp_783),
                "conn_path": str(args.conn_783),
                "w_syn": smb.W_SYN,
                "w_syn_alt": w_alt_for_nulls,
                "type_indices": type_idx_lists,
                "water_idx": water_idx783.tolist(),
                "water_sil_indices": water_sil_lists,
                "mn9_watch": mn9_watch_783.tolist(),
                "n_trials": args.n_trials,
                "ckpt_path": str(args.ckpt_dir / f"rewire_mn9_prot_{9201 + k}.json"),
            }
        )

    results_rw = []
    ctx = mp.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futs = [ex.submit(_rewire_worker, j) for j in jobs]
        for fut in concurrent.futures.as_completed(futs):
            res = fut.result()
            results_rw.append(res)
            nz = sum(1 for x in res["scores_50hz"] if x > 0)
            print(
                f"  [{res['family']:15s} {res['idx']+1:2d}/{args.n_rewires}] seed={res['seed']} "
                f"replaced={res['meta']['replaced_fraction_movable']:.4f} prot={res['meta']['n_protected']} "
                f"nz_types={nz} water_base_r={res['water_160hz']['base_mn9_r']:.2f}Hz ({res['elapsed_sec']:.1f}s)",
                flush=True,
            )

    full_res = sorted([r for r in results_rw if r["family"] == "rewire_full"], key=lambda x: x["idx"])
    prot_res = sorted([r for r in results_rw if r["family"] == "rewire_mn9_prot"], key=lambda x: x["idx"])
    print(f"  All {2 * args.n_rewires} v783 rewired graphs completed in {time.time() - t_rw:.1f}s", flush=True)

    # [7/7] Comprehensive Evaluation & v630 vs v783 Comparison
    print("[7/7] Evaluating subsets, bootstrap, CV, Task 2 water silencing, and v630 vs v783 delta...", flush=True)
    y_max = np.array([rec["y_max"] for rec in st3_records], dtype=np.float64)
    y_mean = np.array([rec["y_mean"] for rec in st3_records], dtype=np.float64)
    y_bin = (y_max > 0.0).astype(np.int32)
    y_bin_ge02 = (y_max >= 0.2).astype(np.int32)
    novel_mask = np.array([not rec["prior_published"] for rec in st3_records], dtype=bool)
    nondirect_v783_mask = ~is_direct_1hop_783
    is_direct_1hop_630 = np.array([bool(row["is_direct_1hop_mn9"]) for row in v630_prev["per_type_task1"]], dtype=bool)
    nondirect_common_mask = (~is_direct_1hop_630) & (~is_direct_1hop_783)

    rewire_full_arr = np.array([r["scores_50hz"] for r in full_res], dtype=np.float64)
    rewire_prot_arr = np.array([r["scores_50hz"] for r in prot_res], dtype=np.float64)

    all_model_scores = {
        **real_freq_scores,
        "rewire_full_ensemble_mean": rewire_full_arr.mean(axis=0),
        "rewire_mn9_prot_ensemble_mean": rewire_prot_arr.mean(axis=0),
        **static_scores_783,
    }

    def eval_subset(mask: np.ndarray, label_bin: np.ndarray, subset_name: str) -> dict:
        ymx = y_max[mask]
        ymn = y_mean[mask]
        yb = label_bin[mask]
        sc_sub = {m: sc[mask] for m, sc in all_model_scores.items()}
        boot = smb.stratified_bootstrap_summary(yb, ymx, ymn, sc_sub, "real_lif_50hz", n_boot=2000, seed=20261123)
        cv_res = {m: smb.evaluate_heldout_cv(yb, sc_sub[m], seed=20261123) for m in sc_sub}

        def null_family_summary(arr: np.ndarray) -> dict:
            auprcs = [smb.average_precision_exact(yb, arr[k][mask]) for k in range(len(arr))]
            aurocs = [smb.auroc_exact(yb, arr[k][mask]) for k in range(len(arr))]
            rhos = [smb.spearman_safe(ymx, arr[k][mask]) for k in range(len(arr))]
            f1_loos = [smb.evaluate_heldout_cv(yb, arr[k][mask], seed=20261123)["loocv"]["f1"] for k in range(len(arr))]
            f1_z = [smb.binary_metrics(yb, (arr[k][mask] > 0.0).astype(np.int32))["f1"] for k in range(len(arr))]
            real_ap = boot["real_lif_50hz"]["point"]["auprc"]
            real_roc = boot["real_lif_50hz"]["point"]["auroc"]
            real_rho = boot["real_lif_50hz"]["point"]["rho_max"]
            return {
                "auprc_per_graph": auprcs,
                "auprc_mean_sd_min_max": [float(np.mean(auprcs)), float(np.std(auprcs)), float(np.min(auprcs)), float(np.max(auprcs))],
                "auroc_per_graph": aurocs,
                "auroc_mean_sd_min_max": [float(np.mean(aurocs)), float(np.std(aurocs)), float(np.min(aurocs)), float(np.max(aurocs))],
                "rho_max_per_graph": rhos,
                "rho_max_mean_sd_min_max": [float(np.mean(rhos)), float(np.std(rhos)), float(np.min(rhos)), float(np.max(rhos))],
                "loocv_f1_per_graph": f1_loos,
                "loocv_f1_mean_min_max": [float(np.mean(f1_loos)), float(np.min(f1_loos)), float(np.max(f1_loos))],
                "zero_thr_f1_per_graph": f1_z,
                "real_exceeds_count_auprc": int(sum(real_ap > x for x in auprcs)),
                "real_exceeds_count_auroc": int(sum(real_roc > x for x in aurocs)),
                "real_exceeds_count_rho": int(sum(real_rho > x for x in rhos)),
                "empirical_upper_p_auprc": float((1 + sum(x >= real_ap for x in auprcs)) / (1 + len(auprcs))),
                "empirical_upper_p_auroc": float((1 + sum(x >= real_roc for x in aurocs)) / (1 + len(aurocs))),
            }

        return {
            "subset": subset_name,
            "n_total": int(mask.sum()),
            "n_pos": int(yb.sum()),
            "n_neg": int((yb == 0).sum()),
            "chance_auprc": float(yb.mean()),
            "bootstrap": boot,
            "cv": cv_res,
            "rewire_full_graphs": null_family_summary(rewire_full_arr),
            "rewire_mn9_prot_graphs": null_family_summary(rewire_prot_arr),
        }

    task1_eval = {
        "all_106": eval_subset(np.ones(106, dtype=bool), y_bin, "all_106"),
        "complete_mapped_105": eval_subset(type_complete_mapped, y_bin, "complete_mapped_105"),
        "all_106_per_ge_02": eval_subset(np.ones(106, dtype=bool), y_bin_ge02, "all_106_per_ge_02"),
        "novel_95": eval_subset(novel_mask, y_bin, "novel_95"),
        "nondirect_v783": eval_subset(nondirect_v783_mask, y_bin, "nondirect_v783"),
        "nondirect_common": eval_subset(nondirect_common_mask, y_bin, "nondirect_common"),
        "novel_and_nondirect_v783": eval_subset(novel_mask & nondirect_v783_mask, y_bin, "novel_and_nondirect_v783"),
    }

    # v630 vs v783 direct per-type comparison
    v630_50_scores = np.array([row["real_lif_50hz_mean"] for row in v630_prev["per_type_task1"]], dtype=np.float64)
    v630_v783_diff = real_50_score - v630_50_scores
    zero_flips = []
    direct_flips = []
    for i, rec in enumerate(st3_records):
        if (v630_50_scores[i] > 0.0) != (real_50_score[i] > 0.0):
            zero_flips.append(
                {
                    "name": rec["name"],
                    "y_max": rec["y_max"],
                    "v630_50hz": float(v630_50_scores[i]),
                    "v783_50hz": float(real_50_score[i]),
                }
            )
        if bool(is_direct_1hop_630[i]) != bool(is_direct_1hop_783[i]):
            direct_flips.append(
                {
                    "name": rec["name"],
                    "y_max": rec["y_max"],
                    "v630_direct": bool(is_direct_1hop_630[i]),
                    "v783_direct": bool(is_direct_1hop_783[i]),
                }
            )

    v630_vs_v783_comp = {
        "pearson_r_50hz": float(stats.pearsonr(v630_50_scores, real_50_score).statistic),
        "spearman_rho_50hz": float(stats.spearmanr(v630_50_scores, real_50_score).statistic),
        "mae_hz": float(np.mean(np.abs(v630_v783_diff))),
        "max_abs_diff_hz": float(np.max(np.abs(v630_v783_diff))),
        "max_abs_diff_type": st3_records[int(np.argmax(np.abs(v630_v783_diff)))]["name"],
        "zero_threshold_flips": zero_flips,
        "direct_1hop_status_flips": direct_flips,
    }

    # Task 2 water silencing evaluation
    w_did = np.array([rec["delta_did"] for rec in st9b_records], dtype=np.float64)
    w_exp = np.array([rec["delta_exp"] for rec in st9b_records], dtype=np.float64)
    w_sig = np.array([1 if rec["sig_impaired"] else 0 for rec in st9b_records], dtype=np.int32)
    no_usnea = np.array([rec["name"] != "Usnea" for rec in st9b_records], dtype=bool)

    def eval_water_vec(rel_change: list[float]) -> dict:
        v = np.asarray(rel_change, dtype=np.float64)
        imp = -v[w_sig == 1]
        non = -v[w_sig == 0]
        mw = stats.mannwhitneyu(imp, non, alternative="two-sided")
        sp_did = stats.spearmanr(w_did, v)
        sp_no_usnea = stats.spearmanr(w_did[no_usnea], v[no_usnea])
        return {
            "spearman_with_did": 0.0 if np.isnan(sp_did.statistic) else float(sp_did.statistic),
            "spearman_with_did_p": 1.0 if np.isnan(sp_did.pvalue) else float(sp_did.pvalue),
            "spearman_with_exp": smb.spearman_safe(w_exp, v),
            "auroc_impaired": smb.auroc_exact(w_sig, -v),
            "auprc_impaired": smb.average_precision_exact(w_sig, -v),
            "mannwhitney_u": float(mw.statistic),
            "mannwhitney_p_two_sided": float(mw.pvalue),
            "diag_excl_usnea_spearman_did": 0.0 if np.isnan(sp_no_usnea.statistic) else float(sp_no_usnea.statistic),
            "diag_excl_usnea_spearman_p": 1.0 if np.isnan(sp_no_usnea.pvalue) else float(sp_no_usnea.pvalue),
            "diag_excl_usnea_auroc": smb.auroc_exact(w_sig[no_usnea], -v[no_usnea]),
        }

    task2_eval = {
        "n_types": len(st9b_records),
        "n_sig_impaired": int(w_sig.sum()),
        "real_160hz_mn9r": {**water_real_160, "metrics": eval_water_vec(water_real_160["rel_change_mn9_r"])},
        "real_160hz_mn9mean": {"metrics": eval_water_vec(water_real_160["rel_change_mn9_mean"])},
        "real_180hz_mn9r": {**water_real_180, "metrics": eval_water_vec(water_real_180["rel_change_mn9_r"])},
        "real_200hz_mn9r": {**water_real_200, "metrics": eval_water_vec(water_real_200["rel_change_mn9_r"])},
        "real_160hz_wcal_mn9r": (
            {**cal_water_160, "metrics": eval_water_vec(cal_water_160["rel_change_mn9_r"])}
            if cal_water_160 is not None
            else None
        ),
        "rewire_full_water": [
            {
                "seed": r["seed"],
                "base_mn9_l": r["water_160hz"]["base_mn9_l"],
                "base_mn9_r": r["water_160hz"]["base_mn9_r"],
                "drive_failed_zero_base": bool(r["water_160hz"]["base_mn9_r"] <= 0.0),
                "metrics": eval_water_vec(r["water_160hz"]["rel_change_mn9_r"]) if r["water_160hz"]["base_mn9_r"] > 0 else None,
            }
            for r in full_res
        ],
        "rewire_mn9_prot_water": [
            {
                "seed": r["seed"],
                "base_mn9_l": r["water_160hz"]["base_mn9_l"],
                "base_mn9_r": r["water_160hz"]["base_mn9_r"],
                "drive_failed_zero_base": bool(r["water_160hz"]["base_mn9_r"] <= 0.0),
                "metrics": eval_water_vec(r["water_160hz"]["rel_change_mn9_r"]) if r["water_160hz"]["base_mn9_r"] > 0 else None,
            }
            for r in prot_res
        ],
        "per_type_records": [
            {
                **rec,
                "v630_160hz_rel_change_mn9_r": v630_prev["task2_water_silencing"]["per_type_records"][i]["sim_160hz_rel_change_mn9_r"],
                "v783_160hz_mn9_r": water_real_160["sil_mn9_r"][i],
                "v783_160hz_rel_change_mn9_r": water_real_160["rel_change_mn9_r"][i],
                "v783_160hz_rel_change_mn9_mean": water_real_160["rel_change_mn9_mean"][i],
            }
            for i, rec in enumerate(st9b_records)
        ],
    }

    per_type_task1 = []
    for i, rec in enumerate(st3_records):
        per_type_task1.append(
            {
                "name": rec["name"],
                "n_v630_fids": type_mapping_audit[i]["n_v630"],
                "n_v783_fids": type_mapping_audit[i]["n_v783"],
                "n_tier1": type_mapping_audit[i]["n_tier1"],
                "n_tier2": type_mapping_audit[i]["n_tier2"],
                "n_tier3": type_mapping_audit[i]["n_tier3"],
                "y_max": rec["y_max"],
                "y_mean": rec["y_mean"],
                "prior_published": rec["prior_published"],
                "is_direct_1hop_v630": bool(is_direct_1hop_630[i]),
                "is_direct_1hop_v783": bool(is_direct_1hop_783[i]),
                "v630_real_lif_50hz_mean": float(v630_50_scores[i]),
                "v783_real_lif_50hz_mean": float(real_50_score[i]),
                "v783_real_lif_50hz_l": float(real_50_m[i, 0]),
                "v783_real_lif_50hz_r": float(real_50_m[i, 1]),
                "v783_real_lif_20hz_mean": float(real_freq_scores["real_lif_20hz"][i]),
                "v783_real_lif_100hz_mean": float(real_freq_scores["real_lif_100hz"][i]),
                "v783_real_lif_200hz_mean": float(real_freq_scores["real_lif_200hz"][i]),
                "v783_real_lif_50hz_wcal": float(cal_real_50_score[i]) if cal_real_50_score is not None else float(real_50_score[i]),
                "v783_rewire_full_mean": float(rewire_full_arr[:, i].mean()),
                "v783_rewire_mn9_prot_mean": float(rewire_prot_arr[:, i].mean()),
                "v783_static_1hop_signed": float(static_scores_783["static_1hop_signed"][i]),
                "v783_static_2hop_signed_fixed": float(static_scores_783["static_2hop_signed_fixed"][i]),
                "v783_static_2hop_relu_signed": float(static_scores_783["static_2hop_relu_signed"][i]),
                "v783_static_2hop_unsigned_fixed": float(static_scores_783["static_2hop_unsigned_fixed"][i]),
                "v783_static_neg_shortest_path": float(static_scores_783["static_neg_shortest_path"][i]),
            }
        )

    tier_summary = Counter(rec["tier"] for rec in id_map_table.values())
    out_payload = {
        "protocol_commit": "2a4a233dd127a9bd3ab88fa32b37b310bd4c2bf3",
        "input_hashes": hashes,
        "mapping_audit": {
            "n_unique_target_v630_ids": len(id_map_table),
            "tier_counts_unique_ids": dict(tier_summary),
            "mn9_l": id_map_table[smb.MN9_L_FID],
            "mn9_r": id_map_table[smb.MN9_R_FID],
            "sugar_grn_tier_counts": dict(Counter(id_map_table[x]["tier"] for x in smb.SUGAR_GRN_FIDS)),
            "water_grn_tier_counts": dict(Counter(id_map_table[x]["tier"] for x in smb.WATER_GRN_FIDS)),
            "zorro_tier_counts": dict(Counter(id_map_table[x]["tier"] for x in smb.ZORRO_FIDS)),
            "sez106_types_with_any_mapped": int(sum(1 for t in type_mapping_audit if t["n_v783"] > 0)),
            "sez106_types_100pct_mapped": int(type_complete_mapped.sum()),
            "water11_types_100pct_mapped": int(sum(1 for t in water_mapping_audit if t["n_tier3"] == 0)),
            "collisions": collisions,
            "per_type_sez106": type_mapping_audit,
            "per_type_water11": water_mapping_audit,
        },
        "graph_delta": graph_delta,
        "sugar_anchor_calibration": {
            "frozen_w_syn_mv": smb.W_SYN,
            "st1d_mn9_mean_target_hz": st1d_mn9_target,
            "v783_frozen_mn9_mean_hz": mn9_mean_0275,
            "diff_from_target_hz": mn9_mean_0275 - st1d_mn9_target,
            "max_abs_z_mn9": mn9_z_max,
            "recalibration_triggered": recal_triggered,
            "anchor_rows_0275": anchor_rows,
            "max_ring_occ_0275": int(s_occ),
            "single_point_1d_sweep": sweep_rows,
            "best_single_point_w_syn_mv": w_syn_cal,
            "best_single_point_row": best_cal,
        },
        "module_a_descriptive_readouts": module_a_readouts,
        "v630_vs_v783_task1_comparison": v630_vs_v783_comp,
        "rewire_full_meta": [r["meta"] for r in full_res],
        "rewire_mn9_prot_meta": [r["meta"] for r in prot_res],
        "task1_activation_v783": task1_eval,
        "task2_water_silencing_v783": task2_eval,
        "per_type_task1": per_type_task1,
        "wall_time_sec": float(time.time() - t_start),
    }

    with open(args.output, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved v783 transfer results to {args.output} in {out_payload['wall_time_sec']:.1f}s", flush=True)

    # Compact summary
    print("\n=== v783 TASK 1 SUMMARY (ALL 106 SEZ TYPES) ===")
    s106 = task1_eval["all_106"]
    for m in all_model_scores:
        pt = s106["bootstrap"][m]["point"]
        cv = s106["cv"][m]["loocv"]
        zt = s106["cv"][m]["zero_threshold"]
        print(
            f"  {m:30s} | AUPRC={pt['auprc']:.4f} | AUROC={pt['auroc']:.4f} | rho={pt['rho_max']:.4f} | "
            f"LOOCV_F1={cv['f1']:.4f} (MCC={cv['mcc']:.4f}) | >0_F1={zt['f1']:.4f}"
        )
    for nf_name in ("rewire_full_graphs", "rewire_mn9_prot_graphs"):
        nf = s106[nf_name]
        print(
            f"  {nf_name:30s} | AUPRC mean={nf['auprc_mean_sd_min_max'][0]:.4f} [{nf['auprc_mean_sd_min_max'][2]:.4f},{nf['auprc_mean_sd_min_max'][3]:.4f}] | "
            f"real>null={nf['real_exceeds_count_auprc']}/{args.n_rewires} (p={nf['empirical_upper_p_auprc']:.4f})"
        )
    nd = task1_eval["nondirect_v783"]
    print(f"\n=== v783 NON-DIRECT SUBSET (N={nd['n_total']}, pos={nd['n_pos']}) ===")
    for m in ("real_lif_50hz", "static_2hop_relu_signed", "rewire_mn9_prot_ensemble_mean"):
        pt = nd["bootstrap"][m]["point"]
        print(f"  {m:30s} | AUPRC={pt['auprc']:.4f} | AUROC={pt['auroc']:.4f} | rho={pt['rho_max']:.4f}")
    print(
        f"  rewire_mn9_prot per-graph      | AUPRC mean={nd['rewire_mn9_prot_graphs']['auprc_mean_sd_min_max'][0]:.4f} | "
        f"real>null={nd['rewire_mn9_prot_graphs']['real_exceeds_count_auprc']}/{args.n_rewires}"
    )

    print("\n=== v783 TASK 2 WATER SILENCING (11/11) ===")
    m2 = task2_eval["real_160hz_mn9r"]["metrics"]
    print(
        f"  real_160hz_mn9r: base={water_real_160['base_mn9_r']:.2f} Hz | AUROC={m2['auroc_impaired']:.4f} "
        f"(U={m2['mannwhitney_u']:.1f}, p={m2['mannwhitney_p_two_sided']:.4f}) | AUPRC={m2['auprc_impaired']:.4f} | "
        f"rho_did={m2['spearman_with_did']:.4f} (p={m2['spearman_with_did_p']:.4f})"
    )
    print(
        f"  rewire_full max base_mn9_r={max(r['base_mn9_r'] for r in task2_eval['rewire_full_water']):.3f} Hz | "
        f"rewire_mn9_prot max base_mn9_r={max(r['base_mn9_r'] for r in task2_eval['rewire_mn9_prot_water']):.3f} Hz"
    )


if __name__ == "__main__":
    main()
