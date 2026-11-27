#!/usr/bin/env python3
"""V5 Reviewer Controls (Opus Major Revision Package — Full-Speed Local CPU).

Never modifies the 6 locked benchmark JSONs or mn9_pre_submission_ab_results.json.
Writes all outputs atomically to `mn9_v5_reviewer_controls_results.json`.

Controls implemented:
  Phase 1 (BANC v888 — Minor #6 + M4 + BANC reference table for M2c):
    - Cell-type-level homolog per-neuron normalization (fafb_cell_type / cell_type -> v783)
      raising coverage from 34.7% (reviewed 1:1) to full cell-type coverage (~75-85%).
    - Fair two-hop baselines on BANC (raw, 1:1 weighted, cell-type weighted; signed & unsigned;
      alpha sweep 0.001..1.0 + pure h2).
  Phase 2 (MaleCNS v1.0 — M4 Fair Two-Hop + M2 Anti-Circularity + Minor #1 Poisson check):
    - M4: Fair two-hop baselines on MaleCNS (consensus_nt raw, predicted_nt raw,
      consensus_nt + homolog weights, predicted_nt + homolog weights; signed ReLU & unsigned;
      alpha sweep 0.001..1.0 + pure h2) + 100k paired bootstrap vs Hom-Norm+Pred-NT LIF.
    - M2a: Shuffled-homolog permutation controls (5 seeds preserving MN9 scale, 3 seeds shuffling MN9 too).
    - M2b: Leave-out benchmark neurons (leave_out_sez106, leave_out_sez106_and_mn9, and
      two-scale control mn9_anchored_rest_density).
    - M2c: Cross-reference normalization (BANC -> MaleCNS, zero FlyWire S_in).
    - Minor #1: Verify stimulated-neuron firing rate invariance across W_syn = 0.157..0.690 mV.
  Phase 3 (FlyWire v783 — M5 K=100 Fixed-Weight Rewired Nulls + M4 v783 alpha sweep + M3/M6 stats):
    - K=100 global MN9-protected fixed-weight (0.275 mV) rewired graphs (empirical p resolution 1/101 < 0.01).
    - K=100 super-class-preserving MN9-protected fixed-weight (0.275 mV) rewired graphs (1/101 < 0.01).
    - FlyWire v783 two-hop alpha sweep (0.001..1.0 + pure h2).
    - M3 label-free calibration selection table + unified BH-FDR family + M6 AUROC & Top-k metrics.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pyarrow.feather as pf
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("PROJC_CACHE_DIR", str(Path.home() / ".cache" / "projc-flywire")))
OUT_JSON = BASE_DIR / "mn9_v5_reviewer_controls_results.json"

sys.path.insert(0, str(BASE_DIR))
import mn9_pre_submission_ab as ab  # noqa: E402
import mn9_portability_hardening as mph  # noqa: E402
import shiu_mn9_benchmark as smb  # noqa: E402


def save_ckpt(payload: dict) -> None:
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT_JSON)


def compute_topk_hits(y: np.ndarray, scores: np.ndarray, ks=(10, 14, 15, 20)) -> dict[str, float]:
    """Expected positive hits in top-k accounting for exact ties conservatively (mid-rank) & strictly."""
    out = {}
    order = np.argsort(-scores, kind="mergesort")
    y_s = y[order]
    s_s = scores[order]
    n = len(y)
    for k in ks:
        # Handle ties spanning boundary k cleanly using expected hypergeometric draw on tied block
        if k >= n:
            out[f"top{k}"] = float(y.sum())
            continue
        thr = s_s[k - 1]
        above = scores > thr
        eq = scores == thr
        n_above = int(above.sum())
        pos_above = int(y[above].sum())
        rem = k - n_above
        n_eq = int(eq.sum())
        pos_eq = int(y[eq].sum())
        exp_hits = pos_above + (rem * pos_eq / float(n_eq) if n_eq > 0 else 0.0)
        out[f"top{k}"] = float(exp_hits)
    return out


def compute_twohop_family(
    indptr: np.ndarray,
    indices: np.ndarray,
    w_edge: np.ndarray,
    n_nodes: int,
    mn9_idx: np.ndarray,
    sez_exc: list[np.ndarray],
    y: np.ndarray,
    prior_pub: np.ndarray,
    is_direct: np.ndarray,
    alphas=(0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0),
) -> dict:
    """Compute signed ReLU and unsigned 1-hop/2-hop static baselines for arbitrary per-edge weights."""
    # Direct incoming signed & unsigned weights onto bilateral MN9
    w1_s = np.zeros(n_nodes, dtype=np.float64)
    w1_u = np.zeros(n_nodes, dtype=np.float64)
    m0, m1 = int(mn9_idx[0]), int(mn9_idx[1])
    for u in range(n_nodes):
        p0, p1 = int(indptr[u]), int(indptr[u + 1])
        if p0 == p1:
            continue
        dst = indices[p0:p1]
        w = w_edge[p0:p1]
        mm = (dst == m0) | (dst == m1)
        if np.any(mm):
            w1_s[u] = float(np.sum(w[mm]))
            w1_u[u] = float(np.sum(np.abs(w[mm])))

    n_types = len(sez_exc)
    h1_s = np.zeros(n_types, dtype=np.float64)
    h2_s = np.zeros(n_types, dtype=np.float64)
    h1_u = np.zeros(n_types, dtype=np.float64)
    h2_u = np.zeros(n_types, dtype=np.float64)

    for i, idxs in enumerate(sez_exc):
        if len(idxs) == 0:
            continue
        h1_s[i] = float(np.sum(w1_s[idxs]))
        h1_u[i] = float(np.sum(w1_u[idxs]))
        # Sum outgoing weights onto intermediate nodes k
        o_s: dict[int, float] = defaultdict(float)
        o_u: dict[int, float] = defaultdict(float)
        for u in idxs:
            p0, p1 = int(indptr[u]), int(indptr[u + 1])
            for e in range(p0, p1):
                dst = int(indices[e])
                wv = float(w_edge[e])
                o_s[dst] += wv
                o_u[dst] += abs(wv)
        s2 = 0.0
        u2 = 0.0
        for k, val in o_s.items():
            if val > 0.0 and w1_s[k] != 0.0:
                s2 += val * w1_s[k]
        for k, val in o_u.items():
            if val > 0.0 and w1_u[k] != 0.0:
                u2 += val * w1_u[k]
        h2_s[i] = s2
        h2_u[i] = u2

    unpub = (prior_pub == 0)
    nd = ~is_direct

    def eval_vec(sc: np.ndarray) -> dict:
        return {
            "all106_auprc": ab.stepwise_auprc(y, sc),
            "all106_auroc": ab.compute_auroc(y, sc),
            "unpub95_auprc": ab.stepwise_auprc(y[unpub], sc[unpub]),
            "nondirect_auprc": ab.stepwise_auprc(y[nd], sc[nd]),
            "topk_hits": compute_topk_hits(y, sc),
        }

    res = {"signed_relu": {}, "unsigned": {}}
    for a in alphas:
        key = f"alpha_{a:g}"
        res["signed_relu"][key] = eval_vec(h1_s + a * h2_s)
        res["unsigned"][key] = eval_vec(h1_u + a * h2_u)
    res["signed_relu"]["pure_h2"] = eval_vec(h2_s)
    res["unsigned"]["pure_h2"] = eval_vec(h2_u)
    res["signed_relu_scores_0p01"] = (h1_s + 0.01 * h2_s).tolist()
    res["unsigned_scores_0p01"] = (h1_u + 0.01 * h2_u).tolist()
    return res


def paired_boot_100k(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, B: int = 100000, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]
    diffs = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos_i, len(pos_i), True), rng.choice(neg_i, len(neg_i), True)])
        yb = y[idx]
        diffs[b] = ab.stepwise_auprc(yb, s1[idx]) - ab.stepwise_auprc(yb, s2[idx])
    pt = ab.stepwise_auprc(y, s1) - ab.stepwise_auprc(y, s2)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p2 = min(1.0, float(2.0 * min(np.mean(diffs <= 0.0), np.mean(diffs >= 0.0))))
    return {"delta": float(pt), "ci95": [float(lo), float(hi)], "p_two_sided": p2}


def strat_ci95(y: np.ndarray, s: np.ndarray, B: int = 2000, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]
    vals = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = np.concatenate([rng.choice(pos_i, len(pos_i), True), rng.choice(neg_i, len(neg_i), True)])
        vals[b] = ab.stepwise_auprc(y[idx], s[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def bh_fdr(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    q = np.empty(m, dtype=np.float64)
    prev = 1.0
    for rank_rev in range(m - 1, -1, -1):
        idx = order[rank_rev]
        val = min(prev, pvals[idx] * m / float(rank_rev + 1))
        q[idx] = val
        prev = val
    return [float(x) for x in q]


# ---------------------------------------------------------------------------
# Phase 1: BANC Cell-Type Homolog Normalization (Minor #6) + Fair Two-Hop + BANC ref
# ---------------------------------------------------------------------------
def run_phase1_banc(meta: dict, v783_ct_mean_sin: dict[str, float], s_in_v783_by_fid: dict[int, float], workers: int) -> tuple[dict, dict[str, float]]:
    t0 = time.time()
    print("\n================ PHASE 1: BANC Cell-Type Homolog Norm (Minor #6) & Fair Two-Hop ================", flush=True)
    bdir = CACHE_DIR / "data/banc"
    bm = pf.read_table(
        bdir / "banc_888_meta.feather",
        columns=[
            "banc_888_id", "root_id", "supervoxel_id", "root_626", "root_850", "root_888", "root_890",
            "super_class", "cell_type", "fafb_cell_type", "fafb_alignment_cell_type", "side",
            "proofread", "roughly_proofread",
        ],
    ).to_pydict()

    sv_to_bid, old_root_to_bid = {}, {}
    main_cand_set = set()
    bid_to_cts: dict[int, set[str]] = {}
    for i in range(len(bm["banc_888_id"])):
        if bm["banc_888_id"][i] is None:
            continue
        bid = int(bm["banc_888_id"][i])
        sv = bm["supervoxel_id"][i]
        if sv and str(sv).isdigit():
            sv_to_bid[int(sv)] = bid
        for rc in ("root_id", "root_626", "root_850", "root_888", "root_890"):
            rv = bm[rc][i]
            if rv and str(rv).isdigit():
                old_root_to_bid[int(rv)] = bid
        if (str(bm["proofread"][i] or "") == "TRUE" or str(bm["roughly_proofread"][i] or "") == "TRUE") and str(bm["super_class"][i] or "") != "glia":
            main_cand_set.add(bid)
        toks = set()
        for col in ("fafb_cell_type", "cell_type", "fafb_alignment_cell_type"):
            toks |= ab.split_tokens(str(bm[col][i] or ""))
        if toks:
            bid_to_cts[bid] = toks

    main_cand_b = np.array(sorted(main_cand_set), dtype=np.int64)
    nt_csv = {}
    with (bdir / "banc_888_neurotransmitter_prediction_v2.csv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            nt_val = row.get("neurotransmitter_predicted") or ""
            if nt_val not in ("", "NA"):
                nt_csv[int(row["root_id"])] = nt_val
    el_b = pf.read_table(bdir / "banc_888_edgelist_simple_v2.feather", columns=["pre", "post"])
    ep_b = el_b["pre"].to_numpy().astype(np.int64)
    eq_b = el_b["post"].to_numpy().astype(np.int64)
    del el_b
    ns_b = ep_b != eq_b
    ep_b, eq_b = ep_b[ns_b], eq_b[ns_b]
    nt_ids_b = np.array(sorted(nt_csv.keys()), dtype=np.int64)
    nt_sgn_b = np.array([ab.nt_to_sign(nt_csv[int(b)]) for b in nt_ids_b], dtype=np.int8)
    p_nt = np.minimum(np.searchsorted(nt_ids_b, ep_b), len(nt_ids_b) - 1)
    v_nt = np.where(nt_ids_b[p_nt] == ep_b, nt_sgn_b[p_nt], 0) != 0
    keep_b = np.isin(ep_b, main_cand_b) & np.isin(eq_b, main_cand_b) & v_nt
    node_ids = np.union1d(ep_b[keep_b], eq_b[keep_b]).astype(np.int64)
    del ep_b, eq_b, ns_b, keep_b, v_nt, p_nt, nt_ids_b, nt_sgn_b, main_cand_b, nt_csv

    gb = np.load(CACHE_DIR / "banc_ckpt/banc_main_graph.npz")
    assert len(node_ids) == int(gb["n_nodes"])
    indptr = gb["indptr"].astype(np.int64)
    indices = gb["post_s"].astype(np.int32)
    w_raw = gb["w_raw_s"].astype(np.float64)
    n_nodes = len(node_ids)
    bid2idx = {int(b): i for i, b in enumerate(node_ids)}
    s_in_banc = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(s_in_banc, indices, np.abs(w_raw))

    # Build BANC cell-type mean S_in lookup table (for M2c: BANC -> MaleCNS reference!)
    banc_ct_vals: dict[str, list[float]] = defaultdict(list)
    for j, bid in enumerate(node_ids):
        if s_in_banc[j] > 0:
            for t in bid_to_cts.get(int(bid), ()):
                banc_ct_vals[t].append(float(s_in_banc[j]))
    banc_ct_mean_sin = {ct: float(np.mean(vs)) for ct, vs in banc_ct_vals.items()}
    print(f"  Built BANC cell-type S_in reference table: {len(banc_ct_mean_sin):,} cell types", flush=True)

    ba_res = json.loads((BASE_DIR / "mn9_banc_results.json").read_text())
    mn9_l_id = int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_l_banc"])
    mn9_r_id = int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_r_banc"])
    mn9_idx = np.array([bid2idx[mn9_l_id], bid2idx[mn9_r_id]], dtype=np.int32)

    names, y = meta["names"], meta["y"]
    is_direct_banc = np.array([int(r["is_direct_banc_m1"]) for r in meta["rows"]], dtype=bool)
    banc_m1_by_name = {r["name"]: [bid2idx[int(b)] for b in r["m1_banc_ids"] if int(b) in bid2idx] for r in ba_res["mapping_audit"]["per_type_sez106"]}
    sez_exc_banc = [np.array(banc_m1_by_name[nm], dtype=np.int32) for nm in names]
    selk_idx = np.array([bid2idx[int(b)] for b in ba_res["selk_full_cns_diagnosis"]["selk_banc_ids"] if int(b) in bid2idx], dtype=np.int32)
    zero_slnc = np.zeros(n_nodes, dtype=np.uint8)

    lb3bc_r = np.array(
        sorted(
            bid2idx[int(bm["banc_888_id"][i])]
            for i in range(len(bm["banc_888_id"]))
            if bm["banc_888_id"][i] is not None
            and int(bm["banc_888_id"][i]) in bid2idx
            and str(bm["cell_type"][i] or "") in ("LB3b", "LB3c")
            and str(bm["side"][i] or "") == "right"
        ),
        dtype=np.int32,
    )

    # 1a. Reviewed 1:1 homolog weights (same as V4)
    banc_to_fafb_sins: dict[int, list[float]] = defaultdict(list)
    with (CACHE_DIR / "data/banc/banc_fafb_reviewed_matches.csv.gz").open("rt") as f:
        for r in csv.DictReader(f):
            if r.get("valid") != "t":
                continue
            try:
                bid_raw, fid = int(r["pt_root_id"]), int(r["match_id"])
            except Exception:
                continue
            sv_s = str(r.get("pt_supervoxel_id") or "")
            if bid_raw in bid2idx:
                bid = bid_raw
            elif sv_s.isdigit() and int(sv_s) in sv_to_bid and sv_to_bid[int(sv_s)] in bid2idx:
                bid = sv_to_bid[int(sv_s)]
            elif bid_raw in old_root_to_bid and old_root_to_bid[bid_raw] in bid2idx:
                bid = old_root_to_bid[bid_raw]
            else:
                continue
            if fid in s_in_v783_by_fid and s_in_v783_by_fid[fid] > 0:
                banc_to_fafb_sins[bid2idx[bid]].append(s_in_v783_by_fid[fid])

    w_post_1to1 = np.full(n_nodes, ab.W_BANC_DENS, dtype=np.float64)
    for j, f_sins in banc_to_fafb_sins.items():
        if s_in_banc[j] > 0:
            w_post_1to1[j] = ab.W_FROZEN * min(max(float(np.mean(f_sins)) / float(s_in_banc[j]), 0.1), 10.0)

    # 1b. Cell-type-level homolog normalization (Minor #6: reviewed 1:1 + fafb_cell_type/cell_type fallback -> v783)
    w_post_ct = w_post_1to1.copy()
    n_matched_1to1 = len([j for j in banc_to_fafb_sins if s_in_banc[j] > 0])
    n_matched_ct_extra = 0
    # Also compute pure cell-type-only (strictly analogous to MaleCNS!)
    w_post_ct_pure = np.full(n_nodes, ab.W_BANC_DENS, dtype=np.float64)
    n_matched_ct_pure = 0
    for j, bid in enumerate(node_ids):
        if s_in_banc[j] <= 0:
            continue
        toks = bid_to_cts.get(int(bid), ())
        refs = [v783_ct_mean_sin[t] for t in toks if t in v783_ct_mean_sin and v783_ct_mean_sin[t] > 0]
        if refs:
            w_ct_val = ab.W_FROZEN * min(max(float(np.mean(refs)) / float(s_in_banc[j]), 0.1), 10.0)
            w_post_ct_pure[j] = w_ct_val
            n_matched_ct_pure += 1
            if j not in banc_to_fafb_sins:
                w_post_ct[j] = w_ct_val
                n_matched_ct_extra += 1
    n_matched_ct_total = n_matched_1to1 + n_matched_ct_extra
    print(
        f"  [BANC Minor #6] Coverage: 1:1={n_matched_1to1:,}/{n_nodes:,} ({100*n_matched_1to1/n_nodes:.1f}%), "
        f"1:1+cell_type={n_matched_ct_total:,}/{n_nodes:,} ({100*n_matched_ct_total/n_nodes:.1f}%), "
        f"pure_cell_type={n_matched_ct_pure:,}/{n_nodes:,} ({100*n_matched_ct_pure/n_nodes:.1f}%)",
        flush=True,
    )

    banc_ct_results = {}
    for variant_name, wp_arr, n_cov in [
        ("banc_homolog_1to1_plus_celltype", w_post_ct, n_matched_ct_total),
        ("banc_homolog_pure_celltype", w_post_ct_pure, n_matched_ct_pure),
    ]:
        w_edge = w_raw * wp_arr[indices]
        cnt_s, b_s, _ = ab.run_lif_exact_cap_slnc(indptr, indices, w_edge, ab.W_FROZEN, lb3bc_r, 100.0, zero_slnc, mn9_idx, 30, 42, 15.0)
        sweeps = {}
        for freq_hz in (50, 200):
            sb = 50000 if freq_hz == 50 else 200000
            m, lr, rr, bh, cc = ab.run_parallel_sweep(
                indptr, indices, w_edge, ab.W_FROZEN, mn9_idx, sez_exc_banc, float(freq_hz), 30, sb, workers, 15.0
            )
            sm = ab.summarize_sweep(names, y, meta["prior_pub"], is_direct_banc, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
            sm["ci95_all106"] = strat_ci95(y, m)
            sm["topk_hits"] = compute_topk_hits(y, m)
            c_sk, _, _ = ab.run_lif_exact_cap_slnc(indptr, indices, w_edge, ab.W_FROZEN, selk_idx, float(freq_hz), zero_slnc, mn9_idx, 30, 42, 15.0)
            sm["selk_mean_hz"] = float(c_sk.mean())
            sweeps[f"{freq_hz}hz"] = sm
            print(
                f"    {variant_name} {freq_hz}Hz: all106={sm['auprc_all106']:.4f} {sm['ci95_all106']}, "
                f"unpub95={sm['auprc_unpub95']:.4f}, nd={sm['auprc_nondirect']:.4f}, auroc={sm['auroc_all106']:.4f}, "
                f"pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, rho={sm['spearman_all106']:.4f}, brain={sm['mean_brain_hz']:.4f}",
                flush=True,
            )
        banc_ct_results[variant_name] = {
            "n_matched": int(n_cov),
            "frac_matched": float(n_cov / n_nodes),
            "sugar_100hz": {"mean_hz": float(cnt_s.mean()), "l_hz": float(cnt_s[:, 0].mean()), "r_hz": float(cnt_s[:, 1].mean()), "brain_hz": float(b_s)},
            "sweeps": sweeps,
        }

    # Fair two-hop baselines on BANC (raw, 1:1 weighted, cell-type weighted)
    print("  [BANC M4] Computing fair two-hop baselines & alpha sweep...", flush=True)
    banc_2hop = {
        "raw": compute_twohop_family(indptr, indices, w_raw, n_nodes, mn9_idx, sez_exc_banc, y, meta["prior_pub"], is_direct_banc),
        "homolog_1to1_weighted": compute_twohop_family(indptr, indices, w_raw * (w_post_1to1[indices] / ab.W_FROZEN), n_nodes, mn9_idx, sez_exc_banc, y, meta["prior_pub"], is_direct_banc),
        "homolog_celltype_weighted": compute_twohop_family(indptr, indices, w_raw * (w_post_ct[indices] / ab.W_FROZEN), n_nodes, mn9_idx, sez_exc_banc, y, meta["prior_pub"], is_direct_banc),
    }
    for k2, v2 in banc_2hop.items():
        print(
            f"    BANC 2-hop [{k2}]: signed(0.01)={v2['signed_relu']['alpha_0.01']['all106_auprc']:.4f} "
            f"(pure_h2={v2['signed_relu']['pure_h2']['all106_auprc']:.4f}), unsigned(0.01)={v2['unsigned']['alpha_0.01']['all106_auprc']:.4f}",
            flush=True,
        )

    del gb, indptr, indices, w_raw, bm
    gc.collect()
    return {
        "minor6_banc_celltype_homolog": banc_ct_results,
        "m4_banc_fair_twohop": banc_2hop,
        "elapsed_sec": float(time.time() - t0),
    }, banc_ct_mean_sin


# ---------------------------------------------------------------------------
# Phase 2: MaleCNS M4 Fair Two-Hop + M2 Anti-Circularity (a, b, c) + Minor #1
# ---------------------------------------------------------------------------
def run_phase2_malecns(meta: dict, v783_ct_mean_sin: dict[str, float], banc_ct_mean_sin: dict[str, float], workers: int) -> dict:
    t0 = time.time()
    print("\n================ PHASE 2: MaleCNS M4 Fair Two-Hop, M2 Anti-Circularity & Minor #1 ================", flush=True)
    mdir = CACHE_DIR / "data/malecns"
    ann_tbl = pf.read_table(mdir / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    nt_tbl = pf.read_table(mdir / "body-neurotransmitters-male-cns-v1.0.feather")
    w_tbl = pf.read_table(mdir / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather")

    cons_nt_map, pred_nt_map = {}, {}
    for b, c_nt, p_nt in zip(nt_tbl["body"].to_pylist(), nt_tbl["consensus_nt"].to_pylist(), nt_tbl["predicted_nt"].to_pylist()):
        if b is None:
            continue
        bi = int(b)
        cons_nt_map[bi] = str(c_nt).strip().lower() if c_nt is not None else ""
        pred_nt_map[bi] = str(p_nt).strip().lower() if p_nt is not None else ""

    def s_str(x) -> str:
        return "" if x is None else str(x).strip()

    traced_meta = {}
    for b, typ, fw, inst, sc, cl, rs, ss, st in zip(
        ann_tbl["bodyId"].to_pylist(), ann_tbl["type"].to_pylist(), ann_tbl["flywireType"].to_pylist(),
        ann_tbl["instance"].to_pylist(), ann_tbl["superclass"].to_pylist(), ann_tbl["class"].to_pylist(),
        ann_tbl["rootSide"].to_pylist(), ann_tbl["somaSide"].to_pylist(), ann_tbl["status"].to_pylist(),
    ):
        if b is None:
            continue
        st_s, sc_s, cl_s = s_str(st), s_str(sc), s_str(cl)
        if st_s != "Traced" or "glia" in sc_s.lower() or "glia" in cl_s.lower():
            continue
        bi = int(b)
        traced_meta[bi] = {
            "id": bi, "type": s_str(typ), "flywireType": s_str(fw),
            "cons_nt": cons_nt_map.get(bi, ""), "pred_nt": pred_nt_map.get(bi, ""),
        }
    del ann_tbl, nt_tbl

    all_bids = np.array(sorted(traced_meta.keys()), dtype=np.int64)
    n_nodes = len(all_bids)
    bid2idx = {int(b): i for i, b in enumerate(all_bids)}
    node_meta = [traced_meta[int(b)] for b in all_bids]
    mn9_l_idx, mn9_r_idx = bid2idx[10331], bid2idx[16949]
    mn9_idx = np.array([mn9_l_idx, mn9_r_idx], dtype=np.int32)

    raw_pre = w_tbl["body_pre"].to_numpy().astype(np.int64)
    raw_post = w_tbl["body_post"].to_numpy().astype(np.int64)
    raw_w = w_tbl["weight"].to_numpy().astype(np.int32)
    del w_tbl

    def map_vec(arr):
        pos = np.minimum(np.searchsorted(all_bids, arr), n_nodes - 1)
        return np.where(all_bids[pos] == arr, pos, -1).astype(np.int32)

    p_idx, q_idx = map_vec(raw_pre), map_vec(raw_post)
    ok = (p_idx >= 0) & (q_idx >= 0) & (p_idx != q_idx) & (raw_w > 0)
    p_idx, q_idx, raw_w = p_idx[ok], q_idx[ok], raw_w[ok]
    del raw_pre, raw_post
    keys = p_idx.astype(np.int64) * np.int64(n_nodes) + q_idx.astype(np.int64)
    ord_k = np.argsort(keys, kind="mergesort")
    keys_s, w_s = keys[ord_k], raw_w[ord_k]
    del keys, ord_k, p_idx, q_idx, raw_w
    uniq_k, first_pos = np.unique(keys_s, return_index=True)
    w_uniq = np.add.reduceat(w_s.astype(np.int64), first_pos).astype(np.int32) if len(uniq_k) < len(keys_s) else w_s.astype(np.int32)
    u_pre = (uniq_k // np.int64(n_nodes)).astype(np.int32)
    u_post = (uniq_k % np.int64(n_nodes)).astype(np.int32)
    del keys_s, w_s, first_pos, uniq_k

    def build_csr(signs):
        es = signs[u_pre]
        keep = es != 0
        pre_s, post_s = u_pre[keep], u_post[keep]
        w_raw_s = (w_uniq[keep] * es[keep]).astype(np.float64)
        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        np.add.at(indptr[1:], pre_s, 1)
        np.cumsum(indptr, out=indptr)
        s_in = np.zeros(n_nodes, dtype=np.float64)
        np.add.at(s_in, post_s, np.abs(w_raw_s))
        return indptr, post_s, w_raw_s, s_in

    base_signs = np.array([ab.nt_to_sign(m["cons_nt"]) for m in node_meta], dtype=np.int8)
    imp_signs = np.array([ab.nt_to_sign(m["cons_nt"]) if ab.nt_to_sign(m["cons_nt"]) != 0 else ab.nt_to_sign(m["pred_nt"]) for m in node_meta], dtype=np.int8)
    indptr0, post0, w0, sin0 = build_csr(base_signs)
    indptr_imp, post_imp, w_imp, sin_imp = build_csr(imp_signs)
    del u_pre, u_post, w_uniq
    gc.collect()

    mc_res = json.loads((BASE_DIR / "mn9_malecns_results.json").read_text())
    mc_bids_by_name = {r["name"]: [int(b) for b in r["malecns_body_ids"]] for r in mc_res["mapping_audit"]["per_type_sez106"]}
    names, y = meta["names"], meta["y"]
    sez_exc_male = [np.array([bid2idx[b] for b in mc_bids_by_name[nm] if b in bid2idx], dtype=np.int32) for nm in names]
    sez106_node_set = {int(u) for arr in sez_exc_male for u in arr}
    is_direct_male0 = np.array([int(r["is_direct_malecns"]) for r in meta["rows"]], dtype=bool)
    direct_imp = {u for u in range(n_nodes) if np.any((post_imp[indptr_imp[u]:indptr_imp[u+1]] == mn9_l_idx) | (post_imp[indptr_imp[u]:indptr_imp[u+1]] == mn9_r_idx))}
    is_direct_male_imp = np.array([any(int(u) in direct_imp for u in sez_exc_male[i]) for i in range(len(names))], dtype=bool)
    zero_slnc = np.zeros(n_nodes, dtype=np.uint8)

    # Build standard homolog weight vector on predicted_nt & consensus_nt graphs
    def build_hom_weights(sin_arr, ref_ct_sin, ref_base_w, fallback_w, mn9_l_w, exclude_set=None):
        wp = np.full(n_nodes, fallback_w, dtype=np.float64)
        matched_idx = []
        for j, m in enumerate(node_meta):
            if exclude_set is not None and j in exclude_set:
                continue
            if sin_arr[j] <= 0:
                continue
            toks = ab.split_tokens(m["flywireType"]) | ab.split_tokens(m["type"])
            refs = [ref_ct_sin[t] for t in toks if t in ref_ct_sin and ref_ct_sin[t] > 0]
            if refs:
                wp[j] = ref_base_w * min(max(float(np.mean(refs)) / float(sin_arr[j]), 0.1), 10.0)
                matched_idx.append(j)
        if mn9_l_w is not None:
            wp[mn9_l_idx] = mn9_l_w
            wp[mn9_r_idx] = mn9_l_w
        return wp, np.array(matched_idx, dtype=np.int32)

    wp_cons, matched_cons = build_hom_weights(sin0, v783_ct_mean_sin, ab.W_FROZEN, ab.W_MALE_DENS, ab.W_MALE_INTACT)
    wp_imp, matched_imp = build_hom_weights(sin_imp, v783_ct_mean_sin, ab.W_FROZEN, ab.W_MALE_JOINT, ab.W_MALE_PREDNT)

    # ---- M4: Fair Two-Hop Baselines on MaleCNS ----
    print("  [M4] Computing MaleCNS fair two-hop baselines (consensus_nt, predicted_nt, homolog-weighted) & alpha sweep...", flush=True)
    m4_male = {
        "consensus_nt_raw": compute_twohop_family(indptr0, post0, w0, n_nodes, mn9_idx, sez_exc_male, y, meta["prior_pub"], is_direct_male0),
        "predicted_nt_raw": compute_twohop_family(indptr_imp, post_imp, w_imp, n_nodes, mn9_idx, sez_exc_male, y, meta["prior_pub"], is_direct_male_imp),
        "consensus_nt_homolog_weighted": compute_twohop_family(indptr0, post0, w0 * (wp_cons[post0] / ab.W_FROZEN), n_nodes, mn9_idx, sez_exc_male, y, meta["prior_pub"], is_direct_male0),
        "predicted_nt_homolog_weighted": compute_twohop_family(indptr_imp, post_imp, w_imp * (wp_imp[post_imp] / ab.W_FROZEN), n_nodes, mn9_idx, sez_exc_male, y, meta["prior_pub"], is_direct_male_imp),
    }
    for k2, v2 in m4_male.items():
        print(
            f"    MaleCNS 2-hop [{k2}]: signed(0.01)={v2['signed_relu']['alpha_0.01']['all106_auprc']:.4f} "
            f"(nd={v2['signed_relu']['alpha_0.01']['nondirect_auprc']:.4f}, pure_h2={v2['signed_relu']['pure_h2']['all106_auprc']:.4f}), "
            f"unsigned(0.01)={v2['unsigned']['alpha_0.01']['all106_auprc']:.4f}",
            flush=True,
        )

    ab_res = json.loads((BASE_DIR / "mn9_pre_submission_ab_results.json").read_text())
    lif_hom_pred = np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"])
    lif_hom_cons = np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_cons"]["sweeps"]["50hz"]["per_type_mean_hz"])
    m4_bootstraps = {
        "Male_HomPred_LIF_vs_Fair2hop_PredHom_signed": paired_boot_100k(y, lif_hom_pred, np.array(m4_male["predicted_nt_homolog_weighted"]["signed_relu_scores_0p01"])),
        "Male_HomPred_LIF_vs_Fair2hop_PredHom_unsigned": paired_boot_100k(y, lif_hom_pred, np.array(m4_male["predicted_nt_homolog_weighted"]["unsigned_scores_0p01"])),
        "Male_HomPred_LIF_vs_2hop_PredRaw_signed": paired_boot_100k(y, lif_hom_pred, np.array(m4_male["predicted_nt_raw"]["signed_relu_scores_0p01"])),
        "Male_HomCons_LIF_vs_Fair2hop_ConsHom_signed": paired_boot_100k(y, lif_hom_cons, np.array(m4_male["consensus_nt_homolog_weighted"]["signed_relu_scores_0p01"])),
    }
    for bk, bv in m4_bootstraps.items():
        print(f"    [M4 100k boot] {bk}: delta={bv['delta']:+.4f} CI [{bv['ci95'][0]:+.4f}, {bv['ci95'][1]:+.4f}] p={bv['p_two_sided']:.4f}", flush=True)

    # Helper to run 50 Hz MaleCNS predicted_nt sweep with custom wp
    def eval_male_wp(label: str, wp_arr: np.ndarray, w_poi_base: float = ab.W_FROZEN, freq_hz: int = 50) -> dict:
        sb = 50000 if freq_hz == 50 else 200000
        m, lr, rr, bh, cc = ab.run_parallel_sweep(
            indptr_imp, post_imp, w_imp * wp_arr[post_imp], w_poi_base, mn9_idx, sez_exc_male, float(freq_hz), 30, sb, workers, 15.0, use_xorshift=1
        )
        sm = ab.summarize_sweep(names, y, meta["prior_pub"], is_direct_male_imp, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
        sm["ci95_all106"] = strat_ci95(y, m)
        sm["topk_hits"] = compute_topk_hits(y, m)
        print(
            f"    [{label}] all106={sm['auprc_all106']:.4f} {sm['ci95_all106']}, unpub95={sm['auprc_unpub95']:.4f}, "
            f"nd={sm['auprc_nondirect']:.4f}, auroc={sm['auroc_all106']:.4f}, pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, "
            f"rho={sm['spearman_all106']:.4f}, brain={sm['mean_brain_hz']:.4f}",
            flush=True,
        )
        return sm

    # ---- M2a: Shuffled-homolog permutation controls ----
    print("\n  [M2a] Shuffled-homolog permutation controls (5 seeds MN9-fixed + 3 seeds MN9-shuffled)...", flush=True)
    non_mn9_matched = matched_imp[(matched_imp != mn9_l_idx) & (matched_imp != mn9_r_idx)]
    m2a_mn9_fixed = []
    for sd in (101, 102, 103, 104, 105):
        rng = np.random.default_rng(sd)
        wp_sh = wp_imp.copy()
        wp_sh[non_mn9_matched] = rng.permutation(wp_imp[non_mn9_matched])
        rec = eval_male_wp(f"M2a_shuffle_nonMN9_s{sd}", wp_sh)
        rec["seed"] = sd
        m2a_mn9_fixed.append(rec)

    m2a_all_shuffled = []
    for sd in (201, 202, 203):
        rng = np.random.default_rng(sd)
        wp_sh = wp_imp.copy()
        wp_sh[matched_imp] = rng.permutation(wp_imp[matched_imp])
        rec = eval_male_wp(f"M2a_shuffle_all_s{sd}", wp_sh)
        rec["seed"] = sd
        m2a_all_shuffled.append(rec)

    # ---- M2b: Leave-out benchmark neurons & two-scale (MN9-anchored + CB-density) control ----
    print("\n  [M2b] Leave-out benchmark neurons & two-scale MN9-vs-CB control...", flush=True)
    wp_lo_sez, _ = build_hom_weights(sin_imp, v783_ct_mean_sin, ab.W_FROZEN, ab.W_MALE_JOINT, ab.W_MALE_PREDNT, exclude_set=sez106_node_set)
    m2b_lo_sez = eval_male_wp("M2b_leave_out_sez106", wp_lo_sez)

    wp_lo_sez_mn9, _ = build_hom_weights(sin_imp, v783_ct_mean_sin, ab.W_FROZEN, ab.W_MALE_JOINT, None, exclude_set=sez106_node_set | {mn9_l_idx, mn9_r_idx})
    m2b_lo_sez_mn9 = eval_male_wp("M2b_leave_out_sez106_and_mn9", wp_lo_sez_mn9)

    wp_two_scale = np.full(n_nodes, ab.W_MALE_JOINT, dtype=np.float64)
    wp_two_scale[mn9_l_idx] = ab.W_MALE_PREDNT
    wp_two_scale[mn9_r_idx] = ab.W_MALE_PREDNT
    m2b_two_scale = eval_male_wp("M2b_two_scale_mn9_02638_rest_01567", wp_two_scale)

    # ---- M2c: Cross-reference normalization (BANC -> MaleCNS, ZERO FlyWire S_in!) ----
    print("\n  [M2c] Cross-reference normalization (BANC -> MaleCNS, zero FlyWire S_in)...", flush=True)
    # Compute median BANC/MaleCNS S_in ratio for fallback weight
    ratios_bm = []
    for j, m in enumerate(node_meta):
        if sin_imp[j] <= 0:
            continue
        toks = ab.split_tokens(m["flywireType"]) | ab.split_tokens(m["type"])
        refs = [banc_ct_mean_sin[t] for t in toks if t in banc_ct_mean_sin and banc_ct_mean_sin[t] > 0]
        if refs:
            ratios_bm.append(float(np.mean(refs)) / float(sin_imp[j]))
    med_bm_ratio = float(np.median(ratios_bm))
    fallback_bm = ab.W_BANC_ANAT * med_bm_ratio
    mn9_bm_w = ab.W_BANC_ANAT * (2283.5 / float(sin_imp[mn9_l_idx]))
    wp_banc_ref, matched_bm = build_hom_weights(sin_imp, banc_ct_mean_sin, ab.W_BANC_ANAT, fallback_bm, mn9_bm_w)
    print(f"    BANC->MaleCNS matched {len(matched_bm):,}/{n_nodes:,} neurons; median S_in(BANC)/S_in(Male)={med_bm_ratio:.4f}, fallback={fallback_bm:.4f} mV, mn9={mn9_bm_w:.4f} mV", flush=True)
    m2c_banc_ref = eval_male_wp("M2c_BANC_ref_to_MaleCNS", wp_banc_ref)
    m2c_banc_ref["n_matched_banc_ref"] = int(len(matched_bm))
    m2c_banc_ref["med_banc_to_male_sin_ratio"] = med_bm_ratio
    m2c_banc_ref["fallback_w_mv"] = float(fallback_bm)

    # ---- Minor #1: Verify stimulated neuron firing-rate invariance across W_syn ----
    print("\n  [Minor #1] Checking stimulated-neuron firing rate invariance across W_syn (0.157 to 0.690 mV)...", flush=True)
    stim_check = {}
    for nm in ("roundup", "diatom", "sink_sync", "FMIn", "rattle"):
        ti = names.index(nm)
        e_idx = sez_exc_male[ti]
        rates_by_w = {}
        for w_lbl, w_val in [("0.1567mV", ab.W_MALE_JOINT), ("0.2683mV", ab.W_MALE_INTACT), ("0.2750mV", ab.W_FROZEN), ("0.6904mV", ab.W_BANC_ANAT)]:
            # Watch the stimulated neurons themselves!
            cnt_st, _, _ = ab.run_lif_exact_cap_slnc(indptr_imp, post_imp, w_imp * ab.W_MALE_JOINT, w_val, e_idx, 50.0, zero_slnc, e_idx, 30, 50000 + ti * 97, 15.0, 1)
            rates_by_w[w_lbl] = float(cnt_st.mean())
        stim_check[nm] = rates_by_w
        print(f"    Stimulated {nm:10s} mean firing @ 50 Hz drive: {rates_by_w}", flush=True)

    # Paired bootstrap of Hom-Pred vs M2a/M2b/M2c
    m2_bootstraps = {
        "HomPred_vs_M2b_two_scale": paired_boot_100k(y, lif_hom_pred, np.array(m2b_two_scale["per_type_mean_hz"])),
        "HomPred_vs_M2b_lo_sez106": paired_boot_100k(y, lif_hom_pred, np.array(m2b_lo_sez["per_type_mean_hz"])),
        "HomPred_vs_M2b_lo_sez_mn9": paired_boot_100k(y, lif_hom_pred, np.array(m2b_lo_sez_mn9["per_type_mean_hz"])),
        "HomPred_vs_M2c_BANC_ref": paired_boot_100k(y, lif_hom_pred, np.array(m2c_banc_ref["per_type_mean_hz"])),
        "M2a_shuffle_nonMN9_mean_auprc": float(np.mean([r["auprc_all106"] for r in m2a_mn9_fixed])),
        "M2a_shuffle_all_mean_auprc": float(np.mean([r["auprc_all106"] for r in m2a_all_shuffled])),
    }
    print(f"    [M2 summary] M2a_nonMN9_mean={m2_bootstraps['M2a_shuffle_nonMN9_mean_auprc']:.4f}, M2a_all_mean={m2_bootstraps['M2a_shuffle_all_mean_auprc']:.4f}", flush=True)

    del indptr0, post0, w0, indptr_imp, post_imp, w_imp
    gc.collect()
    return {
        "m4_malecns_fair_twohop": m4_male,
        "m4_malecns_bootstraps_100k": m4_bootstraps,
        "m2a_shuffle_non_mn9": m2a_mn9_fixed,
        "m2a_shuffle_all_matched": m2a_all_shuffled,
        "m2b_leave_out_sez106": m2b_lo_sez,
        "m2b_leave_out_sez106_and_mn9": m2b_lo_sez_mn9,
        "m2b_two_scale_mn9_02638_rest_01567": m2b_two_scale,
        "m2c_banc_ref_to_malecns": m2c_banc_ref,
        "m2_summary_bootstraps": m2_bootstraps,
        "minor1_poisson_stim_rate_check": stim_check,
        "elapsed_sec": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# Phase 3: M5 FlyWire v783 K=100 Fixed-Weight Rewired Nulls + v783 alpha sweep
# ---------------------------------------------------------------------------
_RW_STATE: dict = {}


def _run_one_rewired_graph_worker(args: tuple[str, int, int]) -> dict:
    mode, k_idx, seed = args
    st = _RW_STATE
    indptr = st["indptr"]
    orig_post = st["post_s"]
    pre_s = st["pre_s"]
    w_raw = st["w_raw_s"]
    w_edge = st["w_edge_0275"]
    node_class = st["node_class"]
    node_comp = st["node_comp"]
    orig_in_pos = st["orig_in_pos"]
    orig_in_neg = st["orig_in_neg"]
    prot_set = st["prot_set"]
    mn9_idx = st["mn9_idx"]
    sez_exc = st["sez_exc"]
    y = st["y"]
    unpub = st["unpub"]
    nd = st["nd"]
    zero_slnc = st["zero_slnc"]

    if mode == "global":
        new_post, rw_meta = smb.build_rewired_graph(
            indptr, orig_post, w_raw, pre_s, node_class, orig_in_pos, orig_in_neg, prot_set, seed
        )
    else:
        new_post, rw_meta = mph.build_compartment_rewired_graph(
            indptr, orig_post, pre_s, node_class, node_comp, orig_in_pos, orig_in_neg, w_raw, prot_set, seed
        )

    means = np.zeros(len(sez_exc), dtype=np.float64)
    brain_sum = 0.0
    for i, e_idx in enumerate(sez_exc):
        if len(e_idx) == 0:
            continue
        cnt, bh, _ = ab.run_lif_exact_cap_slnc(
            indptr, new_post, w_edge, ab.W_FROZEN, e_idx, 50.0, zero_slnc, mn9_idx, 30, 50000 + i * 97, 15.0, 0
        )
        means[i] = float(cnt.mean())
        brain_sum += bh

    return {
        "k": int(k_idx),
        "seed": int(seed),
        "all106_auprc": ab.stepwise_auprc(y, means),
        "nondirect83_auprc": ab.stepwise_auprc(y[nd], means[nd]),
        "unpub95_auprc": ab.stepwise_auprc(y[unpub], means[unpub]),
        "all106_auroc": ab.compute_auroc(y, means),
        "mean_106_hz": float(means.mean()),
        "active_106": int(np.sum(means > 0.0)),
        "mean_brain_hz": float(brain_sum / len(sez_exc)),
        "replaced_frac": float(rw_meta["replaced_fraction_movable"]),
    }


def run_phase3_v783_k100(meta: dict, workers: int, k_graphs: int = 100) -> dict:
    t0 = time.time()
    print(f"\n================ PHASE 3: FlyWire v783 K={k_graphs} Fixed-Weight Rewired Nulls (Global & Compartment) + M4 Alpha Sweep ================", flush=True)
    g783, _, _, _, sez_v783_fids, mn9_783_fids = ab.load_v783_reference()
    idx783 = g783["flyid2idx"]
    fids783 = g783["flyids"]
    n783 = len(fids783)
    indptr = g783["indptr"]
    post_s = g783["post_s"]
    pre_s = g783["pre_s"]
    w_raw_s = g783["w_raw_s"]
    w_raw_f64 = w_raw_s.astype(np.float64)
    mn9_idx = np.array([idx783[f] for f in mn9_783_fids], dtype=np.int32)
    names, y = meta["names"], meta["y"]
    sez_exc = [np.array([idx783[f] for f in sez_v783_fids[nm] if f in idx783], dtype=np.int32) for nm in names]
    is_direct_v7 = np.array([int(r["is_direct_v783"]) for r in meta["rows"]], dtype=bool)

    # Super-class compartment IDs (exact match to mn9_portability_hardening.py!)
    ann783 = CACHE_DIR / "data/Supplemental_file1_neuron_annotations.tsv"
    sc_by_fid = {}
    with ann783.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rs = row.get("root_id", "")
            if rs and rs.isdigit():
                rid = int(rs)
                if rid in idx783:
                    sc_by_fid[rid] = (row.get("super_class") or "").strip() or "unknown"
    sc_list = [sc_by_fid.get(int(fid), "unknown") for fid in fids783]
    sc_names = sorted(set(sc_list))
    sc_to_id = {s: i for i, s in enumerate(sc_names)}
    node_comp = np.array([sc_to_id[s] for s in sc_list], dtype=np.int32)

    # M4: FlyWire v783 two-hop alpha sweep
    v783_2hop = compute_twohop_family(indptr, post_s, w_raw_f64, n783, mn9_idx, sez_exc, y, meta["prior_pub"], is_direct_v7)
    print(
        f"  [v783 M4 2-hop] signed(0.01)={v783_2hop['signed_relu']['alpha_0.01']['all106_auprc']:.4f} "
        f"(pure_h2={v783_2hop['signed_relu']['pure_h2']['all106_auprc']:.4f}), unsigned(0.01)={v783_2hop['unsigned']['alpha_0.01']['all106_auprc']:.4f}",
        flush=True,
    )

    # Warm up Numba JIT before fork so every worker inherits compiled native code!
    zero_slnc = np.zeros(n783, dtype=np.uint8)
    _ = ab.run_lif_exact_cap_slnc(indptr, post_s, w_raw_f64 * ab.W_FROZEN, ab.W_FROZEN, sez_exc[0], 50.0, zero_slnc, mn9_idx, 1, 42, 15.0, 0)
    # Warm up rewire Numba kernels before fork
    _ = smb.build_rewired_graph(indptr, post_s, w_raw_s, pre_s, g783["node_class"], g783["orig_in_pos"], g783["orig_in_neg"], {int(mn9_idx[0]), int(mn9_idx[1])}, 19200)
    _ = mph.build_compartment_rewired_graph(indptr, post_s, pre_s, g783["node_class"], node_comp, g783["orig_in_pos"], g783["orig_in_neg"], w_raw_s, {int(mn9_idx[0]), int(mn9_idx[1])}, 19300)

    _RW_STATE.clear()
    _RW_STATE.update({
        "indptr": indptr,
        "post_s": post_s,
        "pre_s": pre_s,
        "w_raw_s": w_raw_s,
        "w_edge_0275": w_raw_f64 * ab.W_FROZEN,
        "node_class": g783["node_class"],
        "node_comp": node_comp,
        "orig_in_pos": g783["orig_in_pos"],
        "orig_in_neg": g783["orig_in_neg"],
        "prot_set": {int(mn9_idx[0]), int(mn9_idx[1])},
        "mn9_idx": mn9_idx,
        "sez_exc": sez_exc,
        "y": y,
        "unpub": (meta["prior_pub"] == 0),
        "nd": ~is_direct_v7,
        "zero_slnc": zero_slnc,
    })

    real_all = ab.stepwise_auprc(y, meta["v783_50"])
    real_nd = ab.stepwise_auprc(y[~is_direct_v7], meta["v783_50"][~is_direct_v7])
    real_u95 = ab.stepwise_auprc(y[meta["prior_pub"] == 0], meta["v783_50"][meta["prior_pub"] == 0])

    ctx = mp.get_context("fork")
    out_nulls = {}
    for mode, seed_base in [("global", 19201), ("compartment", 19301)]:
        t_m = time.time()
        jobs = [(mode, k, seed_base + k) for k in range(k_graphs)]
        recs = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for idx_done, r in enumerate(ex.map(_run_one_rewired_graph_worker, jobs), 1):
                recs.append(r)
                if idx_done % 25 == 0 or idx_done == k_graphs:
                    print(f"    [{mode} K={k_graphs}] {idx_done}/{k_graphs} done ({time.time()-t_m:.1f}s)", flush=True)
        a_all = np.array([r["all106_auprc"] for r in recs])
        a_nd = np.array([r["nondirect83_auprc"] for r in recs])
        a_u95 = np.array([r["unpub95_auprc"] for r in recs])
        w_all = int(np.sum(a_all < real_all))
        w_nd = int(np.sum(a_nd < real_nd))
        w_u95 = int(np.sum(a_u95 < real_u95))
        t_u95 = int(np.sum(a_u95 == real_u95))
        p_all = float((int(np.sum(a_all >= real_all)) + 1) / (k_graphs + 1))
        p_nd = float((int(np.sum(a_nd >= real_nd)) + 1) / (k_graphs + 1))
        p_u95 = float((int(np.sum(a_u95 >= real_u95)) + 1) / (k_graphs + 1))
        out_nulls[mode] = {
            "k_graphs": k_graphs,
            "all106": {"mean": float(a_all.mean()), "sd": float(a_all.std()), "min": float(a_all.min()), "max": float(a_all.max()), "real_wins": w_all, "p_emp_one_sided": p_all},
            "nondirect83": {"mean": float(a_nd.mean()), "sd": float(a_nd.std()), "min": float(a_nd.min()), "max": float(a_nd.max()), "real_wins": w_nd, "p_emp_one_sided": p_nd},
            "unpub95": {"mean": float(a_u95.mean()), "sd": float(a_u95.std()), "min": float(a_u95.min()), "max": float(a_u95.max()), "real_wins": w_u95, "real_ties": t_u95, "p_emp_one_sided": p_u95},
            "per_graph": recs,
            "elapsed_sec": float(time.time() - t_m),
        }
        print(
            f"  [M5 {mode} K={k_graphs} DONE in {time.time()-t_m:.1f}s] "
            f"all106={a_all.mean():.4f} (max {a_all.max():.4f}, wins {w_all}/{k_graphs}, p={p_all:.4f}) | "
            f"nd83={a_nd.mean():.4f} (max {a_nd.max():.4f}, wins {w_nd}/{k_graphs}, p={p_nd:.4f}) | "
            f"u95={a_u95.mean():.4f} (wins {w_u95}/{k_graphs}, ties {t_u95}, p={p_u95:.4f})",
            flush=True,
        )

    _RW_STATE.clear()
    del g783
    gc.collect()
    return {
        "m4_v783_twohop_alpha_sweep": v783_2hop,
        "m5_v783_fixed_weight_k100": out_nulls,
        "elapsed_sec": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# Phase 4: M3 Label-Free Calibration Selection + Unified BH-FDR + M6 AUROC/Top-k
# ---------------------------------------------------------------------------
def build_unified_m3_m6(meta: dict, res: dict) -> dict:
    print("\n================ PHASE 4: M3 Label-Free Selection, Unified BH-FDR & M6 AUROC/Top-k ================", flush=True)
    ab_res = json.loads((BASE_DIR / "mn9_pre_submission_ab_results.json").read_text())
    port = json.loads((BASE_DIR / "mn9_portability_results.json").read_text())
    rows = meta["rows"]
    y = meta["y"]
    unpub = (meta["prior_pub"] == 0)
    v783_50 = meta["v783_50"]

    # Collect every calibration candidate with LABEL-FREE diagnostics (brain_hz, v783 kappa, v783 rho, binary_agr)
    # and behavioral metrics (AUPRC, AUROC, Top-10/14/15 hits)
    def pack_cand(name: str, connectome: str, arr: np.ndarray, brain_hz: float, sugar_ok: bool) -> dict:
        v_bin = (v783_50 > 0.0).astype(int)
        c_bin = (arr > 0.0).astype(int)
        po = float(np.mean(v_bin == c_bin))
        pe = float(np.mean(v_bin == 1) * np.mean(c_bin == 1) + np.mean(v_bin == 0) * np.mean(c_bin == 0))
        kappa = (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0
        rho = float(stats.spearmanr(v783_50, arr).statistic) if np.std(arr) > 0 else 0.0
        return {
            "name": name,
            "connectome": connectome,
            "label_free": {
                "mean_brain_hz": float(brain_hz),
                "physiological_brain_lt_0p2hz": bool(brain_hz < 0.2),
                "sugar_anchor_physiological": bool(sugar_ok),
                "v783_binary_agreement": po,
                "v783_cohen_kappa": float(kappa),
                "v783_spearman_rho": rho,
            },
            "behavioral": {
                "all106_auprc": ab.stepwise_auprc(y, arr),
                "all106_auroc": ab.compute_auroc(y, arr),
                "unpub95_auprc": ab.stepwise_auprc(y[unpub], arr[unpub]),
                "topk_hits": compute_topk_hits(y, arr),
            },
        }

    b_rows = rows
    cands = [
        pack_cand("BANC_Raw_0.275", "BANC", np.array([float(r["banc_frozen_50hz"]) for r in b_rows]), 0.002, False),
        pack_cand("BANC_Target_0.690", "BANC", np.array([float(r["banc_anat_50hz"]) for r in b_rows]), 0.013, True),
        pack_cand("BANC_Sugar_0.420", "BANC", np.array(ab_res["phase_banc"]["a1_sugar_anchor"]["sweeps"]["50hz"]["per_type_mean_hz"]), ab_res["phase_banc"]["a1_sugar_anchor"]["sweeps"]["50hz"]["mean_brain_hz"], False),
        pack_cand("BANC_Homolog_1to1", "BANC", np.array(ab_res["phase_banc"]["a2a_homolog_per_neuron_norm"]["sweeps"]["50hz"]["per_type_mean_hz"]), ab_res["phase_banc"]["a2a_homolog_per_neuron_norm"]["sweeps"]["50hz"]["mean_brain_hz"], True),
        pack_cand("BANC_Homolog_CellType", "BANC", np.array(res["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["50hz"]["per_type_mean_hz"]), res["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["50hz"]["mean_brain_hz"], True),
        pack_cand("BANC_Intrinsic_1_over_S", "BANC", np.array(ab_res["phase_banc"]["a2b_intrinsic_inv_sin_norm"]["sweeps"]["50hz"]["per_type_mean_hz"]), ab_res["phase_banc"]["a2b_intrinsic_inv_sin_norm"]["sweeps"]["50hz"]["mean_brain_hz"], False),
        pack_cand("MaleCNS_Target_0.268", "MaleCNS", np.array([float(r["malecns_intact_50hz_mean"]) for r in b_rows]), 0.294, False),
        pack_cand("MaleCNS_Density_0.159", "MaleCNS", np.array([float(r["malecns_density_50hz"]) for r in b_rows]), 0.006, True),
        pack_cand("MaleCNS_PredNT_Fixed_0.268", "MaleCNS", np.array(ab_res["phase_malecns"]["experiment_b2_fixed_02683_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"]), ab_res["phase_malecns"]["experiment_b2_fixed_02683_prednt"]["sweeps"]["50hz"]["mean_brain_hz"], False),
        pack_cand("MaleCNS_PredNT_Recalc_0.264", "MaleCNS", np.array([float(r["malecns_predNT_50hz"]) for r in b_rows]), 0.440, False),
        pack_cand("MaleCNS_Joint_0.157", "MaleCNS", np.array([float(r["malecns_predNT_density_50hz"]) for r in b_rows]), 0.0078, True),
        pack_cand("MaleCNS_Sugar_L17_Cons", "MaleCNS", np.array(ab_res["phase_malecns"]["a1_sugar_anchor_consensus"]["sweeps_at_left17_wstar"]["50hz"]["per_type_mean_hz"]), ab_res["phase_malecns"]["a1_sugar_anchor_consensus"]["sweeps_at_left17_wstar"]["50hz"]["mean_brain_hz"], False),
        pack_cand("MaleCNS_Sugar_R17_Cons", "MaleCNS", np.array(ab_res["phase_malecns"]["a1_sugar_anchor_consensus"]["sweeps_at_right17_wstar"]["50hz"]["per_type_mean_hz"]), ab_res["phase_malecns"]["a1_sugar_anchor_consensus"]["sweeps_at_right17_wstar"]["50hz"]["mean_brain_hz"], True),
        pack_cand("MaleCNS_Sugar_R17_PredNT", "MaleCNS", np.array(ab_res["phase_malecns"]["a1_sugar_anchor_prednt"]["sweeps_at_right17_wstar"]["50hz"]["per_type_mean_hz"]), ab_res["phase_malecns"]["a1_sugar_anchor_prednt"]["sweeps_at_right17_wstar"]["50hz"]["mean_brain_hz"], True),
        pack_cand("MaleCNS_Homolog_Cons", "MaleCNS", np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_cons"]["sweeps"]["50hz"]["per_type_mean_hz"]), ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_cons"]["sweeps"]["50hz"]["mean_brain_hz"], True),
        pack_cand("MaleCNS_Homolog_PredNT", "MaleCNS", np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"]), ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["mean_brain_hz"], True),
        pack_cand("MaleCNS_Intrinsic_1_over_S", "MaleCNS", np.array(ab_res["phase_malecns"]["a2b_intrinsic_inv_sin_norm_cons"]["sweeps"]["50hz"]["per_type_mean_hz"]), ab_res["phase_malecns"]["a2b_intrinsic_inv_sin_norm_cons"]["sweeps"]["50hz"]["mean_brain_hz"], False),
    ]
    # Rank within each connectome purely by label-free criteria (physiological brain < 0.2 Hz -> highest v783 kappa & rho)
    lf_winners = {}
    for conn in ("BANC", "MaleCNS"):
        pool = [c for c in cands if c["connectome"] == conn and c["label_free"]["physiological_brain_lt_0p2hz"]]
        pool.sort(key=lambda c: (c["label_free"]["v783_spearman_rho"], c["label_free"]["v783_cohen_kappa"]), reverse=True)
        lf_winners[conn] = [c["name"] for c in pool[:3]]
        print(f"  [M3 Label-Free Top-3 for {conn}]: {lf_winners[conn]} -> #1 AUPRC={pool[0]['behavioral']['all106_auprc']:.4f}, AUROC={pool[0]['behavioral']['all106_auroc']:.4f}", flush=True)

    # Unified BH-FDR across ALL 10 exploratory calibration comparisons (ALL recomputed on identical 100k seed-42 paired draws!)
    pb_ab = ab_res["paired_bootstraps_100k_seed42"]
    banc_tgt = np.array([float(r["banc_anat_50hz"]) for r in b_rows])
    banc_ct_arr = np.array(res["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["50hz"]["per_type_mean_hz"])
    banc_ct_boot = paired_boot_100k(y, banc_ct_arr, banc_tgt)
    banc_ct_vs_v783 = paired_boot_100k(y, banc_ct_arr, v783_50)
    banc_ct_vs_fair2h = paired_boot_100k(y, banc_ct_arr, np.array(res["phase1_banc"]["m4_banc_fair_twohop"]["homolog_celltype_weighted"]["signed_relu_scores_0p01"]))

    male_tgt = np.array([float(r["malecns_intact_50hz_mean"]) for r in b_rows])
    male_den = np.array([float(r["malecns_density_50hz"]) for r in b_rows])
    male_pnt = np.array([float(r["malecns_predNT_50hz"]) for r in b_rows])
    male_jnt = np.array([float(r["malecns_predNT_density_50hz"]) for r in b_rows])
    male_hom_pnt = np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"])
    pb_den_100k = paired_boot_100k(y, male_den, male_tgt)
    pb_pnt_100k = paired_boot_100k(y, male_pnt, male_tgt)
    pb_jnt_100k = paired_boot_100k(y, male_jnt, male_tgt)

    # M1 TOST equivalence check (90% CI of paired delta Hom-PredNT vs v783 on all-106 and unpub-95)
    rng_tost = np.random.default_rng(42)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_b = 100000
    d_all = np.empty(n_b, dtype=np.float64)
    for b in range(n_b):
        ii = np.concatenate([rng_tost.choice(pos_idx, len(pos_idx), True), rng_tost.choice(neg_idx, len(neg_idx), True)])
        d_all[b] = ab.stepwise_auprc(y[ii], male_hom_pnt[ii]) - ab.stepwise_auprc(y[ii], v783_50[ii])
    pos_u = np.where(y[unpub] == 1)[0]
    neg_u = np.where(y[unpub] == 0)[0]
    d_u95 = np.empty(n_b, dtype=np.float64)
    yu, mu, vu = y[unpub], male_hom_pnt[unpub], v783_50[unpub]
    for b in range(n_b):
        ii = np.concatenate([rng_tost.choice(pos_u, len(pos_u), True), rng_tost.choice(neg_u, len(neg_u), True)])
        d_u95[b] = ab.stepwise_auprc(yu[ii], mu[ii]) - ab.stepwise_auprc(yu[ii], vu[ii])
    tost_hom_vs_v783 = {
        "all106": {
            "delta": float(ab.stepwise_auprc(y, male_hom_pnt) - ab.stepwise_auprc(y, v783_50)),
            "ci90": [float(np.percentile(d_all, 5.0)), float(np.percentile(d_all, 95.0))],
            "ci95": [float(np.percentile(d_all, 2.5)), float(np.percentile(d_all, 97.5))],
            "p_tost_0p05": float(max(np.mean(d_all <= -0.05), np.mean(d_all >= 0.05))),
            "p_tost_0p10": float(max(np.mean(d_all <= -0.10), np.mean(d_all >= 0.10))),
        },
        "unpub95": {
            "delta": float(ab.stepwise_auprc(yu, mu) - ab.stepwise_auprc(yu, vu)),
            "ci90": [float(np.percentile(d_u95, 5.0)), float(np.percentile(d_u95, 95.0))],
            "ci95": [float(np.percentile(d_u95, 2.5)), float(np.percentile(d_u95, 97.5))],
            "p_two_sided": float(min(1.0, 2.0 * min(np.mean(d_u95 <= 0.0), np.mean(d_u95 >= 0.0)))),
            "p_tost_0p05": float(max(np.mean(d_u95 <= -0.05), np.mean(d_u95 >= 0.05))),
        },
    }

    unified_cal_family = [
        ("BANC Hom-1:1 vs BANC Target", pb_ab["BANC_A2a_vs_BANC_Target"]["delta"], pb_ab["BANC_A2a_vs_BANC_Target"]["ci95"], pb_ab["BANC_A2a_vs_BANC_Target"]["p_two_sided"]),
        ("BANC Hom-CellType vs BANC Target", banc_ct_boot["delta"], banc_ct_boot["ci95"], banc_ct_boot["p_two_sided"]),
        ("MaleCNS Density vs MaleCNS Target", pb_den_100k["delta"], pb_den_100k["ci95"], pb_den_100k["p_two_sided"]),
        ("MaleCNS PredNT-Fixed vs MaleCNS Target", pb_ab["Male_B2fixed_vs_Male_Target"]["delta"], pb_ab["Male_B2fixed_vs_Male_Target"]["ci95"], pb_ab["Male_B2fixed_vs_Male_Target"]["p_two_sided"]),
        ("MaleCNS PredNT-Recalc vs MaleCNS Target", pb_pnt_100k["delta"], pb_pnt_100k["ci95"], pb_pnt_100k["p_two_sided"]),
        ("MaleCNS Joint vs MaleCNS Target", pb_jnt_100k["delta"], pb_jnt_100k["ci95"], pb_jnt_100k["p_two_sided"]),
        ("MaleCNS R17-Sugar-Cons vs MaleCNS Target", pb_ab["Male_R17sugar_cons_vs_Male_Target"]["delta"], pb_ab["Male_R17sugar_cons_vs_Male_Target"]["ci95"], pb_ab["Male_R17sugar_cons_vs_Male_Target"]["p_two_sided"]),
        ("MaleCNS R17-Sugar-PredNT vs MaleCNS Target", pb_ab["Male_R17sugar_predNT_vs_Male_Target"]["delta"], pb_ab["Male_R17sugar_predNT_vs_Male_Target"]["ci95"], pb_ab["Male_R17sugar_predNT_vs_Male_Target"]["p_two_sided"]),
        ("MaleCNS Hom-Cons vs MaleCNS Target", pb_ab["Male_A2a_cons_vs_Male_Target"]["delta"], pb_ab["Male_A2a_cons_vs_Male_Target"]["ci95"], pb_ab["Male_A2a_cons_vs_Male_Target"]["p_two_sided"]),
        ("MaleCNS Hom-PredNT vs MaleCNS Target", pb_ab["Male_A2a_predNT_vs_Male_Target"]["delta"], pb_ab["Male_A2a_predNT_vs_Male_Target"]["ci95"], pb_ab["Male_A2a_predNT_vs_Male_Target"]["p_two_sided"]),
    ]
    qs = bh_fdr([x[3] for x in unified_cal_family])
    unified_table = [
        {"comparison": nm, "delta": float(d), "ci95": [float(c[0]), float(c[1])], "p_raw": float(p), "q_bh_10test": float(q)}
        for (nm, d, c, p), q in zip(unified_cal_family, qs)
    ]
    print("  [M3 Unified 10-test Calibration BH-FDR (ALL 100k seed-42 draws)]:", flush=True)
    for row in unified_table:
        print(f"    {row['comparison']:42s}: delta={row['delta']:+.4f} CI [{row['ci95'][0]:+.4f},{row['ci95'][1]:+.4f}] p={row['p_raw']:.4f} q={row['q_bh_10test']:.4f}", flush=True)

    return {
        "label_free_selection_candidates": cands,
        "label_free_top3_by_connectome": lf_winners,
        "unified_calibration_bh_fdr_10test": unified_table,
        "m1_tost_hom_prednt_vs_v783": tost_hom_vs_v783,
        "banc_celltype_extra_bootstraps": {
            "vs_banc_target": banc_ct_boot,
            "vs_v783": banc_ct_vs_v783,
            "vs_fair_2hop_signed": banc_ct_vs_fair2h,
        },
        "v783_reference_m6": {
            "all106_auroc": ab.compute_auroc(y, v783_50),
            "topk_hits": compute_topk_hits(y, v783_50),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--k-graphs", type=int, default=100)
    args = ap.parse_args()

    t_all = time.time()
    meta = ab.load_common_meta()
    _, _, v783_ct_mean_sin, s_in_v783_by_fid, _, _ = ab.load_v783_reference()

    out = {"meta": {"workers": args.workers, "k_graphs": args.k_graphs, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}}
    p1, banc_ct_mean_sin = run_phase1_banc(meta, v783_ct_mean_sin, s_in_v783_by_fid, args.workers)
    out["phase1_banc"] = p1
    save_ckpt(out)

    p2 = run_phase2_malecns(meta, v783_ct_mean_sin, banc_ct_mean_sin, args.workers)
    out["phase2_malecns"] = p2
    save_ckpt(out)

    p3 = run_phase3_v783_k100(meta, args.workers, args.k_graphs)
    out["phase3_v783_k100"] = p3
    save_ckpt(out)

    p4 = build_unified_m3_m6(meta, out)
    out["phase4_m3_m6_unified"] = p4
    out["meta"]["total_elapsed_sec"] = float(time.time() - t_all)
    save_ckpt(out)
    print(f"\nALL V5 REVIEWER CONTROLS COMPLETE in {time.time()-t_all:.1f}s -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
