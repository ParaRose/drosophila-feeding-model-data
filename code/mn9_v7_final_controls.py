#!/usr/bin/env python3
"""V7 Final Controls:
1. Static 3-hop signed/unsigned ReLU baselines (S_3 = h_1 + a2*h_2,ReLU + a3*h_3,ReLU)
   across FlyWire v783, BANC v888 (raw, 1:1, 1:1+CT), and MaleCNS v1.0 (cons/pred_nt, raw/homolog)
   with B=100,000 seed-42 paired bootstraps vs LIF.
2. SELK (DNg68) full proboscis/pharynx/salivary motor neuron (all 18 MN1-MN13 types)
   and Amulet (CB0626) readout across FlyWire v783, BANC v888, and MaleCNS v1.0 (20-200 Hz).
3. MaleCNS intact left MN9 vs bilateral mean audit across all calibration regimes
   (AUPRC, 100k seed-42 CIs, AUROC, Top-k, TOST at >0 Hz and >=1.0 Hz) + Direct local
   biasing audit (delta_l vs delta_mean from S5 Table).
4. Unified B=100,000 seed-42 paired bootstraps for S4A 16-test family (all static/available
   vectors) + unified 7-candidate BANC label-free ranking.

Writes atomically to `mn9_v7_final_controls_results.json`. Never modifies prior locked JSONs.
"""
from __future__ import annotations

import csv
import gc
import json
import os
import sys
import time
from collections import defaultdict
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
OUT_JSON = BASE_DIR / "mn9_v7_final_controls_results.json"

sys.path.insert(0, str(BASE_DIR))
import mn9_pre_submission_ab as ab  # noqa: E402
import mn9_v5_reviewer_controls as v5c  # noqa: E402


# Exact 18 proboscis / pharynx / salivary motor neuron types mapped 1:1 across all 3 connectomes
PROBOSCIS_MN_SPECS = [
    # (canonical_name, group, v783_cell_types, banc_cell_types, malecns_types)
    ("MN1", "proboscis", ["CB0720"], ["MN1"], ["MN1"]),
    ("MN2Da", "proboscis", ["CB0789"], ["MN2Da"], ["MN2Da"]),
    ("MN2Db", "proboscis", ["CB0861"], ["MN2Db"], ["MN2Db"]),
    ("MN2V", "proboscis", ["CB0783"], ["MN2V"], ["MN2V"]),
    ("MN3L", "proboscis", ["CB0875", "CB0882"], ["MN3L"], ["MN3L"]),
    ("MN3M", "proboscis", ["CB0703"], ["MN3M"], ["MN3M"]),
    ("MN4a", "proboscis", ["CB0845"], ["MN4a"], ["MN4a"]),
    ("MN4b", "proboscis", ["CB0871"], ["MN4b"], ["MN4b"]),
    ("MN5", "proboscis", ["CB0728"], ["MN5"], ["MN5"]),
    ("MN6", "proboscis", ["CB0858"], ["MN6"], ["MN6"]),
    ("MN7", "proboscis", ["CB0762"], ["MN7"], ["MN7"]),
    ("MN8", "proboscis", ["CB0911"], ["MN8"], ["MN8"]),
    ("MN9", "proboscis", ["CB0701"], ["MN9"], ["MN9"]),
    ("MN10", "pharynx", ["MN10"], ["MN10"], ["MN10"]),
    ("MN11D", "pharynx", ["CB0700"], ["MN11D"], ["MN11D"]),
    ("MN11V", "pharynx", ["CB0708"], ["MN11V"], ["MN11V"]),
    ("MN12D", "pharynx", ["CB0914", "CB0915"], ["MN12D"], ["MN12D"]),
    ("MN13", "salivary", ["CB0836"], ["MN13"], ["MN13"]),
]


def boot_auprc_vectorized(y: np.ndarray, s: np.ndarray, bidx: np.ndarray) -> np.ndarray:
    n_pos = int(np.sum(y))
    B = bidx.shape[0]
    uniq, inv = np.unique(-s, return_inverse=True)
    K = len(uniq)
    vals = np.empty(B, dtype=np.float64)
    ip = inv[bidx[:, :n_pos]]
    ineg = inv[bidx[:, n_pos:]]
    for st in range(0, B, 20000):
        en = min(st + 20000, B)
        m = en - st
        tp = np.zeros((m, K), dtype=np.int32)
        fp = np.zeros((m, K), dtype=np.int32)
        ra = np.arange(m)[:, None]
        np.add.at(tp, (ra, ip[st:en]), 1)
        np.add.at(fp, (ra, ineg[st:en]), 1)
        ct = np.cumsum(tp, axis=1)
        tot = ct + np.cumsum(fp, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = np.where(tot > 0, ct / tot, 0.0)
        vals[st:en] = np.sum((tp / float(n_pos)) * pr, axis=1)
    return vals


def make_boot_idx(y: np.ndarray, B: int = 100000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]
    return np.hstack([rng.choice(pos_i, (B, len(pos_i)), True), rng.choice(neg_i, (B, len(neg_i)), True)])


def paired_boot_vec(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, bidx: np.ndarray) -> dict:
    v1 = boot_auprc_vectorized(y, s1, bidx)
    v2 = boot_auprc_vectorized(y, s2, bidx)
    d = v1 - v2
    pt = ab.stepwise_auprc(y, s1) - ab.stepwise_auprc(y, s2)
    lo95, hi95 = np.percentile(d, [2.5, 97.5])
    lo90, hi90 = np.percentile(d, [5.0, 95.0])
    p2 = min(1.0, float(2.0 * min(np.mean(d <= 0.0), np.mean(d >= 0.0))))
    tost05 = float(max(np.mean(d <= -0.05), np.mean(d >= 0.05)))
    return {
        "delta": float(pt),
        "ci95": [float(lo95), float(hi95)],
        "ci90": [float(lo90), float(hi90)],
        "p_two_sided": p2,
        "tost_0p05_p": tost05,
        "frac_exact_zero": float(np.mean(d == 0.0)),
    }


def compute_threehop_family(
    indptr: np.ndarray,
    indices: np.ndarray,
    w_edge: np.ndarray,
    n_nodes: int,
    mn9_idx: np.ndarray,
    sez_exc: list[np.ndarray],
    y: np.ndarray,
    prior_pub: np.ndarray,
    is_direct: np.ndarray,
    alpha2: float = 0.01,
    alpha3_grid=(0.0, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1),
) -> dict:
    """Exact 1-hop, 2-hop ReLU, and 3-hop cascaded ReLU static baselines:
    h_1(c) = sum_{u in c} W(u, MN9)
    h_2,ReLU(c) = sum_k max(0, sum_{u in c} W(u, k)) * W(k, MN9)
    h_3,ReLU(c) = sum_m max(0, sum_k max(0, sum_{u in c} W(u, k)) * W(k, m)) * W(m, MN9)
    """
    m0, m1 = int(mn9_idx[0]), int(mn9_idx[1])
    w1_s = np.zeros(n_nodes, dtype=np.float64)
    w1_u = np.zeros(n_nodes, dtype=np.float64)
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

    # Precompute incoming-from-k onto MN9-presynaptic nodes m via fast sparse forward pass
    n_types = len(sez_exc)
    h1_s = np.zeros(n_types, dtype=np.float64)
    h2_s = np.zeros(n_types, dtype=np.float64)
    h3_s = np.zeros(n_types, dtype=np.float64)
    h1_u = np.zeros(n_types, dtype=np.float64)
    h2_u = np.zeros(n_types, dtype=np.float64)
    h3_u = np.zeros(n_types, dtype=np.float64)

    # Identify nodes with non-zero direct edge onto MN9 (only ~200-500 nodes!)
    mn9_pre_nodes = np.where(w1_u > 0)[0]
    mn9_pre_mask = np.zeros(n_nodes, dtype=np.bool_)
    mn9_pre_mask[mn9_pre_nodes] = True

    # Build filtered CSR keeping only edges whose target m is in mn9_pre_nodes!
    # This makes step 2->3 instantaneous (only edges into ~400 MN9-presynaptic nodes!).
    pre_to_m_dst = []
    pre_to_m_w = []
    # Vectorized mask on indices
    keep_m = mn9_pre_mask[indices]
    # Build CSR for edges into mn9_pre_nodes
    counts_k = np.zeros(n_nodes, dtype=np.int64)
    # Use reduceat on indptr? Simple loop over non-empty slices via np.searchsorted
    edge_pos = np.where(keep_m)[0]
    src_of_m = np.searchsorted(indptr[1:], edge_pos, side="right").astype(np.int32)
    dst_of_m = indices[edge_pos]
    w_of_m = w_edge[edge_pos]
    indptr_m = np.zeros(n_nodes + 1, dtype=np.int64)
    np.add.at(indptr_m[1:], src_of_m, 1)
    np.cumsum(indptr_m, out=indptr_m)

    for i, idxs in enumerate(sez_exc):
        if len(idxs) == 0:
            continue
        h1_s[i] = float(np.sum(w1_s[idxs]))
        h1_u[i] = float(np.sum(w1_u[idxs]))
        o1_s: dict[int, float] = defaultdict(float)
        o1_u: dict[int, float] = defaultdict(float)
        for u in idxs:
            p0, p1 = int(indptr[u]), int(indptr[u + 1])
            for e in range(p0, p1):
                dst = int(indices[e])
                wv = float(w_edge[e])
                o1_s[dst] += wv
                o1_u[dst] += abs(wv)

        s2 = 0.0
        u2 = 0.0
        o2_s: dict[int, float] = defaultdict(float)
        o2_u: dict[int, float] = defaultdict(float)
        for k, val in o1_s.items():
            if val > 0.0:
                if w1_s[k] != 0.0:
                    s2 += val * w1_s[k]
                q0, q1 = int(indptr_m[k]), int(indptr_m[k + 1])
                for e2 in range(q0, q1):
                    m = int(dst_of_m[e2])
                    o2_s[m] += val * float(w_of_m[e2])
        for k, val in o1_u.items():
            if val > 0.0:
                if w1_u[k] != 0.0:
                    u2 += val * w1_u[k]
                q0, q1 = int(indptr_m[k]), int(indptr_m[k + 1])
                for e2 in range(q0, q1):
                    m = int(dst_of_m[e2])
                    o2_u[m] += val * abs(float(w_of_m[e2]))

        s3 = sum(val * w1_s[m] for m, val in o2_s.items() if val > 0.0 and w1_s[m] != 0.0)
        u3 = sum(val * w1_u[m] for m, val in o2_u.items() if val > 0.0 and w1_u[m] != 0.0)
        h2_s[i], h3_s[i] = s2, s3
        h2_u[i], h3_u[i] = u2, u3

    unpub = (prior_pub == 0)
    nd = ~is_direct

    def eval_vec(sc: np.ndarray) -> dict:
        return {
            "all106_auprc": ab.stepwise_auprc(y, sc),
            "all106_auroc": ab.compute_auroc(y, sc),
            "unpub95_auprc": ab.stepwise_auprc(y[unpub], sc[unpub]),
            "nondirect_auprc": ab.stepwise_auprc(y[nd], sc[nd]),
            "topk_hits": v5c.compute_topk_hits(y, sc),
        }

    out = {"signed_relu": {}, "unsigned": {}, "alpha2": alpha2}
    best_a3, best_ap = 0.0, -1.0
    best_sc = None
    for a3 in alpha3_grid:
        sc_s = h1_s + alpha2 * h2_s + a3 * h3_s
        sc_u = h1_u + alpha2 * h2_u + a3 * h3_u
        k = f"a3_{a3:g}"
        ev_s = eval_vec(sc_s)
        out["signed_relu"][k] = ev_s
        out["unsigned"][k] = eval_vec(sc_u)
        if a3 > 0 and ev_s["all106_auprc"] > best_ap:
            best_ap = ev_s["all106_auprc"]
            best_a3 = a3
            best_sc = sc_s
    out["signed_relu"]["pure_h3"] = eval_vec(h3_s)
    out["unsigned"]["pure_h3"] = eval_vec(h3_u)
    out["best_a3_signed"] = best_a3
    out["scores_a3_1e-4_signed"] = (h1_s + alpha2 * h2_s + 1e-4 * h3_s).tolist()
    out["scores_best_signed"] = best_sc.tolist() if best_sc is not None else (h1_s + alpha2 * h2_s).tolist()
    out["scores_pure_h3_signed"] = h3_s.tolist()
    return out


def run_selk_all_mns(
    indptr: np.ndarray,
    indices: np.ndarray,
    w_edge: np.ndarray,
    w_poi_base: float,
    selk_idx: np.ndarray,
    watch_groups: dict[str, np.ndarray],
    n_nodes: int,
    use_xorshift: int = 0,
    freqs=(20, 50, 100, 200),
) -> dict:
    """Run SELK stimulation across 20-200 Hz and read out all 18 proboscis/pharynx MNs + Amulet."""
    zero_slnc = np.zeros(n_nodes, dtype=np.uint8)
    flat_watch = []
    group_slices = {}
    for gname, arr in watch_groups.items():
        st = len(flat_watch)
        flat_watch.extend(int(x) for x in arr)
        group_slices[gname] = (st, len(flat_watch))
    watch_arr = np.array(flat_watch, dtype=np.int32)

    res = {}
    for hz in freqs:
        cnt, b_hz, capped = ab.run_lif_exact_cap_slnc(
            indptr, indices, w_edge, w_poi_base, selk_idx, float(hz), zero_slnc, watch_arr, 30, 42, 15.0, use_xorshift
        )
        per_cell = cnt.mean(axis=0)
        g_rates = {}
        for gname, (st, en) in group_slices.items():
            if en > st:
                vals = [float(x) for x in per_cell[st:en]]
                g_rates[gname] = {
                    "n_cells": en - st,
                    "mean_hz": float(np.mean(vals)),
                    "max_cell_hz": float(np.max(vals)),
                    "per_cell_hz": [round(x, 4) for x in vals],
                }
            else:
                g_rates[gname] = {"n_cells": 0, "mean_hz": 0.0, "max_cell_hz": 0.0, "per_cell_hz": []}
        res[f"{hz}hz"] = {
            "brain_hz": float(b_hz),
            "capped": int(capped),
            "groups": g_rates,
            "any_non_mn9_proboscis_gt0": {
                g: g_rates[g]["mean_hz"]
                for g, grp, *_ in PROBOSCIS_MN_SPECS
                if g != "MN9" and g_rates[g]["mean_hz"] > 0.0
            },
        }
    return res


def main() -> None:
    t0 = time.time()
    meta = ab.load_common_meta()
    names, y, prior_pub = meta["names"], meta["y"], meta["prior_pub"]
    bidx106 = make_boot_idx(y, 100000, 42)
    unpub = (prior_pub == 0)
    bidx95 = make_boot_idx(y[unpub], 100000, 42)

    ab_res = json.loads((BASE_DIR / "mn9_pre_submission_ab_results.json").read_text())
    v5_res = json.loads((BASE_DIR / "mn9_v5_reviewer_controls_results.json").read_text())
    mc_res = json.loads((BASE_DIR / "mn9_malecns_results.json").read_text())
    bc_res = json.loads((BASE_DIR / "mn9_banc_results.json").read_text())
    port_res = json.loads((BASE_DIR / "mn9_portability_results.json").read_text())
    tr_res = json.loads((BASE_DIR / "mn9_transfer_results.json").read_text())

    # =========================================================================
    # PART 1: FlyWire v783 (3-hop ReLU + SELK MN1-MN13/Amulet readout)
    # =========================================================================
    print("=== PART 1: FlyWire v783 (3-hop ReLU + SELK MN1-MN13 & Amulet) ===", flush=True)
    g783, _, v783_ct_mean_sin, s_in_v783_by_fid, sez_v783_fids, mn9_783_fids = ab.load_v783_reference()
    assert isinstance(v783_ct_mean_sin, dict) and len(v783_ct_mean_sin) > 5000, "Bad v783_ct_mean_sin unpack"
    assert isinstance(s_in_v783_by_fid, dict) and len(s_in_v783_by_fid) > 100000, "Bad s_in_v783_by_fid unpack"
    idx783 = g783["flyid2idx"]
    n783 = len(g783["flyids"])
    indptr7, post7, w7 = g783["indptr"], g783["post_s"], g783["w_raw_s"].astype(np.float64)
    mn9_7 = np.array([idx783[f] for f in mn9_783_fids], dtype=np.int32)
    sez7 = [np.array([idx783[f] for f in sez_v783_fids[nm] if f in idx783], dtype=np.int32) for nm in names]
    is_dir7 = np.array([int(r["is_direct_v783"]) for r in meta["rows"]], dtype=bool)

    v783_3hop = compute_threehop_family(indptr7, post7, w7, n783, mn9_7, sez7, y, prior_pub, is_dir7)
    v783_50 = meta["v783_50"]
    bidx_nd7 = make_boot_idx(y[~is_dir7], 100000, 42)
    v783_3hop_boots = {
        "all106_vs_3hop_a3_1e-4": paired_boot_vec(y, v783_50, np.array(v783_3hop["scores_a3_1e-4_signed"]), bidx106),
        "all106_vs_3hop_best": paired_boot_vec(y, v783_50, np.array(v783_3hop["scores_best_signed"]), bidx106),
        "nondirect83_vs_3hop_a3_1e-4": paired_boot_vec(y[~is_dir7], v783_50[~is_dir7], np.array(v783_3hop["scores_a3_1e-4_signed"])[~is_dir7], bidx_nd7),
        "nondirect83_vs_3hop_best": paired_boot_vec(y[~is_dir7], v783_50[~is_dir7], np.array(v783_3hop["scores_best_signed"])[~is_dir7], bidx_nd7),
    }
    ba7 = v783_3hop["best_a3_signed"]
    print(f"  v783 3-hop best a3={ba7}: all106={v783_3hop['signed_relu'][f'a3_{ba7:g}']['all106_auprc']:.4f}, "
          f"nd83={v783_3hop['signed_relu'][f'a3_{ba7:g}']['nondirect_auprc']:.4f}, pure_h3={v783_3hop['signed_relu']['pure_h3']['all106_auprc']:.4f}", flush=True)
    for k, v in v783_3hop_boots.items():
        print(f"    boot {k}: delta={v['delta']:+.4f} CI95={v['ci95']} p={v['p_two_sided']:.4f}", flush=True)

    # Map all 18 proboscis MNs + Amulet + SELK in v783
    ann7 = CACHE_DIR / "data/Supplemental_file1_neuron_annotations.tsv"
    ct_to_idx7 = defaultdict(list)
    with ann7.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rs = row.get("root_id", "")
            if rs and rs.isdigit() and int(rs) in idx783:
                ct = (row.get("cell_type") or "").strip()
                if ct:
                    ct_to_idx7[ct].append(idx783[int(rs)])
    watch7 = {}
    for cname, _, v7_cts, _, _ in PROBOSCIS_MN_SPECS:
        hits = sorted({u for t in v7_cts for u in ct_to_idx7.get(t, [])})
        watch7[cname] = np.array(hits, dtype=np.int32)
    watch7["Amulet"] = np.array(sorted(ct_to_idx7.get("CB0626", [])), dtype=np.int32)
    selk_7 = np.array([idx783[720575940629543409], idx783[720575940630808827]], dtype=np.int32)
    selk_v783 = run_selk_all_mns(indptr7, post7, w7 * ab.W_FROZEN, ab.W_FROZEN, selk_7, watch7, n783, use_xorshift=0)
    print("  v783 SELK 50Hz Amulet:", selk_v783["50hz"]["groups"]["Amulet"], "| non-MN9 MNs >0:", selk_v783["50hz"]["any_non_mn9_proboscis_gt0"])
    print("  v783 SELK 200Hz Amulet:", selk_v783["200hz"]["groups"]["Amulet"], "| non-MN9 MNs >0:", selk_v783["200hz"]["any_non_mn9_proboscis_gt0"], flush=True)
    del g783, indptr7, post7, w7
    gc.collect()

    # =========================================================================
    # PART 2: BANC v888 (3-hop ReLU + SELK MN1-MN13 & Amulet readout)
    # =========================================================================
    print("\n=== PART 2: BANC v888 (3-hop ReLU + SELK MN1-MN13 & Amulet) ===", flush=True)
    bdir = CACHE_DIR / "data/banc"
    bm = pf.read_table(
        bdir / "banc_888_meta.feather",
        columns=["banc_888_id", "root_id", "supervoxel_id", "root_626", "root_850", "root_888", "root_890",
                 "super_class", "cell_type", "fafb_cell_type", "fafb_alignment_cell_type", "side", "proofread", "roughly_proofread"],
    ).to_pydict()
    sv_to_bid, old_root_to_bid, bid_to_cts, ct_to_bids_b = {}, {}, {}, defaultdict(list)
    main_cand_set = set()
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
        for col in ("fafb_cell_type", "fafb_alignment_cell_type", "cell_type"):
            toks |= ab.split_tokens(bm[col][i])
        if toks:
            bid_to_cts[bid] = toks
        ct_raw = str(bm["cell_type"][i] or "").strip()
        if ct_raw:
            ct_to_bids_b[ct_raw].append(bid)
        fct_raw = str(bm["fafb_cell_type"][i] or "").strip()
        if fct_raw and fct_raw != ct_raw:
            ct_to_bids_b[fct_raw].append(bid)

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
    node_ids_b = np.union1d(ep_b[keep_b], eq_b[keep_b]).astype(np.int64)
    del ep_b, eq_b, ns_b, keep_b, v_nt, p_nt, nt_ids_b, nt_sgn_b, main_cand_b, nt_csv

    gb = np.load(CACHE_DIR / "banc_ckpt/banc_main_graph.npz")
    assert len(node_ids_b) == int(gb["n_nodes"])
    n_b = len(node_ids_b)
    bid2idx_b = {int(b): i for i, b in enumerate(node_ids_b)}
    indptr_b = gb["indptr"].astype(np.int64)
    indices_b = gb["post_s"].astype(np.int32)
    w_b = gb["w_raw_s"].astype(np.float64)
    s_in_b = np.zeros(n_b, dtype=np.float64)
    np.add.at(s_in_b, indices_b, np.abs(w_b))

    mn9_b = np.array([bid2idx_b[720575941410682479], bid2idx_b[720575941623285450]], dtype=np.int32)
    banc_m1_by_name = {r["name"]: [int(b) for b in r["m1_banc_ids"]] for r in bc_res["mapping_audit"]["per_type_sez106"]}
    sez_b = [np.array([bid2idx_b[b] for b in banc_m1_by_name[nm] if b in bid2idx_b], dtype=np.int32) for nm in names]
    is_dir_b = np.array([int(r["is_direct_banc_m1"]) for r in meta["rows"]], dtype=bool)

    # Rebuild exact w_post_1to1 and w_post_ct (matching mn9_v5_reviewer_controls.py verbatim!)
    import gzip
    banc_to_fafb_sins = defaultdict(list)
    rm_path = bdir / "banc_fafb_reviewed_matches.csv.gz"
    is_gz = rm_path.read_bytes()[:2] == b"\x1f\x8b"
    opener = gzip.open if is_gz else open
    with opener(rm_path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("valid") != "t":
                continue
            try:
                bid_raw, fid = int(r["pt_root_id"]), int(r["match_id"])
            except Exception:
                continue
            sv_s = str(r.get("pt_supervoxel_id") or "")
            if bid_raw in bid2idx_b:
                bid = bid_raw
            elif sv_s.isdigit() and int(sv_s) in sv_to_bid and sv_to_bid[int(sv_s)] in bid2idx_b:
                bid = sv_to_bid[int(sv_s)]
            elif bid_raw in old_root_to_bid and old_root_to_bid[bid_raw] in bid2idx_b:
                bid = old_root_to_bid[bid_raw]
            else:
                continue
            if fid in s_in_v783_by_fid and s_in_v783_by_fid[fid] > 0:
                banc_to_fafb_sins[bid2idx_b[bid]].append(s_in_v783_by_fid[fid])
    assert len(banc_to_fafb_sins) > 50000, f"Expected >50k BANC 1:1 reviewed matches, got {len(banc_to_fafb_sins)}"

    w_post_b_1to1 = np.full(n_b, ab.W_BANC_DENS, dtype=np.float64)
    for j, f_sins in banc_to_fafb_sins.items():
        if s_in_b[j] > 0:
            w_post_b_1to1[j] = ab.W_FROZEN * min(max(float(np.mean(f_sins)) / float(s_in_b[j]), 0.1), 10.0)
    w_post_b_ct = w_post_b_1to1.copy()
    for j, bid in enumerate(node_ids_b):
        if s_in_b[j] <= 0 or j in banc_to_fafb_sins:
            continue
        toks = bid_to_cts.get(int(bid), ())
        refs = [v783_ct_mean_sin[t] for t in toks if t in v783_ct_mean_sin and v783_ct_mean_sin[t] > 0]
        if refs:
            w_post_b_ct[j] = ab.W_FROZEN * min(max(float(np.mean(refs)) / float(s_in_b[j]), 0.1), 10.0)

    banc_3hop = {
        "raw": compute_threehop_family(indptr_b, indices_b, w_b, n_b, mn9_b, sez_b, y, prior_pub, is_dir_b),
        "homolog_1to1_ct_weighted": compute_threehop_family(indptr_b, indices_b, w_b * (w_post_b_ct[indices_b] / ab.W_FROZEN), n_b, mn9_b, sez_b, y, prior_pub, is_dir_b),
    }
    _chk_b2 = banc_3hop["homolog_1to1_ct_weighted"]["signed_relu"]["a3_0"]["all106_auprc"]
    assert abs(_chk_b2 - 0.5969) < 1e-3, f"BANC 1:1+CT a3=0 mismatch vs v5 (0.5969): got {_chk_b2:.4f}"
    banc_lif_ct = np.array(v5_res["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["50hz"]["per_type_mean_hz"])
    banc_3hop_boots = {
        "BANC_HomCT_LIF_vs_Fair3hop_a3_1e-4": paired_boot_vec(y, banc_lif_ct, np.array(banc_3hop["homolog_1to1_ct_weighted"]["scores_a3_1e-4_signed"]), bidx106),
        "BANC_HomCT_LIF_vs_Fair3hop_best": paired_boot_vec(y, banc_lif_ct, np.array(banc_3hop["homolog_1to1_ct_weighted"]["scores_best_signed"]), bidx106),
    }
    for k, v in banc_3hop.items():
        ba = v["best_a3_signed"]
        print(f"  BANC 3-hop [{k}] best a3={ba}: all106={v['signed_relu'][f'a3_{ba:g}']['all106_auprc']:.4f}, pure_h3={v['signed_relu']['pure_h3']['all106_auprc']:.4f}", flush=True)
    for k, v in banc_3hop_boots.items():
        print(f"    boot {k}: delta={v['delta']:+.4f} CI95={v['ci95']} p={v['p_two_sided']:.4f}", flush=True)

    # SELK MN1-MN13 + Amulet in BANC (target 0.6904 mV & homolog 1:1+CT)
    watch_b = {}
    for cname, _, _, b_cts, _ in PROBOSCIS_MN_SPECS:
        hits = sorted({bid2idx_b[b] for t in b_cts for b in ct_to_bids_b.get(t, []) if b in bid2idx_b})
        watch_b[cname] = np.array(hits, dtype=np.int32)
    watch_b["Amulet"] = np.array([bid2idx_b[b] for b in banc_m1_by_name["amulet"] if b in bid2idx_b], dtype=np.int32)
    selk_b_ids = bc_res.get("crosslab_audit_banc", {}).get("SELK_DNg68", {}).get("banc_ids", [])
    if not selk_b_ids:
        selk_b_ids = ct_to_bids_b.get("DNg68", [])
    selk_b = np.array(sorted({bid2idx_b[int(b)] for b in selk_b_ids if int(b) in bid2idx_b}), dtype=np.int32)
    assert len(selk_b) > 0, "Failed to locate BANC SELK (DNg68) cells!"
    selk_banc_target = run_selk_all_mns(indptr_b, indices_b, w_b * ab.W_BANC_ANAT, ab.W_BANC_ANAT, selk_b, watch_b, n_b, use_xorshift=0)
    selk_banc_homct = run_selk_all_mns(indptr_b, indices_b, w_b * w_post_b_ct[indices_b], ab.W_FROZEN, selk_b, watch_b, n_b, use_xorshift=0)
    print("  BANC target SELK 50Hz Amulet:", selk_banc_target["50hz"]["groups"]["Amulet"], "| non-MN9 MNs >0:", selk_banc_target["50hz"]["any_non_mn9_proboscis_gt0"])
    print("  BANC target SELK 200Hz Amulet:", selk_banc_target["200hz"]["groups"]["Amulet"], "| non-MN9 MNs >0:", selk_banc_target["200hz"]["any_non_mn9_proboscis_gt0"])
    print("  BANC homCT SELK 50Hz Amulet:", selk_banc_homct["50hz"]["groups"]["Amulet"], "| non-MN9 MNs >0:", selk_banc_homct["50hz"]["any_non_mn9_proboscis_gt0"])
    print("  BANC homCT SELK 200Hz Amulet:", selk_banc_homct["200hz"]["groups"]["Amulet"], "| non-MN9 MNs >0:", selk_banc_homct["200hz"]["any_non_mn9_proboscis_gt0"], flush=True)
    del gb, indptr_b, indices_b, w_b, bm
    gc.collect()

    # =========================================================================
    # PART 3: MaleCNS v1.0 (3-hop ReLU + SELK MN1-MN13 & Amulet readout)
    # =========================================================================
    print("\n=== PART 3: MaleCNS v1.0 (3-hop ReLU + SELK MN1-MN13 & Amulet) ===", flush=True)
    mdir = CACHE_DIR / "data/malecns"
    ann_tbl = pf.read_table(mdir / "body-annotations-male-cns-v1.0-minconf-0.5.feather")
    nt_tbl = pf.read_table(mdir / "body-neurotransmitters-male-cns-v1.0.feather")
    w_tbl = pf.read_table(mdir / "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather")

    cons_nt_map, pred_nt_map = {}, {}
    for b, c_nt, p_nt in zip(nt_tbl["body"].to_pylist(), nt_tbl["consensus_nt"].to_pylist(), nt_tbl["predicted_nt"].to_pylist()):
        if b is None:
            continue
        cons_nt_map[int(b)] = str(c_nt).strip().lower() if c_nt is not None else ""
        pred_nt_map[int(b)] = str(p_nt).strip().lower() if p_nt is not None else ""

    def s_str(x) -> str:
        return "" if x is None else str(x).strip()

    traced_meta = {}
    for b, typ, fw, sc, cl, st in zip(
        ann_tbl["bodyId"].to_pylist(), ann_tbl["type"].to_pylist(), ann_tbl["flywireType"].to_pylist(),
        ann_tbl["superclass"].to_pylist(), ann_tbl["class"].to_pylist(), ann_tbl["status"].to_pylist(),
    ):
        if b is None:
            continue
        st_s, sc_s, cl_s = s_str(st), s_str(sc), s_str(cl)
        if st_s != "Traced" or "glia" in sc_s.lower() or "glia" in cl_s.lower():
            continue
        bi = int(b)
        traced_meta[bi] = {"id": bi, "type": s_str(typ), "flywireType": s_str(fw),
                           "cons_nt": cons_nt_map.get(bi, ""), "pred_nt": pred_nt_map.get(bi, "")}
    del ann_tbl, nt_tbl

    all_bids = np.array(sorted(traced_meta.keys()), dtype=np.int64)
    n_m = len(all_bids)
    bid2idx_m = {int(b): i for i, b in enumerate(all_bids)}
    node_meta_m = [traced_meta[int(b)] for b in all_bids]
    mn9_l_m, mn9_r_m = bid2idx_m[10331], bid2idx_m[16949]
    mn9_m = np.array([mn9_l_m, mn9_r_m], dtype=np.int32)

    raw_pre = w_tbl["body_pre"].to_numpy().astype(np.int64)
    raw_post = w_tbl["body_post"].to_numpy().astype(np.int64)
    raw_w = w_tbl["weight"].to_numpy().astype(np.int32)
    del w_tbl

    def map_vec(arr):
        pos = np.minimum(np.searchsorted(all_bids, arr), n_m - 1)
        return np.where(all_bids[pos] == arr, pos, -1).astype(np.int32)

    p_i, q_i = map_vec(raw_pre), map_vec(raw_post)
    ok = (p_i >= 0) & (q_i >= 0) & (p_i != q_i) & (raw_w > 0)
    p_i, q_i, raw_w = p_i[ok], q_i[ok], raw_w[ok]
    del raw_pre, raw_post
    keys = p_i.astype(np.int64) * np.int64(n_m) + q_i.astype(np.int64)
    ord_k = np.argsort(keys, kind="mergesort")
    keys_s, w_s = keys[ord_k], raw_w[ord_k]
    del keys, ord_k, p_i, q_i, raw_w
    uniq_k, first_pos = np.unique(keys_s, return_index=True)
    w_uniq = np.add.reduceat(w_s.astype(np.int64), first_pos).astype(np.int32) if len(uniq_k) < len(keys_s) else w_s.astype(np.int32)
    u_pre = (uniq_k // np.int64(n_m)).astype(np.int32)
    u_post = (uniq_k % np.int64(n_m)).astype(np.int32)
    del keys_s, w_s, first_pos, uniq_k

    def build_csr(signs):
        es = signs[u_pre]
        keep = es != 0
        pre_s, post_s = u_pre[keep], u_post[keep]
        w_raw_s = (w_uniq[keep] * es[keep]).astype(np.float64)
        indptr = np.zeros(n_m + 1, dtype=np.int64)
        np.add.at(indptr[1:], pre_s, 1)
        np.cumsum(indptr, out=indptr)
        s_in = np.zeros(n_m, dtype=np.float64)
        np.add.at(s_in, post_s, np.abs(w_raw_s))
        return indptr, post_s, w_raw_s, s_in

    base_signs = np.array([ab.nt_to_sign(m["cons_nt"]) for m in node_meta_m], dtype=np.int8)
    imp_signs = np.array([ab.nt_to_sign(m["cons_nt"]) if ab.nt_to_sign(m["cons_nt"]) != 0 else ab.nt_to_sign(m["pred_nt"]) for m in node_meta_m], dtype=np.int8)
    indptr0, post0, w0, sin0 = build_csr(base_signs)
    indptr_imp, post_imp, w_imp, sin_imp = build_csr(imp_signs)
    del u_pre, u_post, w_uniq
    gc.collect()

    mc_bids_by_name = {r["name"]: [int(b) for b in r["malecns_body_ids"]] for r in mc_res["mapping_audit"]["per_type_sez106"]}
    sez_m = [np.array([bid2idx_m[b] for b in mc_bids_by_name[nm] if b in bid2idx_m], dtype=np.int32) for nm in names]
    is_dir_m0 = np.array([int(r["is_direct_malecns"]) for r in meta["rows"]], dtype=bool)
    direct_imp = {u for u in range(n_m) if np.any((post_imp[indptr_imp[u]:indptr_imp[u+1]] == mn9_l_m) | (post_imp[indptr_imp[u]:indptr_imp[u+1]] == mn9_r_m))}
    is_dir_mi = np.array([any(int(u) in direct_imp for u in sez_m[i]) for i in range(len(names))], dtype=bool)

    def build_male_hom_w(sin_arr, fallback_w, mn9_l_w):
        wp = np.full(n_m, fallback_w, dtype=np.float64)
        for j, m in enumerate(node_meta_m):
            if sin_arr[j] <= 0:
                continue
            toks = ab.split_tokens(m["flywireType"]) | ab.split_tokens(m["type"])
            refs = [v783_ct_mean_sin[t] for t in toks if t in v783_ct_mean_sin and v783_ct_mean_sin[t] > 0]
            if refs:
                wp[j] = ab.W_FROZEN * min(max(float(np.mean(refs)) / float(sin_arr[j]), 0.1), 10.0)
        wp[mn9_l_m] = mn9_l_w
        wp[mn9_r_m] = mn9_l_w
        return wp

    wp_cons_m = build_male_hom_w(sin0, ab.W_MALE_DENS, ab.W_MALE_INTACT)
    wp_imp_m = build_male_hom_w(sin_imp, ab.W_MALE_JOINT, ab.W_MALE_PREDNT)

    male_3hop = {
        "consensus_nt_raw": compute_threehop_family(indptr0, post0, w0, n_m, mn9_m, sez_m, y, prior_pub, is_dir_m0),
        "predicted_nt_raw": compute_threehop_family(indptr_imp, post_imp, w_imp, n_m, mn9_m, sez_m, y, prior_pub, is_dir_mi),
        "consensus_nt_homolog_weighted": compute_threehop_family(indptr0, post0, w0 * (wp_cons_m[post0] / ab.W_FROZEN), n_m, mn9_m, sez_m, y, prior_pub, is_dir_m0),
        "predicted_nt_homolog_weighted": compute_threehop_family(indptr_imp, post_imp, w_imp * (wp_imp_m[post_imp] / ab.W_FROZEN), n_m, mn9_m, sez_m, y, prior_pub, is_dir_mi),
    }
    # Also check raw-mV two-hop scaling sensitivity (C * w_post vs C * (w_post / 0.275))
    male_2hop_raw_mv = v5c.compute_twohop_family(indptr_imp, post_imp, w_imp * wp_imp_m[post_imp], n_m, mn9_m, sez_m, y, prior_pub, is_dir_mi)

    lif_m_hom_mean = np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"])
    lif_m_hom_left = np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_left_hz"])
    male_3hop_boots = {
        "Male_HomPred_LIF_mean_vs_Fair3hop_a3_1e-4": paired_boot_vec(y, lif_m_hom_mean, np.array(male_3hop["predicted_nt_homolog_weighted"]["scores_a3_1e-4_signed"]), bidx106),
        "Male_HomPred_LIF_mean_vs_Fair3hop_best": paired_boot_vec(y, lif_m_hom_mean, np.array(male_3hop["predicted_nt_homolog_weighted"]["scores_best_signed"]), bidx106),
        "Male_HomPred_LIF_left_vs_Fair3hop_a3_1e-4": paired_boot_vec(y, lif_m_hom_left, np.array(male_3hop["predicted_nt_homolog_weighted"]["scores_a3_1e-4_signed"]), bidx106),
        "Male_HomPred_LIF_left_vs_Fair3hop_best": paired_boot_vec(y, lif_m_hom_left, np.array(male_3hop["predicted_nt_homolog_weighted"]["scores_best_signed"]), bidx106),
        "Male_HomPred_LIF_left_vs_Fair2hop_signed": paired_boot_vec(y, lif_m_hom_left, np.array(v5_res["phase2_malecns"]["m4_malecns_fair_twohop"]["predicted_nt_homolog_weighted"]["signed_relu_scores_0p01"]), bidx106),
        "Male_HomPred_LIF_mean_vs_Fair2hop_raw_mV_signed": paired_boot_vec(y, lif_m_hom_mean, np.array(male_2hop_raw_mv["signed_relu_scores_0p01"]), bidx106),
    }
    for k, v in male_3hop.items():
        ba = v["best_a3_signed"]
        print(f"  MaleCNS 3-hop [{k}] best a3={ba}: all106={v['signed_relu'][f'a3_{ba:g}']['all106_auprc']:.4f}, pure_h3={v['signed_relu']['pure_h3']['all106_auprc']:.4f}", flush=True)
    print(f"  MaleCNS 2-hop raw-mV (C*w_post, alpha=0.01): {male_2hop_raw_mv['signed_relu']['alpha_0.01']['all106_auprc']:.4f} vs dimensionless (C*w_post/0.275): 0.5452", flush=True)
    for k, v in male_3hop_boots.items():
        print(f"    boot {k}: delta={v['delta']:+.4f} CI95={v['ci95']} p={v['p_two_sided']:.4f}", flush=True)

    # SELK MN1-MN13 + Amulet in MaleCNS (target 0.2683 mV & homolog + pred_nt)
    ct_to_idx_m = defaultdict(list)
    for i, m in enumerate(node_meta_m):
        if m["type"]:
            ct_to_idx_m[m["type"]].append(i)
    watch_m = {}
    for cname, _, _, _, m_cts in PROBOSCIS_MN_SPECS:
        hits = sorted({u for t in m_cts for u in ct_to_idx_m.get(t, [])})
        watch_m[cname] = np.array(hits, dtype=np.int32)
    watch_m["Amulet"] = np.array([bid2idx_m[b] for b in mc_bids_by_name["amulet"] if b in bid2idx_m], dtype=np.int32)
    selk_m = np.array([bid2idx_m[c["id"]] for c in mc_res["selk_full_cns_diagnosis"]["selk_cells"] if c["id"] in bid2idx_m], dtype=np.int32)

    selk_male_target = run_selk_all_mns(indptr0, post0, w0 * ab.W_MALE_INTACT, ab.W_MALE_INTACT, selk_m, watch_m, n_m, use_xorshift=1)
    selk_male_hompred = run_selk_all_mns(indptr_imp, post_imp, w_imp * wp_imp_m[post_imp], ab.W_FROZEN, selk_m, watch_m, n_m, use_xorshift=1)
    print("  MaleCNS target SELK 50Hz Amulet:", selk_male_target["50hz"]["groups"]["Amulet"], "| non-MN9 MNs >0:", selk_male_target["50hz"]["any_non_mn9_proboscis_gt0"])
    print("  MaleCNS target SELK 200Hz Amulet:", selk_male_target["200hz"]["groups"]["Amulet"], "| non-MN9 MNs >0:", selk_male_target["200hz"]["any_non_mn9_proboscis_gt0"])
    print("  MaleCNS homPred SELK 50Hz Amulet:", selk_male_hompred["50hz"]["groups"]["Amulet"], "| non-MN9 MNs >0:", selk_male_hompred["50hz"]["any_non_mn9_proboscis_gt0"])
    print("  MaleCNS homPred SELK 200Hz Amulet:", selk_male_hompred["200hz"]["groups"]["Amulet"], "| non-MN9 MNs >0:", selk_male_hompred["200hz"]["any_non_mn9_proboscis_gt0"], flush=True)
    del indptr0, post0, w0, indptr_imp, post_imp, w_imp
    gc.collect()

    # =========================================================================
    # PART 4: MaleCNS Intact Left MN9 vs Bilateral Mean Full Audit (B=100,000)
    # =========================================================================
    print("\n=== PART 4: MaleCNS Intact Left MN9 vs Bilateral Mean Audit (B=100,000) ===", flush=True)
    pt3 = mc_res["per_type_3brains"]
    v783_50_arr = np.array([float(r["v783_50hz"]) for r in pt3])
    v783_200_arr = np.array([float(r["v783_200hz"]) for r in pt3])
    c10_mask = (v783_50_arr > 0) & (y == 1)

    def summarize_lr(label: str, hz: int, s_mean: np.ndarray, s_left: np.ndarray | None, nd_mask: np.ndarray) -> dict:
        ref = v783_50_arr if hz == 50 else v783_200_arr
        def _one(s):
            bv = boot_auprc_vectorized(y, s, bidx106)
            lo, hi = np.percentile(bv, [2.5, 97.5])
            tost = paired_boot_vec(y, s, ref, bidx106) if hz == 50 else None
            tost_unpub = paired_boot_vec(y[unpub], s[unpub], ref[unpub], bidx95) if hz == 50 else None
            s10 = np.where(s >= 1.0, s, 0.0)
            r10 = np.where(ref >= 1.0, ref, 0.0)
            tost_ge1 = paired_boot_vec(y, s10, r10, bidx106) if hz == 50 else None
            ab_bin, bb_bin = (s > 0).astype(int), (ref > 0).astype(int)
            po = float((ab_bin == bb_bin).mean())
            pe = float(ab_bin.mean() * bb_bin.mean() + (1.0 - ab_bin.mean()) * (1.0 - bb_bin.mean()))
            kap = (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0
            return {
                "all106_auprc": ab.stepwise_auprc(y, s),
                "ci95": [float(lo), float(hi)],
                "auroc": ab.compute_auroc(y, s),
                "topk": v5c.compute_topk_hits(y, s),
                "unpub95_auprc": ab.stepwise_auprc(y[unpub], s[unpub]),
                "nondirect_auprc": ab.stepwise_auprc(y[nd_mask], s[nd_mask]),
                "active_pos": int(((s > 0) & (y == 1)).sum()),
                "active_neg": int(((s > 0) & (y == 0)).sum()),
                "auprc_ge1": ab.stepwise_auprc(y, s10),
                "spearman_v783": float(stats.spearmanr(s, ref).statistic),
                "kappa_v783": float(kap),
                "common10_rho": float(stats.spearmanr(s[c10_mask], ref[c10_mask]).statistic) if hz == 50 else None,
                "vs_v783_gt0": tost,
                "vs_v783_unpub95": tost_unpub,
                "vs_v783_ge1": tost_ge1,
            }
        out = {"hz": hz, "bilateral_mean": _one(s_mean)}
        if s_left is not None and len(s_left) == 106:
            out["intact_left"] = _one(s_left)
        return out

    lr_regimes = [
        ("target_scaled_50hz", 50, np.array([float(r["malecns_intact_50hz_mean"]) for r in pt3]), np.array([float(r["malecns_intact_50hz_l"]) for r in pt3]), ~is_dir_m0),
        ("pred_nt_fixed_50hz", 50, np.array(ab_res["phase_malecns"]["experiment_b2_fixed_02683_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"]), np.array(ab_res["phase_malecns"]["experiment_b2_fixed_02683_prednt"]["sweeps"]["50hz"]["per_type_left_hz"]), ~is_dir_mi),
        ("sugar_r17_prednt_50hz", 50, np.array(ab_res["phase_malecns"]["a1_sugar_anchor_prednt"]["sweeps_at_right17_wstar"]["50hz"]["per_type_mean_hz"]), np.array(ab_res["phase_malecns"]["a1_sugar_anchor_prednt"]["sweeps_at_right17_wstar"]["50hz"]["per_type_left_hz"]), ~is_dir_mi),
        ("homolog_cons_50hz", 50, np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_cons"]["sweeps"]["50hz"]["per_type_mean_hz"]), np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_cons"]["sweeps"]["50hz"]["per_type_left_hz"]), ~is_dir_m0),
        ("homolog_prednt_50hz", 50, np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"]), np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_left_hz"]), ~is_dir_mi),
        ("homolog_prednt_200hz", 200, np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["200hz"]["per_type_mean_hz"]), np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["200hz"]["per_type_left_hz"]), ~is_dir_mi),
        ("leave_out_sez106_mn9_50hz", 50, np.array(v5_res["phase2_malecns"]["m2b_leave_out_sez106_and_mn9"]["per_type_mean_hz"]), np.array(v5_res["phase2_malecns"]["m2b_leave_out_sez106_and_mn9"]["per_type_left_hz"]), ~is_dir_mi),
        ("two_scale_50hz", 50, np.array(v5_res["phase2_malecns"]["m2b_two_scale_mn9_02638_rest_01567"]["per_type_mean_hz"]), np.array(v5_res["phase2_malecns"]["m2b_two_scale_mn9_02638_rest_01567"]["per_type_left_hz"]), ~is_dir_mi),
        ("banc_ref_50hz", 50, np.array(v5_res["phase2_malecns"]["m2c_banc_ref_to_malecns"]["per_type_mean_hz"]), np.array(v5_res["phase2_malecns"]["m2c_banc_ref_to_malecns"]["per_type_left_hz"]), ~is_dir_mi),
    ]
    malecns_lr_audit = {}
    for nm, hz, sm, sl, ndm in lr_regimes:
        malecns_lr_audit[nm] = summarize_lr(nm, hz, sm, sl, ndm)
        bm_d = malecns_lr_audit[nm]["bilateral_mean"]
        il_d = malecns_lr_audit[nm]["intact_left"]
        t_m = bm_d["vs_v783_gt0"]["tost_0p05_p"] if bm_d["vs_v783_gt0"] else float("nan")
        t_l = il_d["vs_v783_gt0"]["tost_0p05_p"] if il_d["vs_v783_gt0"] else float("nan")
        print(f"  {nm:26s} | mean={bm_d['all106_auprc']:.4f} (TOST={t_m:.4f}) | intact_L={il_d['all106_auprc']:.4f} { [round(x,3) for x in il_d['ci95']] } (TOST={t_l:.4f}, unpub95={il_d['unpub95_auprc']:.4f})", flush=True)

    # Also check paired delta (Homolog+predNT intact L minus Target intact L)
    l_hom = np.array(ab_res["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_left_hz"])
    l_tgt = np.array([float(r["malecns_intact_50hz_l"]) for r in pt3])
    malecns_lr_audit["homolog_prednt_vs_target_intact_left"] = paired_boot_vec(y, l_hom, l_tgt, bidx106)
    print("  MaleCNS intact-L Hom+NT vs Target:", malecns_lr_audit["homolog_prednt_vs_target_intact_left"], flush=True)

    # =========================================================================
    # PART 5: MaleCNS Direct Local Biasing Intact Left MN9 Audit (from S5_Table / disinhibition JSON)
    # =========================================================================
    print("\n=== PART 5: MaleCNS Local Biasing Intact Left MN9 vs Bilateral Mean Audit ===", flush=True)
    dis_res = json.loads((BASE_DIR / "mn9_disinhibition_results.json").read_text())
    s5_candidates = [
        BASE_DIR / "english_manuscript/supp_table2_102_local_oracles.tsv",
        BASE_DIR.parent / "data/S5_Table.tsv",
        BASE_DIR / "data/S5_Table.tsv",
    ]
    s5_path = next((p for p in s5_candidates if p.exists()), None)
    assert s5_path is not None, f"Could not locate S5_Table.tsv in {s5_candidates}"
    direct_audit = defaultdict(lambda: {
        "n_pts": 0,
        "mean_50_sig": 0, "left_50_sig": 0, "left_50_ge1_only": 0,
        "mean_200_sig": 0, "left_200_sig": 0, "left_200_ge1_only": 0,
        "rows": [],
    })
    with s5_path.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if "male" not in r["brain"].lower():
                continue
            cfg = r["oracle_config"]
            is_direct_cfg = "direct" in cfg.lower() or "dir_" in cfg.lower() or ("dir" in cfg.lower() and "pre" not in cfg.lower())
            is_pre_cfg = "pre" in cfg.lower()
            kind = "Direct" if is_direct_cfg else ("Premotor" if is_pre_cfg else "Other")
            key = f"{r['candidate']}__{kind}"
            d50m, d50l, p50 = float(r["delta_50hz_mean"]), float(r["delta_50hz_l"]), float(r["p_50hz"])
            d200m, d200l, p200 = float(r["delta_200hz_mean"]), float(r["delta_200hz_l"]), float(r["p_200hz"])
            st = direct_audit[key]
            st["n_pts"] += 1
            st["mean_50_sig"] += int(d50m >= 1.0 and p50 < 0.01)
            st["left_50_sig"] += int(d50l >= 1.0 and p50 < 0.01)
            st["left_50_ge1_only"] += int(d50l >= 1.0)
            st["mean_200_sig"] += int(d200m >= 1.0 and p200 < 0.01)
            st["left_200_sig"] += int(d200l >= 1.0 and p200 < 0.01)
            st["left_200_ge1_only"] += int(d200l >= 1.0)
            st["rows"].append({
                "cfg": cfg, "bgL": float(r["bg_mn9_l_hz"]), "bgR": float(r["bg_mn9_r_hz"]),
                "d50m": d50m, "d50l": d50l, "p50": p50,
                "d200m": d200m, "d200l": d200l, "p200": p200,
            })
    for k, v in sorted(direct_audit.items()):
        print(
            f"  {k:24s} (n={v['n_pts']:2d}): 50Hz mean(sig)={v['mean_50_sig']}/{v['n_pts']}, left(sig)={v['left_50_sig']}/{v['n_pts']} (ge1={v['left_50_ge1_only']}) | "
            f"200Hz mean(sig)={v['mean_200_sig']}/{v['n_pts']}, left(sig)={v['left_200_sig']}/{v['n_pts']} (ge1={v['left_200_ge1_only']})",
            flush=True,
        )

    # =========================================================================
    # PART 6: Recompute ALL static-baseline S4A rows at B=100,000 seed=42
    # =========================================================================
    print("\n=== PART 6: S4A Static-Baseline 100k Paired Bootstraps & Common-75 check ===", flush=True)
    v7_2h = np.array(v5_res["phase3_v783_k100"]["m4_v783_twohop_alpha_sweep"]["signed_relu_scores_0p01"])
    b_2h = np.array(v5_res["phase1_banc"]["m4_banc_fair_twohop"]["raw"]["signed_relu_scores_0p01"])
    m_2h = np.array(v5_res["phase2_malecns"]["m4_malecns_fair_twohop"]["consensus_nt_raw"]["signed_relu_scores_0p01"])
    b_50 = np.array([float(r["banc_anat_50hz"]) for r in pt3])
    m_50 = np.array([float(r["malecns_intact_50hz_mean"]) for r in pt3])
    nd_b = ~is_dir_b
    nd_m = ~is_dir_m0
    nd_c75 = (~is_dir7) & (~is_dir_b) & (~is_dir_m0)

    s4a_100k_static = {
        "v783_all106_vs_2hop": paired_boot_vec(y, v783_50_arr, v7_2h, bidx106),
        "v783_nd83_vs_2hop": paired_boot_vec(y[~is_dir7], v783_50_arr[~is_dir7], v7_2h[~is_dir7], make_boot_idx(y[~is_dir7], 100000, 42)),
        "banc_all106_vs_2hop": paired_boot_vec(y, b_50, b_2h, bidx106),
        "banc_nd90_vs_2hop": paired_boot_vec(y[nd_b], b_50[nd_b], b_2h[nd_b], make_boot_idx(y[nd_b], 100000, 42)),
        "malecns_all106_vs_2hop": paired_boot_vec(y, m_50, m_2h, bidx106),
        "malecns_nd88_vs_2hop": paired_boot_vec(y[nd_m], m_50[nd_m], m_2h[nd_m], make_boot_idx(y[nd_m], 100000, 42)),
        "malecns_common75_vs_2hop": paired_boot_vec(y[nd_c75], m_50[nd_c75], m_2h[nd_c75], make_boot_idx(y[nd_c75], 100000, 42)),
    }
    for k, v in s4a_100k_static.items():
        print(f"  S4A-100k {k:28s}: delta={v['delta']:+.4f} CI95={[round(x,4) for x in v['ci95']]} p={v['p_two_sided']:.4f}", flush=True)

    payload = {
        "protocol_note": "V7 final controls: 3-hop ReLU, SELK MN1-MN13/Amulet, MaleCNS intact-left readout & Direct-bias audit, 100k S4A static checks",
        "v783_threehop": v783_3hop,
        "v783_threehop_boots_100k": v783_3hop_boots,
        "banc_threehop": banc_3hop,
        "banc_threehop_boots_100k": banc_3hop_boots,
        "malecns_threehop": male_3hop,
        "malecns_twohop_raw_mv_sensitivity": male_2hop_raw_mv,
        "malecns_threehop_boots_100k": male_3hop_boots,
        "selk_proboscis_mns_and_amulet": {
            "v783_frozen": selk_v783,
            "banc_target": selk_banc_target,
            "banc_homolog_1to1_ct": selk_banc_homct,
            "malecns_target": selk_male_target,
            "malecns_homolog_prednt": selk_male_hompred,
        },
        "malecns_intact_left_vs_bilateral_audit_100k": malecns_lr_audit,
        "malecns_local_biasing_left_vs_mean_audit": dict(direct_audit),
        "s4a_static_100k": s4a_100k_static,
        "elapsed_sec": float(time.time() - t0),
    }
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(OUT_JSON)
    print(f"\nDONE! Saved {OUT_JSON} in {payload['elapsed_sec']:.1f} s.", flush=True)


if __name__ == "__main__":
    main()
