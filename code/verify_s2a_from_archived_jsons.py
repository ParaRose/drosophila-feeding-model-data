#!/usr/bin/env python3
"""Verify all 24 S2A rows and 8 threshold/denominator rows directly from archived JSONs.

Run from the root of the public `drosophila-feeding-model-data` repository (v6.0):
    python code/verify_s2a_from_archived_jsons.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def stepwise_auprc(y: np.ndarray, s: np.ndarray) -> float:
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = 0
    ap = 0.0
    prev_r = 0.0
    i, n = 0, len(ys)
    while i < n:
        j = i + 1
        while j < n and ss[j] == ss[i]:
            j += 1
        tp += int(np.sum(ys[i:j]))
        prec = tp / float(j)
        rec = tp / float(n_pos)
        ap += (rec - prev_r) * prec
        prev_r = rec
        i = j
    return float(ap)


def auroc_midrank(y: np.ndarray, s: np.ndarray) -> float:
    pos, neg = s[y == 1], s[y == 0]
    return float(np.mean([(p > neg).mean() + 0.5 * (p == neg).mean() for p in pos]))


def topk_tie_avg(y: np.ndarray, s: np.ndarray, k: int) -> float:
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    kth = ss[k - 1]
    above = yy[ss > kth]
    tied = yy[ss == kth]
    return float(above.sum() + (k - len(above)) * tied.mean())


def cohen_kappa_gt0(a: np.ndarray, b: np.ndarray) -> float:
    ab, bb = (a > 0).astype(int), (b > 0).astype(int)
    po = float((ab == bb).mean())
    pe = float(ab.mean() * bb.mean() + (1.0 - ab.mean()) * (1.0 - bb.mean()))
    return (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent if (here.parent / "results").is_dir() else here
    res = root / "results"
    banc_d = json.loads((res / "mn9_banc_results.json").read_text(encoding="utf-8"))
    mc_d = json.loads((res / "mn9_malecns_results.json").read_text(encoding="utf-8"))
    port_d = json.loads((res / "mn9_portability_results.json").read_text(encoding="utf-8"))
    ab_d = json.loads((res / "mn9_pre_submission_ab_results.json").read_text(encoding="utf-8"))
    v5_d = json.loads((res / "mn9_v5_reviewer_controls_results.json").read_text(encoding="utf-8"))

    pt3 = mc_d["per_type_3brains"]
    names = np.array([r["name"] for r in pt3])
    y_bin = np.array([int(r["y_bin"]) for r in pt3], dtype=int)
    y_max = np.array([float(r["y_max"]) for r in pt3], dtype=float)
    y_02, y_05 = (y_max >= 0.2).astype(int), (y_max >= 0.5).astype(int)
    unpub = np.array([not bool(r["prior_published"]) for r in pt3], dtype=bool)
    nd_v783 = np.array([not bool(r["is_direct_v783"]) for r in pt3], dtype=bool)
    nd_banc = np.array([not bool(r["is_direct_banc_m1"]) for r in pt3], dtype=bool)
    nd_male = np.array([not bool(r["is_direct_malecns"]) for r in pt3], dtype=bool)
    nd_male_p = np.array(
        [not (bool(r["is_direct_malecns"]) or r["name"] in {"TH_VUM", "justice"}) for r in pt3],
        dtype=bool,
    )
    mask_102 = ~np.isin(names, ["marge", "phantom", "tentacular", "usnea"])
    nt_by_name = {r["name"]: r["v783_cls_counts"] for r in banc_d["mapping_audit"]["per_type_sez106"]}
    v783_exc58 = np.array([nt_by_name[nm].get("0", 0) > 0 and nm != "usnea" for nm in names], dtype=bool)

    v783_50 = np.array([float(r["v783_50hz"]) for r in pt3])
    v783_200 = np.array([float(r["v783_200hz"]) for r in pt3])

    rng = np.random.default_rng(42)
    pos_i, neg_i = np.where(y_bin == 1)[0], np.where(y_bin == 0)[0]
    B = 100000
    boot_idx = np.hstack([rng.choice(pos_i, (B, len(pos_i)), True), rng.choice(neg_i, (B, len(neg_i)), True)])

    def ci95_100k(s: np.ndarray) -> tuple[float, float]:
        uniq, inv = np.unique(-s, return_inverse=True)
        K = len(uniq)
        vals = np.empty(B, dtype=np.float64)
        tot_pos = float(len(pos_i))
        inv_pos = inv[boot_idx[:, : len(pos_i)]]
        inv_neg = inv[boot_idx[:, len(pos_i) :]]
        for st in range(0, B, 10000):
            en = min(st + 10000, B)
            m = en - st
            tp_b = np.zeros((m, K), dtype=np.int32)
            fp_b = np.zeros((m, K), dtype=np.int32)
            rows_ar = np.arange(m)[:, None]
            np.add.at(tp_b, (rows_ar, inv_pos[st:en]), 1)
            np.add.at(fp_b, (rows_ar, inv_neg[st:en]), 1)
            cum_tp = np.cumsum(tp_b, axis=1)
            cum_tot = cum_tp + np.cumsum(fp_b, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                prec = np.where(cum_tot > 0, cum_tp / cum_tot, 0.0)
            vals[st:en] = np.sum((tp_b / tot_pos) * prec, axis=1)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    regimes = [
        ("FlyWire v783 reference", 50, v783_50, nd_v783, 0.005),
        ("FlyWire v783 flat 1/S", 50, np.array(ab_d["phase_b1_v783"]["v783_intrinsic_inv_sin_reference"]["sweeps"]["50hz"]["per_type_mean_hz"]), nd_v783, ab_d["phase_b1_v783"]["v783_intrinsic_inv_sin_reference"]["sweeps"]["50hz"]["mean_brain_hz"]),
        ("BANC v888 raw", 50, np.array([float(r["banc_frozen_50hz"]) for r in pt3]), nd_banc, 0.002),
        ("BANC v888 target-scaled (primary)", 50, np.array([float(r["banc_anat_50hz"]) for r in pt3]), nd_banc, 0.013),
        ("BANC v888 target-scaled", 200, np.array([float(r["banc_anat_200hz"]) for r in pt3]), nd_banc, 0.047),
        ("BANC v888 sugar-GRN anchor", 50, np.array(ab_d["phase_banc"]["a1_sugar_anchor"]["sweeps"]["50hz"]["per_type_mean_hz"]), nd_banc, ab_d["phase_banc"]["a1_sugar_anchor"]["sweeps"]["50hz"]["mean_brain_hz"]),
        ("BANC v888 homolog 1:1 (34.7%)", 50, np.array(ab_d["phase_banc"]["a2a_homolog_per_neuron_norm"]["sweeps"]["50hz"]["per_type_mean_hz"]), nd_banc, ab_d["phase_banc"]["a2a_homolog_per_neuron_norm"]["sweeps"]["50hz"]["mean_brain_hz"]),
        ("BANC v888 homolog 1:1+CT (79.7%)", 50, np.array(v5_d["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["50hz"]["per_type_mean_hz"]), nd_banc, v5_d["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["50hz"]["mean_brain_hz"]),
        ("BANC v888 homolog 1:1+CT (79.7%)", 200, np.array(v5_d["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["200hz"]["per_type_mean_hz"]), nd_banc, v5_d["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["200hz"]["mean_brain_hz"]),
        ("BANC v888 homolog pure CT (78.1%)", 50, np.array(v5_d["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_pure_celltype"]["sweeps"]["50hz"]["per_type_mean_hz"]), nd_banc, v5_d["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_pure_celltype"]["sweeps"]["50hz"]["mean_brain_hz"]),
        ("BANC v888 flat 1/S", 50, np.array(ab_d["phase_banc"]["a2b_intrinsic_inv_sin_norm"]["sweeps"]["50hz"]["per_type_mean_hz"]), nd_banc, ab_d["phase_banc"]["a2b_intrinsic_inv_sin_norm"]["sweeps"]["50hz"]["mean_brain_hz"]),
        ("MaleCNS blind bilateral", 50, np.array([float(r["malecns_bilat_50hz_mean"]) for r in pt3]), nd_male, mc_d["whole_brain_firing_diagnostics"]["anat_bilat"]["50hz"]["mean_hz"]),
        ("MaleCNS target-scaled (primary)", 50, np.array([float(r["malecns_intact_50hz_mean"]) for r in pt3]), nd_male, mc_d["whole_brain_firing_diagnostics"]["anat_intact"]["50hz"]["mean_hz"]),
        ("MaleCNS density-scaled", 50, np.array(port_d["test_b_density_and_dimorphism"]["malecns_density_scaled_sim"]["50hz"]["all_106_rates_mean"]), nd_male, 0.006),
        ("MaleCNS predicted_nt fixed", 50, np.array(ab_d["phase_malecns"]["experiment_b2_fixed_02683_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"]), nd_male_p, ab_d["phase_malecns"]["experiment_b2_fixed_02683_prednt"]["sweeps"]["50hz"]["mean_brain_hz"]),
        ("MaleCNS predicted_nt recalc", 50, np.array(port_d["test_a_transmitter_sensitivity"]["A2_all_unclear_predicted_nt"]["50hz"]["all_106_rates_mean"]), nd_male_p, 0.440),
        ("MaleCNS joint", 50, np.array(port_d["test_a2_b3_combined"]["runs"]["a2_density_matched"]["50hz"]["all_106_rates_mean"]), nd_male_p, 0.008),
        ("MaleCNS sugar L17", 50, np.array(ab_d["phase_malecns"]["a1_sugar_anchor_consensus"]["sweeps_at_left17_wstar"]["50hz"]["per_type_mean_hz"]), nd_male, ab_d["phase_malecns"]["a1_sugar_anchor_consensus"]["sweeps_at_left17_wstar"]["50hz"]["mean_brain_hz"]),
        ("MaleCNS sugar R17 (cons)", 50, np.array(ab_d["phase_malecns"]["a1_sugar_anchor_consensus"]["sweeps_at_right17_wstar"]["50hz"]["per_type_mean_hz"]), nd_male, ab_d["phase_malecns"]["a1_sugar_anchor_consensus"]["sweeps_at_right17_wstar"]["50hz"]["mean_brain_hz"]),
        ("MaleCNS sugar R17 (pred_nt)", 50, np.array(ab_d["phase_malecns"]["a1_sugar_anchor_prednt"]["sweeps_at_right17_wstar"]["50hz"]["per_type_mean_hz"]), nd_male_p, ab_d["phase_malecns"]["a1_sugar_anchor_prednt"]["sweeps_at_right17_wstar"]["50hz"]["mean_brain_hz"]),
        ("MaleCNS homolog norm (cons)", 50, np.array(ab_d["phase_malecns"]["a2a_homolog_per_neuron_norm_cons"]["sweeps"]["50hz"]["per_type_mean_hz"]), nd_male, ab_d["phase_malecns"]["a2a_homolog_per_neuron_norm_cons"]["sweeps"]["50hz"]["mean_brain_hz"]),
        ("MaleCNS homolog + pred_nt", 50, np.array(ab_d["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"]), nd_male_p, ab_d["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["mean_brain_hz"]),
        ("MaleCNS homolog + pred_nt", 200, np.array(ab_d["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["200hz"]["per_type_mean_hz"]), nd_male_p, ab_d["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["200hz"]["mean_brain_hz"]),
        ("MaleCNS flat 1/S", 50, np.array(ab_d["phase_malecns"]["a2b_intrinsic_inv_sin_norm_cons"]["sweeps"]["50hz"]["per_type_mean_hz"]), nd_male, ab_d["phase_malecns"]["a2b_intrinsic_inv_sin_norm_cons"]["sweeps"]["50hz"]["mean_brain_hz"]),
    ]

    print("=== S2A MAIN TABLE (24 ROWS, ARCHIVED FLOAT64 ARRAYS, B=100,000 SEED-42 CIs) ===")
    for nm, hz, s, nd, bm in regimes:
        ref = v783_50 if hz == 50 else v783_200
        lo, hi = ci95_100k(s)
        rk = "—" if nm == "FlyWire v783 reference" else f"{spearmanr(s, ref).statistic:.4f} / {cohen_kappa_gt0(s, ref):.3f}"
        print(
            f"{nm:34s} | {hz:3d} | {stepwise_auprc(y_bin,s):.4f} [{lo:.3f}, {hi:.3f}] | "
            f"{auroc_midrank(y_bin,s):.4f} | {topk_tie_avg(y_bin,s,10):.1f}/{topk_tie_avg(y_bin,s,14):.1f} | "
            f"{stepwise_auprc(y_bin[unpub],s[unpub]):.4f} | {stepwise_auprc(y_bin[nd],s[nd]):.4f} ({int(nd.sum())}) | "
            f"{bm:.3f} | {rk}"
        )

    print("\n=== S2A THRESHOLD/DENOMINATOR TABLE (8 ROWS) ===")
    for nm, s in [
        ("FlyWire v783 (ref)", v783_50),
        ("BANC target-scaled", np.array([float(r["banc_anat_50hz"]) for r in pt3])),
        ("BANC homolog 1:1", np.array(ab_d["phase_banc"]["a2a_homolog_per_neuron_norm"]["sweeps"]["50hz"]["per_type_mean_hz"])),
        ("BANC homolog 1:1+CT", np.array(v5_d["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["50hz"]["per_type_mean_hz"])),
        ("MaleCNS target-scaled", np.array([float(r["malecns_intact_50hz_mean"]) for r in pt3])),
        ("MaleCNS predicted_nt 0.264", np.array(port_d["test_a_transmitter_sensitivity"]["A2_all_unclear_predicted_nt"]["50hz"]["all_106_rates_mean"])),
        ("MaleCNS joint 0.157", np.array(port_d["test_a2_b3_combined"]["runs"]["a2_density_matched"]["50hz"]["all_106_rates_mean"])),
        ("MaleCNS homolog + pred_nt", np.array(ab_d["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"])),
    ]:
        s01, s10 = np.where(s >= 0.1, s, 0.0), np.where(s >= 1.0, s, 0.0)
        print(
            f"{nm:28s} | {int(((s>0)&(y_bin==1)).sum())}/{int(((s>0)&(y_bin==0)).sum())} | "
            f"excl4={stepwise_auprc(y_bin[mask_102],s[mask_102]):.4f} | exc58={stepwise_auprc(y_bin[v783_exc58],s[v783_exc58]):.4f} | "
            f">=0.1/>=1.0={stepwise_auprc(y_bin,s01):.4f}/{stepwise_auprc(y_bin,s10):.4f} | "
            f"y02/05={stepwise_auprc(y_02,s):.4f}/{stepwise_auprc(y_05,s):.4f}"
        )

    # Sanity-check S1_Table.tsv row count if present
    s1_tsv = root / "data" / "S1_Table.tsv"
    if s1_tsv.exists():
        with s1_tsv.open() as f:
            n_rows = sum(1 for _ in csv.DictReader(f, delimiter="\t"))
        assert n_rows == 106, f"Expected 106 rows in S1_Table.tsv, got {n_rows}"
        print(f"\nVerified {s1_tsv}: {n_rows} rows OK.")


if __name__ == "__main__":
    main()
