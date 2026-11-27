#!/usr/bin/env python3
"""Deterministic figure, supplementary-table, and estimate builder for the
Drosophila SEZ->MN9 connectome portability study.

Reads the six locked JSON result files plus the pre-submission Experiment A/B
calibration/control JSON (`mn9_pre_submission_ab_results.json`) in this directory:
  - shiu_mn9_results.json
  - mn9_transfer_results.json
  - mn9_banc_results.json
  - mn9_disinhibition_results.json
  - mn9_malecns_results.json
  - mn9_portability_results.json
  - mn9_pre_submission_ab_results.json

Outputs committed to `english_manuscript/`:
  - estimates.json (every manuscript statistic with programmatic assertions)
  - fig1.svg, fig2.svg, fig3.svg, fig4.svg, figS1.svg (`svg.fonttype = 'none'`)
  - supp_table1_three_brain_sez106.tsv
  - supp_table2_102_local_oracles.tsv
  - supp_table3_44ctrl_disinhibition.tsv
  - supp_table4_rewired_graphs_v783.tsv

Optional `--preview-dir /tmp/mn9_fig_preview` writes high-DPI PNG previews outside Git
for layout inspection and PDF embedding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr, spearmanr

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "english_manuscript"

# Consistent colorblind-safe publication palette
TEAL = "#00796b"
TEAL_LIGHT = "#5bb3a8"
ORANGE = "#c85a17"
ORANGE_LIGHT = "#e89c68"
INDIGO = "#48548c"
INDIGO_LIGHT = "#8f98c4"
SLATE = "#6f7f85"
SLATE_LIGHT = "#b5c2c6"
CRIMSON = "#b73832"
GOLD = "#9e7b2f"
PURPLE = "#6f3c8c"
DARK = "#17262b"
GRID = "#dce4e6"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.4,
        "axes.titlesize": 9.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.3,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.2,
        "svg.fonttype": "none",
        "svg.hashsalt": "projc-mn9-v2",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#425459",
        "axes.linewidth": 0.8,
        "xtick.color": DARK,
        "ytick.color": DARK,
        "text.color": DARK,
    }
)


def load_locked_jsons() -> dict[str, dict]:
    return {
        "v630": json.loads((BASE_DIR / "shiu_mn9_results.json").read_text(encoding="utf-8")),
        "v783": json.loads((BASE_DIR / "mn9_transfer_results.json").read_text(encoding="utf-8")),
        "banc": json.loads((BASE_DIR / "mn9_banc_results.json").read_text(encoding="utf-8")),
        "disinh": json.loads((BASE_DIR / "mn9_disinhibition_results.json").read_text(encoding="utf-8")),
        "malecns": json.loads((BASE_DIR / "mn9_malecns_results.json").read_text(encoding="utf-8")),
        "port": json.loads((BASE_DIR / "mn9_portability_results.json").read_text(encoding="utf-8")),
        "ab": json.loads((BASE_DIR / "mn9_pre_submission_ab_results.json").read_text(encoding="utf-8")),
        "v5": json.loads((BASE_DIR / "mn9_v5_reviewer_controls_results.json").read_text(encoding="utf-8")),
    }


def _stepwise_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(np.sum(y_true))
    if n_pos == 0:
        return 0.0
    order = np.argsort(-scores, kind="mergesort")
    y_s = y_true[order]
    s_s = scores[order]
    tp = 0
    ap = 0.0
    prev_r = 0.0
    i = 0
    n = len(y_s)
    while i < n:
        j = i + 1
        while j < n and s_s[j] == s_s[i]:
            j += 1
        tp += int(np.sum(y_s[i:j]))
        prec = tp / float(j)
        rec = tp / float(n_pos)
        ap += (rec - prev_r) * prec
        prev_r = rec
        i = j
    return float(ap)


def _strat_auprc_ci95(y_true: np.ndarray, scores: np.ndarray, n_boot: int = 100000, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    pos_i = np.where(y_true == 1)[0]
    neg_i = np.where(y_true == 0)[0]
    n_pos = len(pos_i)
    if n_pos == 0:
        return [0.0, 0.0]
    bidx = np.hstack([rng.choice(pos_i, (n_boot, n_pos), True), rng.choice(neg_i, (n_boot, len(neg_i)), True)])
    uniq, inv = np.unique(-scores, return_inverse=True)
    K = len(uniq)
    vals = np.empty(n_boot, dtype=np.float64)
    ip = inv[bidx[:, :n_pos]]
    ineg = inv[bidx[:, n_pos:]]
    for st in range(0, n_boot, 20000):
        en = min(st + 20000, n_boot)
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
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def save_figure(fig: plt.Figure, name: str, preview_dir: Path | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{name}.svg", bbox_inches="tight", facecolor="white", metadata={"Date": None})
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{name}.png", bbox_inches="tight", dpi=220, facecolor="white")
    plt.close(fig)


def export_supp_tables(data: dict[str, dict]) -> None:
    """Export byte-deterministic TSV supplementary tables directly from locked JSONs + AB results."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mcns = data["malecns"]
    disinh = data["disinh"]
    port = data["port"]
    ab = data["ab"]

    # Supplementary Table 1: All 106 SEZ types across 3 connectomes + sensitivity/joint + Experiment A/B regimes
    b3_rates = port["test_b_density_and_dimorphism"]["malecns_density_scaled_sim"]["50hz"]["all_106_rates_mean"]
    b3_200 = port["test_b_density_and_dimorphism"]["malecns_density_scaled_sim"]["200hz"]["all_106_rates_mean"]
    a1a_rates = port["test_a_transmitter_sensitivity"]["A1a_diatom18_thvum_plus1"]["50hz"]["all_106_rates_mean"]
    a2_rates = port["test_a_transmitter_sensitivity"]["A2_all_unclear_predicted_nt"]["50hz"]["all_106_rates_mean"]
    a2_200 = port["test_a_transmitter_sensitivity"]["A2_all_unclear_predicted_nt"]["200hz"]["all_106_rates_mean"]
    a2b3_rates = port["test_a2_b3_combined"]["runs"]["a2_density_matched"]["50hz"]["all_106_rates_mean"]
    a2b3_200 = port["test_a2_b3_combined"]["runs"]["a2_density_matched"]["200hz"]["all_106_rates_mean"]

    ab_v783_int50 = ab["phase_b1_v783"]["v783_intrinsic_inv_sin_reference"]["sweeps"]["50hz"]["per_type_mean_hz"]
    ab_b_sug50 = ab["phase_banc"]["a1_sugar_anchor"]["sweeps"]["50hz"]["per_type_mean_hz"]
    ab_b_sug200 = ab["phase_banc"]["a1_sugar_anchor"]["sweeps"]["200hz"]["per_type_mean_hz"]
    ab_b_hom50 = ab["phase_banc"]["a2a_homolog_per_neuron_norm"]["sweeps"]["50hz"]["per_type_mean_hz"]
    ab_b_hom200 = ab["phase_banc"]["a2a_homolog_per_neuron_norm"]["sweeps"]["200hz"]["per_type_mean_hz"]
    ab_b_int50 = ab["phase_banc"]["a2b_intrinsic_inv_sin_norm"]["sweeps"]["50hz"]["per_type_mean_hz"]
    ab_m_b2_50 = ab["phase_malecns"]["experiment_b2_fixed_02683_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"]
    ab_m_b2_200 = ab["phase_malecns"]["experiment_b2_fixed_02683_prednt"]["sweeps"]["200hz"]["per_type_mean_hz"]
    ab_m_l17c_50 = ab["phase_malecns"]["a1_sugar_anchor_consensus"]["sweeps_at_left17_wstar"]["50hz"]["per_type_mean_hz"]
    ab_m_r17c_50 = ab["phase_malecns"]["a1_sugar_anchor_consensus"]["sweeps_at_right17_wstar"]["50hz"]["per_type_mean_hz"]
    ab_m_r17p_50 = ab["phase_malecns"]["a1_sugar_anchor_prednt"]["sweeps_at_right17_wstar"]["50hz"]["per_type_mean_hz"]
    ab_m_r17p_200 = ab["phase_malecns"]["a1_sugar_anchor_prednt"]["sweeps_at_right17_wstar"]["200hz"]["per_type_mean_hz"]
    ab_m_homc_50 = ab["phase_malecns"]["a2a_homolog_per_neuron_norm_cons"]["sweeps"]["50hz"]["per_type_mean_hz"]
    ab_m_homc_200 = ab["phase_malecns"]["a2a_homolog_per_neuron_norm_cons"]["sweeps"]["200hz"]["per_type_mean_hz"]
    ab_m_homp_50 = ab["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"]
    ab_m_homp_200 = ab["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["200hz"]["per_type_mean_hz"]
    ab_m_int50 = ab["phase_malecns"]["a2b_intrinsic_inv_sin_norm_cons"]["sweeps"]["50hz"]["per_type_mean_hz"]

    # V5 reviewer control columns (BANC 79.7% cell-type homolog, fair 2-hop, M2 anti-circularity controls)
    v5 = data["v5"]
    v5_b_homct_50 = v5["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["50hz"]["per_type_mean_hz"]
    v5_b_homct_200 = v5["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["200hz"]["per_type_mean_hz"]
    v5_b_fair2h = v5["phase1_banc"]["m4_banc_fair_twohop"]["homolog_celltype_weighted"]["signed_relu_scores_0p01"]
    v5_m_fair2h = v5["phase2_malecns"]["m4_malecns_fair_twohop"]["predicted_nt_homolog_weighted"]["signed_relu_scores_0p01"]
    v5_m_lo_sez_mn9 = v5["phase2_malecns"]["m2b_leave_out_sez106_and_mn9"]["per_type_mean_hz"]
    v5_m_two_scale = v5["phase2_malecns"]["m2b_two_scale_mn9_02638_rest_01567"]["per_type_mean_hz"]
    v5_m_banc_ref = v5["phase2_malecns"]["m2c_banc_ref_to_malecns"]["per_type_mean_hz"]

    dim_map = {
        r["name"]: ";".join(r["dimorphism"] + r["fruDsx"])
        for r in port["test_b_density_and_dimorphism"]["malecns_vs_v783_density_and_dimorphism"]["all106_any_dimorphism_or_fruDsx"]
    }
    ma_map = {r["name"]: r for r in mcns["mapping_audit"]["per_type_sez106"]}
    v7_map = {r["name"]: r for r in data["v783"]["per_type_task1"]}
    # True 1-hop adjacency under whole-brain predicted_nt (primary direct + TH_VUM + justice)
    prednt_extra_direct = {"TH_VUM", "justice"}

    t1_path = OUT_DIR / "supp_table1_three_brain_sez106.tsv"
    cols1 = [
        "name", "y_bin", "y_max", "y_mean", "prior_published",
        "n_v783", "n_banc_m1", "n_banc_m2", "n_malecns",
        "malecns_dimorphism_fruDsx", "malecns_nt_classes",
        "is_direct_v783", "is_direct_banc_m1", "is_direct_malecns", "is_direct_malecns_predNT",
        "v783_static_2hop_relu", "banc_static_2hop_relu", "malecns_static_2hop_relu",
        "banc_fair_2hop_hom_ct_relu", "malecns_fair_2hop_predNT_hom_relu",
        "v630_recomputed_50hz", "v783_50hz", "v783_50hz_l", "v783_50hz_r", "v783_100hz", "v783_200hz",
        "banc_frozen_50hz", "banc_anat_50hz", "banc_anat_100hz", "banc_anat_200hz",
        "malecns_intact_50hz_mean", "malecns_intact_50hz_l", "malecns_intact_50hz_r",
        "malecns_intact_100hz_mean", "malecns_intact_200hz_mean",
        "malecns_frozen_50hz_mean", "malecns_bilat_50hz_mean",
        "malecns_density_50hz", "malecns_density_200hz",
        "malecns_focal_nt_50hz",
        "malecns_predNT_fixed02683_50hz", "malecns_predNT_fixed02683_200hz",
        "malecns_predNT_50hz", "malecns_predNT_200hz",
        "malecns_predNT_density_50hz", "malecns_predNT_density_200hz",
        "v783_int_norm_50hz",
        "banc_sugar_50hz", "banc_sugar_200hz",
        "banc_hom_norm_50hz", "banc_hom_norm_200hz",
        "banc_hom_ct_50hz", "banc_hom_ct_200hz", "banc_int_norm_50hz",
        "malecns_sugar_l17_cons_50hz", "malecns_sugar_r17_cons_50hz",
        "malecns_sugar_r17_predNT_50hz", "malecns_sugar_r17_predNT_200hz",
        "malecns_hom_norm_cons_50hz", "malecns_hom_norm_cons_200hz",
        "malecns_hom_norm_predNT_50hz", "malecns_hom_norm_predNT_200hz",
        "malecns_leave_out_sez_mn9_50hz", "malecns_two_scale_50hz", "malecns_banc_ref_50hz",
        "malecns_int_norm_50hz",
    ]
    with t1_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols1)
        for i, r in enumerate(mcns["per_type_3brains"]):
            nm = r["name"]
            cls_c = ma_map[nm]["malecns_cls_counts"]
            nt_str = f"E={cls_c.get('0', 0)},I={cls_c.get('1', 0)},U={cls_c.get('2', 0)}"
            is_dir_pnt = int(bool(r["is_direct_malecns"]) or (nm in prednt_extra_direct))
            v7r = v7_map[nm]
            w.writerow([
                nm, int(r["y_bin"]), f"{r['y_max']:.4f}", f"{r['y_mean']:.4f}", int(bool(r["prior_published"])),
                int(r["n_v783"]), int(r["n_banc_m1"]), int(r["n_banc_m2"]), int(r["n_malecns"]),
                "unmapped" if int(r["n_malecns"]) == 0 else dim_map.get(nm, "none"), nt_str,
                int(bool(r["is_direct_v783"])), int(bool(r["is_direct_banc_m1"])), int(bool(r["is_direct_malecns"])), is_dir_pnt,
                f"{r['v783_static_2hop_relu']:.3f}", f"{r['banc_static_2hop_relu']:.3f}", f"{r['malecns_static_2hop_relu']:.3f}",
                f"{v5_b_fair2h[i]:.3f}", f"{v5_m_fair2h[i]:.3f}",
                f"{float(v7r['v630_real_lif_50hz_mean']):.4f}", f"{r['v783_50hz']:.4f}",
                f"{float(v7r['v783_real_lif_50hz_l']):.4f}", f"{float(v7r['v783_real_lif_50hz_r']):.4f}",
                f"{r['v783_100hz']:.4f}", f"{r['v783_200hz']:.4f}",
                f"{r['banc_frozen_50hz']:.4f}", f"{r['banc_anat_50hz']:.4f}", f"{r['banc_anat_100hz']:.4f}", f"{r['banc_anat_200hz']:.4f}",
                f"{r['malecns_intact_50hz_mean']:.4f}", f"{r['malecns_intact_50hz_l']:.4f}", f"{r['malecns_intact_50hz_r']:.4f}",
                f"{r['malecns_intact_100hz_mean']:.4f}", f"{r['malecns_intact_200hz_mean']:.4f}",
                f"{r['malecns_frozen_50hz_mean']:.4f}", f"{r['malecns_bilat_50hz_mean']:.4f}",
                f"{b3_rates[i]:.4f}", f"{b3_200[i]:.4f}",
                f"{a1a_rates[i]:.4f}",
                f"{ab_m_b2_50[i]:.4f}", f"{ab_m_b2_200[i]:.4f}",
                f"{a2_rates[i]:.4f}", f"{a2_200[i]:.4f}",
                f"{a2b3_rates[i]:.4f}", f"{a2b3_200[i]:.4f}",
                f"{ab_v783_int50[i]:.4f}",
                f"{ab_b_sug50[i]:.4f}", f"{ab_b_sug200[i]:.4f}",
                f"{ab_b_hom50[i]:.4f}", f"{ab_b_hom200[i]:.4f}",
                f"{v5_b_homct_50[i]:.4f}", f"{v5_b_homct_200[i]:.4f}", f"{ab_b_int50[i]:.4f}",
                f"{ab_m_l17c_50[i]:.4f}", f"{ab_m_r17c_50[i]:.4f}",
                f"{ab_m_r17p_50[i]:.4f}", f"{ab_m_r17p_200[i]:.4f}",
                f"{ab_m_homc_50[i]:.4f}", f"{ab_m_homc_200[i]:.4f}",
                f"{ab_m_homp_50[i]:.4f}", f"{ab_m_homp_200[i]:.4f}",
                f"{v5_m_lo_sez_mn9[i]:.4f}", f"{v5_m_two_scale[i]:.4f}", f"{v5_m_banc_ref[i]:.4f}",
                f"{ab_m_int50[i]:.4f}",
            ])

    # Supplementary Table 2: All 102 diagnostic local premotor Oracles across 3 connectomes
    t2_path = OUT_DIR / "supp_table2_102_local_oracles.tsv"
    cols2 = [
        "brain", "candidate", "oracle_config",
        "bg_mn9_l_hz", "bg_mn9_r_hz", "bg_mn9_mean_hz",
        "delta_50hz_mean", "delta_50hz_se", "delta_50hz_l", "p_50hz", "unlocked_50hz",
        "delta_200hz_mean", "delta_200hz_se", "delta_200hz_l", "p_200hz", "unlocked_200hz",
        "compute_backend",
    ]
    ora_all = []
    for src in (disinh["module2_local_oracle_realizability"], mcns["disinhibition_module4"]["local_oracle_realizability_intact"]):
        for key in sorted(src.keys()):
            brain, cand, cfg = key.split("|", 2)
            rec = src[key]
            r50 = rec["rates"]["50hz"]
            r200 = rec["rates"]["200hz"]
            ora_all.append([
                brain, cand, cfg,
                f"{rec['bg_mn9_l_hz']:.4f}", f"{rec['bg_mn9_r_hz']:.4f}", f"{rec['bg_mn9_mean_hz']:.4f}",
                f"{r50['delta_mean_hz']:.4f}", f"{r50['delta_se_hz']:.4f}", f"{r50['delta_l_hz']:.4f}",
                f"{r50['paired_p']:.4e}", int(bool(r50["unlocked_ge1hz_p01"])),
                f"{r200['delta_mean_hz']:.4f}", f"{r200['delta_se_hz']:.4f}", f"{r200['delta_l_hz']:.4f}",
                f"{r200['paired_p']:.4e}", int(bool(r200["unlocked_ge1hz_p01"])),
                str(rec.get("compute_backend", "")),
            ])
    assert len(ora_all) == 102, f"Expected 102 Oracle rows, got {len(ora_all)}"
    with t2_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols2)
        w.writerows(ora_all)

    # Supplementary Table 3 (machine-readable TSV): Full 44-control + 4-candidate three-brain disinhibition pool
    t3_path = OUT_DIR / "supp_table3_44ctrl_disinhibition.tsv"
    pd_null = port["test_d_44ctrl_three_brain_null"]
    strict36 = set(port["test_d_strict_3brain_inh_sensitivity"]["strict36_names"])
    cols3 = [
        "name", "role", "in_strict_3brain_inh_36",
        "hop2_net_pct_v783", "hop2_net_pct_banc", "hop2_net_pct_malecns",
        "hop3_disinh_pct_v783", "hop3_disinh_pct_banc", "hop3_disinh_pct_malecns",
        "n_vb_hop2", "n_all3_hop2", "ret_3rd_hop2",
        "n_vb_hop3", "n_all3_hop3", "ret_3rd_hop3",
        "shared_all3_hop2", "shared_all3_hop3",
    ]
    rows3 = []
    for nm in ("tentacular", "phantom", "marge", "usnea"):
        rec = pd_null["candidates"][nm]
        h2 = rec["hop2_net_pct_3brains"]
        h3 = rec["hop3_disinh_pct_3brains"]
        rows3.append([
            nm, "positive_candidate", "NA",
            f"{h2[0]:.4f}", f"{h2[1]:.4f}", f"{h2[2]:.4f}",
            f"{h3[0]:.4f}", f"{h3[1]:.4f}", f"{h3[2]:.4f}",
            int(rec["n_vb_hop2"]), int(rec["n_all3_hop2"]), f"{rec['ret_3rd_hop2']:.4f}",
            int(rec["n_vb_hop3"]), int(rec["n_all3_hop3"]), f"{rec['ret_3rd_hop3']:.4f}",
            ";".join(rec["shared_all3_hop2"]), ";".join(rec["shared_all3_hop3"]),
        ])
    for nm in sorted(pd_null["per_control_table"].keys()):
        rec = pd_null["per_control_table"][nm]
        h2 = rec["hop2_net_pct_3brains"]
        h3 = rec["hop3_disinh_pct_3brains"]
        rows3.append([
            nm, "neg_inh_control", int(nm in strict36),
            f"{h2[0]:.4f}", f"{h2[1]:.4f}", f"{h2[2]:.4f}",
            f"{h3[0]:.4f}", f"{h3[1]:.4f}", f"{h3[2]:.4f}",
            int(rec["n_vb_hop2"]), int(rec["n_all3_hop2"]), f"{rec['ret_3rd_hop2']:.4f}",
            int(rec["n_vb_hop3"]), int(rec["n_all3_hop3"]), f"{rec['ret_3rd_hop3']:.4f}",
            ";".join(rec["shared_all3_hop2"]), ";".join(rec["shared_all3_hop3"]),
        ])
    assert len(rows3) == 48, f"Expected 48 disinhibition rows (4+44), got {len(rows3)}"
    with t3_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols3)
        w.writerows(rows3)

    # Supplementary Table 4 (machine-readable TSV): Per-rewired-graph v783 predictions & per-hemisphere MN9 rates
    t4_path = OUT_DIR / "supp_table4_rewired_graphs_v783.tsv"
    glob_det = port["test_c_v783_activity_matched_rewire"]["per_graph_details"]
    comp_det = port["test_c_v783_compartment_preserving_rewire"]["per_graph_details"]
    b_hom_l = ab["phase_banc"]["a2a_homolog_per_neuron_norm"]["sweeps"]["50hz"]["per_type_left_hz"]
    b_hom_r = ab["phase_banc"]["a2a_homolog_per_neuron_norm"]["sweeps"]["50hz"]["per_type_right_hz"]
    m_homp_l = ab["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_left_hz"]
    m_homp_r = ab["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_right_hz"]
    cols4 = (
        ["name", "y_bin", "prior_published", "is_direct_v783",
         "v783_real_50hz_l", "v783_real_50hz_r",
         "banc_hom_norm_50hz_l", "banc_hom_norm_50hz_r",
         "malecns_intact_50hz_l", "malecns_intact_50hz_r",
         "malecns_hom_norm_predNT_50hz_l", "malecns_hom_norm_predNT_50hz_r"]
        + [f"v783_global_matched_k{g['k']}" for g in glob_det]
        + [f"v783_comp_fixed0275_k{g['k']}" for g in comp_det]
        + [f"v783_comp_matched_k{g['k']}" for g in comp_det]
    )
    with t4_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(cols4)
        for i, r in enumerate(mcns["per_type_3brains"]):
            nm = r["name"]
            v7r = v7_map[nm]
            row = [
                nm, int(r["y_bin"]), int(bool(r["prior_published"])), int(bool(r["is_direct_v783"])),
                f"{float(v7r['v783_real_lif_50hz_l']):.4f}", f"{float(v7r['v783_real_lif_50hz_r']):.4f}",
                f"{b_hom_l[i]:.4f}", f"{b_hom_r[i]:.4f}",
                f"{r['malecns_intact_50hz_l']:.4f}", f"{r['malecns_intact_50hz_r']:.4f}",
                f"{m_homp_l[i]:.4f}", f"{m_homp_r[i]:.4f}",
            ]
            row += [f"{float(g['rates_106_matched'][i]):.4f}" for g in glob_det]
            row += [f"{float(g['rates_106_fixed_0275'][i]):.4f}" for g in comp_det]
            row += [f"{float(g['rates_106_matched'][i]):.4f}" for g in comp_det]
            w.writerow(row)


def build_estimates(data: dict[str, dict]) -> dict:
    """Extract and assert every quantitative figure cited in the manuscript."""
    v630 = data["v630"]
    v783 = data["v783"]
    banc = data["banc"]
    disinh = data["disinh"]
    mcns = data["malecns"]
    port = data["port"]

    v7_all = v783["task1_activation_v783"]["all_106"]
    v7_nd = v783["task1_activation_v783"]["nondirect_v783"]
    v7_cm = v783["task1_activation_v783"]["complete_mapped_105"]
    v7_nov = v783["task1_activation_v783"]["novel_95"]
    v7_w2 = v783["task2_water_silencing_v783"]
    cmp_ver = v783["v630_vs_v783_task1_comparison"]

    mn9_anat = banc["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]
    b_main = banc["graph_construction_and_anatomy"]["main_graph_banc_v888"]
    b_froz_all = banc["task1_activation_banc"]["M1_frozen_0.275"]["all_106"]
    b_froz_nd = banc["task1_activation_banc"]["M1_frozen_0.275"]["nondirect_banc"]
    b_anat_all = banc["task1_activation_banc"]["M1_anat_scale"]["all_106"]
    b_anat_105 = banc["task1_activation_banc"]["M1_anat_scale"]["M1_105"]
    b_anat_nd = banc["task1_activation_banc"]["M1_anat_scale"]["nondirect_banc"]
    b_anat_nc = banc["task1_activation_banc"]["M1_anat_scale"]["nondirect_common_v783_banc"]
    b_anat_nov = banc["task1_activation_banc"]["M1_anat_scale"]["novel_95"]
    b_m2_all = banc["task1_activation_banc"]["M2_anat_scale"]["all_106"]
    b_m2_nd = banc["task1_activation_banc"]["M2_anat_scale"]["nondirect_banc"]
    b_w2 = banc["task2_water_silencing_banc_bounded"]
    b_w2_160 = b_w2["silencing_runs"]["right_unilateral_14_anat_scale_160hz"]

    b_rows = banc["per_type_task1"]
    xv_b = np.array([r["v783_50hz"] for r in b_rows], dtype=float)
    yf_b = np.array([r["banc_m1_frozen_50hz"] for r in b_rows], dtype=float)
    ya50_b = np.array([r["banc_m1_anat_50hz"] for r in b_rows], dtype=float)
    ya100_b = np.array([r["banc_m1_anat_100hz"] for r in b_rows], dtype=float)
    ya200_b = np.array([r["banc_m1_anat_200hz"] for r in b_rows], dtype=float)
    no_ru = np.array([r["name"] != "roundup" for r in b_rows], dtype=bool)
    b_froz_r = float(pearsonr(xv_b, yf_b)[0])
    b_froz_rho = float(spearmanr(xv_b, yf_b)[0])
    b_froz_rho_noru = float(spearmanr(xv_b[no_ru], yf_b[no_ru])[0])
    b_anat50_r_noru = float(pearsonr(xv_b[no_ru], ya50_b[no_ru])[0])
    b_anat50_rho_noru = float(spearmanr(xv_b[no_ru], ya50_b[no_ru])[0])
    b_anat100_r_noru = float(pearsonr(xv_b[no_ru], ya100_b[no_ru])[0])
    b_anat100_rho_noru = float(spearmanr(xv_b[no_ru], ya100_b[no_ru])[0])
    b_anat200_rho_noru = float(spearmanr(xv_b[no_ru], ya200_b[no_ru])[0])
    b_pos_awakened = [
        r["name"] for r in b_rows if r["y_bin"] and r["banc_m1_frozen_50hz"] == 0.0 and r["banc_m1_anat_50hz"] > 0.0
    ]

    m_main = mcns["graph_construction_and_anatomy"]["main_graph_malecns_v1"]
    m_mn9 = mcns["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]
    m_int_all = mcns["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["all_106"]
    m_int_104 = mcns["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["mapped_104"]
    m_int_102 = mcns["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["isomorphic_102"]
    m_int_02 = mcns["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["per_ge_0_2"]
    m_int_nd = mcns["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["nondirect_malecns"]
    m_int_nd3 = mcns["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["nondirect_common_3brains"]
    m_int_nov = mcns["task1_activation_malecns"]["anat_intact"]["bilateral_mean"]["novel_95"]
    m_frz_all = mcns["task1_activation_malecns"]["frozen_0.275"]["bilateral_mean"]["all_106"]
    m_frz_nd = mcns["task1_activation_malecns"]["frozen_0.275"]["bilateral_mean"]["nondirect_malecns"]
    m_bil_all = mcns["task1_activation_malecns"]["anat_bilat"]["bilateral_mean"]["all_106"]
    m_bil_nd = mcns["task1_activation_malecns"]["anat_bilat"]["bilateral_mean"]["nondirect_malecns"]
    m_corr = mcns["cross_brain_correlations"]
    m_wb = mcns["whole_brain_firing_diagnostics"]
    m_3b = {r["name"]: r for r in mcns["per_type_3brains"]}
    m_ma = {r["name"]: r for r in mcns["mapping_audit"]["per_type_sez106"]}

    m_selk = mcns["selk_full_cns_diagnosis"]
    m_reg = m_selk["selk_postsynaptic_region_ei_summary"]
    m_selk_tot = float(m_selk["selk_total_outgoing_synapses"])
    m_selk_vnc = float(m_reg["ventral_nerve_cord_E"]["sum_syn"] + m_reg["ventral_nerve_cord_I"]["sum_syn"] + m_reg["ventral_nerve_cord_U"]["sum_syn"])
    m_selk_valid_e = float(m_reg["ventral_nerve_cord_E"]["sum_syn"] + m_reg["central_brain_E"]["sum_syn"])
    m_selk_valid_i = float(m_reg["ventral_nerve_cord_I"]["sum_syn"] + m_reg["central_brain_I"]["sum_syn"])
    m_selk_top15 = m_selk["selk_top20_postsynaptic_targets"][:15]
    m_selk_top15_inh = sum(1 for r in m_selk_top15 if r["ei_class"] == "I")
    m_selk_top15_valid = sum(1 for r in m_selk_top15 if r["ei_class"] in ("E", "I"))

    selk_b = banc["selk_full_cns_diagnosis"]
    reg_ei = selk_b["selk_postsynaptic_region_ei_summary"]
    selk_vnc_syn = float(reg_ei["ventral_nerve_cord_E"]["sum_syn"] + reg_ei["ventral_nerve_cord_I"]["sum_syn"])
    selk_tot_syn = float(selk_b["selk_total_outgoing_synapses"])
    selk_top15 = selk_b["selk_top20_postsynaptic_targets"][:15]
    selk_top15_inh = sum(1 for r in selk_top15 if r["ei_class"] == "I")
    selk_top15_asc = [r for r in selk_top15 if r["super_class"] == "ascending"]
    selk_top15_asc_gaba = sum(1 for r in selk_top15_asc if r["nt_predicted"] == "gaba")

    bg_states = disinh["module2_v783_background_states"]
    n_v783_bg_preserve = sum(1 for v in bg_states.values() if not v["all_106_delta_metrics"]["specificity_broken"])
    n_v783_bg_unlock_preserve = sum(
        1 for v in bg_states.values()
        if (not v["all_106_delta_metrics"]["specificity_broken"]) and any(fr["unlocked_ge1hz_p01"] for fr in v["focal_30tr"].values())
    )
    banc_bg = disinh["module2_banc_confirmation"]
    n_banc_bg_unlock_focal = sum(
        1 for bo in banc_bg.values()
        if any(bo["focal_and_refs_30tr"][f"{fn}_{hz}hz"]["unlocked_ge1hz_p01"] for fn in ("phantom", "tentacular", "SELK_DNg68", "marge", "usnea") for hz in (50, 200))
    )
    ora = disinh["module2_local_oracle_realizability"]
    ora_m = mcns["disinhibition_module4"]["local_oracle_realizability_intact"]

    def count_oracle_dict(ora_dict: dict, brain: str | None, cand: str, mode_substr: str = "", freq: str | None = None):
        keys = [k for k in ora_dict if (brain is None or k.startswith(f"{brain}|")) and k.split("|")[1] == cand and (mode_substr in k)]
        if freq is not None:
            return {"unlocked": sum(1 for k in keys if ora_dict[k]["rates"][freq]["unlocked_ge1hz_p01"]), "total": len(keys)}
        return {
            "unlocked_both_50_200": sum(1 for k in keys if ora_dict[k]["rates"]["50hz"]["unlocked_ge1hz_p01"] and ora_dict[k]["rates"]["200hz"]["unlocked_ge1hz_p01"]),
            "unlocked_any": sum(1 for k in keys if ora_dict[k]["rates"]["50hz"]["unlocked_ge1hz_p01"] or ora_dict[k]["rates"]["200hz"]["unlocked_ge1hz_p01"]),
            "total": len(keys),
        }

    m4 = mcns["disinhibition_module4"]
    m4_pct = m4["percentiles_vs_mapped_neg_inh"]

    # Portability hardening tests A/B/C/D
    pa = port["test_a_transmitter_sensitivity"]
    pb = port["test_b_density_and_dimorphism"]
    pc = port["test_c_v783_activity_matched_rewire"]
    pd = port["test_d_44ctrl_three_brain_null"]
    c2 = pc["c2_activity_matched_summary"]

    est = {
        "graphs": {
            "v630_n_nodes": int(v630["graph_summary"]["n_nodes"]),
            "v630_n_edges": int(v630["graph_summary"]["n_edges"]),
            "v783_n_nodes": int(v783["graph_delta"]["whole_brain_v783"]["n_nodes"]),
            "v783_n_edges": int(v783["graph_delta"]["whole_brain_v783"]["n_edges"]),
            "banc_n_nodes": int(b_main["n_nodes"]),
            "banc_n_edges": int(b_main["n_edges"]),
            "banc_region_counts": b_main["region_counts"],
            "banc_nt_edge_coverage": float(b_main["nt_coverage_on_main_non_self_edges"]),
            "malecns_n_nodes": int(m_main["n_nodes"]),
            "malecns_n_edges": int(m_main["n_edges_valid_nt"]),
            "malecns_region_counts": m_main["region_counts"],
            "malecns_nt_edge_coverage": float(m_main["nt_coverage_on_non_self_edges"]),
        },
        "co_primary_endpoints_50hz": port["co_primary_endpoints_50hz"],
        "secondary_bh_fdr": port["secondary_bh_fdr"],
        "v630_to_v783": {
            "unique_targets_mapped": 411,
            "unique_targets_total": int(v783["mapping_audit"]["n_unique_target_v630_ids"]),
            "pearson_r_50hz": float(cmp_ver["pearson_r_50hz"]),
            "spearman_rho_50hz": float(cmp_ver["spearman_rho_50hz"]),
            "mae_hz": float(cmp_ver["mae_hz"]),
            "zero_threshold_flips": len(cmp_ver["zero_threshold_flips"]),
            "sugar_anchor_v783_hz": float(v783["sugar_anchor_calibration"]["v783_frozen_mn9_mean_hz"]),
            "v783_all106_auprc": float(v7_all["bootstrap"]["real_lif_50hz"]["point"]["auprc"]),
            "v783_all106_auprc_ci95": [float(x) for x in v7_all["bootstrap"]["real_lif_50hz"]["ci95"]["auprc"]],
            "v783_all106_auroc": float(v7_all["bootstrap"]["real_lif_50hz"]["point"]["auroc"]),
            "v783_all106_loocv_f1": float(v7_all["cv"]["real_lif_50hz"]["loocv"]["f1"]),
            "v783_all106_loocv_mcc": float(v7_all["cv"]["real_lif_50hz"]["loocv"]["mcc"]),
            "v783_all106_static2hop_relu_auprc": float(v7_all["bootstrap"]["static_2hop_relu_signed"]["point"]["auprc"]),
            "v783_all106_rewire_mn9_per_graph_mean": float(v7_all["rewire_mn9_prot_graphs"]["auprc_mean_sd_min_max"][0]),
            "v783_all106_rewire_mn9_wins": int(v7_all["rewire_mn9_prot_graphs"]["real_exceeds_count_auprc"]),
            "v783_all106_rewire_full_per_graph_mean": float(v7_all["rewire_full_graphs"]["auprc_mean_sd_min_max"][0]),
            "v783_all106_complete105_auprc": float(v7_cm["bootstrap"]["real_lif_50hz"]["point"]["auprc"]),
            "v783_nondirect83_auprc": float(v7_nd["bootstrap"]["real_lif_50hz"]["point"]["auprc"]),
            "v783_nondirect83_auprc_ci95": [float(x) for x in v7_nd["bootstrap"]["real_lif_50hz"]["ci95"]["auprc"]],
            "v783_nondirect83_static2hop_relu_auprc": float(v7_nd["bootstrap"]["static_2hop_relu_signed"]["point"]["auprc"]),
            "v783_nondirect83_rewire_mn9_per_graph_mean": float(v7_nd["rewire_mn9_prot_graphs"]["auprc_mean_sd_min_max"][0]),
            "v783_nondirect83_rewire_mn9_wins": int(v7_nd["rewire_mn9_prot_graphs"]["real_exceeds_count_auprc"]),
            "v783_novel95_auprc": float(v7_nov["bootstrap"]["real_lif_50hz"]["point"]["auprc"]),
            "v783_novel95_ci95": [float(x) for x in v7_nov["bootstrap"]["real_lif_50hz"]["ci95"]["auprc"]],
            "v783_water11_auroc": float(v7_w2["real_160hz_mn9r"]["metrics"]["auroc_impaired"]),
            "v783_water11_auprc": float(v7_w2["real_160hz_mn9r"]["metrics"]["auprc_impaired"]),
            "v783_water11_mwu_u": float(v7_w2["real_160hz_mn9r"]["metrics"]["mannwhitney_u"]),
            "v783_water11_mwu_p": float(v7_w2["real_160hz_mn9r"]["metrics"]["mannwhitney_p_two_sided"]),
            "v783_water11_spearman_did": float(v7_w2["real_160hz_mn9r"]["metrics"]["spearman_with_did"]),
            "v783_water11_spearman_p": float(v7_w2["real_160hz_mn9r"]["metrics"]["spearman_with_did_p"]),
        },
        "banc_cross_individual": {
            "s_in_mn9_v783_mean": float(mn9_anat["s_in_mn9_v783_mean"]),
            "s_in_mn9_v783_l": float(mn9_anat["v783_mn9_l"]["sum_abs"]),
            "s_in_mn9_v783_r": float(mn9_anat["v783_mn9_r"]["sum_abs"]),
            "s_in_mn9_banc_mean": float(mn9_anat["s_in_mn9_banc_mean"]),
            "s_in_mn9_banc_l": float(mn9_anat["banc_mn9_l"]["sum_abs"]),
            "s_in_mn9_banc_r": float(mn9_anat["banc_mn9_r"]["sum_abs"]),
            "anat_scale_ratio": float(mn9_anat["anat_scale_ratio"]),
            "w_syn_anat_mv": float(mn9_anat["w_syn_anat_mv"]),
            "v783_vs_banc_m1_frozen_pearson_r": b_froz_r,
            "v783_vs_banc_m1_frozen_spearman_rho": b_froz_rho,
            "v783_vs_banc_m1_frozen_spearman_rho_excl_roundup": b_froz_rho_noru,
            "v783_vs_banc_m1_anat_pearson_r": float(banc["cross_individual_v783_vs_banc"]["M1_anat_50hz"]["pearson_r_all106"]),
            "v783_vs_banc_m1_anat_spearman_rho": float(banc["cross_individual_v783_vs_banc"]["M1_anat_50hz"]["spearman_rho_all106"]),
            "v783_vs_banc_m1_anat_pearson_r_excl_roundup": b_anat50_r_noru,
            "v783_vs_banc_m1_anat_spearman_rho_excl_roundup": b_anat50_rho_noru,
            "v783_vs_banc_m1_anat_100hz_pearson_r_excl_roundup": b_anat100_r_noru,
            "v783_vs_banc_m1_anat_100hz_spearman_rho_excl_roundup": b_anat100_rho_noru,
            "v783_vs_banc_m1_anat_200hz_spearman_rho_excl_roundup": b_anat200_rho_noru,
            "positives_awakened_from_zero_at_50hz": b_pos_awakened,
            "m1_frozen_all106_auprc": float(b_froz_all["models"]["frozen_50hz"]["auprc"]),
            "m1_frozen_nondirect90_auprc": float(b_froz_nd["models"]["frozen_50hz"]["auprc"]),
            "m1_anat_all106_50hz_auprc": float(b_anat_all["models"]["anat_50hz"]["auprc"]),
            "m1_anat_all106_50hz_auprc_ci95": [float(x) for x in b_anat_all["models"]["anat_50hz"]["ci95"]["auprc"]],
            "m1_anat_all106_100hz_auprc": float(b_anat_all["models"]["anat_100hz"]["auprc"]),
            "m1_anat_all106_200hz_auprc": float(b_anat_all["models"]["anat_200hz"]["auprc"]),
            "m1_anat_all106_static2hop_relu_auprc": float(b_anat_all["models"]["static_2hop_relu_signed"]["auprc"]),
            "m1_anat_all106_rewire_mn9_mean": float(b_anat_all["rewire_mn9_protected"]["per_graph_auprc_mean"]),
            "m1_anat_all106_rewire_mn9_wins": int(b_anat_all["rewire_mn9_protected"]["graph_level_rank_vs_primary"]["wins"]),
            "m1_anat_105_50hz_auprc": float(b_anat_105["models"]["anat_50hz"]["auprc"]),
            "m1_anat_nondirect90_50hz_auprc": float(b_anat_nd["models"]["anat_50hz"]["auprc"]),
            "m1_anat_nondirect90_200hz_auprc": float(b_anat_nd["models"]["anat_200hz"]["auprc"]),
            "m1_anat_nondirect90_static2hop_relu_auprc": float(b_anat_nd["models"]["static_2hop_relu_signed"]["auprc"]),
            "m1_anat_nondirect90_rewire_mn9_mean": float(b_anat_nd["rewire_mn9_protected"]["per_graph_auprc_mean"]),
            "m1_anat_nondirect90_rewire_mn9_wins": int(b_anat_nd["rewire_mn9_protected"]["graph_level_rank_vs_primary"]["wins"]),
            "m1_anat_nondirect_common78_auprc": float(b_anat_nc["models"]["anat_50hz"]["auprc"]),
            "m1_anat_novel95_50hz_auprc": float(b_anat_nov["models"]["anat_50hz"]["auprc"]),
            "m1_anat_novel95_200hz_auprc": float(b_anat_nov["models"]["anat_200hz"]["auprc"]),
            "m2_anat_all106_50hz_auprc": float(b_m2_all["models"]["anat_50hz"]["auprc"]),
            "m2_anat_nondirect90_50hz_auprc": float(b_m2_nd["models"]["anat_50hz"]["auprc"]),
            "water_160hz_base_mn9_l": float(b_w2_160["base_mn9_l"]),
            "water_160hz_contra_mn9l_all11": b_w2_160["metrics_by_readout"]["mn9_l_contra_to_right"]["all_11"],
            "water_160hz_contra_mn9l_excl_usnea10": b_w2_160["metrics_by_readout"]["mn9_l_contra_to_right"]["diag_excl_usnea_10"],
        },
        "malecns_third_connectome": {
            "sez106_mapped": int(mcns["mapping_audit"]["sez106_mapped_count"]),
            "s_in_mn9_l": float(m_mn9["s_in_mn9_l"]),
            "s_in_mn9_r": float(m_mn9["s_in_mn9_r"]),
            "mn9_lr_asymmetry_ratio": float(m_mn9["mn9_lr_asymmetry_ratio"]),
            "w_syn_intact_mv": float(m_mn9["w_syn_intact_mv"]),
            "w_syn_bilat_mv": float(m_mn9["w_syn_bilat_mv"]),
            "equiv_max_spike_diff": max(int(r["max_abs_spike_diff"]) for r in mcns["compute_provenance"]["equivalence_check"]["checks"].values()),
            "brain_hz_50hz_intact": float(m_wb["anat_intact"]["50hz"]["mean_hz"]),
            "brain_hz_50hz_bilat": float(m_wb["anat_bilat"]["50hz"]["mean_hz"]),
            "v783_vs_intact_50hz": m_corr["v783_vs_anat_intact_50hz_mean"],
            "v783_vs_bilat_50hz": m_corr["v783_vs_anat_bilat_50hz_mean"],
            "intact_all106_50hz_auprc": float(m_int_all["models"]["50hz"]["auprc"]),
            "intact_all106_50hz_auprc_ci95": [float(x) for x in m_int_all["models"]["50hz"]["ci95"]["auprc"]],
            "intact_all106_100hz_auprc": float(m_int_all["models"]["100hz"]["auprc"]),
            "intact_all106_200hz_auprc": float(m_int_all["models"]["200hz"]["auprc"]),
            "intact_all106_static2hop_relu_auprc": float(m_int_all["models"]["static_2hop_relu_signed"]["auprc"]),
            "intact_all106_rewire_mn9_mean": float(m_int_all["rewire_mn9_protected"]["per_graph_auprc_mean"]),
            "intact_all106_rewire_mn9_wins": int(m_int_all["rewire_mn9_protected"]["graph_level_rank_vs_primary"]["wins"]),
            "intact_mapped104_50hz_auprc": float(m_int_104["models"]["50hz"]["auprc"]),
            "intact_isomorphic102_50hz_auprc": float(m_int_102["models"]["50hz"]["auprc"]),
            "intact_per02_50hz_auprc": float(m_int_02["models"]["50hz"]["auprc"]),
            "intact_nondirect88_50hz_auprc": float(m_int_nd["models"]["50hz"]["auprc"]),
            "intact_nondirect88_200hz_auprc": float(m_int_nd["models"]["200hz"]["auprc"]),
            "intact_nondirect88_static2hop_relu_auprc": float(m_int_nd["models"]["static_2hop_relu_signed"]["auprc"]),
            "intact_nondirect88_rewire_mn9_mean": float(m_int_nd["rewire_mn9_protected"]["per_graph_auprc_mean"]),
            "intact_nondirect88_rewire_mn9_wins": int(m_int_nd["rewire_mn9_protected"]["graph_level_rank_vs_primary"]["wins"]),
            "intact_nondirect_common3_50hz_auprc": float(m_int_nd3["models"]["50hz"]["auprc"]),
            "intact_nondirect_common3_200hz_auprc": float(m_int_nd3["models"]["200hz"]["auprc"]),
            "intact_nondirect_common3_unsigned2h_auprc": float(m_int_nd3["models"]["static_2hop_unsigned_fixed"]["auprc"]),
            "intact_novel95_50hz_auprc": float(m_int_nov["models"]["50hz"]["auprc"]),
            "intact_novel95_200hz_auprc": float(m_int_nov["models"]["200hz"]["auprc"]),
            "intact_novel95_rewire_mn9_mean": float(m_int_nov["rewire_mn9_protected"]["per_graph_auprc_mean"]),
            "frozen_all106_50hz_auprc": float(m_frz_all["models"]["50hz"]["auprc"]),
            "frozen_nondirect88_50hz_auprc": float(m_frz_nd["models"]["50hz"]["auprc"]),
            "bilat_all106_50hz_auprc": float(m_bil_all["models"]["50hz"]["auprc"]),
            "bilat_all106_rewire_mn9_wins": int(m_bil_all["rewire_mn9_protected"]["graph_level_rank_vs_primary"]["wins"]),
            "bilat_nondirect88_50hz_auprc": float(m_bil_nd["models"]["50hz"]["auprc"]),
            "key_types_3brains": {
                nm: {
                    "n_malecns": int(m_3b[nm]["n_malecns"]),
                    "is_direct": bool(m_3b[nm]["is_direct_malecns"]),
                    "intact_50hz_mean": float(m_3b[nm]["malecns_intact_50hz_mean"]),
                    "intact_50hz_l": float(m_3b[nm]["malecns_intact_50hz_l"]),
                    "intact_200hz_mean": float(m_3b[nm]["malecns_intact_200hz_mean"]),
                    "cls_counts": m_ma[nm]["malecns_cls_counts"],
                }
                for nm in ("Fdg", "rattle", "diatom", "TH_VUM", "roundup", "sink_sync", "peacock", "puddle", "spirit", "mime", "peahen", "crab")
            },
        },
        "portability_hardening": {
            "test_a_A1a_50hz_all106": float(pa["A1a_diatom18_thvum_plus1"]["50hz"]["all_106_auprc"]),
            "test_a_A1a_50hz_novel95": float(pa["A1a_diatom18_thvum_plus1"]["50hz"]["novel_95_auprc"]),
            "test_a_A1a_50hz_nd88": float(pa["A1a_diatom18_thvum_plus1"]["50hz"]["nondirect_malecns_auprc"]),
            "test_a_A1a_200hz_all106": float(pa["A1a_diatom18_thvum_plus1"]["200hz"]["all_106_auprc"]),
            "test_a_A1a_200hz_novel95": float(pa["A1a_diatom18_thvum_plus1"]["200hz"]["novel_95_auprc"]),
            "test_a_A1a_50hz_diatom_hz": float(pa["A1a_diatom18_thvum_plus1"]["50hz"]["key_rates_mean"]["diatom"]),
            "test_a_A1a_50hz_thvum_hz": float(pa["A1a_diatom18_thvum_plus1"]["50hz"]["key_rates_mean"]["TH_VUM"]),
            "test_a_A2_w_syn_mv": float(pa["A2_all_unclear_predicted_nt"]["w_syn_intact_a2_mv"]),
            "test_a_A2_50hz_all106": float(pa["A2_all_unclear_predicted_nt"]["50hz"]["all_106_auprc"]),
            "test_a_A2_50hz_novel95": float(pa["A2_all_unclear_predicted_nt"]["50hz"]["novel_95_auprc"]),
            "test_a_A2_50hz_nd88": float(pa["A2_all_unclear_predicted_nt"]["50hz"]["nondirect_malecns_auprc"]),
            "test_a_A2_200hz_all106": float(pa["A2_all_unclear_predicted_nt"]["200hz"]["all_106_auprc"]),
            "test_a_A2_200hz_novel95": float(pa["A2_all_unclear_predicted_nt"]["200hz"]["novel_95_auprc"]),
            "test_a_A2_selk_50hz": float(pa["A2_all_unclear_predicted_nt"]["selk_dng68_20_to_200hz"]["50hz"]["mn9_mean_hz"]),
            "test_a_A2_selk_200hz": float(pa["A2_all_unclear_predicted_nt"]["selk_dng68_20_to_200hz"]["200hz"]["mn9_mean_hz"]),
            "test_b_banc_cb_median_ratio": float(pb["banc_vs_v783_density"]["single_cell_1to1_central_brain"]["median"]),
            "test_b_banc_cb_iqr": [
                float(pb["banc_vs_v783_density"]["single_cell_1to1_central_brain"]["q25"]),
                float(pb["banc_vs_v783_density"]["single_cell_1to1_central_brain"]["q75"]),
            ],
            "test_b_malecns_cb_median_ratio": float(pb["malecns_vs_v783_density_and_dimorphism"]["r_med_central_brain"]),
            "test_b_malecns_cb_iqr": [
                float(pb["malecns_vs_v783_density_and_dimorphism"]["cell_type_per_cell_central_brain"]["q25"]),
                float(pb["malecns_vs_v783_density_and_dimorphism"]["cell_type_per_cell_central_brain"]["q75"]),
            ],
            "test_b_malecns_w_syn_density_mv": float(pb["malecns_density_scaled_sim"]["w_syn_density_mv"]),
            "test_b_B3_50hz_all106": float(pb["malecns_density_scaled_sim"]["50hz"]["all_106_auprc"]),
            "test_b_B3_50hz_novel95": float(pb["malecns_density_scaled_sim"]["50hz"]["novel_95_auprc"]),
            "test_b_B3_50hz_nd88": float(pb["malecns_density_scaled_sim"]["50hz"]["nondirect_malecns_auprc"]),
            "test_b_B3_50hz_rho": float(pb["malecns_density_scaled_sim"]["50hz"]["v783_spearman_rho"]),
            "test_b_B3_200hz_rho": float(pb["malecns_density_scaled_sim"]["200hz"]["v783_spearman_rho"]),
            "test_b_B3_50hz_fp6_silenced": sum(
                1 for k in ("peacock", "puddle", "spirit", "mime", "peahen", "crab")
                if pb["malecns_density_scaled_sim"]["50hz"]["key_rates_mean"][k] == 0.0
            ),
            "test_b_fp6_with_dimorphism_or_fru": sum(
                1 for d in pb["malecns_vs_v783_density_and_dimorphism"]["male_fp6_audit"].values()
                if d["dimorphism"] or d["fruDsx"]
            ),
            "test_c_w_syn_matched_mean_mv": float(c2["w_syn_matched_mean_mv"]),
            "test_c_matched_mean_106_hz": float(c2["matched_mean_106_hz"]),
            "test_c_all106_matched_mean": float(c2["all_106"]["per_graph_auprc_mean"]),
            "test_c_all106_real_wins": int(c2["all_106"]["real_wins_out_of_10"]),
            "test_c_nd83_matched_mean": float(c2["nondirect_83"]["per_graph_auprc_mean"]),
            "test_c_nd83_real_wins": int(c2["nondirect_83"]["real_wins_out_of_10"]),
            "test_c_nov95_matched_mean": float(c2["novel_95"]["per_graph_auprc_mean"]),
            "test_c_nov95_real_wins": int(c2["novel_95"]["real_wins_out_of_10"]),
            "test_d_tentacular_joint_pass_of_44": int(pd["null_criteria_among_44_neg_inh"]["joint_tentacular_level_h3ge90_h2nge90_shared2h5_3h5"]["k_pass_of_44"]),
            "test_d_tentacular_joint_p_emp": float(pd["null_criteria_among_44_neg_inh"]["joint_tentacular_level_h3ge90_h2nge90_shared2h5_3h5"]["p_emp_k1_over_45"]),
            "test_d_phantom_joint_pass_of_44": int(pd["null_criteria_among_44_neg_inh"]["joint_phantom_level_h3ge80_shared3h_ge4"]["k_pass_of_44"]),
            "test_d_phantom_joint_p_emp": float(pd["null_criteria_among_44_neg_inh"]["joint_phantom_level_h3ge80_shared3h_ge4"]["p_emp_k1_over_45"]),
            "test_d_hop2_shared_ge4_pass_of_44": int(pd["null_criteria_among_44_neg_inh"]["shared_all3_hop2_ge_4"]["k_pass_of_44"]),
            "test_d_hop3_shared_ge4_pass_of_44": int(pd["null_criteria_among_44_neg_inh"]["shared_all3_hop3_ge_4"]["k_pass_of_44"]),
            "test_d_hop3_shared_ge5_pass_of_44": int(pd["null_criteria_among_44_neg_inh"]["shared_all3_hop3_ge_5"]["k_pass_of_44"]),
        },
        "cross_lab_and_selk": {
            "non_scott_studies_audited": 13,
            "admitted_upstream_sez_types": 2,
            "selk_banc_vnc_frac": selk_vnc_syn / selk_tot_syn,
            "selk_banc_top15_inh_count": selk_top15_inh,
            "selk_banc_top15_asc_gaba_count": selk_top15_asc_gaba,
            "selk_banc_via_vnc_exc_bottleneck": float(selk_b["selk_exc_shortest_paths_to_mn9"]["forced_via_vnc"]["bottleneck_weight_from_selk"]),
            "selk_malecns_vnc_frac": m_selk_vnc / m_selk_tot,
            "selk_malecns_valid_inh_frac": m_selk_valid_i / (m_selk_valid_e + m_selk_valid_i),
            "selk_malecns_top15_inh_count": m_selk_top15_inh,
            "selk_malecns_top15_valid_count": m_selk_top15_valid,
            "selk_malecns_via_vnc_exc_bottleneck": float(m_selk["selk_exc_shortest_paths_to_mn9"]["forced_via_vnc"]["bottleneck_weight_from_selk"]),
        },
        "disinhibition_and_states": {
            "n_v783_bg_regimes": len(bg_states),
            "n_v783_bg_specificity_preserved": n_v783_bg_preserve,
            "n_v783_bg_unlock_while_preserving_specificity": n_v783_bg_unlock_preserve,
            "n_banc_bg_regimes": len(banc_bg),
            "n_banc_bg_unlock_focal": n_banc_bg_unlock_focal,
            "anat_percentiles": {
                **{
                    brain: {
                        cand: {
                            "hop2_net_pct": float(disinh["module1_anatomy"][f"percentiles_vs_44_inh_neg_{brain}"]["hop2_net_disinh_minus_antag_prod"]["candidates"][cand]["pct_among_44_neg_inh"]),
                            "hop3_disinh_pct": float(disinh["module1_anatomy"][f"percentiles_vs_44_inh_neg_{brain}"]["hop3_disinh_epre_prod_sum"]["candidates"][cand]["pct_among_44_neg_inh"]),
                        }
                        for cand in ("tentacular", "phantom", "marge", "usnea")
                    }
                    for brain in ("v783", "banc")
                },
                "malecns": {
                    cand: {
                        "hop2_net_pct": float(m4_pct["hop2_net_disinh_minus_antag_prod"]["candidates"][cand]["pct_among_mapped_neg_inh"]),
                        "hop3_disinh_pct": float(m4_pct["hop3_disinh_epre_prod_sum"]["candidates"][cand]["pct_among_mapped_neg_inh"]),
                    }
                    for cand in ("tentacular", "phantom", "marge", "usnea")
                },
            },
            "three_brain_overlap_counts": {
                cand: {k: len(v) for k, v in ov.items()} for cand, ov in m4["three_brain_top15_intermediate_overlap"].items()
            },
            "oracle_total_jobs_3brains": len(ora) + len(ora_m),
            "oracle_summary": {
                "tentacular_v783_banc": count_oracle_dict(ora, None, "tentacular"),
                "tentacular_malecns": count_oracle_dict(ora_m, "malecns", "tentacular"),
                "phantom_v783_banc": count_oracle_dict(ora, None, "phantom"),
                "phantom_malecns": count_oracle_dict(ora_m, "malecns", "phantom"),
                "selk_malecns_via_vnc_200hz": count_oracle_dict(ora_m, "malecns", "SELK_DNg68", "via_vnc", "200hz"),
            },
        },
        "revision_hardening": {
            "selk_vnc_exact_widest_paths": {
                "banc_by_hop": {k: v["bottleneck_weight"] for k, v in port["selk_vnc_exact_widest_paths"]["banc_v888"]["via_any_vnc_by_hop"].items()},
                "banc_global_max": port["selk_vnc_exact_widest_paths"]["banc_v888"]["global_widest_via_any_vnc"]["bottleneck_weight"],
                "banc_global_hops": port["selk_vnc_exact_widest_paths"]["banc_v888"]["global_widest_via_any_vnc"]["hops"],
                "malecns_primary_by_hop": {k: v["bottleneck_weight"] for k, v in port["selk_vnc_exact_widest_paths"]["malecns_v1_0"]["primary_consensus_nt"]["via_any_vnc_by_hop"].items()},
                "malecns_a2_by_hop": {k: v["bottleneck_weight"] for k, v in port["selk_vnc_exact_widest_paths"]["malecns_v1_0"]["a2_predicted_nt"]["via_any_vnc_by_hop"].items()},
                "malecns_global_max": port["selk_vnc_exact_widest_paths"]["malecns_v1_0"]["primary_consensus_nt"]["global_widest_via_any_vnc"]["bottleneck_weight"],
                "malecns_global_hops": port["selk_vnc_exact_widest_paths"]["malecns_v1_0"]["primary_consensus_nt"]["global_widest_via_any_vnc"]["hops"],
            },
            "rewire_compartment_scatter_audit": port["rewire_compartment_scatter_audit"],
            "test_a_nondirect_variant_audit": port["test_a_nondirect_variant_audit"],
            "test_a2_b3_combined_summary": {
                "r_med_cb_a2": port["test_a2_b3_combined"]["r_med_cb_a2"],
                "w_syn_a2_b3_mv": port["test_a2_b3_combined"]["w_syn_a2_b3_mv"],
                "50hz": {
                    k: port["test_a2_b3_combined"]["runs"]["a2_density_matched"]["50hz"][k]
                    for k in ("all_106_auprc", "all_106_ci95", "novel_95_auprc", "novel_95_ci95", "nondirect_malecns_auprc", "nondirect_excl_thvum_87_auprc", "nondirect_variant_auprc", "nondirect_variant_ci95", "v783_spearman_rho", "v783_spearman_rho_excl_roundup", "mean_brain_hz", "nonzero_106")
                },
                "200hz": {
                    k: port["test_a2_b3_combined"]["runs"]["a2_density_matched"]["200hz"][k]
                    for k in ("all_106_auprc", "all_106_ci95", "novel_95_auprc", "novel_95_ci95", "nondirect_variant_auprc", "v783_spearman_rho", "mean_brain_hz")
                },
                "selk_20_to_200hz": {
                    f: port["test_a2_b3_combined"]["runs"]["a2_density_matched"]["selk_dng68_20_to_200hz"][f]["mn9_mean_hz"]
                    for f in ("20hz", "50hz", "100hz", "200hz")
                },
            },
            "compartment_preserving_rewire_v783": {
                "fixed_0275": port["test_c_v783_compartment_preserving_rewire"]["fixed_0275_summary"],
                "activity_matched": port["test_c_v783_compartment_preserving_rewire"]["c2_activity_matched_summary"],
            },
            "secondary_bh_fdr_16tests": port["secondary_bh_fdr_16tests_with_compartment"],
            "test_d_strict_3brain_inh_sensitivity": port["test_d_strict_3brain_inh_sensitivity"],
            "malecns_calibration_paired_bootstrap_50hz": port["malecns_calibration_paired_bootstrap_50hz"],
            "unpublished_95_key_positive_ranks_50hz": port["unpublished_95_key_positive_ranks_50hz"],
        },
    }

    # Programmatic assertions covering all locked + hardening + revision headline metrics
    assert abs(est["v630_to_v783"]["pearson_r_50hz"] - 0.9999024) < 1e-5
    assert abs(est["v630_to_v783"]["v783_all106_auprc"] - 0.7215201) < 1e-5
    assert abs(est["v630_to_v783"]["v783_nondirect83_auprc"] - 0.6461975) < 1e-5
    assert abs(est["v630_to_v783"]["v783_novel95_auprc"] - 0.5210526) < 1e-5
    assert abs(est["banc_cross_individual"]["m1_anat_novel95_50hz_auprc"] - 0.1149123) < 1e-5
    assert abs(est["malecns_third_connectome"]["intact_novel95_50hz_auprc"] - 0.0815789) < 1e-5
    ph = est["portability_hardening"]
    assert abs(ph["test_a_A2_50hz_all106"] - 0.6534) < 1e-3
    assert abs(ph["test_a_A2_50hz_novel95"] - 0.4377) < 1e-3
    assert ph["test_a_A2_selk_50hz"] == 0.0
    assert abs(ph["test_b_banc_cb_median_ratio"] - 0.4508) < 1e-3
    assert abs(ph["test_b_malecns_cb_median_ratio"] - 1.7253) < 1e-3
    assert ph["test_b_B3_50hz_fp6_silenced"] == 5
    assert ph["test_b_fp6_with_dimorphism_or_fru"] == 0
    assert ph["test_c_all106_real_wins"] == 10
    assert ph["test_c_nd83_real_wins"] == 10
    assert ph["test_c_nov95_real_wins"] == 10
    assert ph["test_d_tentacular_joint_pass_of_44"] == 0
    assert ph["test_d_phantom_joint_pass_of_44"] == 0
    # Programmatic assertion of phantom 16/18 any-freq & 13/18 both-freq Oracle unlock counts
    os_ph = est["disinhibition_and_states"]["oracle_summary"]
    assert os_ph["phantom_v783_banc"]["unlocked_any"] + os_ph["phantom_malecns"]["unlocked_any"] == 16
    assert os_ph["phantom_v783_banc"]["unlocked_both_50_200"] + os_ph["phantom_malecns"]["unlocked_both_50_200"] == 13
    rh = est["revision_hardening"]
    assert abs(rh["test_a2_b3_combined_summary"]["50hz"]["all_106_auprc"] - 0.5926) < 1e-3
    assert abs(rh["test_a2_b3_combined_summary"]["50hz"]["v783_spearman_rho"] - 0.8095) < 1e-3
    assert rh["test_a2_b3_combined_summary"]["selk_20_to_200hz"]["50hz"] == 0.0
    s36 = rh["test_d_strict_3brain_inh_sensitivity"]
    assert s36["n_controls_strict36"] == 36
    assert abs(s36["candidates"]["tentacular"]["hop2_net_pct_3brains_strict36"][2] - 88.8889) < 1e-2
    assert abs(s36["candidates"]["tentacular"]["hop3_disinh_pct_3brains_strict36"][2] - 97.2222) < 1e-2
    assert s36["null_criteria_among_36"]["joint_tentacular_3h_only"]["k"] == 0
    assert s36["null_criteria_among_36"]["joint_phantom_3h"]["k"] == 0
    pb_mc = rh["malecns_calibration_paired_bootstrap_50hz"]
    assert abs(pb_mc["predNT_vs_primary_all106"]["delta_point"] - 0.1927) < 1e-3
    assert abs(pb_mc["joint_vs_primary_all106"]["delta_point"] - 0.1320) < 1e-3

    # Pre-submission Experiments A & B + MaleCNS LB3a water audit
    ab = data["ab"]
    b1_50 = ab["phase_b1_v783"]["experiment_b1_diatom_downsampling"]["50hz"]
    ab_v783_int = ab["phase_b1_v783"]["v783_intrinsic_inv_sin_reference"]["sweeps"]["50hz"]
    ab_b_sug = ab["phase_banc"]["a1_sugar_anchor"]["sweeps"]["50hz"]
    ab_b_hom = ab["phase_banc"]["a2a_homolog_per_neuron_norm"]["sweeps"]["50hz"]
    ab_b_int = ab["phase_banc"]["a2b_intrinsic_inv_sin_norm"]["sweeps"]["50hz"]
    ab_m_b2 = ab["phase_malecns"]["experiment_b2_fixed_02683_prednt"]["sweeps"]["50hz"]
    ab_m_r17c = ab["phase_malecns"]["a1_sugar_anchor_consensus"]["sweeps_at_right17_wstar"]["50hz"]
    ab_m_r17p = ab["phase_malecns"]["a1_sugar_anchor_prednt"]["sweeps_at_right17_wstar"]["50hz"]
    ab_m_homc = ab["phase_malecns"]["a2a_homolog_per_neuron_norm_cons"]["sweeps"]["50hz"]
    ab_m_homp = ab["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]
    ab_m_int = ab["phase_malecns"]["a2b_intrinsic_inv_sin_norm_cons"]["sweeps"]["50hz"]
    est["pre_submission_ab"] = {
        "b1_diatom_50hz": {
            "exact_banc_matched_3_hz": float(b1_50["exact_banc_matched_3"]["rate_hz_median"]),
            "random_4_of_14_frac_zero": float(b1_50["random_4_of_14"]["frac_exact_zero"]),
            "random_4_of_14_unpub95_median": float(b1_50["random_4_of_14"]["auprc_unpub95_median"]),
            "random_7_of_14_frac_zero": float(b1_50["random_7_of_14"]["frac_exact_zero"]),
            "random_7_of_14_unpub95_median": float(b1_50["random_7_of_14"]["auprc_unpub95_median"]),
        },
        "v783_a2b_intrinsic_50hz_all106": float(ab_v783_int["auprc_all106"]),
        "banc_a1_sugar_50hz_all106": float(ab_b_sug["auprc_all106"]),
        "banc_a2a_homolog_50hz_all106": float(ab_b_hom["auprc_all106"]),
        "banc_a2a_homolog_50hz_unpub95": float(ab_b_hom["auprc_unpub95"]),
        "banc_a2b_intrinsic_50hz_all106": float(ab_b_int["auprc_all106"]),
        "malecns_b2_fixed02683_50hz_all106": float(ab_m_b2["auprc_all106"]),
        "malecns_r17_sugar_cons_50hz_all106": float(ab_m_r17c["auprc_all106"]),
        "malecns_r17_sugar_prednt_50hz_all106": float(ab_m_r17p["auprc_all106"]),
        "malecns_a2a_homolog_cons_50hz_all106": float(ab_m_homc["auprc_all106"]),
        "malecns_a2a_homolog_prednt_50hz_all106": float(ab_m_homp["auprc_all106"]),
        "malecns_a2a_homolog_prednt_50hz_rho": float(ab_m_homp["spearman_all106"]),
        "malecns_a2b_intrinsic_50hz_all106": float(ab_m_int["auprc_all106"]),
        "fixed_common_coactive_50hz": ab["fixed_common_coactive_50hz"],
        "paired_bootstraps_100k_seed42": ab["paired_bootstraps_100k_seed42"],
    }
    assert abs(est["pre_submission_ab"]["banc_a2a_homolog_50hz_all106"] - 0.5891) < 1e-3
    assert abs(est["pre_submission_ab"]["malecns_b2_fixed02683_50hz_all106"] - 0.6475) < 1e-3
    assert abs(est["pre_submission_ab"]["malecns_a2a_homolog_prednt_50hz_all106"] - 0.7215) < 1e-3
    assert abs(est["pre_submission_ab"]["malecns_a2a_homolog_prednt_50hz_rho"] - 0.8912) < 1e-3
    assert abs(est["pre_submission_ab"]["v783_a2b_intrinsic_50hz_all106"] - 0.3667) < 1e-3

    # V5 reviewer controls (M1 TOST, M2 anti-circularity, M3 label-free & 10-test BH-FDR, M4 fair 2-hop, M5 K=100, Minor #6 BANC cell-type)
    v5 = data["v5"]
    b_ct = v5["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]
    m_p2 = v5["phase2_malecns"]
    m_p3 = v5["phase3_v783_k100"]
    m_p4 = v5["phase4_m3_m6_unified"]
    est["v5_reviewer_controls"] = {
        "m1_tost_hom_prednt_vs_v783": m_p4["m1_tost_hom_prednt_vs_v783"],
        "minor6_banc_celltype_homolog_frac": float(b_ct["frac_matched"]),
        "minor6_banc_celltype_50hz_all106": float(b_ct["sweeps"]["50hz"]["auprc_all106"]),
        "minor6_banc_celltype_50hz_auroc": float(b_ct["sweeps"]["50hz"]["auroc_all106"]),
        "m4_malecns_fair_2hop_predNT_hom_signed_0p01": float(m_p2["m4_malecns_fair_twohop"]["predicted_nt_homolog_weighted"]["signed_relu"]["alpha_0.01"]["all106_auprc"]),
        "m4_malecns_fair_2hop_predNT_hom_unsigned_0p01": float(m_p2["m4_malecns_fair_twohop"]["predicted_nt_homolog_weighted"]["unsigned"]["alpha_0.01"]["all106_auprc"]),
        "m4_malecns_bootstraps_100k": m_p2["m4_malecns_bootstraps_100k"],
        "m2a_shuffle_non_mn9_mean": float(m_p2["m2_summary_bootstraps"]["M2a_shuffle_nonMN9_mean_auprc"]),
        "m2a_shuffle_all_mean": float(m_p2["m2_summary_bootstraps"]["M2a_shuffle_all_mean_auprc"]),
        "m2b_leave_out_sez106_and_mn9_auprc": float(m_p2["m2b_leave_out_sez106_and_mn9"]["auprc_all106"]),
        "m2b_two_scale_auprc": float(m_p2["m2b_two_scale_mn9_02638_rest_01567"]["auprc_all106"]),
        "m2c_banc_ref_to_malecns_auprc": float(m_p2["m2c_banc_ref_to_malecns"]["auprc_all106"]),
        "m5_k100_global_all106_wins": int(m_p3["m5_v783_fixed_weight_k100"]["global"]["all106"]["real_wins"]),
        "m5_k100_global_all106_p": float(m_p3["m5_v783_fixed_weight_k100"]["global"]["all106"]["p_emp_one_sided"]),
        "m5_k100_comp_all106_wins": int(m_p3["m5_v783_fixed_weight_k100"]["compartment"]["all106"]["real_wins"]),
        "m5_k100_comp_all106_p": float(m_p3["m5_v783_fixed_weight_k100"]["compartment"]["all106"]["p_emp_one_sided"]),
        "m3_label_free_top3": m_p4["label_free_top3_by_connectome"],
        "m3_unified_10test_bh_fdr": m_p4["unified_calibration_bh_fdr_10test"],
    }
    assert abs(est["v5_reviewer_controls"]["minor6_banc_celltype_50hz_all106"] - 0.5894) < 1e-3
    assert abs(est["v5_reviewer_controls"]["m4_malecns_fair_2hop_predNT_hom_signed_0p01"] - 0.5452) < 1e-3
    assert abs(est["v5_reviewer_controls"]["m2b_leave_out_sez106_and_mn9_auprc"] - 0.6994) < 1e-3
    assert abs(est["v5_reviewer_controls"]["m2b_two_scale_auprc"] - 0.5926) < 1e-3
    assert abs(est["v5_reviewer_controls"]["m2c_banc_ref_to_malecns_auprc"] - 0.6398) < 1e-3
    assert est["v5_reviewer_controls"]["m5_k100_global_all106_wins"] == 100
    assert est["v5_reviewer_controls"]["m5_k100_comp_all106_wins"] == 100
    assert est["v5_reviewer_controls"]["m3_label_free_top3"]["MaleCNS"][0] == "MaleCNS_Homolog_PredNT"
    assert est["v5_reviewer_controls"]["m3_label_free_top3"]["BANC"][0] == "BANC_Homolog_CellType"
    return est


def make_fig1(data: dict[str, dict], est: dict, preview_dir: Path | None) -> None:
    """Fig. 1: Proofreading transfer (v630->v783), multi-hop specificity, & water silencing (v783 + BANC)."""
    v783 = data["v783"]
    banc = data["banc"]
    port = data["port"]
    c2 = port["test_c_v783_activity_matched_rewire"]["c2_activity_matched_summary"]

    fig = plt.figure(figsize=(12.2, 7.45), layout="constrained")
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.18], height_ratios=[1.05, 0.96], hspace=0.26, wspace=0.18)

    # --- Panel A (top-left): v630 vs v783 rate conservation ---
    ax = fig.add_subplot(gs[0, 0])
    recs = v783["per_type_task1"]
    x630 = np.array([r["v630_real_lif_50hz_mean"] for r in recs], dtype=float)
    y783 = np.array([r["v783_real_lif_50hz_mean"] for r in recs], dtype=float)
    pos = np.array([r["y_max"] > 0.0 for r in recs], dtype=bool)

    lim = max(x630.max(), y783.max()) * 1.08
    ax.plot([0, lim], [0, lim], ls="--", lw=1.0, color=SLATE_LIGHT, zorder=1)
    ax.scatter(x630[~pos], y783[~pos], s=22, fc="white", ec=SLATE, lw=0.85, alpha=0.85, label="Negative (n = 92)", zorder=2)
    ax.scatter(x630[pos], y783[pos], s=36, fc=TEAL, ec="white", lw=0.5, label="PER positive (n = 14)", zorder=3)

    offsets = {
        "roundup": (-12, -12),
        "sink_sync": (-46, 5),
        "diatom": (7, -9),
        "Fdg": (-26, 6),
        "bract": (7, -6),
        "clavicle": (8, -11),
        "G2N_1": (-38, 6),
        "kitty": (46, -4),
    }
    for r in recs:
        nm = r["name"]
        if nm in offsets:
            dx, dy = offsets[nm]
            lbl = "kitty (neg FP)" if nm == "kitty" else nm
            ax.annotate(
                lbl,
                (r["v630_real_lif_50hz_mean"], r["v783_real_lif_50hz_mean"]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=6.9,
                color=CRIMSON if nm == "kitty" else DARK,
                arrowprops=(
                    dict(arrowstyle="-", color=CRIMSON, lw=0.6)
                    if nm == "kitty"
                    else (dict(arrowstyle="-", color=SLATE_LIGHT, lw=0.55) if nm in ("sink_sync", "Fdg", "G2N_1") else None)
                ),
            )

    ax.text(
        0.04, 0.78,
        f"r = {est['v630_to_v783']['pearson_r_50hz']:.5f}\n"
        f"MAE = {est['v630_to_v783']['mae_hz']:.3f} Hz\n"
        f"0 sign flips (N = 106)",
        transform=ax.transAxes, fontsize=7.3, va="top",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f4f8f8", ec=SLATE_LIGHT, lw=0.7),
    )
    ax.set(
        xlim=(-2.5, lim), ylim=(-2.5, lim),
        xlabel="FlyWire v630 MN9 rate (Hz, recomputed)",
        ylabel="FlyWire v783 MN9 rate (Hz)",
        title="A   Proofreading transfer (v630 → v783)",
    )
    ax.legend(frameon=False, loc="lower right", handletextpad=0.3, fontsize=7.0)

    # --- Panel B (top-right): v783 Multi-hop specificity vs K=100 fixed nulls & K=10 gain-adjusted nulls ---
    ax = fig.add_subplot(gs[0, 1])
    comp_c2 = port["test_c_v783_compartment_preserving_rewire"]["c2_activity_matched_summary"]
    k100 = data["v5"]["phase3_v783_k100"]["m5_v783_fixed_weight_k100"]
    subsets = [
        ("All 106 types\n(14/106 pos)", v783["task1_activation_v783"]["all_106"], "all106", c2["all_106"], comp_c2["all_106"], 0.1321),
        ("Non-direct ≥2-hop\n(9/83 pos)", v783["task1_activation_v783"]["nondirect_v783"], "nondirect83", c2["nondirect_83"], comp_c2["nondirect_83"], 0.1084),
    ]
    model_specs = [
        ("Frozen LIF (50 Hz)", TEAL, "real"),
        ("Signed 2-hop ReLU", ORANGE, "static"),
        ("Global 0.275 mV (K=100)", INDIGO, "k100_glob"),
        ("Compartment 0.275 mV (K=100)", "#2c7fb8", "k100_comp"),
        ("Global gain-adj. W* (K=10)", CRIMSON, "rew_matched"),
        ("Compartment gain-adj. W* (K=10)", PURPLE, "rew_comp"),
    ]
    x_centers = np.array([0.0, 2.02])
    bar_w = 0.23
    rng = np.random.default_rng(7)

    # Uniform B=100,000 seed-42 stratified bootstrap CIs for Fig 1B (matching Fig 2C-D & S2A)
    pt3_f1 = data["malecns"]["per_type_3brains"]
    y_f1 = np.array([int(r["y_bin"]) for r in pt3_f1], dtype=int)
    v7_f1 = np.array([float(r["v783_50hz"]) for r in pt3_f1], dtype=float)
    s2_f1 = np.array([float(r["v783_static_2hop_relu"]) for r in pt3_f1], dtype=float)
    nd_f1 = np.array([not bool(r["is_direct_v783"]) for r in pt3_f1], dtype=bool)
    f1b_ci100k = {
        ("all106", "real"): _strat_auprc_ci95(y_f1, v7_f1, n_boot=100000, seed=42),
        ("all106", "static"): _strat_auprc_ci95(y_f1, s2_f1, n_boot=100000, seed=42),
        ("nondirect83", "real"): _strat_auprc_ci95(y_f1[nd_f1], v7_f1[nd_f1], n_boot=100000, seed=42),
        ("nondirect83", "static"): _strat_auprc_ci95(y_f1[nd_f1], s2_f1[nd_f1], n_boot=100000, seed=42),
    }

    for s_idx, (s_label, s_obj, k100_key, c2_obj, comp_obj, chance) in enumerate(subsets):
        xc = x_centers[s_idx]
        ax.hlines(chance, xc - 0.80, xc + 0.80, colors=SLATE, linestyles=":", lw=1.0, zorder=1)
        ax.text(xc + 0.82, chance, "chance", fontsize=6.3, color=SLATE, va="center")
        for m_idx, (mlab, col, mtype) in enumerate(model_specs):
            xp = xc + (m_idx - 2.5) * (bar_w + 0.035)
            lab = mlab if s_idx == 0 else None
            if mtype == "real":
                pt = s_obj["bootstrap"]["real_lif_50hz"]["point"]["auprc"]
                lo, hi = f1b_ci100k[(k100_key, "real")]
                ax.bar(xp, pt, width=bar_w, color=col, alpha=0.88, label=lab, zorder=2)
                ax.errorbar(xp, pt, yerr=[[pt - lo], [hi - pt]], fmt="none", ecolor=DARK, elinewidth=0.95, capsize=2.4, zorder=4)
                ax.text(xp, hi + 0.022, f"{pt:.3f}", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=TEAL)
            elif mtype == "static":
                pt = s_obj["bootstrap"]["static_2hop_relu_signed"]["point"]["auprc"]
                lo, hi = f1b_ci100k[(k100_key, "static")]
                ax.bar(xp, pt, width=bar_w, color=col, alpha=0.82, label=lab, zorder=2)
                ax.errorbar(xp, pt, yerr=[[pt - lo], [hi - pt]], fmt="none", ecolor=DARK, elinewidth=0.95, capsize=2.4, zorder=4)
                ax.text(xp, hi + 0.022, f"{pt:.3f}", ha="center", va="bottom", fontsize=6.6, color=ORANGE)
            elif mtype in ("k100_glob", "k100_comp"):
                mode = "global" if mtype == "k100_glob" else "compartment"
                pg = k100[mode]["per_graph"]
                field = "all106_auprc" if k100_key == "all106" else "nondirect83_auprc"
                vals = np.array([r[field] for r in pg], dtype=float)
                mean_v = float(vals.mean())
                ax.bar(xp, mean_v, width=bar_w, color=col, alpha=0.25, edgecolor=col, lw=0.9, label=lab, zorder=2)
                ax.scatter(xp + rng.uniform(-0.065, 0.065, len(vals)), vals, s=5, color=col, alpha=0.45, edgecolors="none", zorder=4)
                ax.scatter([xp], [mean_v], s=26, marker="D", color=col, edgecolors=DARK, lw=0.5, zorder=5)
                ax.text(xp, max(vals.max(), mean_v) + 0.024, f"{mean_v:.3f}\n100/100\np=.0099", ha="center", va="bottom", fontsize=5.2, color=col)
            elif mtype == "rew_matched":
                vals = np.array(c2_obj["per_graph_auprc_list"], dtype=float)
                mean_v = float(vals.mean())
                wins = int(c2_obj["real_wins_out_of_10"])
                ax.bar(xp, mean_v, width=bar_w, color=col, alpha=0.22, edgecolor=col, lw=1.0, ls="--", label=lab, zorder=2)
                ax.scatter(xp + rng.uniform(-0.05, 0.05, len(vals)), vals, s=11, color=col, alpha=0.85, edgecolors="white", lw=0.3, zorder=4)
                ax.scatter([xp], [mean_v], s=26, marker="D", color=col, edgecolors=DARK, lw=0.5, zorder=5)
                ax.text(xp, max(vals.max(), mean_v) + 0.024, f"{mean_v:.3f}\n({wins}/10;\n6 conv)", ha="center", va="bottom", fontsize=5.2, color=col)
            else:
                vals = np.array(comp_obj["per_graph"], dtype=float)
                mean_v = float(vals.mean())
                wins = int(comp_obj["real_wins_of_10"])
                ax.bar(xp, mean_v, width=bar_w, color=col, alpha=0.22, edgecolor=col, lw=1.0, ls="-.", label=lab, zorder=2)
                ax.scatter(xp + rng.uniform(-0.05, 0.05, len(vals)), vals, s=11, color=col, alpha=0.85, edgecolors="white", lw=0.3, zorder=4)
                ax.scatter([xp], [mean_v], s=26, marker="D", color=col, edgecolors=DARK, lw=0.5, zorder=5)
                ax.text(xp, max(vals.max(), mean_v) + 0.024, f"{mean_v:.3f}\n({wins}/10;\n5 conv)", ha="center", va="bottom", fontsize=5.2, color=col)

    ax.set(
        xticks=x_centers, xticklabels=[s[0] for s in subsets],
        xlim=(-0.92, 3.16), ylim=(0.0, 1.38),
        ylabel="Stepwise AUPRC",
        title="B   Multi-hop specificity vs K=100 fixed & K=10 gain-adj. nulls",
    )
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=6.2, columnspacing=0.65, handletextpad=0.3)

    # --- Panel C (bottom-left): FlyWire v783 water silencing (all 11 types) ---
    ax = fig.add_subplot(gs[1, 0])
    w_recs = sorted(v783["task2_water_silencing_v783"]["per_type_records"], key=lambda r: r["v783_160hz_rel_change_mn9_r"])
    y_pos = np.arange(len(w_recs))
    vals = [100.0 * r["v783_160hz_rel_change_mn9_r"] for r in w_recs]
    cols = [CRIMSON if r["name"] == "Usnea" else (TEAL if r["sig_impaired"] else SLATE_LIGHT) for r in w_recs]

    ax.barh(y_pos, vals, color=cols, height=0.68, zorder=2)
    ax.axvline(0.0, color=DARK, lw=0.8, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['name']}*" if r["name"] == "Usnea" else r["name"] for r in w_recs], fontsize=7.5)
    ax.invert_yaxis()
    ax.set(xlim=(-118, 185), xlabel="FlyWire v783 right MN9 rate change (%)", title="C   FlyWire v783 water silencing (all 11 types)")
    ax.text(
        0.025, 0.04,
        f"11-type AUROC = {est['v630_to_v783']['v783_water11_auroc']:.3f}\n"
        f"U = {est['v630_to_v783']['v783_water11_mwu_u']:.1f}, p = {est['v630_to_v783']['v783_water11_mwu_p']:.4f}\n"
        f"5/6 impaired suppressed\n"
        f"20/20 rewired: 0.00 Hz base",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.30", fc="#f4f8f8", ec=SLATE_LIGHT, lw=0.7),
    )
    leg_p = [
        mpatches.Patch(color=TEAL, label="Impaired (5 non-pep)"),
        mpatches.Patch(color=CRIMSON, label="Impaired (Usnea*, pep)"),
        mpatches.Patch(color=SLATE_LIGHT, label="Unimpaired (5)"),
    ]
    ax.legend(handles=leg_p, frameon=False, loc="upper right", fontsize=6.5)

    # --- Panel D (bottom-right): BANC v888 water silencing (all 11 types, moved from old Fig. 2D) ---
    ax = fig.add_subplot(gs[1, 1])
    r160 = banc["task2_water_silencing_banc_bounded"]["silencing_runs"]["right_unilateral_14_anat_scale_160hz"]
    w_meta = v783["task2_water_silencing_v783"]["per_type_records"]
    rows = [{"name": m["name"], "sig": m["sig_impaired"], "rel_l": 100.0 * rc} for m, rc in zip(w_meta, r160["rel_change_mn9_l"])]
    rows.sort(key=lambda r: r["rel_l"])
    y_pos = np.arange(len(rows))
    cols = [CRIMSON if r["name"] == "Usnea" else (ORANGE if r["sig"] else SLATE_LIGHT) for r in rows]
    ax.barh(y_pos, [r["rel_l"] for r in rows], color=cols, height=0.68, zorder=2)
    ax.axvline(0.0, color=DARK, lw=0.8, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['name']}*" if r["name"] == "Usnea" else r["name"] for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    z_idx = next(i for i, r in enumerate(rows) if r["name"] == "Zorro")
    ax.annotate(
        "+0.20 Hz (40→46 spikes)\nlow-baseline fluctuation",
        xy=(15.0, z_idx), xytext=(22.0, z_idx - 1.35),
        fontsize=6.1, color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=0.7),
    )
    ax.set(xlim=(-102, 85), xlabel="BANC v888 left MN9 rate change (%)", title="D   BANC v888 water silencing (all 11 types; base = 1.33 Hz, 40 spikes)")
    m_l = r160["metrics_by_readout"]["mn9_l_contra_to_right"]["all_11"]
    ax.text(
        0.025, 0.04,
        f"11-type AUROC = {m_l['auroc']:.3f}\n"
        f"U = {m_l['mannwhitney_u']:.1f}, p = {m_l['mannwhitney_p_twosided']:.4f}\n"
        f"4/6 impaired suppressed\n"
        f"(Usnea* & Zorro opposite)\n"
        f"10/10 MN9-prot rew: 0.00 Hz",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6.4,
        bbox=dict(boxstyle="round,pad=0.30", fc="#fdf7f2", ec=ORANGE_LIGHT, lw=0.7),
    )
    leg_d = [
        mpatches.Patch(color=ORANGE, label="Impaired (5 non-pep)"),
        mpatches.Patch(color=CRIMSON, label="Impaired (Usnea*, pep)"),
        mpatches.Patch(color=SLATE_LIGHT, label="Unimpaired (5)"),
    ]
    ax.legend(handles=leg_d, frameon=False, loc="upper right", fontsize=6.5)

    save_figure(fig, "fig1", preview_dir)


def make_fig2(data: dict[str, dict], est: dict, preview_dir: Path | None) -> None:
    """Fig. 2: Cross-connectome transfer — BANC & MaleCNS scatters (top) + split default-transfer (C) & calibration/control (D) panels."""
    banc = data["banc"]
    mcns = data["malecns"]
    port = data["port"]
    ab = data["ab"]
    v5 = data["v5"]
    pb = port["test_b_density_and_dimorphism"]

    fig = plt.figure(figsize=(13.2, 8.75), layout="constrained")
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.06], height_ratios=[0.96, 1.08], hspace=0.28, wspace=0.16)

    xticks_hz = [0, 1, 5, 20, 80]

    # --- Panel A (top-left): FlyWire v783 vs BANC v888 + MN9 L/R & CB density inset ---
    ax = fig.add_subplot(gs[0, 0])
    recs = banc["per_type_task1"]
    xv = np.array([r["v783_50hz"] for r in recs], dtype=float)
    yf = np.array([r["banc_m1_frozen_50hz"] for r in recs], dtype=float)
    ya = np.array([r["banc_m1_anat_50hz"] for r in recs], dtype=float)
    pos = np.array([r["y_bin"] == 1 for r in recs], dtype=bool)
    lx, lyf, lya = np.log1p(xv), np.log1p(yf), np.log1p(ya)

    ax.plot([0, math.log1p(45.0)], [0, math.log1p(45.0)], ls="--", lw=0.9, color=SLATE_LIGHT, zorder=1)
    ax.axhline(0.0, color=GRID, lw=0.7, zorder=0)
    ax.axvline(0.0, color=GRID, lw=0.7, zorder=0)
    for r in recs:
        if r["y_bin"] and (r["banc_m1_anat_50hz"] - r["banc_m1_frozen_50hz"]) > 0.25:
            x0 = math.log1p(r["v783_50hz"])
            ax.plot([x0, x0], [math.log1p(r["banc_m1_frozen_50hz"]), math.log1p(r["banc_m1_anat_50hz"])], color=ORANGE_LIGHT, lw=0.9, zorder=2)

    ax.scatter(lx[pos], lyf[pos], s=22, marker="^", fc="none", ec=SLATE, lw=0.9, alpha=0.9, label="Raw 0.275 mV (pos)", zorder=3)
    ax.scatter(lx[~pos], lya[~pos], s=19, fc="white", ec=SLATE_LIGHT, lw=0.8, alpha=0.85, label="Target 0.690 mV (neg)", zorder=2)
    ax.scatter(lx[pos], lya[pos], s=38, fc=ORANGE, ec="white", lw=0.5, label="Target 0.690 mV (pos)", zorder=4)

    ann_pts = {
        "roundup": (-22, -16), "rattle": (7, -9), "bract": (7, 6),
        "clavicle": (-30, 8), "G2N_1": (7, -7), "sink_sync": (12, 12),
        "Fdg": (-26, 10), "diatom": (20, 4),
    }
    for r in recs:
        nm = r["name"]
        if nm in ann_pts:
            dx, dy = ann_pts[nm]
            ax.annotate(
                nm, (math.log1p(r["v783_50hz"]), math.log1p(r["banc_m1_anat_50hz"])),
                xytext=(dx, dy), textcoords="offset points", fontsize=6.9,
                fontweight="bold" if nm in ("diatom", "rattle", "Fdg") else "normal",
                color=CRIMSON if nm == "diatom" else DARK,
                arrowprops=dict(arrowstyle="-", color=CRIMSON if nm == "diatom" else SLATE_LIGHT, lw=0.6)
                if nm in ("diatom", "Fdg", "clavicle", "sink_sync") else None,
            )

    ax_in = ax.inset_axes([0.055, 0.59, 0.43, 0.36])
    x_in = np.arange(3)
    w_in = 0.34
    l_vals = [est["banc_cross_individual"]["s_in_mn9_v783_l"], est["banc_cross_individual"]["s_in_mn9_banc_l"], est["malecns_third_connectome"]["s_in_mn9_l"]]
    r_vals = [est["banc_cross_individual"]["s_in_mn9_v783_r"], est["banc_cross_individual"]["s_in_mn9_banc_r"], est["malecns_third_connectome"]["s_in_mn9_r"]]
    ax_in.bar(x_in - w_in / 2, l_vals, width=w_in, color=[TEAL, ORANGE, INDIGO])
    ax_in.bar(x_in + w_in / 2, r_vals, width=w_in, color=[TEAL_LIGHT, ORANGE_LIGHT, CRIMSON])
    ax_in.text(2 + w_in / 2 + 0.04, 720, "521\n11.3×↓", ha="center", va="bottom", fontsize=5.8, color=CRIMSON, fontweight="bold")
    ax_in.text(0, 6100, "5.7k", ha="center", fontsize=6.0, color=TEAL, fontweight="bold")
    ax_in.text(1, 2820, "2.3k", ha="center", fontsize=6.0, color=ORANGE, fontweight="bold")
    ax_in.text(2 - w_in / 2, 6080, "5.9k", ha="center", fontsize=6.0, color=INDIGO, fontweight="bold")
    ax_in.set_xlim(-0.55, 2.72)
    ax_in.set_xticks(x_in)
    ax_in.set_xticklabels(["v783\n1.0× CB", "BANC\n0.45× CB", "Male\n1.73× CB"], fontsize=6.0)
    ax_in.set_ylim(0, 7700)
    ax_in.set_yticks([0, 3000, 6000])
    ax_in.set_yticklabels(["0", "3k", "6k"], fontsize=6.0)
    ax_in.set_title("MN9 L / R in-synapses", fontsize=6.6, fontweight="bold", pad=2)
    ax_in.set_facecolor("#f9fbfb")

    b_ct_sw = v5["phase1_banc"]["minor6_banc_celltype_homolog"]["banc_homolog_1to1_plus_celltype"]["sweeps"]["50hz"]
    b1_50 = ab["phase_b1_v783"]["experiment_b1_diatom_downsampling"]["50hz"]
    ax.text(
        0.98, 0.56,
        f"Target: ρ = {est['banc_cross_individual']['v783_vs_banc_m1_anat_spearman_rho']:.2f}, 8/14 pos, AUROC 0.75\n"
        f"Hom. CT (80%): AUPRC {b_ct_sw['auprc_all106']:.2f}, diatom 0 Hz\n"
        f"v783 diatom 4/14: {int(round(100*b1_50['random_4_of_14']['frac_exact_zero']))}% zero (exact 3: {b1_50['exact_banc_matched_3']['rate_hz_median']:.1f} Hz)",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
        bbox=dict(boxstyle="round,pad=0.30", fc="#fdf7f2", ec=ORANGE_LIGHT, lw=0.75),
    )
    yticks_hz = [0, 1, 5, 15, 40]
    ax.set(
        xlim=(-0.18, math.log1p(105.0)), ylim=(-0.18, math.log1p(68.0)),
        xticks=[math.log1p(v) for v in xticks_hz], xticklabels=[str(v) for v in xticks_hz],
        yticks=[math.log1p(v) for v in yticks_hz], yticklabels=[str(v) for v in yticks_hz],
        xlabel="FlyWire v783 MN9 rate at 50 Hz (Hz, log1p)",
        ylabel="BANC v888 MN9 rate at 50 Hz (Hz, log1p)",
        title="A   BANC v888 transfer & MN9 synapse QC",
    )
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.99, 1.00), fontsize=6.4, handletextpad=0.25, borderaxespad=0.2)

    # --- Panel B (top-right): FlyWire v783 vs MaleCNS v1.0 (intact 0.268 mV) ---
    ax = fig.add_subplot(gs[0, 1])
    m_rows = mcns["per_type_3brains"]
    m_ma = {r["name"]: r for r in mcns["mapping_audit"]["per_type_sez106"]}
    xv_m = np.array([r["v783_50hz"] for r in m_rows], dtype=float)
    ym_m = np.array([r["malecns_intact_50hz_mean"] for r in m_rows], dtype=float)
    pos_m = np.array([r["y_bin"] == 1 for r in m_rows], dtype=bool)
    unc_pos = np.array([r["y_bin"] == 1 and m_ma[r["name"]]["malecns_cls_counts"].get("2", 0) > 0 for r in m_rows], dtype=bool)
    lx_m, ly_m = np.log1p(xv_m), np.log1p(ym_m)

    ax.plot([0, math.log1p(65.0)], [0, math.log1p(65.0)], ls="--", lw=0.9, color=SLATE_LIGHT, zorder=1)
    ax.axhline(0.0, color=GRID, lw=0.7, zorder=0)
    ax.axvline(0.0, color=GRID, lw=0.7, zorder=0)

    ax.scatter(lx_m[~pos_m], ly_m[~pos_m], s=20, fc="white", ec=SLATE, lw=0.8, alpha=0.85, label="Negative (n = 92)", zorder=2)
    ax.scatter(lx_m[pos_m & (~unc_pos)], ly_m[pos_m & (~unc_pos)], s=38, fc=INDIGO, ec="white", lw=0.5, label="PER pos, valid NT (n = 12)", zorder=4)
    ax.scatter(lx_m[unc_pos], ly_m[unc_pos], s=44, marker="D", fc="white", ec=CRIMSON, lw=1.2, label="Unclear consensus NT (diatom, TH_VUM)", zorder=5)

    ann_m = {
        "roundup": (-26, -15), "clavicle": (-28, 8), "Fdg": (-18, -14), "rattle": (7, -4),
        "G2N_1": (7, 5), "FMIn": (8, -2), "sink_sync": (7, -8),
        "diatom": (-6, 12), "TH_VUM": (9, 10),
        "peacock": (8, 5), "puddle": (8, -3), "mime": (8, -11),
    }
    for r in m_rows:
        nm = r["name"]
        if nm in ann_m:
            dx, dy = ann_m[nm]
            col_t = CRIMSON if nm in ("diatom", "TH_VUM", "peacock", "puddle", "mime") else DARK
            lbl = "peacock (isomorphic neg)" if nm == "peacock" else nm
            ax.annotate(
                lbl, (math.log1p(r["v783_50hz"]), math.log1p(r["malecns_intact_50hz_mean"])),
                xytext=(dx, dy), textcoords="offset points", fontsize=6.7,
                fontweight="bold" if nm in ("Fdg", "rattle", "diatom") else "normal",
                color=col_t,
                arrowprops=dict(arrowstyle="-", color=CRIMSON if nm in ("diatom", "TH_VUM") else SLATE_LIGHT, lw=0.6)
                if nm in ("diatom", "TH_VUM", "Fdg", "clavicle") else None,
            )

    mc_c = est["malecns_third_connectome"]["v783_vs_intact_50hz"]
    a2b3_50 = port["test_a2_b3_combined"]["runs"]["a2_density_matched"]["50hz"]
    m_homp = ab["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]
    ax.text(
        0.98, 0.16,
        f"Target (0.268 mV): ρ = {mc_c['spearman_rho_all106']:.2f}, 8/14 pos, 17/92 neg\n"
        f"Density + Pred-NT: ρ = {a2b3_50['v783_spearman_rho']:.2f}, 9/14 pos, 3/92 neg\n"
        f"Homolog + Pred-NT: ρ = {m_homp['spearman_all106']:.2f}, 10/14 pos, 2/92 neg",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.30", fc="#f4f5fa", ec=INDIGO_LIGHT, lw=0.75),
    )
    ax.set(
        xlim=(-0.18, math.log1p(105.0)), ylim=(-0.18, math.log1p(78.0)),
        xticks=[math.log1p(v) for v in xticks_hz], xticklabels=[str(v) for v in xticks_hz],
        yticks=[math.log1p(v) for v in [0, 1, 5, 20, 50]], yticklabels=["0", "1", "5", "20", "50"],
        xlabel="FlyWire v783 MN9 rate at 50 Hz (Hz, log1p)",
        ylabel="MaleCNS v1.0 MN9 rate at 50 Hz (Hz, log1p)",
        title="B   MaleCNS v1.0 transfer: NT & density mismatch",
    )
    ax.legend(frameon=False, loc="upper left", fontsize=6.4, handletextpad=0.25)

    # Helpers for C and D
    co = port["co_primary_endpoints_50hz"]
    b_f_all = banc["task1_activation_banc"]["M1_frozen_0.275"]["all_106"]
    m_bil_all = mcns["task1_activation_malecns"]["anat_bilat"]["bilateral_mean"]["all_106"]
    ci_b3_50 = port["test_a_nondirect_variant_audit"]["test_b3_bootstrap_ci95"]["50hz"]
    y_all = np.array([int(r["y_bin"]) for r in m_rows], dtype=np.int32)
    unpub_m = np.array([not bool(r["prior_published"]) for r in m_rows], dtype=bool)

    def _ci_ab(arr_list: list[float], subset_mask: np.ndarray | None = None) -> tuple[float, list[float]]:
        arr = np.asarray(arr_list, dtype=np.float64)
        if subset_mask is not None:
            return _stepwise_auprc(y_all[subset_mask], arr[subset_mask]), _strat_auprc_ci95(y_all[subset_mask], arr[subset_mask])
        return _stepwise_auprc(y_all, arr), _strat_auprc_ci95(y_all, arr)

    v783_ref_pt, v783_ref_ci = _ci_ab([r["v783_50hz"] for r in m_rows])
    v783_ref95_pt, v783_ref95_ci = _ci_ab([r["v783_50hz"] for r in m_rows], unpub_m)
    b_raw_pt, b_raw_ci = _ci_ab([r["banc_frozen_50hz"] for r in m_rows])
    b_tgt_pt, b_tgt_ci = _ci_ab([r["banc_anat_50hz"] for r in m_rows])
    b_tgt95_pt, b_tgt95_ci = _ci_ab([r["banc_anat_50hz"] for r in m_rows], unpub_m)
    m_bil_pt, m_bil_ci = _ci_ab([r["malecns_bilat_50hz_mean"] for r in m_rows])
    m_tgt_pt, m_tgt_ci = _ci_ab([r["malecns_intact_50hz_mean"] for r in m_rows])
    m_tgt95_pt, m_tgt95_ci = _ci_ab([r["malecns_intact_50hz_mean"] for r in m_rows], unpub_m)
    m_dens_pt, m_dens_ci = _ci_ab(pb["malecns_density_scaled_sim"]["50hz"]["all_106_rates_mean"])
    m_b2_pt, m_b2_ci = _ci_ab(ab["phase_malecns"]["experiment_b2_fixed_02683_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"])
    b1_4_med = float(b1_50["random_4_of_14"]["auprc_unpub95_median"])
    b1_4_iqr = [float(x) for x in b1_50["random_4_of_14"]["auprc_unpub95_iqr"]]
    m_homp95_pt, m_homp95_ci = _ci_ab(ab["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"], unpub_m)

    # --- Panel C (bottom-left): Default transfer & perturbation controls ---
    ax_c = fig.add_subplot(gs[1, 0])
    c_groups = [
        (
            "All 106 types (chance 0.132)",
            14.0 / 106.0,
            [
                ("v783\nRef.", v783_ref_pt, v783_ref_ci, TEAL, None),
                ("BANC\nRaw", b_raw_pt, b_raw_ci, SLATE, None),
                ("BANC\nTarget", b_tgt_pt, b_tgt_ci, ORANGE, None),
                ("Male\nBlind", m_bil_pt, m_bil_ci, CRIMSON, "///"),
                ("Male\nTarget", m_tgt_pt, m_tgt_ci, INDIGO, None),
                ("Male\nDensity", m_dens_pt, m_dens_ci, INDIGO_LIGHT, None),
                ("Male Fix.\nPred-NT", m_b2_pt, m_b2_ci, PURPLE, None),
            ],
        ),
        (
            "Unpub. 95 (chance 0.042)",
            4.0 / 95.0,
            [
                ("v783\nRef.", v783_ref95_pt, v783_ref95_ci, TEAL, None),
                ("v783\ndia 4/14", b1_4_med, b1_4_iqr, TEAL_LIGHT, "///"),
                ("BANC\nTarget", b_tgt95_pt, b_tgt95_ci, ORANGE, None),
                ("Male\nTarget", m_tgt95_pt, m_tgt95_ci, INDIGO, None),
                ("Male\nHom.+NT", m_homp95_pt, m_homp95_ci, "#00574b", None),
            ],
        ),
    ]
    cur_x = 0.0
    xt_c, xl_c = [], []
    for g_idx, (gt, ch, items) in enumerate(c_groups):
        g0 = cur_x
        for lbl, pt, ci, col, hatch in items:
            xt_c.append(cur_x)
            xl_c.append(lbl)
            ax_c.bar(cur_x, pt, width=0.66, color=col, alpha=0.88, edgecolor=DARK if hatch else "none", lw=0.6 if hatch else 0.0, hatch=hatch, zorder=2)
            ax_c.errorbar(cur_x, pt, yerr=[[max(0.0, pt - ci[0])], [max(0.0, ci[1] - pt)]], fmt="none", ecolor=DARK, elinewidth=0.9, capsize=2.2, zorder=4)
            y_txt = min(ci[1] + 0.024, 1.02)
            tcol = DARK if col in (TEAL_LIGHT, SLATE_LIGHT) else (INDIGO if col == INDIGO_LIGHT else col)
            ax_c.text(cur_x, y_txt, f"{pt:.2f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=tcol, bbox=dict(boxstyle="round,pad=0.04", fc="white", ec="none", alpha=0.85), zorder=5)
            cur_x += 0.94
        g1 = cur_x - 0.94
        ax_c.hlines(ch, g0 - 0.38, g1 + 0.38, colors=SLATE, linestyles=":", lw=1.0, zorder=1)
        if g_idx == 0:
            ax_c.hlines(0.7520, g0 - 0.38, g1 + 0.38, colors=TEAL, linestyles="--", lw=0.85, alpha=0.65, zorder=1)
            ax_c.axvline(g1 + 0.62, color=GRID, lw=1.3)
            cur_x += 0.36
        ax_c.text((g0 + g1) / 2.0, 1.14, gt, ha="center", va="top", fontsize=7.2, fontweight="bold", color=DARK)

    ax_c.set(
        xticks=xt_c, xticklabels=xl_c,
        xlim=(-0.55, cur_x - 0.35), ylim=(0.0, 1.18),
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ylabel="Stepwise AUPRC (50 Hz)",
        title="C   Default transfer & diagnostic perturbation controls",
    )
    ax_c.tick_params(axis="x", labelsize=6.2)

    # --- Panel D (bottom-right): Calibration & anti-circularity controls (All 106 types, 50 Hz) ---
    ax_d = fig.add_subplot(gs[1, 1])
    v783_int_pt, v783_int_ci = _ci_ab(ab["phase_b1_v783"]["v783_intrinsic_inv_sin_reference"]["sweeps"]["50hz"]["per_type_mean_hz"])
    b_sug_pt, b_sug_ci = _ci_ab(ab["phase_banc"]["a1_sugar_anchor"]["sweeps"]["50hz"]["per_type_mean_hz"])
    b_ct_pt, b_ct_ci = _ci_ab(b_ct_sw["per_type_mean_hz"])
    m_r17p_pt, m_r17p_ci = _ci_ab(ab["phase_malecns"]["a1_sugar_anchor_prednt"]["sweeps_at_right17_wstar"]["50hz"]["per_type_mean_hz"])
    m_ts = v5["phase2_malecns"]["m2b_two_scale_mn9_02638_rest_01567"]
    m_ts_pt, m_ts_ci = _ci_ab(m_ts["per_type_mean_hz"])
    m_shuf_recs = v5["phase2_malecns"]["m2a_shuffle_non_mn9"]
    m_shuf_vals = np.array([r["auprc_all106"] for r in m_shuf_recs], dtype=float)
    m_bref = v5["phase2_malecns"]["m2c_banc_ref_to_malecns"]
    m_bref_pt, m_bref_ci = _ci_ab(m_bref["per_type_mean_hz"])
    m_lo = v5["phase2_malecns"]["m2b_leave_out_sez106_and_mn9"]
    m_lo_pt, m_lo_ci = _ci_ab(m_lo["per_type_mean_hz"])
    m_homp_pt, m_homp_ci = _ci_ab(ab["phase_malecns"]["a2a_homolog_per_neuron_norm_prednt"]["sweeps"]["50hz"]["per_type_mean_hz"])
    fair_2h = float(v5["phase2_malecns"]["m4_malecns_fair_twohop"]["predicted_nt_homolog_weighted"]["signed_relu"]["alpha_0.01"]["all106_auprc"])

    d_items = [
        ("v783\nRef.", v783_ref_pt, v783_ref_ci, TEAL, None, None),
        ("v783\nFlat 1/S", v783_int_pt, v783_int_ci, SLATE_LIGHT, "///", None),
        ("BANC\nSugar", b_sug_pt, b_sug_ci, ORANGE_LIGHT, None, None),
        ("BANC\nHom. CT", b_ct_pt, b_ct_ci, "#9e3f08", None, None),
        ("Male\nSugar+NT", m_r17p_pt, m_r17p_ci, "#8b5eb0", None, None),
        ("Male\n2-Scale", m_ts_pt, m_ts_ci, GOLD, "///", None),
        ("Male\nShuffled", float(m_shuf_vals.mean()), None, CRIMSON, None, m_shuf_vals),
        ("Male\nBANC-Ref", m_bref_pt, m_bref_ci, "#2f6b8a", None, None),
        ("Male\nLeave-Out", m_lo_pt, m_lo_ci, "#00838f", None, None),
        ("Male\nHom.+NT", m_homp_pt, m_homp_ci, "#00574b", None, None),
    ]
    rng_d = np.random.default_rng(11)
    xd = np.arange(len(d_items), dtype=float)
    for xi, (lbl, pt, ci, col, hatch, dots) in zip(xd, d_items):
        if dots is not None:
            ax_d.bar(xi, pt, width=0.66, color=col, alpha=0.28, edgecolor=col, lw=1.0, zorder=2)
            ax_d.scatter(xi + rng_d.uniform(-0.14, 0.14, len(dots)), dots, s=16, color=col, edgecolors="white", lw=0.4, zorder=4)
            ax_d.scatter([xi], [pt], s=30, marker="D", color=col, edgecolors=DARK, lw=0.5, zorder=5)
            y_txt = float(dots.max()) + 0.032
        else:
            ax_d.bar(xi, pt, width=0.66, color=col, alpha=0.88, edgecolor=DARK if hatch else "none", lw=0.6 if hatch else 0.0, hatch=hatch, zorder=2)
            ax_d.errorbar(xi, pt, yerr=[[max(0.0, pt - ci[0])], [max(0.0, ci[1] - pt)]], fmt="none", ecolor=DARK, elinewidth=0.9, capsize=2.2, zorder=4)
            y_txt = min(ci[1] + 0.024, 1.02)
        tcol = DARK if col in (SLATE_LIGHT, ORANGE_LIGHT) else col
        ax_d.text(xi, y_txt, f"{pt:.2f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold", color=tcol, bbox=dict(boxstyle="round,pad=0.04", fc="white", ec="none", alpha=0.85), zorder=5)

    ax_d.hlines(14.0 / 106.0, -0.45, len(d_items) - 0.55, colors=SLATE, linestyles=":", lw=1.0, zorder=1)
    ax_d.hlines(fair_2h, -0.45, len(d_items) - 0.55, colors=PURPLE, linestyles="-.", lw=1.05, alpha=0.85, zorder=3)
    ax_d.hlines(0.7520, -0.45, len(d_items) - 0.55, colors=TEAL, linestyles="--", lw=0.85, alpha=0.65, zorder=1)
    ax_d.text(len(d_items) - 0.52, fair_2h + 0.015, f"Male fair 2-hop ({fair_2h:.2f})", ha="right", va="bottom", fontsize=6.1, color=PURPLE, fontweight="bold", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=PURPLE, lw=0.5, alpha=0.94), zorder=5)
    ax_d.text(-0.40, 0.770, "0.752 ceiling", ha="left", va="bottom", fontsize=6.0, color=TEAL)

    ax_d.set(
        xticks=xd, xticklabels=[x[0] for x in d_items],
        xlim=(-0.55, len(d_items) - 0.40), ylim=(0.0, 1.18),
        yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ylabel="All-106 stepwise AUPRC (50 Hz)",
        title="D   Calibration & anti-circularity controls (All 106 types)",
    )
    ax_d.tick_params(axis="x", labelsize=6.2)

    save_figure(fig, "fig2", preview_dir)


def make_fig3(data: dict[str, dict], est: dict, preview_dir: Path | None) -> None:
    """Fig. 3: Cross-lab PER audit & two-full-CNS SELK (DNg68) bottleneck (with log1p sweep & explicit side labels)."""
    v783 = data["v783"]
    banc = data["banc"]
    mcns = data["malecns"]
    port = data["port"]
    selk_a2 = port["test_a_transmitter_sensitivity"]["A2_all_unclear_predicted_nt"]["selk_dng68_20_to_200hz"]
    selk_a2b3 = port["test_a2_b3_combined"]["runs"]["a2_density_matched"]["selk_dng68_20_to_200hz"]

    fig = plt.figure(figsize=(12.2, 5.95), layout="constrained")
    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[0.96, 1.40], height_ratios=[0.80, 1.24], hspace=0.30, wspace=0.14)

    # --- Panel A top: 13 independent-lab PER study audit funnel ---
    ax_a1 = fig.add_subplot(gs[0, 0])
    cats = ["Missing FlyWire ID / k/n", "Non-PER endpoint", "Terminal MN / mixed GAL4", "Admitted: Fdg + SELK"]
    cnts = [5, 4, 2, 2]
    ccols = [SLATE_LIGHT, "#a6b3b8", "#8d9ca2", TEAL]
    y = np.arange(len(cats))
    ax_a1.barh(y, cnts, color=ccols, height=0.62, zorder=2)
    for i, c in enumerate(cnts):
        ax_a1.text(c + 0.14, i, f"{c}/13", va="center", fontsize=7.3, fontweight="bold" if i == 3 else "normal", color=TEAL if i == 3 else DARK)
    ax_a1.set_yticks(y)
    ax_a1.set_yticklabels(cats, fontsize=7.3)
    ax_a1.invert_yaxis()
    ax_a1.set(xlim=(0, 6.4), xlabel="Independent PER studies audited (N = 13)", title="A   Retrospective cross-lab audit")

    # --- Panel A bottom: Three-brain frequency sweep on log1p axis + zoomed linear SELK inset ---
    ax_a2 = fig.add_subplot(gs[1, 0])
    freqs = [20, 50, 100, 200]
    fdg_v7_raw = [v783["module_a_descriptive_readouts"]["Fdg_CB0038"]["v783_lif"][f"{f}hz"]["mn9_mean"] for f in freqs]
    selk_v7_raw = [v783["module_a_descriptive_readouts"]["SELK_DNg68"]["v783_lif"][f"{f}hz"]["mn9_mean"] for f in freqs]
    fdg_ba_raw = [banc["selk_full_cns_diagnosis"]["lif_responses"]["anat_scale"]["Fdg_CB0038"][f"{f}hz"]["mn9_mean_hz"] for f in freqs]
    selk_ba_raw = [banc["selk_full_cns_diagnosis"]["lif_responses"]["anat_scale"]["SELK_DNg68"][f"{f}hz"]["mn9_mean_hz"] for f in freqs]
    fdg_mc_raw = [mcns["selk_full_cns_diagnosis"]["lif_responses"]["anat_intact"]["Fdg_CB0038"][f"{f}hz"]["mn9_mean_hz"] for f in freqs]
    selk_mc_raw = [mcns["selk_full_cns_diagnosis"]["lif_responses"]["anat_intact"]["SELK_DNg68"][f"{f}hz"]["mn9_mean_hz"] for f in freqs]
    selk_mc_a2_raw = [selk_a2[f"{f}hz"]["mn9_mean_hz"] for f in freqs]
    selk_mc_a2b3_raw = [selk_a2b3[f"{f}hz"]["mn9_mean_hz"] for f in freqs]

    ax_a2.plot(freqs, np.log1p(fdg_v7_raw), "o-", color=TEAL, lw=1.7, ms=4.5, label="Fdg (v783)")
    ax_a2.plot(freqs, np.log1p(fdg_mc_raw), "s-", color=INDIGO, lw=1.6, ms=4.2, label="Fdg (MaleCNS)")
    ax_a2.plot(freqs, np.log1p(fdg_ba_raw), "^--", color="#009e73", lw=1.7, ms=4.8, label="Fdg (BANC)")
    ax_a2.plot(freqs, np.log1p(selk_v7_raw), "D-", color=CRIMSON, lw=1.3, ms=3.5, label="SELK (v783)")
    ax_a2.plot(freqs, np.log1p(selk_ba_raw), "v:", color=ORANGE, lw=1.3, ms=3.5, label="SELK (BANC)")
    ax_a2.plot(freqs, np.log1p(selk_mc_raw), "x-.", color=PURPLE, lw=1.4, ms=4.4, label="SELK (Male Target)")
    ax_a2.plot(freqs, np.log1p(selk_mc_a2_raw), "P--", color="#4a145c", lw=1.4, ms=4.2, label="SELK (Male Pred-NT)")
    ax_a2.plot(freqs, np.log1p(selk_mc_a2b3_raw), "*:", color=GOLD, lw=1.3, ms=4.8, label="SELK (Male Joint)")

    # Zoomed linear inset placed in the empty mid-right white gap between Fdg (MaleCNS) and SELK curves
    ax_in3 = ax_a2.inset_axes([0.46, 0.30, 0.50, 0.24])
    ax_in3.plot(freqs, selk_mc_a2_raw, "P--", color="#4a145c", lw=1.3, ms=3.8)
    ax_in3.plot(freqs, selk_mc_raw, "x-.", color=PURPLE, lw=1.3, ms=4.0)
    ax_in3.plot(freqs, [v + 0.05 for v in selk_ba_raw], "v:", color=ORANGE, lw=1.2, ms=3.2)
    ax_in3.plot(freqs, selk_v7_raw, "D-", color=CRIMSON, lw=1.1, ms=2.8)
    ax_in3.plot(freqs, [v - 0.05 for v in selk_mc_a2b3_raw], "*:", color=GOLD, lw=1.2, ms=3.8)
    ax_in3.set(
        xticks=[20, 50, 100, 200], xticklabels=["20", "50", "100", "200"],
        ylim=(-0.22, 3.05), yticks=[0, 1, 2], yticklabels=["0", "1", "2"],
    )
    ax_in3.text(0.04, 0.92, "SELK zoom (Hz, linear)", transform=ax_in3.transAxes, ha="left", va="top", fontsize=5.6, fontweight="bold", color=DARK)
    ax_in3.tick_params(labelsize=5.3, pad=1)
    ax_in3.set_facecolor("#fafcfd")
    for sp in ax_in3.spines.values():
        sp.set_visible(True)
        sp.set_color(SLATE_LIGHT)
        sp.set_linewidth(0.7)

    yticks_r = [0, 1, 5, 20, 70]
    ax_a2.set(
        xticks=freqs,
        ylim=(-0.18, math.log1p(280.0)),
        yticks=[math.log1p(v) for v in yticks_r],
        yticklabels=[str(v) for v in yticks_r],
        xlabel="Optogenetic drive frequency (Hz)",
        ylabel="Predicted MN9 rate (Hz, log1p)",
        title="Three-brain sweep (Fdg vs SELK, log1p)",
    )
    ax_a2.legend(frameon=False, loc="upper left", ncol=2, fontsize=5.3, columnspacing=0.55, handletextpad=0.25, labelspacing=0.20)

    # --- Panel B (right, 3 clean stacked full-width rows) ---
    gs_b = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[:, 1], height_ratios=[0.36, 0.48, 0.48], hspace=0.40)
    ax_b1 = fig.add_subplot(gs_b[0, 0])

    b_reg = banc["selk_full_cns_diagnosis"]["selk_postsynaptic_region_ei_summary"]
    m_reg = mcns["selk_full_cns_diagnosis"]["selk_postsynaptic_region_ei_summary"]
    rows_split = [
        ("BANC female\n(3,629 syn)", b_reg["central_brain_E"]["sum_syn"], b_reg["central_brain_I"]["sum_syn"],
         b_reg["ventral_nerve_cord_E"]["sum_syn"], b_reg["ventral_nerve_cord_I"]["sum_syn"], 0.0),
        ("MaleCNS male\n(13,078 syn)", m_reg["central_brain_E"]["sum_syn"], m_reg["central_brain_I"]["sum_syn"],
         m_reg["ventral_nerve_cord_E"]["sum_syn"], m_reg["ventral_nerve_cord_I"]["sum_syn"],
         m_reg["ventral_nerve_cord_U"]["sum_syn"] + m_reg["central_brain_U"]["sum_syn"]),
    ]
    seg_meta = [
        ("Brain E", TEAL_LIGHT),
        ("Brain I", ORANGE_LIGHT),
        ("VNC E", TEAL),
        ("VNC I", CRIMSON),
        ("Unclear NT", SLATE_LIGHT),
    ]
    for r_i, (rlbl, be, bi, ve, vi, un) in enumerate(rows_split):
        tot = be + bi + ve + vi + un
        vals = [100.0 * v / tot for v in (be, bi, ve, vi, un)]
        left = 0.0
        for v_pct, (slbl, col) in zip(vals, seg_meta):
            if v_pct <= 0.0:
                continue
            ax_b1.barh([r_i], [v_pct], left=left, color=col, height=0.58, edgecolor="white", lw=0.7)
            if v_pct >= 12.0:
                ax_b1.text(left + v_pct / 2.0, r_i, f"{v_pct:.0f}%", ha="center", va="center", color="white" if col != SLATE_LIGHT else DARK, fontsize=6.4, fontweight="bold")
            left += v_pct
    ax_b1.set_yticks([0, 1])
    ax_b1.set_yticklabels([r[0] for r in rows_split], fontsize=6.8)
    ax_b1.invert_yaxis()
    ax_b1.set(xlim=(0, 100), ylim=(1.45, -0.75), xticks=[0, 20, 40, 60, 80, 100],
              xlabel="% of SELK output synapses by target neuron region (BANC 80.0%, MaleCNS 76.3% VNC-assigned)",
              title="B   Full-CNS SELK outputs & top-8 individual neuron partners")
    ax_b1.tick_params(axis="x", labelsize=6.4)
    leg_s = [mpatches.Patch(color=c, label=l) for l, c in seg_meta]
    ax_b1.legend(handles=leg_s, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=5, fontsize=5.9, columnspacing=0.65, handletextpad=0.3)

    # Helper to disambiguate duplicate cell types by hemisphere/instance
    def format_partner_labels(top8_list: list[dict], is_banc: bool) -> list[str]:
        side_counts: dict[tuple[str, str], int] = {}
        for r in top8_list:
            sd = "L" if str(r.get("side", "")).lower().startswith("l") else ("R" if str(r.get("side", "")).lower().startswith("r") else "?")
            side_counts[(r["cell_type"], sd)] = side_counts.get((r["cell_type"], sd), 0) + 1
        seen: dict[tuple[str, str], int] = {}
        out = []
        for i, r in enumerate(top8_list, 1):
            sd = "L" if str(r.get("side", "")).lower().startswith("l") else ("R" if str(r.get("side", "")).lower().startswith("r") else "?")
            key = (r["cell_type"], sd)
            seen[key] = seen.get(key, 0) + 1
            sd_tag = f"{sd}#{seen[key]}" if side_counts[key] > 1 else sd
            if is_banc:
                reg_tag = "VNC→GNG" if r["super_class"] == "ascending" else ("VNC" if r["region"] == "ventral_nerve_cord" else "Brain")
                nm = "CB0626/Amulet" if r["cell_type"] == "CB0626" else r["cell_type"]
            else:
                reg_tag = "VNC→GNG" if r["is_ascending"] else ("VNC" if r["region"] == "ventral_nerve_cord" else "Brain")
                nm = r["cell_type"]
            out.append(f"{i}. {nm} ({sd_tag}) [{reg_tag}]")
        return out

    # Middle row: BANC Top 8
    ax_b2 = fig.add_subplot(gs_b[1, 0])
    top8_b = banc["selk_full_cns_diagnosis"]["selk_top20_postsynaptic_targets"][:8]
    y8 = np.arange(8)
    cols_b = [CRIMSON if r["super_class"] == "ascending" else (ORANGE if r["ei_class"] == "I" else TEAL) for r in top8_b]
    ax_b2.barh(y8, [r["syn_from_selk"] for r in top8_b], color=cols_b, height=0.68, zorder=2)
    for i, r in enumerate(top8_b):
        nt = "GABA" if r["nt_predicted"] == "gaba" else "ACh"
        ax_b2.text(r["syn_from_selk"] + 1.2, i, f"{int(r['syn_from_selk'])} ({nt})", va="center", fontsize=6.2, color=DARK)
    ax_b2.set_yticks(y8)
    ax_b2.set_yticklabels(format_partner_labels(top8_b, True), fontsize=6.5)
    ax_b2.invert_yaxis()
    ax_b2.set(xlim=(0, 110), xlabel="BANC synapses from bilateral SELK (10/15 GABA; 2-hop VNC exc. = 1, ≥3-hop subthreshold)")
    ax_b2.xaxis.label.set_fontsize(6.3)

    # Bottom row: MaleCNS Top 8
    ax_b3 = fig.add_subplot(gs_b[2, 0])
    top8_m = mcns["selk_full_cns_diagnosis"]["selk_top20_postsynaptic_targets"][:8]
    cols_m = [
        SLATE_LIGHT if r["ei_class"] == "U" else (CRIMSON if (r["is_ascending"] and r["ei_class"] == "I") else (ORANGE if r["ei_class"] == "I" else TEAL))
        for r in top8_m
    ]
    ax_b3.barh(y8, [r["syn_from_selk"] for r in top8_m], color=cols_m, height=0.68, zorder=2)
    nt_map = {"gaba": "GABA", "acetylcholine": "ACh", "unclear": "Unclear"}
    for i, r in enumerate(top8_m):
        nt = nt_map.get(r["nt_predicted"], r["nt_predicted"])
        ax_b3.text(r["syn_from_selk"] + 2.5, i, f"{int(r['syn_from_selk'])} ({nt})", va="center", fontsize=6.2, color=DARK)
    ax_b3.set_yticks(y8)
    ax_b3.set_yticklabels(format_partner_labels(top8_m, False), fontsize=6.5)
    ax_b3.invert_yaxis()
    ax_b3.set(xlim=(0, 232), xlabel="MaleCNS synapses from bilateral SELK (10/15 GABA; 3-hop VNC exc. = 9, ≥3-hop subthreshold)")
    ax_b3.xaxis.label.set_fontsize(6.3)

    leg_b = [
        mpatches.Patch(color=CRIMSON, label="VNC→GNG asc. GABA"),
        mpatches.Patch(color=ORANGE, label="Other GABA"),
        mpatches.Patch(color=TEAL, label="ACh (excitatory)"),
        mpatches.Patch(color=SLATE_LIGHT, label="Unclear NT"),
    ]
    ax_b2.legend(handles=leg_b, frameon=False, loc="lower right", fontsize=6.0, handletextpad=0.35)

    save_figure(fig, "fig3", preview_dir)


def make_fig4(data: dict[str, dict], est: dict, preview_dir: Path | None) -> None:
    """Fig. 4: Background states (A, B with colorbar), 44-ctrl 3-brain disinhibition null (C, 0-100%),
    & informative Premotor Oracle lock on VNC SELK vs tentacular (D; full 17x8 matrix in Fig. S1)."""
    disinh = data["disinh"]
    mcns = data["malecns"]
    port = data["port"]
    pd = port["test_d_44ctrl_three_brain_null"]

    fig = plt.figure(figsize=(13.4, 5.55), layout="constrained")
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[0.90, 0.90, 1.32, 1.32], wspace=0.20)

    # --- Panel A: 11 v783 zero-tuning background scans (AUPRC + FP count) ---
    ax_a = fig.add_subplot(gs[0, 0])
    cond_order = [
        ("F1_global_sub_5.50", "Global 5.5 mV"),
        ("F1_global_supra_7.10", "Global 7.1 mV"),
        ("F1_global_supra_7.30", "Global 7.3 mV"),
        ("F1_global_supra_7.40", "Global 7.4 mV"),
        ("F2_ei_E5.50_I7.50", "E 5.5 / I 7.5 mV"),
        ("F2_ei_E7.30_I8.00", "E 7.3 / I 8.0 mV"),
        ("F2_ei_E7.45_I8.00", "E 7.45 / I 8.0 mV"),
        ("F2_ei_E7.55_I8.00", "E 7.55 / I 8.0 mV"),
        ("F3a_sugar_bg_40hz", "Sugar 40 Hz"),
        ("F3a_sugar_bg_50hz", "Sugar 50 Hz"),
        ("F3a_water_bg_160hz", "Water 160 Hz"),
    ]
    bg_map = disinh["module2_v783_background_states"]
    y11 = np.arange(len(cond_order))
    aps = [bg_map[k]["all_106_delta_metrics"]["auprc"] for k, _ in cond_order]
    nfp = [bg_map[k]["all_106_delta_metrics"]["n_sig_neg_fp_ge1hz_p01"] for k, _ in cond_order]
    brk = [bg_map[k]["all_106_delta_metrics"]["specificity_broken"] for k, _ in cond_order]
    cols_a = [TEAL if not b else SLATE_LIGHT for b in brk]

    ax_a.barh(y11, aps, color=cols_a, height=0.68, zorder=2)
    zero_ap = disinh["v783_zero_bg_reference"]["auprc_all106"]
    ax_a.axvline(zero_ap, color=TEAL, ls="--", lw=1.0, zorder=3)
    ax_a.axvline(zero_ap - 0.05, color=CRIMSON, ls=":", lw=1.0, zorder=3)
    for i, (apv, fpv) in enumerate(zip(aps, nfp)):
        xpos = max(apv, zero_ap) + 0.03 if apv >= 0.50 else apv + 0.025
        ax_a.text(xpos, i, f"FP={fpv}", va="center", fontsize=6.1, color=CRIMSON if fpv >= 5 else DARK, fontweight="bold" if fpv >= 5 else "normal")
    ax_a.set_yticks(y11)
    ax_a.set_yticklabels([lbl for _, lbl in cond_order], fontsize=6.7)
    ax_a.invert_yaxis()
    ax_a.set(xlim=(0.0, 1.18), xticks=[0.0, 0.4, 0.8], xticklabels=["0", "0.4", "0.8"], xlabel="AUPRC (106 types)", title="A   Background states")

    # --- Panel B: 200 Hz focal Delta r heatmap WITH explicit colorbar ---
    ax_b = fig.add_subplot(gs[0, 1])
    cands5 = [("tentacular", "Tent"), ("phantom", "Phan"), ("SELK_DNg68", "SELK"), ("marge", "Mrg"), ("usnea", "Usn")]
    mat_b = np.zeros((len(cond_order), len(cands5)))
    sig_b = np.zeros((len(cond_order), len(cands5)), dtype=bool)
    for i, (ck, _) in enumerate(cond_order):
        for j, (fn, _) in enumerate(cands5):
            r = bg_map[ck]["focal_30tr"][f"{fn}_200hz"]
            mat_b[i, j] = r["delta_mean_hz"]
            sig_b[i, j] = bool(r["unlocked_ge1hz_p01"])

    norm_b = TwoSlopeNorm(vmin=-12.0, vcenter=0.0, vmax=6.0)
    im_b = ax_b.imshow(mat_b, cmap="RdBu", norm=norm_b, aspect="auto")
    for i in range(len(cond_order)):
        for j in range(len(cands5)):
            val = mat_b[i, j]
            txt = "0.0" if abs(val) < 0.05 else f"{val:+.1f}{'*' if sig_b[i, j] else ''}"
            ax_b.text(j, i, txt, ha="center", va="center", fontsize=5.9, color="white" if val < -5.5 or val > 3.0 else DARK, fontweight="bold" if sig_b[i, j] else "normal")
    ax_b.set_xticks(range(len(cands5)))
    ax_b.set_xticklabels([c[1] for c in cands5], fontsize=6.8, rotation=32, ha="right")
    short_bg = ["G 5.5", "G 7.1", "G 7.3", "G 7.4", "E5.5/I7.5", "E7.3/I8", "E7.45/I8", "E7.55/I8", "Sug 40", "Sug 50", "Wat 160"]
    ax_b.set_yticks(y11)
    ax_b.set_yticklabels(short_bg, fontsize=5.8)
    ax_b.set_xlabel("200 Hz focal Δr (Hz)")
    ax_b.set_title("B   Focal Δr (0/11 unlock+spec)")
    for spine in ax_b.spines.values():
        spine.set_visible(True)
        spine.set_color(SLATE_LIGHT)
    cb_b = fig.colorbar(im_b, ax=ax_b, orientation="horizontal", fraction=0.055, pad=0.14, aspect=18)
    cb_b.set_ticks([-12, -6, 0, 3, 6])
    cb_b.ax.tick_params(labelsize=6.0)
    cb_b.set_label("MN9 Δr (Hz)", fontsize=6.5)

    # --- Panel C: Three-brain disinhibition percentiles (strictly 0-100%) + 44-ctrl null rates ---
    gs_c = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 2], hspace=0.46)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[1, 0])

    c4 = ["tentacular", "phantom", "marge", "usnea"]
    c4_top = ["Tent", "Phan", "Mrg", "Usn"]
    c4_lbl = ["Tent\n(6 / 5)", "Phan\n(2 / 4)", "Mrg\n(2 / 2)", "Usn\n(0 / 0)"]
    apct = est["disinhibition_and_states"]["anat_percentiles"]
    x4 = np.arange(len(c4))
    bw = 0.25
    b_specs = [("v783", "v783", TEAL), ("banc", "BANC", ORANGE), ("malecns", "MaleCNS", INDIGO)]

    for ax_sub, mkey, sub_ttl, show_leg in (
        (ax_c1, "hop2_net_pct", "C   3-brain motifs: 2-hop net", True),
        (ax_c2, "hop3_disinh_pct", "3-hop premotor disinhibition", False),
    ):
        for s_i, (bk, blab, col) in enumerate(b_specs):
            xp = x4 + (s_i - 1) * bw
            vals = [apct[bk][c][mkey] for c in c4]
            ax_sub.bar(xp, vals, width=bw * 0.84, color=col, alpha=0.88, label=blab if show_leg else None, zorder=2)
            for idx_c, (xi, vi) in enumerate(zip(xp, vals)):
                if vi >= 80:
                    y_off = 5.4 if (s_i == 1 and mkey == "hop3_disinh_pct" and idx_c == 1) else 1.8
                    ax_sub.text(xi, vi + y_off, f"{vi:.0f}", ha="center", va="bottom", fontsize=4.6, fontweight="bold", color=col)
        ax_sub.axhline(50, color=SLATE_LIGHT, ls=":", lw=0.85, zorder=1)
        ax_sub.set_ylim(0, 116)
        ax_sub.set_yticks([0, 25, 50, 75, 100])
        ax_sub.set_ylabel("Percentile (%)", fontsize=7.4)
        ax_sub.set_title(sub_ttl, fontsize=8.3, fontweight="bold", pad=12 if show_leg else 4)
        ax_sub.set_xticks(x4)
        if not show_leg:
            ax_sub.set_xticklabels(c4_lbl, fontsize=6.7)
            ax_sub.set_xlabel("Candidate (3-brain shared 2h / 3h)", fontsize=7.0)
            nc = pd["null_criteria_among_44_neg_inh"]
            s36 = port["test_d_strict_3brain_inh_sensitivity"]["null_criteria_among_36"]
            ax_sub.text(
                0.98, 0.96,
                f"44-ctrl (strict 36-ctrl) 3h null:\n"
                f"Tent 3h: {nc['shared_all3_hop3_ge_5']['k_pass_of_44']}/44 ({s36['joint_tentacular_3h_only']['k']}/36)\n"
                f"Phan 3h: {nc['joint_phantom_level_h3ge80_shared3h_ge4']['k_pass_of_44']}/44 ({s36['joint_phantom_3h']['k']}/36)\n"
                f"2h ≥4: {nc['shared_all3_hop2_ge_4']['k_pass_of_44']}/44 (27%)",
                transform=ax_sub.transAxes, ha="right", va="top", fontsize=5.2,
                bbox=dict(boxstyle="round,pad=0.22", fc="#f4f8f8", ec=SLATE_LIGHT, lw=0.65),
            )
        else:
            ax_sub.set_xticklabels(c4_top, fontsize=6.4)
            ax_sub.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=6.0, columnspacing=0.6, handletextpad=0.25)

    # --- Panel D: Informative Premotor Oracle contrast on signed log1p axis (full 17x8 in Fig. S1) ---
    ax_d = fig.add_subplot(gs[0, 3])
    ora = disinh["module2_local_oracle_realizability"]
    ora_m = mcns["disinhibition_module4"]["local_oracle_realizability_intact"]

    d_rows = [
        ("v783 tentacular", ora["v783|tentacular|oracle_premotor_K5"]["rates"], ora["v783|tentacular|oracle_premotor_K10"]["rates"], TEAL),
        ("BANC tentacular", ora["banc|tentacular|oracle_premotor_K5"]["rates"], ora["banc|tentacular|oracle_premotor_K10"]["rates"], ORANGE),
        ("MaleCNS tentacular", ora_m["malecns|tentacular|oracle_premotor_K5"]["rates"], ora_m["malecns|tentacular|oracle_premotor_K10"]["rates"], INDIGO),
        ("BANC SELK (VNC)", ora["banc|SELK_DNg68|oracle_premotor_via_vnc_K5"]["rates"], ora["banc|SELK_DNg68|oracle_premotor_via_vnc_K10"]["rates"], CRIMSON),
        ("MaleCNS SELK (VNC)", ora_m["malecns|SELK_DNg68|oracle_premotor_via_vnc_K5"]["rates"], ora_m["malecns|SELK_DNg68|oracle_premotor_via_vnc_K10"]["rates"], "#8b1e28"),
        ("v783 marge (ctrl)", ora["v783|marge|oracle_premotor_K5"]["rates"], ora["v783|marge|oracle_premotor_K10"]["rates"], SLATE),
        ("BANC marge (ctrl)", ora["banc|marge|oracle_premotor_K5"]["rates"], ora["banc|marge|oracle_premotor_K10"]["rates"], SLATE),
        ("MaleCNS marge (ctrl)", ora_m["malecns|marge|oracle_premotor_K5"]["rates"], ora_m["malecns|marge|oracle_premotor_K10"]["rates"], SLATE),
    ]
    def slog1p(v: float) -> float:
        return math.copysign(math.log1p(abs(v)), v) if v != 0.0 else 0.0

    yd = np.arange(len(d_rows))
    h = 0.36
    raw_k5 = [r[1]["200hz"]["delta_mean_hz"] for r in d_rows]
    raw_k10 = [r[2]["200hz"]["delta_mean_hz"] for r in d_rows]
    v_k5 = [slog1p(x) for x in raw_k5]
    v_k10 = [slog1p(x) for x in raw_k10]
    cols_d = [r[3] for r in d_rows]

    ax_d.barh(yd - h / 2, v_k5, height=h, color=cols_d, alpha=0.48, edgecolor=cols_d, lw=0.7, label="Premotor K = 5 (200 Hz)", zorder=2)
    ax_d.barh(yd + h / 2, v_k10, height=h, color=cols_d, alpha=0.92, edgecolor="none", label="Premotor K = 10 (200 Hz)", zorder=3)
    ax_d.axvline(0.0, color=DARK, lw=0.85, zorder=4)
    ax_d.axhline(2.5, color=GRID, lw=1.0)
    ax_d.axhline(4.5, color=GRID, lw=1.0)
    for i, (rk5, rk10, sk5, sk10) in enumerate(zip(raw_k5, raw_k10, v_k5, v_k10)):
        pick_k10 = abs(rk10) >= abs(rk5)
        r_best = rk10 if pick_k10 else rk5
        s_best = sk10 if pick_k10 else sk5
        ha = "left" if r_best >= 0 else "right"
        pad = 0.14 if r_best >= 0 else -0.14
        ax_d.text(s_best + pad, i + (h / 2 if pick_k10 else -h / 2), f"{r_best:+.1f}", va="center", ha=ha, fontsize=6.0, fontweight="bold", color=cols_d[i])
    ax_d.set_yticks(yd)
    ax_d.set_yticklabels([r[0] for r in d_rows], fontsize=6.7)
    ax_d.invert_yaxis()
    xt_raw = [-30, -10, -3, 0, 3, 10, 30, 80]
    ax_d.set(
        xlim=(-4.6, 5.55),
        xticks=[slog1p(x) for x in xt_raw],
        xticklabels=[f"{x:+d}" if x != 0 else "0" for x in xt_raw],
        xlabel="Premotor upper-bound MN9 Δr at 200 Hz (Hz, signed log1p)",
        title="D   Premotor upper bounds (full 17×12: S1 Fig)",
    )
    ax_d.tick_params(axis="x", labelsize=6.2)
    ax_d.legend(frameon=False, loc="lower right", fontsize=5.8, handletextpad=0.3)
    ax_d.text(
        0.98, 0.24,
        "VNC SELK locked 11/12\nin both full-CNS flies\n(−63.1 Hz left MN9 Male)",
        transform=ax_d.transAxes, ha="right", va="bottom", fontsize=5.7, color=CRIMSON,
        bbox=dict(boxstyle="round,pad=0.22", fc="#fdf3f2", ec="#e2a8a5", lw=0.65),
    )

    save_figure(fig, "fig4", preview_dir)


def make_figS1(data: dict[str, dict], preview_dir: Path | None) -> None:
    """Supplementary Fig. S1: Complete diagnostic 17 x 8 three-brain local Oracle matrix with colorbar."""
    disinh = data["disinh"]
    mcns = data["malecns"]
    ora = disinh["module2_local_oracle_realizability"]
    ora_m = mcns["disinhibition_module4"]["local_oracle_realizability_intact"]

    row_specs = [
        (ora, "v783|tentacular", "FlyWire v783 — tentacular", "direct", "premotor"),
        (ora, "banc|tentacular", "BANC v888 — tentacular", "direct", "premotor"),
        (ora_m, "malecns|tentacular", "MaleCNS v1.0 — tentacular", "direct", "premotor"),
        (ora, "v783|phantom", "FlyWire v783 — phantom", "direct", "premotor"),
        (ora, "banc|phantom", "BANC v888 — phantom", "direct", "premotor"),
        (ora_m, "malecns|phantom", "MaleCNS v1.0 — phantom", "direct", "premotor"),
        (ora, "v783|SELK_DNg68", "FlyWire v783 — SELK (brain only)", "direct_all", "premotor_all"),
        (ora, "banc|SELK_DNg68", "BANC v888 — SELK (unconstrained)", "direct_all", "premotor_all"),
        (ora_m, "malecns|SELK_DNg68", "MaleCNS v1.0 — SELK (unconstrained)", "direct_all", "premotor_all"),
        (ora, "banc|SELK_DNg68", "BANC v888 — SELK (VNC-targeted)", "direct_via_vnc", "premotor_via_vnc"),
        (ora_m, "malecns|SELK_DNg68", "MaleCNS v1.0 — SELK (VNC-targeted)", "direct_via_vnc", "premotor_via_vnc"),
        (ora, "v783|marge", "FlyWire v783 — marge (antag. ctrl)", "direct", "premotor"),
        (ora, "banc|marge", "BANC v888 — marge (antag. ctrl)", "direct", "premotor"),
        (ora_m, "malecns|marge", "MaleCNS v1.0 — marge (antag. ctrl)", "direct", "premotor"),
        (ora, "v783|usnea", "FlyWire v783 — usnea (peptidergic ctrl)", "direct", "premotor"),
        (ora, "banc|usnea", "BANC v888 — usnea (peptidergic ctrl)", "direct", "premotor"),
        (ora_m, "malecns|usnea", "MaleCNS v1.0 — usnea (peptidergic ctrl)", "direct", "premotor"),
    ]
    col_specs = [
        ("Dir K=5\n7.5 mV\n50 Hz", "d", "K5_vmn9_7.5", "50hz"),
        ("Dir K=5\n7.5 mV\n200 Hz", "d", "K5_vmn9_7.5", "200hz"),
        ("Dir K=5\n8.5 mV\n50 Hz", "d", "K5_vmn9_8.5", "50hz"),
        ("Dir K=5\n8.5 mV\n200 Hz", "d", "K5_vmn9_8.5", "200hz"),
        ("Dir K=10\n7.5 mV\n50 Hz", "d", "K10_vmn9_7.5", "50hz"),
        ("Dir K=10\n7.5 mV\n200 Hz", "d", "K10_vmn9_7.5", "200hz"),
        ("Dir K=10\n8.5 mV\n50 Hz", "d", "K10_vmn9_8.5", "50hz"),
        ("Dir K=10\n8.5 mV\n200 Hz", "d", "K10_vmn9_8.5", "200hz"),
        ("Prem K=5\n\n50 Hz", "p", "K5", "50hz"),
        ("Prem K=5\n\n200 Hz", "p", "K5", "200hz"),
        ("Prem K=10\n\n50 Hz", "p", "K10", "50hz"),
        ("Prem K=10\n\n200 Hz", "p", "K10", "200hz"),
    ]

    mat = np.zeros((len(row_specs), len(col_specs)))
    sig = np.zeros((len(row_specs), len(col_specs)), dtype=bool)
    for i, (odict, pref, _, d_tag, p_tag) in enumerate(row_specs):
        for j, (_, kind, suf, hz) in enumerate(col_specs):
            key = f"{pref}|oracle_{d_tag if kind == 'd' else p_tag}_{suf}"
            rec = odict[key]["rates"][hz]
            mat[i, j] = rec["delta_mean_hz"]
            sig[i, j] = bool(rec["unlocked_ge1hz_p01"])

    fig, ax = plt.subplots(figsize=(13.2, 7.6), layout="constrained")
    disp = np.sign(mat) * np.log1p(np.abs(mat))
    norm = TwoSlopeNorm(vmin=-math.log1p(35.0), vcenter=0.0, vmax=math.log1p(90.0))
    im = ax.imshow(disp, cmap="RdBu", norm=norm, aspect="auto")

    for i in range(len(row_specs)):
        for j in range(len(col_specs)):
            val = mat[i, j]
            if abs(val) < 0.05:
                txt = "0.0"
            elif abs(val) >= 10:
                txt = f"{val:+.1f}{'*' if sig[i, j] else ''}"
            else:
                txt = f"{val:+.2f}{'*' if sig[i, j] else ''}"
            dark_bg = disp[i, j] > 2.3 or disp[i, j] < -1.9
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.4, color="white" if dark_bg else DARK, fontweight="bold" if sig[i, j] else "normal")

    ax.axhline(5.5, color=DARK, lw=1.3)
    ax.axhline(8.5, color=SLATE, lw=0.9, ls=":")
    ax.axhline(10.5, color=DARK, lw=1.3)
    ax.axvline(7.5, color=DARK, lw=1.3, ls="--")
    ax.set_xticks(range(len(col_specs)))
    ax.set_xticklabels([c[0] for c in col_specs], fontsize=6.8)
    ax.set_yticks(range(len(row_specs)))
    ax.set_yticklabels([r[2] for r in row_specs], fontsize=7.6)
    ax.set_title("S1 Fig. Complete diagnostic three-brain local premotor upper-bound matrix (17 rows × 12 regimes; * = Δr ≥ +1.0 Hz, paired p < 0.01)", fontsize=9.0, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SLATE_LIGHT)

    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    raw_ticks = [-30, -10, -3, 0, 3, 10, 30, 80]
    cb.set_ticks([math.copysign(math.log1p(abs(v)), v) if v != 0 else 0.0 for v in raw_ticks])
    cb.set_ticklabels([f"{v:+d}" if v != 0 else "0" for v in raw_ticks])
    cb.set_label("Bilateral MN9 Δr (Hz, signed log1p color scale)", fontsize=8.2)

    save_figure(fig, "figS1", preview_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build estimates.json, TSV tables, fig1..fig4.svg, and figS1.svg.")
    ap.add_argument("--preview-dir", type=Path, default=None, help="Optional outside-Git directory for PNG previews")
    args = ap.parse_args()

    data = load_locked_jsons()
    est = build_estimates(data)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "estimates.json").write_text(json.dumps(est, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    export_supp_tables(data)

    make_fig1(data, est, args.preview_dir)
    make_fig2(data, est, args.preview_dir)
    make_fig3(data, est, args.preview_dir)
    make_fig4(data, est, args.preview_dir)
    make_figS1(data, args.preview_dir)
    print(f"Wrote estimates.json, TSV tables, fig1.svg..fig4.svg, and figS1.svg to {OUT_DIR}")
    if args.preview_dir:
        print(f"Wrote PNG previews to {args.preview_dir}")


if __name__ == "__main__":
    main()
