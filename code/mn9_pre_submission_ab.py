#!/usr/bin/env python3
"""Pre-submission Experiments A & B + MaleCNS LB3a Water Audit (Zero-New-A100, Local CPU).

Resource-safe exact FP64 Numba lazy-exact LIF engine (bit-identical to Brian 2 / A100):
  - Default 6 worker processes via Linux COW fork + os.nice(15) + single-threaded BLAS/Numba
    to keep CPU load gentle (~20-30%) and RAM footprint low (<3-4 GB).
  - Never modifies the 6 locked benchmark result JSONs; writes all new outputs to
    `mn9_pre_submission_ab_results.json` with per-phase atomic checkpoints.

Experiments implemented:
  - Phase B1 (FlyWire v783):
      * Exact BANC reviewed-matched diatom subset (3 unique v783 cells matched by BANC's 4 cells)
      * 100 deterministic random subsets each of 3/14, 4/14, and 7/14 diatom cells (50 & 200 Hz)
      * Individual 14-cell diatom outgoing-synapse & 2-hop ReLU decomposition
      * FlyWire v783 intrinsic 1/S_in per-neuron normalization reference (sugar-anchored to 58.52 Hz)
  - Phase BANC (BANC v888):
      * A1: Sugar-GRN 58.52 Hz anchor recalibration (canonical right LB3b+LB3c 45 cells;
            plus diagnostic curves for right LB3b 40 and left LB3b+LB3c 33) -> 106 types + SELK
      * A2a: Homolog-matched per-neuron input normalization (1:1 reviewed FAFB matches) -> 106 types + SELK
      * A2b: Intrinsic reference-free 1/S_in per-neuron normalization (sugar-anchored) -> 106 types + SELK
  - Phase MaleCNS (MaleCNS v1.0):
      * B2: Fixed-weight (0.268284 mV) whole-brain predicted_nt pure ablation -> 106 types + SELK
      * A1: Sugar-GRN 58.52 Hz anchor recalibration (canonical Traced LB3b+LB3c: left 17 & right 17
            diagnostic scan; bisection to intact left MN9 58.52 Hz) -> 106 types + SELK
      * A2a: Homolog-matched per-neuron input normalization (consensus_nt & predicted_nt;
             preserving intact left MN9 scale on truncated right MN9) -> 106 types + SELK
      * A2b: Intrinsic reference-free 1/S_in per-neuron normalization (sugar-anchored) -> 106 types + SELK
      * Water LB3a audit (Review #2): Traced type=='LB3a' (right 8 & left 9) 160 Hz baselines across
             calibrations + full 11-type acute silencing wherever baseline >= 0.1 Hz.
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
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Enforce gentle single-threaded BLAS/Numba inside each worker before importing numpy/numba
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import pyarrow.feather as pf
from numba import njit
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = Path(os.environ.get("PROJC_CACHE_DIR", str(Path.home() / ".cache" / "projc-flywire")))
OUT_JSON = BASE_DIR / "mn9_pre_submission_ab_results.json"

sys.path.insert(0, str(BASE_DIR))
import shiu_mn9_benchmark as smb  # noqa: E402
import mn9_transfer_benchmark as mtb  # noqa: E402

INH_NTS = {"gaba", "glutamate"}
EXC_NTS = {"acetylcholine", "dopamine", "serotonin", "octopamine", "tyramine", "histamine"}

TARGET_SUGAR_HZ = 58.52
W_FROZEN = 0.275
W_BANC_ANAT = 0.275 * (5732.5 / 2283.5)       # 0.6903601927 mV
W_MALE_INTACT = 0.275 * (5732.5 / 5876.0)     # 0.2682841219 mV
W_MALE_PREDNT = 0.275 * (5732.5 / 5976.0)     # 0.2637947624 mV
W_MALE_DENS = 0.275 / 1.72526                 # 0.1593962649 mV
W_MALE_JOINT = 0.275 / 1.75521                # 0.1566764091 mV
W_BANC_DENS = 0.275 / 0.4508                  # 0.6100266193 mV


# ---------------------------------------------------------------------------
# Bit-identical lazy-exact Numba LIF kernel with silencing mask + seizure cap
# ---------------------------------------------------------------------------
@njit(cache=False)
def run_lif_exact_cap_slnc(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    w_poi_base: float,
    exc_idx: np.ndarray,
    rate_hz: float,
    slnc_mask: np.ndarray,
    watch_idx: np.ndarray,
    n_trials: int,
    seed_base: int,
    max_brain_hz_cap: float,
    use_xorshift: int = 0,
) -> tuple[np.ndarray, float, int]:
    """Exact Brian2/Shiu LIF schedule with arbitrary per-edge float64 weights,
    binary silencing mask, whole-brain spike count, and runaway seizure cap."""
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
    w_poi = w_poi_base * 250.0
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
    inv_2_53 = 1.0 / 9007199254740992.0
    peak_gain = 0.160
    safe_th = 6.95
    cap_spikes = int(max_brain_hz_cap * n_nodes) if max_brain_hz_cap > 0.0 else 0
    total_brain_spikes = 0
    capped_trials = 0
    trials_done = 0

    for tr in range(n_trials):
        s_opto = np.uint64(seed_base) + np.uint64(tr) * np.uint64(65537) + np.uint64(0x9E3779B97F4A7C15)
        if s_opto == np.uint64(0):
            s_opto = np.uint64(0x9E3779B97F4A7C15)
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
                if use_xorshift != 0:
                    for k in range(len(exc_idx)):
                        s_opto ^= s_opto >> np.uint64(12)
                        s_opto ^= s_opto << np.uint64(25)
                        s_opto ^= s_opto >> np.uint64(27)
                        val = (s_opto * np.uint64(2685821657736338717)) >> np.uint64(11)
                        if float(val) * inv_2_53 < p_poi:
                            u[exc_idx[k]] += w_poi
                else:
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

            if cap_spikes > 0 and tr_spikes >= cap_spikes:
                hit_cap = True
                break

        for t_i in range(n_touched):
            idx = touched[t_i]
            u[idx] = 0.0
            g[idx] = 0.0
            last_spk[idx] = -100000
            last_upd[idx] = 0
            in_active[idx] = 0
            is_touched[idx] = 0

        total_brain_spikes += tr_spikes
        trials_done += 1
        if hit_cap:
            capped_trials += 1
            if capped_trials >= 2:
                break

    mean_brain_hz = (total_brain_spikes / (trials_done * n_nodes)) if trials_done > 0 else 0.0
    return watch_counts, mean_brain_hz, capped_trials


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def stepwise_auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.float64)
    pos = int(y.sum())
    if pos == 0:
        return 0.0
    uniq = np.sort(np.unique(s))[::-1]
    tp = fp = prev_r = ap = 0.0
    for u in uniq:
        m = s == u
        tp += float(y[m].sum())
        fp += float((1 - y[m]).sum())
        r = tp / pos
        p = tp / (tp + fp)
        ap += (r - prev_r) * p
        prev_r = r
    return float(ap)


def compute_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int32)
    s = np.asarray(s, dtype=np.float64)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    u_stat = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u_stat / (len(pos) * len(neg)))


def rank_1based_ties(scores: np.ndarray, target_i: int) -> tuple[int, int]:
    """Exact 1-based tie range (hi_rank, lo_rank) descending."""
    val = float(scores[target_i])
    greater = int(np.sum(scores > val))
    equal = int(np.sum(scores == val))
    return greater + 1, greater + equal


def summarize_sweep(
    names: list[str],
    y: np.ndarray,
    prior_pub: np.ndarray,
    is_direct: np.ndarray,
    v783_ref_50: np.ndarray,
    v783_ref_200: np.ndarray,
    v783_exc_any: np.ndarray,
    rates: np.ndarray,
    l_rates: np.ndarray,
    r_rates: np.ndarray,
    brain_hzs: np.ndarray,
    capped_counts: np.ndarray,
    freq_hz: int,
) -> dict:
    ref = v783_ref_50 if freq_hz == 50 else v783_ref_200
    nov_mask = ~prior_pub.astype(bool)
    nd_mask = ~is_direct.astype(bool)
    excl4_mask = ~np.isin(names, ["marge", "phantom", "tentacular", "usnea"])

    ap_all = stepwise_auprc(y, rates)
    ap_nov = stepwise_auprc(y[nov_mask], rates[nov_mask])
    ap_nd = stepwise_auprc(y[nd_mask], rates[nd_mask])
    ap_excl4 = stepwise_auprc(y[excl4_mask], rates[excl4_mask])
    ap_exc58 = stepwise_auprc(y[v783_exc_any], rates[v783_exc_any])
    auroc = compute_auroc(y, rates)

    # Active counts
    act_pos = int(np.sum((rates > 0) & (y == 1)))
    act_neg = int(np.sum((rates > 0) & (y == 0)))
    act_pos_01 = int(np.sum((rates >= 0.1) & (y == 1)))
    act_neg_01 = int(np.sum((rates >= 0.1) & (y == 0)))
    act_pos_1 = int(np.sum((rates >= 1.0) & (y == 1)))
    act_neg_1 = int(np.sum((rates >= 1.0) & (y == 0)))

    # Threshold sensitivity
    ap_01 = stepwise_auprc(y, np.where(rates >= 0.1, rates, 0.0))
    ap_10 = stepwise_auprc(y, np.where(rates >= 1.0, rates, 0.0))

    # Concordance with v783
    rho_all = float(stats.spearmanr(ref, rates).statistic) if np.std(rates) > 0 else 0.0
    tau_all = float(stats.kendalltau(ref, rates).statistic) if np.std(rates) > 0 else 0.0
    shared_zeros = int(np.sum((ref == 0) & (rates == 0)))
    bin_agree = float(np.mean((ref > 0) == (rates > 0)))
    # Cohen's kappa
    a = int(np.sum((ref > 0) & (rates > 0)))
    b = int(np.sum((ref > 0) & (rates == 0)))
    c = int(np.sum((ref == 0) & (rates > 0)))
    d = int(np.sum((ref == 0) & (rates == 0)))
    n = a + b + c + d
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    kappa = float((bin_agree - pe) / (1.0 - pe)) if pe < 1.0 else 1.0

    co_mask = (ref > 0) & (rates > 0)
    any_mask = (ref > 0) | (rates > 0)
    rho_co = float(stats.spearmanr(ref[co_mask], rates[co_mask]).statistic) if co_mask.sum() >= 3 and np.std(rates[co_mask]) > 0 else float("nan")
    rho_any = float(stats.spearmanr(ref[any_mask], rates[any_mask]).statistic) if any_mask.sum() >= 3 and np.std(rates[any_mask]) > 0 else float("nan")

    # Key positive ranks in unpublished-95
    nov_names = [names[i] for i in range(len(names)) if nov_mask[i]]
    nov_rates = rates[nov_mask]
    di_rank_nov = rank_1based_ties(nov_rates, nov_names.index("diatom"))
    ss_rank_nov = rank_1based_ties(nov_rates, nov_names.index("sink_sync"))
    di_rank_all = rank_1based_ties(rates, names.index("diatom"))
    ss_rank_all = rank_1based_ties(rates, names.index("sink_sync"))

    # Focal rates
    focal = {nm: float(rates[names.index(nm)]) for nm in ["roundup", "diatom", "TH_VUM", "sink_sync", "FMIn", "rattle", "Fdg", "peacock", "puddle", "spirit", "mime", "peahen", "crab"]}

    return {
        "freq_hz": freq_hz,
        "auprc_all106": ap_all,
        "auprc_unpub95": ap_nov,
        "auprc_nondirect": ap_nd,
        "n_nondirect": int(nd_mask.sum()),
        "auprc_excl4_nonexc_pos_102": ap_excl4,
        "auprc_strictly_exc_58": ap_exc58,
        "auroc_all106": auroc,
        "auprc_thr_0p1": ap_01,
        "auprc_thr_1p0": ap_10,
        "active_pos_gt0": act_pos,
        "active_neg_gt0": act_neg,
        "active_pos_ge0p1": act_pos_01,
        "active_neg_ge0p1": act_neg_01,
        "active_pos_ge1p0": act_pos_1,
        "active_neg_ge1p0": act_neg_1,
        "spearman_all106": rho_all,
        "kendall_tau_b_all106": tau_all,
        "shared_zeros_vs_v783": shared_zeros,
        "binary_agreement_vs_v783": bin_agree,
        "cohen_kappa_vs_v783": kappa,
        "coactive_n": int(co_mask.sum()),
        "spearman_coactive": rho_co,
        "anyactive_n": int(any_mask.sum()),
        "spearman_anyactive": rho_any,
        "mean_brain_hz": float(np.mean(brain_hzs)),
        "max_brain_hz": float(np.max(brain_hzs)),
        "n_types_capped": int(np.sum(capped_counts > 0)),
        "rank_unpub95_diatom": list(di_rank_nov),
        "rank_unpub95_sink_sync": list(ss_rank_nov),
        "rank_all106_diatom": list(di_rank_all),
        "rank_all106_sink_sync": list(ss_rank_all),
        "focal_rates_hz": focal,
        "per_type_mean_hz": [float(x) for x in rates],
        "per_type_left_hz": [float(x) for x in l_rates],
        "per_type_right_hz": [float(x) for x in r_rates],
        "per_type_brain_hz": [float(x) for x in brain_hzs],
    }


# ---------------------------------------------------------------------------
# Linux COW-fork parallel sweep worker (zero RAM duplication!)
# ---------------------------------------------------------------------------
_SWEEP_CTX: dict = {}


def _worker_sim_one(job: tuple[int, np.ndarray, float, int, int, np.ndarray | None]) -> tuple[int, float, float, float, float, int]:
    """Run one (idx, exc_idx, rate_hz, n_trials, seed, slnc_idx) on shared COW graph."""
    job_id, exc_idx, rate_hz, n_trials, seed, slnc_idx = job
    ctx = _SWEEP_CTX
    n_nodes = ctx["n_nodes"]
    if len(exc_idx) == 0:
        return job_id, 0.0, 0.0, 0.0, 0.0, 0
    slnc_mask = ctx["zero_slnc"]
    if slnc_idx is not None and len(slnc_idx) > 0:
        slnc_mask = np.zeros(n_nodes, dtype=np.uint8)
        slnc_mask[slnc_idx] = 1
    cnt, b_hz, cap = run_lif_exact_cap_slnc(
        ctx["indptr"],
        ctx["indices"],
        ctx["weights"],
        ctx["w_poi_base"],
        exc_idx.astype(np.int32),
        float(rate_hz),
        slnc_mask,
        ctx["watch_idx"],
        int(n_trials),
        int(seed),
        float(ctx["max_brain_hz_cap"]),
        int(ctx["use_xorshift"]),
    )
    l_r = float(cnt[:, 0].mean())
    r_r = float(cnt[:, 1].mean())
    m_r = 0.5 * (l_r + r_r)
    return job_id, m_r, l_r, r_r, float(b_hz), int(cap)


def run_parallel_sweep(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    w_poi_base: float,
    mn9_idx: np.ndarray,
    exc_list: list[np.ndarray],
    rate_hz: float,
    n_trials: int,
    seed_base: int,
    workers: int,
    max_brain_hz_cap: float = 15.0,
    use_xorshift: int = 0,
    seed_step: int = 97,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = len(indptr) - 1
    _SWEEP_CTX.clear()
    _SWEEP_CTX.update({
        "n_nodes": n_nodes,
        "indptr": indptr,
        "indices": indices,
        "weights": weights,
        "w_poi_base": float(w_poi_base),
        "watch_idx": mn9_idx.astype(np.int32),
        "zero_slnc": np.zeros(n_nodes, dtype=np.uint8),
        "max_brain_hz_cap": float(max_brain_hz_cap),
        "use_xorshift": int(use_xorshift),
    })
    jobs = [(i, exc_list[i], rate_hz, n_trials, seed_base + i * seed_step, None) for i in range(len(exc_list))]
    n = len(jobs)
    means = np.zeros(n, dtype=np.float64)
    l_arr = np.zeros(n, dtype=np.float64)
    r_arr = np.zeros(n, dtype=np.float64)
    b_arr = np.zeros(n, dtype=np.float64)
    c_arr = np.zeros(n, dtype=np.int32)
    ctx_mp = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx_mp) as ex:
        for jid, m_r, l_r, r_r, b_hz, cap in ex.map(_worker_sim_one, jobs, chunksize=2):
            means[jid] = m_r
            l_arr[jid] = l_r
            r_arr[jid] = r_r
            b_arr[jid] = b_hz
            c_arr[jid] = cap
    _SWEEP_CTX.clear()
    return means, l_arr, r_arr, b_arr, c_arr


def _worker_eval_sugar_w(job: tuple[int, float, int]) -> tuple[int, dict]:
    idx, w, n_trials = job
    ctx = _SWEEP_CTX
    w_poi = w if ctx["w_poi_fixed"] is None else ctx["w_poi_fixed"]
    cnt, b_hz, cap = run_lif_exact_cap_slnc(
        ctx["indptr"],
        ctx["indices"],
        ctx["base_weights_unit"] * w,
        w_poi,
        ctx["sugar_idx"],
        100.0,
        ctx["zero_slnc"],
        ctx["watch_idx"],
        n_trials,
        42,
        ctx["max_brain_hz_cap"],
        ctx["use_xorshift"],
    )
    l_r = float(cnt[:, 0].mean())
    r_r = float(cnt[:, 1].mean())
    m_r = 0.5 * (l_r + r_r)
    val = l_r if ctx["readout_mode"] == "left_intact" else m_r
    return idx, {"w": float(w), "val_hz": val, "mn9_mean_hz": m_r, "mn9_l_hz": l_r, "mn9_r_hz": r_r, "brain_hz": float(b_hz), "capped": int(cap)}


def bisect_sugar_weight(
    indptr: np.ndarray,
    indices: np.ndarray,
    base_weights_unit: np.ndarray,
    sugar_idx: np.ndarray,
    mn9_idx: np.ndarray,
    target_hz: float,
    readout_mode: str,  # "bilateral" or "left_intact"
    w_poi_fixed: float | None,  # None -> use candidate scalar w; float -> fixed w_poi_base
    grid: list[float],
    max_brain_hz_cap: float = 15.0,
    workers: int = 10,
    use_xorshift: int = 0,
) -> dict:
    """Scan coarse grid in parallel (fast COW fork!), then bisect 8 steps to match target_hz."""
    n_nodes = len(indptr) - 1
    zero_slnc = np.zeros(n_nodes, dtype=np.uint8)
    s_idx = sugar_idx.astype(np.int32)
    w_idx = mn9_idx.astype(np.int32)

    # Parallel coarse grid scan across workers (turns 8 min into ~25 sec!)
    _SWEEP_CTX.clear()
    _SWEEP_CTX.update({
        "indptr": indptr,
        "indices": indices,
        "base_weights_unit": base_weights_unit,
        "sugar_idx": s_idx,
        "watch_idx": w_idx,
        "zero_slnc": zero_slnc,
        "readout_mode": readout_mode,
        "w_poi_fixed": w_poi_fixed,
        "max_brain_hz_cap": float(max_brain_hz_cap),
        "use_xorshift": int(use_xorshift),
    })
    scan = [None] * len(grid)
    ctx_mp = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=min(workers, len(grid)), mp_context=ctx_mp) as ex:
        for idx, row in ex.map(_worker_eval_sugar_w, [(i, float(w), 10) for i, w in enumerate(grid)]):
            scan[idx] = row
    _SWEEP_CTX.clear()

    def eval_w(w: float, n_trials: int) -> dict:
        w_poi = w if w_poi_fixed is None else w_poi_fixed
        cnt, b_hz, cap = run_lif_exact_cap_slnc(
            indptr, indices, base_weights_unit * w, w_poi,
            s_idx, 100.0, zero_slnc, w_idx, n_trials, 42, max_brain_hz_cap, use_xorshift,
        )
        l_r = float(cnt[:, 0].mean())
        r_r = float(cnt[:, 1].mean())
        m_r = 0.5 * (l_r + r_r)
        val = l_r if readout_mode == "left_intact" else m_r
        return {"w": float(w), "val_hz": val, "mn9_mean_hz": m_r, "mn9_l_hz": l_r, "mn9_r_hz": r_r, "brain_hz": float(b_hz), "capped": int(cap)}

    lo_row = None
    hi_row = None
    for row in scan:
        print(f"    [sugar scan w={row['w']:.5f}] val={row['val_hz']:.2f} Hz (L={row['mn9_l_hz']:.2f}, R={row['mn9_r_hz']:.2f}, brain={row['brain_hz']:.3f} Hz, cap={row['capped']})", flush=True)
    for row in scan:
        if row["capped"] > 0:
            break
        if row["val_hz"] < target_hz:
            lo_row = row
        elif hi_row is None:
            hi_row = row
            break

    status = "converged"
    if hi_row is None:
        # Target unreachable before seizure cap or grid max -> pick physiological peak (brain < 5.0 Hz preferred)
        phys = [r for r in scan if r["capped"] == 0 and r["brain_hz"] < 5.0]
        pool = phys if phys else [r for r in scan if r["capped"] == 0]
        best = max(pool, key=lambda r: r["val_hz"]) if pool else scan[0]
        best_w = best["w"]
        status = "unreachable_peak_selected"
    elif lo_row is None:
        best_w = hi_row["w"]
        status = "above_target_at_grid_min"
    else:
        lo_w, hi_w = lo_row["w"], hi_row["w"]
        for _ in range(8):
            mid = 0.5 * (lo_w + hi_w)
            rm = eval_w(mid, 10)
            if rm["capped"] > 0 or rm["val_hz"] >= target_hz:
                hi_w = mid
            else:
                lo_w = mid
            if abs(rm["val_hz"] - target_hz) < 0.35 and rm["capped"] == 0:
                lo_w = hi_w = mid
                break
        best_w = 0.5 * (lo_w + hi_w)

    final_30 = eval_w(best_w, 30)
    print(f"    [sugar FINAL w*={best_w:.6f} ({status})] 30-trial val={final_30['val_hz']:.2f} Hz (L={final_30['mn9_l_hz']:.2f}, R={final_30['mn9_r_hz']:.2f}, brain={final_30['brain_hz']:.3f} Hz, cap={final_30['capped']})", flush=True)
    return {"status": status, "target_hz": target_hz, "readout_mode": readout_mode, "w_star": float(best_w), "final_30trial": final_30, "coarse_scan": scan}


def load_common_meta() -> dict:
    rows = list(csv.DictReader((BASE_DIR / "english_manuscript/supp_table1_three_brain_sez106.tsv").open(), delimiter="\t"))
    names = [r["name"] for r in rows]
    y = np.array([int(r["y_bin"]) for r in rows], dtype=np.int32)
    prior_pub = np.array([int(r["prior_published"]) for r in rows], dtype=np.int32)
    v783_50 = np.array([float(r["v783_50hz"]) for r in rows], dtype=np.float64)
    v783_200 = np.array([float(r["v783_200hz"]) for r in rows], dtype=np.float64)
    banc_res = json.loads((BASE_DIR / "mn9_banc_results.json").read_text())
    nt_by_name = {r["name"]: r["v783_cls_counts"] for r in banc_res["mapping_audit"]["per_type_sez106"]}
    v783_exc_any = np.array([nt_by_name[nm].get("0", 0) > 0 and nm != "usnea" for nm in names], dtype=bool)
    return {
        "names": names, "y": y, "prior_pub": prior_pub,
        "v783_50": v783_50, "v783_200": v783_200, "v783_exc_any": v783_exc_any,
        "rows": rows,
    }


def save_checkpoint(out_dict: dict) -> None:
    tmp = OUT_JSON.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out_dict, indent=2), encoding="utf-8")
    tmp.replace(OUT_JSON)


def load_v783_reference() -> tuple[dict, np.ndarray, dict[str, float], dict[int, float], dict[str, list[int]], list[int]]:
    comp783 = CACHE_DIR / "data/Completeness_783.csv"
    conn783 = CACHE_DIR / "data/Connectivity_783.parquet"
    ann783 = CACHE_DIR / "data/Supplemental_file1_neuron_annotations.tsv"
    g783 = mtb.load_graph_csr(comp783, conn783)
    fids783 = g783["flyids"]
    n783 = int(len(fids783))
    idx783 = g783["flyid2idx"]

    s_in_783 = np.zeros(n783, dtype=np.float64)
    np.add.at(s_in_783, g783["post_s"], np.abs(g783["w_raw_s"].astype(np.float64)))
    s_in_by_fid = {int(fids783[i]): float(s_in_783[i]) for i in range(n783)}

    ct_vals: dict[str, list[float]] = defaultdict(list)
    with ann783.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rid_s = row.get("root_id", "")
            if not rid_s or not rid_s.isdigit():
                continue
            rid = int(rid_s)
            if rid in idx783:
                ct = (row.get("cell_type") or "").strip() or (row.get("hemibrain_type") or "").strip()
                if ct:
                    ct_vals[ct].append(float(s_in_783[idx783[rid]]))
    v783_ct_mean_sin = {ct: float(np.mean(vs)) for ct, vs in ct_vals.items()}

    v7_res = json.loads((BASE_DIR / "mn9_transfer_results.json").read_text())
    ba_res = json.loads((BASE_DIR / "mn9_banc_results.json").read_text())
    sez_v783_fids = {r["name"]: [int(x) for x in r["v783_ids"]] for r in v7_res["mapping_audit"]["per_type_sez106"]}
    mn9_783_fids = [
        int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_l_v783"]),
        int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_r_v783"]),
    ]
    return g783, s_in_783, v783_ct_mean_sin, s_in_by_fid, sez_v783_fids, mn9_783_fids


# ---------------------------------------------------------------------------
# PHASE 1: Experiment B1 (FlyWire v783 diatom downsampling + v783 intrinsic 1/S_in ref)
# ---------------------------------------------------------------------------
def run_phase_b1(meta: dict, workers: int) -> tuple[dict, dict[str, float], dict[int, float]]:
    t0 = time.time()
    print("\n================ PHASE B1: FlyWire v783 diatom downsampling & 1/S_in ref ================", flush=True)
    g783, s_in_783, v783_ct_mean_sin, s_in_by_fid, sez_v783_fids, mn9_783_fids = load_v783_reference()
    idx783 = g783["flyid2idx"]
    indptr = g783["indptr"]
    indices = g783["post_s"]
    w_raw = g783["w_raw_s"].astype(np.float64)
    n783 = len(indptr) - 1
    mn9_idx = np.array([idx783[f] for f in mn9_783_fids], dtype=np.int32)
    sugar_783_idx = np.array(
        [idx783[720575940639259967 if int(f) == 720575940620900446 else int(f)] for f in smb.SUGAR_GRN_FIDS],
        dtype=np.int32,
    )

    # Warm up Numba & parity check on 100 Hz sugar GRN (must match 58.3000 Hz!)
    zero_slnc = np.zeros(n783, dtype=np.uint8)
    cnt_s, b_s, cap_s = run_lif_exact_cap_slnc(
        indptr, indices, w_raw * W_FROZEN, W_FROZEN, sugar_783_idx, 100.0, zero_slnc, mn9_idx, 30, 42, 15.0
    )
    sugar_hz = float(cnt_s.mean())
    print(f"  [PARITY] v783 100 Hz sugar GRN = {sugar_hz:.4f} Hz (expected 58.3000 Hz)", flush=True)
    assert abs(sugar_hz - 58.30) < 1e-4, f"Parity mismatch: {sugar_hz}"

    # Full diatom 14/14 parity check using exact unsorted order & per-type seed (50000 + di_idx * 97 / 200000 + di_idx * 97)
    diatom_fids = sez_v783_fids["diatom"]
    assert len(diatom_fids) == 14
    di_pos_all = meta["names"].index("diatom")
    diatom_idx = np.array([idx783[f] for f in diatom_fids], dtype=np.int32)
    di_seed_50 = 50000 + di_pos_all * 97
    di_seed_200 = 200000 + di_pos_all * 97
    cnt_d50, _, _ = run_lif_exact_cap_slnc(indptr, indices, w_raw * W_FROZEN, W_FROZEN, diatom_idx, 50.0, zero_slnc, mn9_idx, 30, di_seed_50, 15.0)
    cnt_d200, _, _ = run_lif_exact_cap_slnc(indptr, indices, w_raw * W_FROZEN, W_FROZEN, diatom_idx, 200.0, zero_slnc, mn9_idx, 30, di_seed_200, 15.0)
    di_ref_50 = meta["v783_50"][di_pos_all]
    di_ref_200 = meta["v783_200"][di_pos_all]
    print(f"  [PARITY] v783 diatom 14/14: 50 Hz={cnt_d50.mean():.4f} (ref {di_ref_50:.4f}), 200 Hz={cnt_d200.mean():.4f} (ref {di_ref_200:.4f})", flush=True)
    assert abs(float(cnt_d50.mean()) - di_ref_50) < 1e-3, f"diatom 50Hz mismatch: {cnt_d50.mean()} vs {di_ref_50}"

    # Exact 3 unique FlyWire cells matched by BANC's 4 reviewed diatom cells
    banc_matched_fids = [720575940605699104, 720575940641058523, 720575940619697134]
    for fid in banc_matched_fids:
        assert fid in diatom_fids, f"Matched fid {fid} not in diatom_fids"
    banc_matched_idx = np.array([idx783[f] for f in banc_matched_fids], dtype=np.int32)

    # Individual diatom cell anatomy (outgoing synapses & signed 2-hop ReLU onto MN9)
    w1_mn9 = np.zeros(n783, dtype=np.float64)
    for u in range(n783):
        p0, p1 = indptr[u], indptr[u + 1]
        for e in range(p0, p1):
            if indices[e] == mn9_idx[0] or indices[e] == mn9_idx[1]:
                w1_mn9[u] += w_raw[e]
    indiv_cells = []
    for fid in diatom_fids:
        u = idx783[fid]
        p0, p1 = indptr[u], indptr[u + 1]
        out_syn = float(np.sum(np.abs(w_raw[p0:p1])))
        h1 = float(w1_mn9[u])
        h2 = float(np.sum(np.maximum(0.0, w_raw[p0:p1]) * w1_mn9[indices[p0:p1]]))
        indiv_cells.append({
            "v783_root_id": int(fid),
            "in_banc_reviewed_4": bool(fid in banc_matched_fids),
            "out_syn": out_syn,
            "h1_direct": h1,
            "s2hop_relu": h1 + 0.01 * h2,
        })

    # Build random subset jobs: exact-3 + 100x(3/14) + 100x(4/14) + 100x(7/14)
    rng = np.random.default_rng(2026)
    subsets = {"exact_banc_matched_3": [banc_matched_idx]}
    for k in (3, 4, 7):
        subsets[f"random_{k}_of_14"] = [np.sort(rng.choice(diatom_idx, size=k, replace=False)).astype(np.int32) for _ in range(100)]

    names = meta["names"]
    y = meta["y"]
    nov_mask = ~meta["prior_pub"].astype(bool)
    di_pos_all = names.index("diatom")
    nov_names = [names[i] for i in range(len(names)) if nov_mask[i]]
    di_pos_nov = nov_names.index("diatom")

    diatom_results = {"full_14_of_14": {"50hz": float(cnt_d50.mean()), "200hz": float(cnt_d200.mean())}, "individual_14_cells": indiv_cells}
    for freq_hz, base_vec in [(50, meta["v783_50"]), (200, meta["v783_200"])]:
        fkey = f"{freq_hz}hz"
        sb_di = di_seed_50 if freq_hz == 50 else di_seed_200
        diatom_results[fkey] = {}
        for label, sub_list in subsets.items():
            means, l_arr, r_arr, _, _ = run_parallel_sweep(
                indptr, indices, w_raw * W_FROZEN, W_FROZEN, mn9_idx, sub_list, float(freq_hz), 30, sb_di, workers, 15.0, use_xorshift=0, seed_step=0
            )
            aps_all, aps_nov, ranks_nov = [], [], []
            for m_r in means:
                vec = base_vec.copy()
                vec[di_pos_all] = m_r
                aps_all.append(stepwise_auprc(y, vec))
                aps_nov.append(stepwise_auprc(y[nov_mask], vec[nov_mask]))
                ranks_nov.append(rank_1based_ties(vec[nov_mask], di_pos_nov)[0])
            aps_all = np.array(aps_all)
            aps_nov = np.array(aps_nov)
            ranks_nov = np.array(ranks_nov)
            diatom_results[fkey][label] = {
                "n_draws": int(len(means)),
                "rate_hz_median": float(np.median(means)),
                "rate_hz_iqr": [float(np.percentile(means, 25)), float(np.percentile(means, 75))],
                "rate_hz_min_max": [float(np.min(means)), float(np.max(means))],
                "rate_hz_mean": float(np.mean(means)),
                "frac_exact_zero": float(np.mean(means == 0.0)),
                "frac_lt_0p1hz": float(np.mean(means < 0.1)),
                "frac_ge_1p0hz": float(np.mean(means >= 1.0)),
                "auprc_all106_median": float(np.median(aps_all)),
                "auprc_all106_iqr": [float(np.percentile(aps_all, 25)), float(np.percentile(aps_all, 75))],
                "auprc_unpub95_median": float(np.median(aps_nov)),
                "auprc_unpub95_iqr": [float(np.percentile(aps_nov, 25)), float(np.percentile(aps_nov, 75))],
                "rank_unpub95_median": float(np.median(ranks_nov)),
                "rank_unpub95_min_max": [int(np.min(ranks_nov)), int(np.max(ranks_nov))],
            }
            print(f"  [B1 {fkey} {label}] median={np.median(means):.2f} Hz (IQR {np.percentile(means,25):.2f}-{np.percentile(means,75):.2f}), zero={100*np.mean(means==0):.1f}%, all106={np.median(aps_all):.4f}, unpub95={np.median(aps_nov):.4f}", flush=True)

    # Also run FlyWire v783 intrinsic scale-invariant per-neuron normalization: W(u,j) / max(S_in(j), S_med)
    # (plus log unfloored 1/S_in sugar scan showing low-degree amplification triggers seizure before 58.5 Hz)
    print("\n  [A2b ref] FlyWire v783 intrinsic median-floored per-neuron norm W/max(S_in, S_med)...", flush=True)
    s_med_783 = float(np.median(s_in_783[s_in_783 > 0]))
    inv_floor_783 = 1.0 / np.maximum(s_in_783, s_med_783)
    w_unit_inv = w_raw * inv_floor_783[indices]
    bis_v783_inv = bisect_sugar_weight(
        indptr, indices, w_unit_inv, sugar_783_idx, mn9_idx, TARGET_SUGAR_HZ,
        readout_mode="bilateral", w_poi_fixed=W_FROZEN,
        grid=[100.0, 250.0, 450.0, 700.0, 1000.0, 1400.0, 2000.0, 2800.0],
    )
    w_inv_star = bis_v783_inv["w_star"]
    sez_exc_783 = [np.array([idx783[f] for f in sez_v783_fids[nm] if f in idx783], dtype=np.int32) for nm in names]
    is_direct_v783 = np.array([int(r["is_direct_v783"]) for r in meta["rows"]], dtype=bool)
    v783_inv_sweeps = {}
    for freq_hz in (50, 200):
        sb = 50000 if freq_hz == 50 else 200000
        m, lr, rr, bh, cc = run_parallel_sweep(
            indptr, indices, w_unit_inv * w_inv_star, W_FROZEN, mn9_idx, sez_exc_783, float(freq_hz), 30, sb, workers, 15.0
        )
        sm = summarize_sweep(names, y, meta["prior_pub"], is_direct_v783, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
        v783_inv_sweeps[f"{freq_hz}hz"] = sm
        print(f"    v783 intrinsic norm {freq_hz}Hz (S_med={s_med_783:.1f}, W*={w_inv_star:.1f}): all106={sm['auprc_all106']:.4f}, unpub95={sm['auprc_unpub95']:.4f}, pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, brain={sm['mean_brain_hz']:.4f}", flush=True)

    del g783, w_raw, w_unit_inv
    gc.collect()
    return {
        "experiment_b1_diatom_downsampling": diatom_results,
        "v783_intrinsic_inv_sin_reference": {"s_med": s_med_783, "sugar_anchor_calibration": bis_v783_inv, "sweeps": v783_inv_sweeps},
        "elapsed_sec": float(time.time() - t0),
    }, v783_ct_mean_sin, s_in_by_fid


# ---------------------------------------------------------------------------
# PHASE 2: BANC v888 Experiment A1 (Sugar anchor) + A2a/A2b (Per-neuron norm)
# ---------------------------------------------------------------------------
def run_phase_banc(meta: dict, s_in_v783_by_fid: dict[int, float], workers: int) -> dict:
    t0 = time.time()
    print("\n================ PHASE BANC: A1 Sugar-GRN Anchor & A2 Per-Neuron Norm ================", flush=True)
    bdir = CACHE_DIR / "data/banc"
    bm = pf.read_table(
        bdir / "banc_888_meta.feather",
        columns=["banc_888_id", "root_id", "supervoxel_id", "root_626", "root_850", "root_888", "root_890", "super_class", "cell_type", "side", "proofread", "roughly_proofread"],
    ).to_pydict()
    sv_to_bid: dict[int, int] = {}
    old_root_to_bid: dict[int, int] = {}
    main_cand_set: set[int] = set()
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
    main_cand_b = np.array(sorted(main_cand_set), dtype=np.int64)
    nt_csv: dict[int, str] = {}
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
    nt_sgn_b = np.array([nt_to_sign(nt_csv[int(b)]) for b in nt_ids_b], dtype=np.int8)
    p_nt = np.minimum(np.searchsorted(nt_ids_b, ep_b), len(nt_ids_b) - 1)
    v_nt = np.where(nt_ids_b[p_nt] == ep_b, nt_sgn_b[p_nt], 0) != 0
    keep_b = np.isin(ep_b, main_cand_b) & np.isin(eq_b, main_cand_b) & v_nt
    node_ids = np.union1d(ep_b[keep_b], eq_b[keep_b]).astype(np.int64)
    del ep_b, eq_b, ns_b, keep_b, v_nt, p_nt, nt_ids_b, nt_sgn_b, main_cand_b, nt_csv

    gb = np.load(CACHE_DIR / "banc_ckpt/banc_main_graph.npz")
    assert len(node_ids) == int(gb["n_nodes"]), f"BANC node count mismatch: {len(node_ids)} vs {int(gb['n_nodes'])}"
    indptr = gb["indptr"].astype(np.int64)
    indices = gb["post_s"].astype(np.int32)
    w_raw = gb["w_raw_s"].astype(np.float64)
    n_nodes = len(node_ids)
    bid2idx = {int(b): i for i, b in enumerate(node_ids)}

    s_in_banc = np.zeros(n_nodes, dtype=np.float64)
    np.add.at(s_in_banc, indices, np.abs(w_raw))

    ba_res = json.loads((BASE_DIR / "mn9_banc_results.json").read_text())
    mn9_l_id = int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_l_banc"])
    mn9_r_id = int(ba_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["mn9_r_banc"])
    mn9_idx = np.array([bid2idx[mn9_l_id], bid2idx[mn9_r_id]], dtype=np.int32)

    names = meta["names"]
    y = meta["y"]
    is_direct_banc = np.array([int(r["is_direct_banc_m1"]) for r in meta["rows"]], dtype=bool)
    banc_m1_by_name = {r["name"]: [bid2idx[int(b)] for b in r["m1_banc_ids"] if int(b) in bid2idx] for r in ba_res["mapping_audit"]["per_type_sez106"]}
    sez_exc_banc = [np.array(banc_m1_by_name[nm], dtype=np.int32) for nm in names]
    selk_banc_ids = [int(b) for b in ba_res["selk_full_cns_diagnosis"]["selk_banc_ids"]]
    selk_idx = np.array([bid2idx[b] for b in selk_banc_ids if b in bid2idx], dtype=np.int32)

    # Parity check on roundup at 50 Hz (anat 0.690360 mV, exact per-type seed 50000 + ru_i * 97)
    zero_slnc = np.zeros(n_nodes, dtype=np.uint8)
    ru_i = names.index("roundup")
    cnt_ru, _, _ = run_lif_exact_cap_slnc(indptr, indices, w_raw * W_BANC_ANAT, W_BANC_ANAT, sez_exc_banc[ru_i], 50.0, zero_slnc, mn9_idx, 30, 50000 + ru_i * 97, 15.0)
    ru_ref = float(meta["rows"][ru_i]["banc_anat_50hz"])
    print(f"  [PARITY] BANC roundup 50 Hz anat = {cnt_ru.mean():.4f} Hz (locked ref = {ru_ref:.4f} Hz)", flush=True)
    assert abs(float(cnt_ru.mean()) - ru_ref) < 1.0, f"BANC parity mismatch: {cnt_ru.mean()} vs {ru_ref}"

    # Extract BANC sugar GRNs from already-loaded banc_888_meta dict (using banc_888_id!)
    lb3b_r, lb3bc_r, lb3bc_l = [], [], []
    for i in range(len(bm["banc_888_id"])):
        if bm["banc_888_id"][i] is None:
            continue
        rid = int(bm["banc_888_id"][i])
        if rid not in bid2idx:
            continue
        ct = str(bm["cell_type"][i] or "")
        sd = str(bm["side"][i] or "")
        if ct == "LB3b" and sd == "right":
            lb3b_r.append(bid2idx[rid])
            lb3bc_r.append(bid2idx[rid])
        elif ct == "LB3c" and sd == "right":
            lb3bc_r.append(bid2idx[rid])
        elif ct in ("LB3b", "LB3c") and sd == "left":
            lb3bc_l.append(bid2idx[rid])
    lb3b_r = np.array(sorted(lb3b_r), dtype=np.int32)
    lb3bc_r = np.array(sorted(lb3bc_r), dtype=np.int32)
    lb3bc_l = np.array(sorted(lb3bc_l), dtype=np.int32)
    print(f"  BANC sugar GRNs in main graph: right LB3b={len(lb3b_r)}, right LB3b+LB3c={len(lb3bc_r)}, left LB3b+LB3c={len(lb3bc_l)}", flush=True)

    # A1: Sugar-GRN 58.52 Hz anchor recalibration (primary: right LB3b+LB3c 45 cells)
    print("  [BANC A1] Scanning W_syn (parallel) on canonical right LB3b+LB3c (45 cells) toward 58.52 Hz...", flush=True)
    bis_banc_sugar = bisect_sugar_weight(
        indptr, indices, w_raw, lb3bc_r, mn9_idx, TARGET_SUGAR_HZ,
        readout_mode="bilateral", w_poi_fixed=None,
        grid=[0.275, 0.35, 0.42, 0.50, 0.58, 0.69036, 0.85, 1.05, 1.35, 1.75],
        workers=workers,
    )
    # Parallel sensitivity scan on right LB3b (40) & left LB3b+LB3c (31)
    print("  [BANC A1 sensitivity] Checking right LB3b (40) & left LB3b+LB3c (31)...", flush=True)
    bis_banc_lb3b_r = bisect_sugar_weight(
        indptr, indices, w_raw, lb3b_r, mn9_idx, TARGET_SUGAR_HZ,
        readout_mode="bilateral", w_poi_fixed=None,
        grid=[0.275, 0.40, 0.50, 0.60, 0.69036, 0.90, 1.25],
        workers=workers,
    )
    bis_banc_lb3bc_l = bisect_sugar_weight(
        indptr, indices, w_raw, lb3bc_l, mn9_idx, TARGET_SUGAR_HZ,
        readout_mode="bilateral", w_poi_fixed=None,
        grid=[0.275, 0.40, 0.50, 0.60, 0.69036, 0.90, 1.25],
        workers=workers,
    )

    w_sugar_banc = bis_banc_sugar["w_star"]
    banc_a1_sweeps = {}
    for freq_hz in (50, 200):
        sb = 50000 if freq_hz == 50 else 200000
        m, lr, rr, bh, cc = run_parallel_sweep(
            indptr, indices, w_raw * w_sugar_banc, w_sugar_banc, mn9_idx, sez_exc_banc, float(freq_hz), 30, sb, workers, 15.0
        )
        sm = summarize_sweep(names, y, meta["prior_pub"], is_direct_banc, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
        # SELK check
        c_sk, b_sk, _ = run_lif_exact_cap_slnc(indptr, indices, w_raw * w_sugar_banc, w_sugar_banc, selk_idx, float(freq_hz), zero_slnc, mn9_idx, 30, 42, 15.0)
        sm["selk_mean_hz"] = float(c_sk.mean())
        banc_a1_sweeps[f"{freq_hz}hz"] = sm
        print(f"    BANC A1 sugar-anchor (W={w_sugar_banc:.4f}) {freq_hz}Hz: all106={sm['auprc_all106']:.4f}, unpub95={sm['auprc_unpub95']:.4f}, nd={sm['auprc_nondirect']:.4f}, rho={sm['spearman_all106']:.4f}, pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, brain={sm['mean_brain_hz']:.4f}", flush=True)

    # A2a: BANC homolog-matched per-neuron input normalization (1:1 reviewed FAFB matches)
    print("\n  [BANC A2a] Homolog-matched per-neuron input normalization...", flush=True)
    banc_to_fafb_sins: dict[int, list[float]] = defaultdict(list)
    with (CACHE_DIR / "data/banc/banc_fafb_reviewed_matches.csv.gz").open("rt") as f:
        for r in csv.DictReader(f):
            if r.get("valid") != "t":
                continue
            try:
                bid_raw = int(r["pt_root_id"])
                fid = int(r["match_id"])
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

    w_post_hom = np.full(n_nodes, W_BANC_DENS, dtype=np.float64)
    n_matched_norm = 0
    for j, f_sins in banc_to_fafb_sins.items():
        if s_in_banc[j] > 0:
            ratio = float(np.mean(f_sins)) / float(s_in_banc[j])
            w_post_hom[j] = W_FROZEN * min(max(ratio, 0.1), 10.0)
            n_matched_norm += 1
    w_edge_hom = w_raw * w_post_hom[indices]
    cnt_hs, b_hs, cap_hs = run_lif_exact_cap_slnc(indptr, indices, w_edge_hom, W_FROZEN, lb3bc_r, 100.0, zero_slnc, mn9_idx, 30, 42, 15.0)
    print(f"    Matched {n_matched_norm:,}/{n_nodes:,} BANC neurons; 100 Hz sugar GRN = {cnt_hs.mean():.2f} Hz (L={cnt_hs[:,0].mean():.2f}, R={cnt_hs[:,1].mean():.2f}, brain={b_hs:.4f})", flush=True)

    banc_a2a_sweeps = {}
    for freq_hz in (50, 200):
        sb = 50000 if freq_hz == 50 else 200000
        m, lr, rr, bh, cc = run_parallel_sweep(
            indptr, indices, w_edge_hom, W_FROZEN, mn9_idx, sez_exc_banc, float(freq_hz), 30, sb, workers, 15.0
        )
        sm = summarize_sweep(names, y, meta["prior_pub"], is_direct_banc, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
        c_sk, _, _ = run_lif_exact_cap_slnc(indptr, indices, w_edge_hom, W_FROZEN, selk_idx, float(freq_hz), zero_slnc, mn9_idx, 30, 42, 15.0)
        sm["selk_mean_hz"] = float(c_sk.mean())
        banc_a2a_sweeps[f"{freq_hz}hz"] = sm
        print(f"    BANC A2a homolog-norm {freq_hz}Hz: all106={sm['auprc_all106']:.4f}, unpub95={sm['auprc_unpub95']:.4f}, nd={sm['auprc_nondirect']:.4f}, rho={sm['spearman_all106']:.4f}, pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, brain={sm['mean_brain_hz']:.4f}", flush=True)

    # A2b: BANC intrinsic reference-free median-floored per-neuron norm W/max(S_in, S_med)
    print("\n  [BANC A2b] Intrinsic median-floored per-neuron norm W/max(S_in, S_med) (sugar-anchored)...", flush=True)
    s_med_banc = float(np.median(s_in_banc[s_in_banc > 0]))
    inv_floor_banc = 1.0 / np.maximum(s_in_banc, s_med_banc)
    w_unit_inv = w_raw * inv_floor_banc[indices]
    bis_banc_inv = bisect_sugar_weight(
        indptr, indices, w_unit_inv, lb3bc_r, mn9_idx, TARGET_SUGAR_HZ,
        readout_mode="bilateral", w_poi_fixed=W_FROZEN,
        grid=[100.0, 250.0, 450.0, 700.0, 1000.0, 1400.0, 2000.0, 2800.0],
        workers=workers,
    )
    w_inv_b = bis_banc_inv["w_star"]
    banc_a2b_sweeps = {}
    for freq_hz in (50, 200):
        sb = 50000 if freq_hz == 50 else 200000
        m, lr, rr, bh, cc = run_parallel_sweep(
            indptr, indices, w_unit_inv * w_inv_b, W_FROZEN, mn9_idx, sez_exc_banc, float(freq_hz), 30, sb, workers, 15.0
        )
        sm = summarize_sweep(names, y, meta["prior_pub"], is_direct_banc, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
        c_sk, _, _ = run_lif_exact_cap_slnc(indptr, indices, w_unit_inv * w_inv_b, W_FROZEN, selk_idx, float(freq_hz), zero_slnc, mn9_idx, 30, 42, 15.0)
        sm["selk_mean_hz"] = float(c_sk.mean())
        banc_a2b_sweeps[f"{freq_hz}hz"] = sm
        print(f"    BANC A2b intrinsic norm {freq_hz}Hz (S_med={s_med_banc:.1f}, W*={w_inv_b:.1f}): all106={sm['auprc_all106']:.4f}, unpub95={sm['auprc_unpub95']:.4f}, nd={sm['auprc_nondirect']:.4f}, rho={sm['spearman_all106']:.4f}, pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, brain={sm['mean_brain_hz']:.4f}", flush=True)

    del gb, indptr, indices, w_raw, w_edge_hom, w_unit_inv
    gc.collect()
    return {
        "sugar_grn_counts": {"right_lb3b": int(len(lb3b_r)), "right_lb3b_lb3c": int(len(lb3bc_r)), "left_lb3b_lb3c": int(len(lb3bc_l))},
        "a1_sugar_anchor": {
            "primary_right_lb3b_lb3c": bis_banc_sugar,
            "sensitivity_right_lb3b": bis_banc_lb3b_r,
            "sensitivity_left_lb3b_lb3c": bis_banc_lb3bc_l,
            "sweeps": banc_a1_sweeps,
        },
        "a2a_homolog_per_neuron_norm": {
            "n_matched_neurons": int(n_matched_norm),
            "n_total_neurons": int(n_nodes),
            "fallback_w_syn_mv": float(W_BANC_DENS),
            "mn9_l_w_post_mv": float(w_post_hom[mn9_idx[0]]),
            "mn9_r_w_post_mv": float(w_post_hom[mn9_idx[1]]),
            "sugar_100hz_check": {"mn9_mean_hz": float(cnt_hs.mean()), "mn9_l_hz": float(cnt_hs[:, 0].mean()), "mn9_r_hz": float(cnt_hs[:, 1].mean()), "brain_hz": float(b_hs)},
            "sweeps": banc_a2a_sweeps,
        },
        "a2b_intrinsic_inv_sin_norm": {
            "s_med": s_med_banc,
            "sugar_anchor_calibration": bis_banc_inv,
            "sweeps": banc_a2b_sweeps,
        },
        "elapsed_sec": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# PHASE 3: MaleCNS v1.0 Experiment B2 + A1 + A2a/A2b + Review #2 LB3a Water
# ---------------------------------------------------------------------------
def nt_to_sign(nt: str) -> int:
    s = (nt or "").strip().lower()
    if not s or s in ("none", "nan", "na", "unclear", "unknown"):
        return 0
    if s in INH_NTS:
        return -1
    if s in EXC_NTS:
        return 1
    return 0


def split_tokens(val: str) -> set[str]:
    s = (val or "").strip()
    if not s or s.lower() in ("none", "nan", "na"):
        return set()
    return {p.strip() for p in s.replace(";", ",").split(",") if p.strip()}


def run_phase_malecns(meta: dict, v783_ct_mean_sin: dict[str, float], workers: int) -> dict:
    t0 = time.time()
    print("\n================ PHASE MALECNS: B2 Fixed-Weight, A1 Sugar, A2 Norm, LB3a Water ================", flush=True)
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
        ann_tbl["bodyId"].to_pylist(),
        ann_tbl["type"].to_pylist(),
        ann_tbl["flywireType"].to_pylist(),
        ann_tbl["instance"].to_pylist(),
        ann_tbl["superclass"].to_pylist(),
        ann_tbl["class"].to_pylist(),
        ann_tbl["rootSide"].to_pylist(),
        ann_tbl["somaSide"].to_pylist(),
        ann_tbl["status"].to_pylist(),
    ):
        if b is None:
            continue
        st_s = s_str(st)
        sc_s = s_str(sc)
        cl_s = s_str(cl)
        if st_s != "Traced" or "glia" in sc_s.lower() or "glia" in cl_s.lower():
            continue
        bi = int(b)
        inst_s = s_str(inst)
        side_s = s_str(rs) or s_str(ss) or ("R" if inst_s.endswith("_R") else "L" if inst_s.endswith("_L") else "")
        traced_meta[bi] = {
            "id": bi, "type": s_str(typ), "flywireType": s_str(fw), "instance": inst_s,
            "superclass": sc_s, "side": side_s,
            "cons_nt": cons_nt_map.get(bi, ""), "pred_nt": pred_nt_map.get(bi, ""),
        }
    del ann_tbl, nt_tbl

    all_bids = np.array(sorted(traced_meta.keys()), dtype=np.int64)
    n_nodes = int(len(all_bids))
    bid2idx = {int(b): i for i, b in enumerate(all_bids)}
    node_meta = [traced_meta[int(b)] for b in all_bids]
    mn9_l_idx, mn9_r_idx = bid2idx[10331], bid2idx[16949]
    mn9_idx = np.array([mn9_l_idx, mn9_r_idx], dtype=np.int32)

    raw_pre = w_tbl["body_pre"].to_numpy().astype(np.int64)
    raw_post = w_tbl["body_post"].to_numpy().astype(np.int64)
    raw_w = w_tbl["weight"].to_numpy().astype(np.int32)
    del w_tbl

    def map_vec(arr: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(all_bids, arr)
        pos_c = np.minimum(pos, n_nodes - 1)
        hit = all_bids[pos_c] == arr
        return np.where(hit, pos_c, -1).astype(np.int32)

    p_idx = map_vec(raw_pre)
    q_idx = map_vec(raw_post)
    ok = (p_idx >= 0) & (q_idx >= 0) & (p_idx != q_idx) & (raw_w > 0)
    p_idx, q_idx, raw_w = p_idx[ok], q_idx[ok], raw_w[ok]
    del raw_pre, raw_post

    keys = p_idx.astype(np.int64) * np.int64(n_nodes) + q_idx.astype(np.int64)
    ord_k = np.argsort(keys, kind="mergesort")
    keys_s, w_s = keys[ord_k], raw_w[ord_k]
    del keys, ord_k, p_idx, q_idx, raw_w
    uniq_k, first_pos = np.unique(keys_s, return_index=True)
    w_uniq = np.add.reduceat(w_s.astype(np.int64), first_pos).astype(np.int32) if len(uniq_k) < len(keys_s) else w_s.astype(np.int32)
    del keys_s, w_s, first_pos
    u_pre = (uniq_k // np.int64(n_nodes)).astype(np.int32)
    u_post = (uniq_k % np.int64(n_nodes)).astype(np.int32)
    del uniq_k
    gc.collect()
    print(f"  MaleCNS raw non-self graph: N={n_nodes:,}, E_uniq={len(u_pre):,}", flush=True)

    def build_signed_csr(signs: np.ndarray):
        es = signs[u_pre]
        keep = es != 0
        pre_s = u_pre[keep]
        post_s = u_post[keep]
        w_raw_s = (w_uniq[keep] * es[keep]).astype(np.float64)
        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        np.add.at(indptr[1:], pre_s, 1)
        np.cumsum(indptr, out=indptr)
        s_in = np.zeros(n_nodes, dtype=np.float64)
        np.add.at(s_in, post_s, np.abs(w_raw_s))
        return indptr, post_s, w_raw_s, s_in

    base_signs = np.array([nt_to_sign(m["cons_nt"]) for m in node_meta], dtype=np.int8)
    imp_signs = np.array([nt_to_sign(m["cons_nt"]) if nt_to_sign(m["cons_nt"]) != 0 else nt_to_sign(m["pred_nt"]) for m in node_meta], dtype=np.int8)

    indptr0, post0, w0, sin0 = build_signed_csr(base_signs)
    indptr_imp, post_imp, w_imp, sin_imp = build_signed_csr(imp_signs)
    print(f"  [ASSERT] MN9_L S_in: consensus={sin0[mn9_l_idx]:.1f} (ref 5876.0), predNT={sin_imp[mn9_l_idx]:.1f} (ref 5976.0)", flush=True)
    assert abs(sin0[mn9_l_idx] - 5876.0) < 1e-6
    assert abs(sin_imp[mn9_l_idx] - 5976.0) < 1e-6
    del u_pre, u_post, w_uniq
    gc.collect()

    # Exact SEZ-106 mapping from mn9_malecns_results.json!
    mc_res = json.loads((BASE_DIR / "mn9_malecns_results.json").read_text())
    names = meta["names"]
    y = meta["y"]
    mc_bids_by_name = {r["name"]: [int(b) for b in r["malecns_body_ids"]] for r in mc_res["mapping_audit"]["per_type_sez106"]}
    sez_exc_male = [np.array([bid2idx[b] for b in mc_bids_by_name[nm] if b in bid2idx], dtype=np.int32) for nm in names]
    is_direct_male0 = np.array([int(r["is_direct_malecns"]) for r in meta["rows"]], dtype=bool)
    # True 1-hop direct mask under predNT (excluding TH_VUM and justice which become direct)
    direct_nodes_imp = set()
    for u in range(n_nodes):
        p0, p1 = indptr_imp[u], indptr_imp[u + 1]
        for e in range(p0, p1):
            if post_imp[e] == mn9_l_idx or post_imp[e] == mn9_r_idx:
                direct_nodes_imp.add(u)
                break
    is_direct_male_imp = np.array([any(int(u) in direct_nodes_imp for u in sez_exc_male[i]) for i in range(len(names))], dtype=bool)

    selk_idx = np.array([i for i, m in enumerate(node_meta) if "DNg68" in split_tokens(m["type"]) | split_tokens(m["flywireType"])], dtype=np.int32)
    assert len(selk_idx) == 2

    # Parity check on roundup at 50 Hz (intact 0.268284 mV -> 47.9333 Hz using exact MaleCNS xorshift64* seed 50000 + ru_i * 97)
    zero_slnc = np.zeros(n_nodes, dtype=np.uint8)
    ru_i = names.index("roundup")
    cnt_ru, _, _ = run_lif_exact_cap_slnc(indptr0, post0, w0 * W_MALE_INTACT, W_MALE_INTACT, sez_exc_male[ru_i], 50.0, zero_slnc, mn9_idx, 30, 50000 + ru_i * 97, 15.0, 1)
    ru_ref = float(meta["rows"][ru_i]["malecns_intact_50hz_mean"])
    print(f"  [PARITY] MaleCNS roundup 50 Hz intact = {cnt_ru.mean():.4f} Hz (locked ref = {ru_ref:.4f} Hz)", flush=True)
    assert abs(float(cnt_ru.mean()) - ru_ref) < 2.0, f"MaleCNS parity mismatch: {cnt_ru.mean()} vs {ru_ref}"

    # Identify canonical sugar GRNs (LB3b+LB3c) & water GRNs (LB3a) in MaleCNS
    lb3a_r = np.array([i for i, m in enumerate(node_meta) if m["type"] == "LB3a" and m["side"] == "R"], dtype=np.int32)
    lb3a_l = np.array([i for i, m in enumerate(node_meta) if m["type"] == "LB3a" and m["side"] == "L"], dtype=np.int32)
    lb3bc_r = np.array([i for i, m in enumerate(node_meta) if m["type"] in ("LB3b", "LB3c") and m["side"] == "R"], dtype=np.int32)
    lb3bc_l = np.array([i for i, m in enumerate(node_meta) if m["type"] in ("LB3b", "LB3c") and m["side"] == "L"], dtype=np.int32)
    print(f"  MaleCNS GRNs: LB3a R={len(lb3a_r)} L={len(lb3a_l)} | LB3b+LB3c R={len(lb3bc_r)} L={len(lb3bc_l)}", flush=True)

    # ------------------------------------------------------------------
    # EXPERIMENT B2: MaleCNS FIXED-WEIGHT (0.268284 mV) whole-brain predicted_nt pure ablation
    # ------------------------------------------------------------------
    print("\n  [MaleCNS B2] Fixed-weight (0.268284 mV) whole-brain predicted_nt pure ablation...", flush=True)
    b2_sweeps = {}
    for freq_hz in (50, 200):
        sb = 50000 if freq_hz == 50 else 200000
        m, lr, rr, bh, cc = run_parallel_sweep(
            indptr_imp, post_imp, w_imp * W_MALE_INTACT, W_MALE_INTACT, mn9_idx, sez_exc_male, float(freq_hz), 30, sb, workers, 15.0, use_xorshift=1
        )
        sm = summarize_sweep(names, y, meta["prior_pub"], is_direct_male_imp, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
        sm["auprc_nondirect_fixed88"] = stepwise_auprc(y[~is_direct_male0], m[~is_direct_male0])
        c_sk, _, _ = run_lif_exact_cap_slnc(indptr_imp, post_imp, w_imp * W_MALE_INTACT, W_MALE_INTACT, selk_idx, float(freq_hz), zero_slnc, mn9_idx, 30, 42, 15.0)
        sm["selk_mean_hz"] = float(c_sk.mean())
        sm["selk_left_hz"] = float(c_sk[:, 0].mean())
        b2_sweeps[f"{freq_hz}hz"] = sm
        print(f"    MaleCNS B2 fixed-0.2683 predNT {freq_hz}Hz: all106={sm['auprc_all106']:.4f}, unpub95={sm['auprc_unpub95']:.4f}, nd86={sm['auprc_nondirect']:.4f}, pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, rho={sm['spearman_all106']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # EXPERIMENT A1: MaleCNS Sugar-GRN 58.52 Hz Anchor Recalibration
    # ------------------------------------------------------------------
    print("\n  [MaleCNS A1] Scanning Left-17 & Right-17 LB3b+LB3c sugar GRNs (consensus_nt)...", flush=True)
    # Use intact Left MN9 = 58.52 Hz (consistent with symmetry-screened intact left MN9 protocol!)
    bis_male_l17 = bisect_sugar_weight(
        indptr0, post0, w0, lb3bc_l, mn9_idx, TARGET_SUGAR_HZ,
        readout_mode="left_intact", w_poi_fixed=None,
        grid=[0.08, 0.12, 0.16, 0.20, 0.24, 0.268284, 0.32, 0.40, 0.55],
        workers=workers, use_xorshift=1,
    )
    bis_male_r17 = bisect_sugar_weight(
        indptr0, post0, w0, lb3bc_r, mn9_idx, TARGET_SUGAR_HZ,
        readout_mode="left_intact", w_poi_fixed=None,
        grid=[0.08, 0.12, 0.16, 0.20, 0.24, 0.268284, 0.32, 0.40, 0.55],
        workers=workers, use_xorshift=1,
    )
    # Pick left-17 (ipsilateral to intact left MN9, and matches FlyWire Shiu's left-side 21 sugar GRNs!)
    # Also check predicted_nt left-17 sugar anchor!
    print("  [MaleCNS A1 predNT] Scanning Left-17 LB3b+LB3c on predicted_nt graph...", flush=True)
    bis_male_l17_imp = bisect_sugar_weight(
        indptr_imp, post_imp, w_imp, lb3bc_l, mn9_idx, TARGET_SUGAR_HZ,
        readout_mode="left_intact", w_poi_fixed=None,
        grid=[0.08, 0.12, 0.16, 0.20, 0.24, 0.263795, 0.32, 0.40, 0.55],
        workers=workers, use_xorshift=1,
    )

    w_sugar_male = bis_male_l17["w_star"]
    male_a1_sweeps = {}
    for freq_hz in (50, 200):
        sb = 50000 if freq_hz == 50 else 200000
        m, lr, rr, bh, cc = run_parallel_sweep(
            indptr0, post0, w0 * w_sugar_male, w_sugar_male, mn9_idx, sez_exc_male, float(freq_hz), 30, sb, workers, 15.0, use_xorshift=1
        )
        sm = summarize_sweep(names, y, meta["prior_pub"], is_direct_male0, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
        c_sk, _, _ = run_lif_exact_cap_slnc(indptr0, post0, w0 * w_sugar_male, w_sugar_male, selk_idx, float(freq_hz), zero_slnc, mn9_idx, 30, 42, 15.0, 1)
        sm["selk_mean_hz"] = float(c_sk.mean())
        male_a1_sweeps[f"{freq_hz}hz"] = sm
        print(f"    MaleCNS A1 sugar-anchor cons (W={w_sugar_male:.4f}) {freq_hz}Hz: all106={sm['auprc_all106']:.4f}, unpub95={sm['auprc_unpub95']:.4f}, nd={sm['auprc_nondirect']:.4f}, pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, rho={sm['spearman_all106']:.4f}, brain={sm['mean_brain_hz']:.4f}", flush=True)

    w_sugar_male_imp = bis_male_l17_imp["w_star"]
    male_a1_imp_sweeps = {}
    for freq_hz in (50, 200):
        sb = 50000 if freq_hz == 50 else 200000
        m, lr, rr, bh, cc = run_parallel_sweep(
            indptr_imp, post_imp, w_imp * w_sugar_male_imp, w_sugar_male_imp, mn9_idx, sez_exc_male, float(freq_hz), 30, sb, workers, 15.0, use_xorshift=1
        )
        sm = summarize_sweep(names, y, meta["prior_pub"], is_direct_male_imp, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
        c_sk, _, _ = run_lif_exact_cap_slnc(indptr_imp, post_imp, w_imp * w_sugar_male_imp, w_sugar_male_imp, selk_idx, float(freq_hz), zero_slnc, mn9_idx, 30, 42, 15.0, 1)
        sm["selk_mean_hz"] = float(c_sk.mean())
        male_a1_imp_sweeps[f"{freq_hz}hz"] = sm
        print(f"    MaleCNS A1 sugar-anchor predNT (W={w_sugar_male_imp:.4f}) {freq_hz}Hz: all106={sm['auprc_all106']:.4f}, unpub95={sm['auprc_unpub95']:.4f}, nd={sm['auprc_nondirect']:.4f}, pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, rho={sm['spearman_all106']:.4f}", flush=True)

    # ------------------------------------------------------------------
    # EXPERIMENT A2a: MaleCNS Homolog-Matched Per-Neuron Input Normalization (cons & predNT)
    # ------------------------------------------------------------------
    print("\n  [MaleCNS A2a] Homolog-matched per-neuron input normalization (cons & predNT)...", flush=True)
    def build_male_hom_weights(sin_arr: np.ndarray, fallback_w: float, mn9_l_w: float) -> tuple[np.ndarray, int]:
        wp = np.full(n_nodes, fallback_w, dtype=np.float64)
        n_m = 0
        for j, m in enumerate(node_meta):
            if sin_arr[j] <= 0:
                continue
            toks = split_tokens(m["flywireType"]) | split_tokens(m["type"])
            refs = [v783_ct_mean_sin[t] for t in toks if t in v783_ct_mean_sin and v783_ct_mean_sin[t] > 0]
            if refs:
                ratio = float(np.mean(refs)) / float(sin_arr[j])
                wp[j] = W_FROZEN * min(max(ratio, 0.1), 10.0)
                n_m += 1
        # Keep truncated right MN9 (521 syn) at intact left MN9 scale (do not artificially 10x boost stub)
        wp[mn9_l_idx] = mn9_l_w
        wp[mn9_r_idx] = mn9_l_w
        return wp, n_m

    wp_m0, nm_m0 = build_male_hom_weights(sin0, W_MALE_DENS, W_MALE_INTACT)
    w_edge_m0_hom = w0 * wp_m0[post0]
    cnt_mh0, b_mh0, _ = run_lif_exact_cap_slnc(indptr0, post0, w_edge_m0_hom, W_FROZEN, lb3bc_l, 100.0, zero_slnc, mn9_idx, 30, 42, 15.0, 1)
    print(f"    MaleCNS cons matched {nm_m0:,}/{n_nodes:,}; 100 Hz sugar L17 -> L={cnt_mh0[:,0].mean():.2f}, R={cnt_mh0[:,1].mean():.2f}, brain={b_mh0:.4f}", flush=True)
    male_a2a_cons = {}
    for freq_hz in (50, 200):
        sb = 50000 if freq_hz == 50 else 200000
        m, lr, rr, bh, cc = run_parallel_sweep(indptr0, post0, w_edge_m0_hom, W_FROZEN, mn9_idx, sez_exc_male, float(freq_hz), 30, sb, workers, 15.0, use_xorshift=1)
        sm = summarize_sweep(names, y, meta["prior_pub"], is_direct_male0, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
        c_sk, _, _ = run_lif_exact_cap_slnc(indptr0, post0, w_edge_m0_hom, W_FROZEN, selk_idx, float(freq_hz), zero_slnc, mn9_idx, 30, 42, 15.0, 1)
        sm["selk_mean_hz"] = float(c_sk.mean())
        male_a2a_cons[f"{freq_hz}hz"] = sm
        print(f"    MaleCNS A2a homolog-norm cons {freq_hz}Hz: all106={sm['auprc_all106']:.4f}, unpub95={sm['auprc_unpub95']:.4f}, nd={sm['auprc_nondirect']:.4f}, pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, rho={sm['spearman_all106']:.4f}", flush=True)

    wp_mi, nm_mi = build_male_hom_weights(sin_imp, W_MALE_JOINT, W_MALE_PREDNT)
    w_edge_mi_hom = w_imp * wp_mi[post_imp]
    cnt_mhi, b_mhi, _ = run_lif_exact_cap_slnc(indptr_imp, post_imp, w_edge_mi_hom, W_FROZEN, lb3bc_l, 100.0, zero_slnc, mn9_idx, 30, 42, 15.0, 1)
    print(f"    MaleCNS predNT matched {nm_mi:,}/{n_nodes:,}; 100 Hz sugar L17 -> L={cnt_mhi[:,0].mean():.2f}, R={cnt_mhi[:,1].mean():.2f}, brain={b_mhi:.4f}", flush=True)
    male_a2a_imp = {}
    for freq_hz in (50, 200):
        sb = 50000 if freq_hz == 50 else 200000
        m, lr, rr, bh, cc = run_parallel_sweep(indptr_imp, post_imp, w_edge_mi_hom, W_FROZEN, mn9_idx, sez_exc_male, float(freq_hz), 30, sb, workers, 15.0, use_xorshift=1)
        sm = summarize_sweep(names, y, meta["prior_pub"], is_direct_male_imp, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
        c_sk, _, _ = run_lif_exact_cap_slnc(indptr_imp, post_imp, w_edge_mi_hom, W_FROZEN, selk_idx, float(freq_hz), zero_slnc, mn9_idx, 30, 42, 15.0, 1)
        sm["selk_mean_hz"] = float(c_sk.mean())
        male_a2a_imp[f"{freq_hz}hz"] = sm
        print(f"    MaleCNS A2a homolog-norm predNT {freq_hz}Hz: all106={sm['auprc_all106']:.4f}, unpub95={sm['auprc_unpub95']:.4f}, nd={sm['auprc_nondirect']:.4f}, pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, rho={sm['spearman_all106']:.4f}", flush=True)
    del w_edge_m0_hom, w_edge_mi_hom

    # ------------------------------------------------------------------
    # EXPERIMENT A2b: MaleCNS Intrinsic Reference-Free Median-Floored Per-Neuron Norm (cons)
    # ------------------------------------------------------------------
    print("\n  [MaleCNS A2b] Intrinsic median-floored per-neuron norm W/max(S_in, S_med) (sugar-anchored)...", flush=True)
    s_med_m0 = float(np.median(sin0[sin0 > 0]))
    inv_floor_m0 = 1.0 / np.maximum(sin0, s_med_m0)
    # Do not amplify truncated MN9_R by 1/521; anchor MN9_R denominator to MN9_L's 5876
    inv_floor_m0[mn9_r_idx] = 1.0 / sin0[mn9_l_idx]
    w_unit_inv0 = w0 * inv_floor_m0[post0]
    bis_male_inv = bisect_sugar_weight(
        indptr0, post0, w_unit_inv0, lb3bc_l, mn9_idx, TARGET_SUGAR_HZ,
        readout_mode="left_intact", w_poi_fixed=W_FROZEN,
        grid=[100.0, 250.0, 450.0, 700.0, 1000.0, 1400.0, 2000.0, 2800.0],
        workers=workers, use_xorshift=1,
    )
    w_inv_m = bis_male_inv["w_star"]
    male_a2b_cons = {}
    for freq_hz in (50, 200):
        sb = 50000 if freq_hz == 50 else 200000
        m, lr, rr, bh, cc = run_parallel_sweep(indptr0, post0, w_unit_inv0 * w_inv_m, W_FROZEN, mn9_idx, sez_exc_male, float(freq_hz), 30, sb, workers, 15.0, use_xorshift=1)
        sm = summarize_sweep(names, y, meta["prior_pub"], is_direct_male0, meta["v783_50"], meta["v783_200"], meta["v783_exc_any"], m, lr, rr, bh, cc, freq_hz)
        c_sk, _, _ = run_lif_exact_cap_slnc(indptr0, post0, w_unit_inv0 * w_inv_m, W_FROZEN, selk_idx, float(freq_hz), zero_slnc, mn9_idx, 30, 42, 15.0, 1)
        sm["selk_mean_hz"] = float(c_sk.mean())
        male_a2b_cons[f"{freq_hz}hz"] = sm
        print(f"    MaleCNS A2b intrinsic norm cons {freq_hz}Hz (S_med={s_med_m0:.1f}, W*={w_inv_m:.1f}): all106={sm['auprc_all106']:.4f}, unpub95={sm['auprc_unpub95']:.4f}, nd={sm['auprc_nondirect']:.4f}, pos={sm['active_pos_gt0']}/14, neg={sm['active_neg_gt0']}/92, rho={sm['spearman_all106']:.4f}, brain={sm['mean_brain_hz']:.4f}", flush=True)
    del w_unit_inv0

    # ------------------------------------------------------------------
    # BONUS REVIEW #2: MaleCNS LB3a Water Stimulation & 11-Type Acute Silencing
    # ------------------------------------------------------------------
    print("\n  [MaleCNS Water LB3a] Auditing 160 Hz right-8 & left-9 LB3a baselines + 11-type silencing...", flush=True)
    tr_res = json.loads((BASE_DIR / "mn9_transfer_results.json").read_text())
    w11_rows = tr_res["task2_water_silencing_v783"]["per_type_records"]

    # Also read v783 cell_type lookup for any water11 type whose sez_key is outside sez106 (e.g. Zorro)
    v783_fid_to_ct: dict[int, str] = {}
    with (CACHE_DIR / "data/Supplemental_file1_neuron_annotations.tsv").open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rs = row.get("root_id", "")
            if rs and rs.isdigit():
                v783_fid_to_ct[int(rs)] = (row.get("cell_type") or "").strip() or (row.get("hemibrain_type") or "").strip()

    # Map all 11 water types in MaleCNS (using SEZ106 mapping or exact flywireType/type)
    by_tok: dict[str, list[int]] = defaultdict(list)
    for i, m in enumerate(node_meta):
        for t in split_tokens(m["flywireType"]) | split_tokens(m["type"]):
            by_tok[t].append(i)
    w11_idx_male = {}
    for wr in w11_rows:
        nm = wr["name"]
        sk = wr.get("sez_key", nm.lower())
        if sk in mc_bids_by_name and len(mc_bids_by_name[sk]) > 0:
            w11_idx_male[nm] = np.array([bid2idx[b] for b in mc_bids_by_name[sk] if b in bid2idx], dtype=np.int32)
        else:
            cts = {v783_fid_to_ct.get(int(fid), "") for fid in wr.get("fids", [])} - {""}
            hits = sorted({u for ct in cts for u in by_tok.get(ct, [])})
            w11_idx_male[nm] = np.array(hits, dtype=np.int32)

    water_audit = {
        "lb3a_right_body_ids": [int(all_bids[i]) for i in lb3a_r],
        "lb3a_left_body_ids": [int(all_bids[i]) for i in lb3a_l],
        "water11_mapped_counts": {nm: int(len(ix)) for nm, ix in w11_idx_male.items()},
        "baselines_160hz": {},
        "silencing_11type": {},
    }
    print("    MaleCNS water11 mapped counts:", water_audit["water11_mapped_counts"], flush=True)
    regimes_water = [
        ("intact_0.2683", indptr0, post0, w0 * W_MALE_INTACT, W_MALE_INTACT),
        ("density_0.1594", indptr0, post0, w0 * W_MALE_DENS, W_MALE_DENS),
        ("predNT_fixed_0.2683", indptr_imp, post_imp, w_imp * W_MALE_INTACT, W_MALE_INTACT),
        ("joint_0.1567", indptr_imp, post_imp, w_imp * W_MALE_JOINT, W_MALE_JOINT),
        ("sugar_anchored_cons", indptr0, post0, w0 * w_sugar_male, w_sugar_male),
    ]
    for rname, ip, po, ww, wp_b in regimes_water:
        for side_lbl, g_idx in [("right_8", lb3a_r), ("left_9", lb3a_l)]:
            cnt, b_hz, cap = run_lif_exact_cap_slnc(ip, po, ww, wp_b, g_idx, 160.0, zero_slnc, mn9_idx, 30, 42, 15.0)
            l_hz, r_hz, m_hz = float(cnt[:, 0].mean()), float(cnt[:, 1].mean()), float(cnt.mean())
            key = f"{rname}__{side_lbl}"
            water_audit["baselines_160hz"][key] = {
                "mn9_l_hz": l_hz, "mn9_r_hz": r_hz, "mn9_mean_hz": m_hz, "brain_hz": float(b_hz), "capped": int(cap)
            }
            print(f"    Water 160Hz {key}: L={l_hz:.3f} Hz, R={r_hz:.3f} Hz, mean={m_hz:.3f} Hz, brain={b_hz:.4f}", flush=True)
            # If baseline >= 0.5 Hz on intact left MN9 (and not capped), run full 11-type silencing!
            if l_hz >= 0.5 and cap == 0 and rname in ("intact_0.2683", "predNT_fixed_0.2683", "joint_0.1567", "sugar_anchored_cons"):
                sl_rows = []
                for wr in w11_rows:
                    nm = wr["name"]
                    s_ix = w11_idx_male[nm]
                    smask = np.zeros(n_nodes, dtype=np.uint8)
                    if len(s_ix) > 0:
                        smask[s_ix] = 1
                    c_s, b_s, cap_s = run_lif_exact_cap_slnc(ip, po, ww, wp_b, g_idx, 160.0, smask, mn9_idx, 30, 42, 15.0)
                    sl_l, sl_r = float(c_s[:, 0].mean()), float(c_s[:, 1].mean())
                    rel_l = (sl_l - l_hz) / l_hz if l_hz > 0 else 0.0
                    sl_rows.append({
                        "name": nm,
                        "n_cells_malecns": int(len(s_ix)),
                        "impaired": bool(wr["sig_impaired"]),
                        "emp_did": float(wr["delta_did"]),
                        "mn9_l_hz": sl_l, "mn9_r_hz": sl_r,
                        "abs_diff_l_hz": sl_l - l_hz, "rel_change_l": rel_l,
                    })
                imp_flags = np.array([1 if x["impaired"] else 0 for x in sl_rows])
                rel_vals = np.array([x["rel_change_l"] for x in sl_rows])
                did_vals = np.array([x["emp_did"] for x in sl_rows])
                # Lower rel_change (more suppression) predicts impaired=1 -> score = -rel_vals
                auroc_11 = compute_auroc(imp_flags, -rel_vals)
                u_11 = stats.mannwhitneyu(-rel_vals[imp_flags == 1], -rel_vals[imp_flags == 0], alternative="two-sided")
                rho_11 = stats.spearmanr(rel_vals, did_vals)
                no_usn = [i for i, x in enumerate(sl_rows) if x["name"].lower() != "usnea"]
                auroc_10 = compute_auroc(imp_flags[no_usn], -rel_vals[no_usn])
                rho_10 = stats.spearmanr(rel_vals[no_usn], did_vals[no_usn])
                water_audit["silencing_11type"][key] = {
                    "baseline_l_hz": l_hz, "baseline_r_hz": r_hz,
                    "auroc_11": float(auroc_11), "mannwhitney_u_11": float(u_11.statistic), "p_11": float(u_11.pvalue),
                    "spearman_rho_11": float(rho_11.statistic), "spearman_p_11": float(rho_11.pvalue),
                    "auroc_10_excl_usnea": float(auroc_10), "spearman_rho_10": float(rho_10.statistic), "spearman_p_10": float(rho_10.pvalue),
                    "per_type": sl_rows,
                }
                print(f"      -> 11-type silencing ({key}): AUROC11={auroc_11:.4f} (p={u_11.pvalue:.4f}), rho11={rho_11.statistic:+.4f}, AUROC10={auroc_10:.4f}, rho10={rho_10.statistic:+.4f}", flush=True)

    return {
        "grn_counts": {"lb3a_r": int(len(lb3a_r)), "lb3a_l": int(len(lb3a_l)), "lb3b_lb3c_r": int(len(lb3bc_r)), "lb3b_lb3c_l": int(len(lb3bc_l))},
        "experiment_b2_fixed_02683_prednt": {
            "w_syn_mv": float(W_MALE_INTACT),
            "sweeps": b2_sweeps,
        },
        "a1_sugar_anchor_consensus": {
            "left17_bisection": bis_male_l17,
            "right17_bisection": bis_male_r17,
            "sweeps_at_left17_wstar": male_a1_sweeps,
        },
        "a1_sugar_anchor_prednt": {
            "left17_bisection": bis_male_l17_imp,
            "sweeps_at_left17_wstar": male_a1_imp_sweeps,
        },
        "a2a_homolog_per_neuron_norm_cons": {
            "n_matched_neurons": int(nm_m0), "n_total_neurons": int(n_nodes),
            "fallback_w_syn_mv": float(W_MALE_DENS),
            "sugar_100hz_l17_check": {"mn9_l_hz": float(cnt_mh0[:, 0].mean()), "mn9_r_hz": float(cnt_mh0[:, 1].mean()), "brain_hz": float(b_mh0)},
            "sweeps": male_a2a_cons,
        },
        "a2a_homolog_per_neuron_norm_prednt": {
            "n_matched_neurons": int(nm_mi), "n_total_neurons": int(n_nodes),
            "fallback_w_syn_mv": float(W_MALE_JOINT),
            "sugar_100hz_l17_check": {"mn9_l_hz": float(cnt_mhi[:, 0].mean()), "mn9_r_hz": float(cnt_mhi[:, 1].mean()), "brain_hz": float(b_mhi)},
            "sweeps": male_a2a_imp,
        },
        "a2b_intrinsic_inv_sin_norm_cons": {
            "sugar_anchor_calibration": bis_male_inv,
            "sweeps": male_a2b_cons,
        },
        "water_lb3a_audit": water_audit,
        "elapsed_sec": float(time.time() - t0),
    }


def main() -> None:
    try:
        os.nice(15)
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6, help="Worker processes (default 6 out of 32 threads for low CPU/RAM load)")
    ap.add_argument("--phase", choices=["all", "b1", "banc", "malecns"], default="all")
    args = ap.parse_args()

    out = json.loads(OUT_JSON.read_text()) if OUT_JSON.is_file() else {"meta": {}}
    out["meta"] = {"workers": args.workers, "cache_dir_env": bool(os.environ.get("PROJC_CACHE_DIR")), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta = load_common_meta()

    v783_ct_mean_sin = None
    s_in_v783_by_fid = None

    if args.phase in ("all", "b1"):
        res_b1, v783_ct_mean_sin, s_in_v783_by_fid = run_phase_b1(meta, args.workers)
        out["phase_b1_v783"] = res_b1
        save_checkpoint(out)
        print(f"[CHECKPOINT SAVED] phase_b1_v783 -> {OUT_JSON}", flush=True)

    if args.phase in ("all", "banc"):
        if s_in_v783_by_fid is None:
            _, _, v783_ct_mean_sin, s_in_v783_by_fid, _, _ = load_v783_reference()
        out["phase_banc"] = run_phase_banc(meta, s_in_v783_by_fid, args.workers)
        save_checkpoint(out)
        print(f"[CHECKPOINT SAVED] phase_banc -> {OUT_JSON}", flush=True)

    if args.phase in ("all", "malecns"):
        if v783_ct_mean_sin is None:
            _, _, v783_ct_mean_sin, _, _, _ = load_v783_reference()
        out["phase_malecns"] = run_phase_malecns(meta, v783_ct_mean_sin, args.workers)
        save_checkpoint(out)
        print(f"[CHECKPOINT SAVED] phase_malecns -> {OUT_JSON}", flush=True)

    out["meta"]["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    save_checkpoint(out)
    print(f"\nALL REQUESTED PHASES DONE -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
