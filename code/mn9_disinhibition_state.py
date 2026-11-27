#!/usr/bin/env python3
"""Cross-brain disinhibitory pathway & background-state discriminative study (mn9-disinhibition-state-01).

Implements MN9_DISINHIBITION_PROTOCOL.md (commit deea00bbc222a4e47360dbc9677fbf0726491c0c):
Module 1 (Anatomical disinhibition audit on FlyWire v783 & BANC v888):
  - Quantify 1-hop direct inhibition, 2-hop disinhibition (X - I_pre - MN9),
    antagonistic 2-hop inhibition of excitatory premotors (X - E_pre + MN9),
    and 3-hop disinhibitory excitation (X - I_1 - E_pre + MN9) for phantom (CB0062),
    tentacular (CB0821), usnea (CB0008), marge (CB0812), and all 44 purely-inhibitory
    negative SEZ control types, with cross-brain homologous intermediate overlap.
  - Quantify 3-hop (SELK + I_1 - I_2 - MN9) and 4-hop (SELK + I_1 - I_2 - E_pre + MN9)
    disinhibitory paths plus antagonistic shortcuts (SELK + I_1 - MN9, SELK + I_1 - E_pre + MN9)
    for SELK (DNg68) in v783 and full-CNS BANC v888 (splitting brain-only vs via-VNC).
Module 2 (Background-state discriminative tests on FlyWire v783 & BANC v888 confirmation):
  - Exact common-random-number paired trials Delta_r_MN9 = r_on - r_bg across:
    * State Family 1 (Global uniform DC bias): 5.50, 7.10, 7.30, 7.40 mV (all 106 SEZ + SELK)
    * State Family 2 (Class-uniform E/I tonic bias): (5.50,7.50), (7.30,8.00), (7.45,8.00), (7.55,8.00) mV
    * State Family 3a (Endogenous gustatory background): sugar 40/50 Hz, water 160 Hz (all 106 SEZ + SELK)
    * State Family 3b (Non-predictive local Oracle realizability bound on top Module-1 intermediates)
    * Cross-individual BANC v888 (W_syn_anat = 0.690360 mV) confirmation on focal + reference types.
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.feather as pf
from numba import njit
from scipy import stats

import mn9_banc_benchmark as mbb
import mn9_transfer_benchmark as mtb
import shiu_mn9_benchmark as smb

PROTOCOL_COMMIT = "deea00bbc222a4e47360dbc9677fbf0726491c0c"
_WORKER_CACHE: dict[str, dict] = {}


def _save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, allow_nan=False)
    tmp.replace(path)


@njit
def run_lif_dc_paired(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    w_syn: float,
    v_bias: np.ndarray,
    init_u_max: float,
    exc_idx: np.ndarray,
    exc_rate_hz: float,
    bg_exc_idx: np.ndarray,
    bg_exc_rate_hz: float,
    watch_idx: np.ndarray,
    n_trials: int,
    seed_bg: int,
    seed_opto: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Brian2 LIF with per-node DC bias and exact common-random-number trial pairing.

    At the start of every trial tr:
    - np.random.seed(seed_bg + tr * 1009) drives initial u(0) and bg_exc_idx Poisson events.
    - An independent xorshift64* state initialized from (seed_opto + tr * 65537) drives
      exc_idx optogenetic Poisson events, so background RNG draws are 100% identical
      between baseline (exc_rate_hz == 0) and stimulated (exc_rate_hz > 0) trials.
    Bit-identical to mtb.run_lif_brian2_wsyn when v_bias == 0 and bg_exc_rate_hz == 0 (for matching stream).
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
    one_minus_av = 1.0 - a_v
    v_th = 7.0
    w_poi = w_syn * 250.0
    p_poi = exc_rate_hz * 1e-3 * dt
    p_bg = bg_exc_rate_hz * 1e-3 * dt

    n_nodes = len(indptr) - 1
    n_watch = len(watch_idx)
    watch_counts = np.zeros((n_trials, n_watch), dtype=np.int32)
    total_spikes = np.zeros(n_trials, dtype=np.int64)

    is_exc = np.zeros(n_nodes, dtype=np.uint8)
    for k in range(len(exc_idx)):
        is_exc[exc_idx[k]] = 1
    for k in range(len(bg_exc_idx)):
        is_exc[bg_exc_idx[k]] = 1

    n_bias = 0
    for i in range(n_nodes):
        if v_bias[i] != 0.0:
            n_bias += 1
    use_dense = n_bias > (n_nodes // 16)

    u = np.zeros(n_nodes, dtype=np.float64)
    g = np.zeros(n_nodes, dtype=np.float64)
    last_spk = np.full(n_nodes, -100000, dtype=np.int32)
    spiked_now = np.empty(n_nodes, dtype=np.int32)
    ring_len = dly_steps + 1
    ring_spk = np.empty((ring_len, n_nodes), dtype=np.int32)
    ring_cnt = np.zeros(ring_len, dtype=np.int32)

    in_active = np.zeros(n_nodes, dtype=np.uint8)
    active_list = np.empty(n_nodes, dtype=np.int32)
    next_active = np.empty(n_nodes, dtype=np.int32)
    bias_nodes = np.empty(n_bias, dtype=np.int32)
    if not use_dense:
        c = 0
        for i in range(n_nodes):
            if v_bias[i] != 0.0:
                bias_nodes[c] = i
                c += 1

    inv_2_53 = 1.0 / 9007199254740992.0  # 2^-53

    for tr in range(n_trials):
        np.random.seed(seed_bg + tr * 1009)
        # Independent 64-bit xorshift64* PRNG for opto Poisson stream
        rng_state = np.uint64(seed_opto + tr * 65537 + 0x9E3779B97F4A7C15)
        if rng_state == np.uint64(0):
            rng_state = np.uint64(0x9E3779B97F4A7C15)

        for slot in range(ring_len):
            ring_cnt[slot] = 0
        n_act = 0
        if use_dense:
            for i in range(n_nodes):
                g[i] = 0.0
                last_spk[i] = -100000
                if init_u_max > 0.0 and v_bias[i] > 0.0:
                    u[i] = np.random.random() * init_u_max
                else:
                    u[i] = 0.0
        else:
            for i in range(n_nodes):
                u[i] = 0.0
                g[i] = 0.0
                last_spk[i] = -100000
                in_active[i] = 0
            for k in range(len(exc_idx)):
                ii = exc_idx[k]
                if in_active[ii] == 0:
                    in_active[ii] = 1
                    active_list[n_act] = ii
                    n_act += 1
            for k in range(len(bg_exc_idx)):
                ii = bg_exc_idx[k]
                if in_active[ii] == 0:
                    in_active[ii] = 1
                    active_list[n_act] = ii
                    n_act += 1
            for k in range(n_bias):
                ii = bias_nodes[k]
                if init_u_max > 0.0 and v_bias[ii] > 0.0:
                    u[ii] = np.random.random() * init_u_max
                if in_active[ii] == 0:
                    in_active[ii] = 1
                    active_list[n_act] = ii
                    n_act += 1

        tr_spk = 0
        for step in range(n_steps):
            n_spk_now = 0
            if use_dense:
                for i in range(n_nodes):
                    if is_exc[i] != 0 or (step - last_spk[i]) >= rfc_steps:
                        ui = u[i]
                        gi = g[i]
                        ui_new = a_v * ui + one_minus_av * v_bias[i] + b_vg * gi
                        gi_new = a_g * gi
                        u[i] = ui_new
                        g[i] = gi_new
                        if ui_new > v_th:
                            spiked_now[n_spk_now] = i
                            n_spk_now += 1
            else:
                n_next = 0
                for a_i in range(n_act):
                    i = active_list[a_i]
                    if is_exc[i] != 0 or (step - last_spk[i]) >= rfc_steps:
                        ui = u[i]
                        gi = g[i]
                        vb = v_bias[i]
                        ui_new = a_v * ui + one_minus_av * vb + b_vg * gi
                        gi_new = a_g * gi
                        u[i] = ui_new
                        g[i] = gi_new
                        if ui_new > v_th:
                            spiked_now[n_spk_now] = i
                            n_spk_now += 1
                            next_active[n_next] = i
                            n_next += 1
                        elif is_exc[i] != 0 or vb != 0.0 or abs(ui_new) > 1e-6 or abs(gi_new) > 1e-6:
                            next_active[n_next] = i
                            n_next += 1
                        else:
                            u[i] = 0.0
                            g[i] = 0.0
                            in_active[i] = 0
                    else:
                        next_active[n_next] = i
                        n_next += 1

            tr_spk += n_spk_now
            fut_slot = (step + dly_steps) % ring_len
            for s_i in range(n_spk_now):
                i = spiked_now[s_i]
                c_f = ring_cnt[fut_slot]
                ring_spk[fut_slot, c_f] = i
                ring_cnt[fut_slot] = c_f + 1
                for w_i in range(n_watch):
                    if watch_idx[w_i] == i:
                        watch_counts[tr, w_i] += 1

            if p_bg > 0.0:
                for k in range(len(bg_exc_idx)):
                    if np.random.random() < p_bg:
                        u[bg_exc_idx[k]] += w_poi
            if p_poi > 0.0:
                for k in range(len(exc_idx)):
                    rng_state ^= rng_state >> np.uint64(12)
                    rng_state ^= rng_state << np.uint64(25)
                    rng_state ^= rng_state >> np.uint64(27)
                    val = (rng_state * np.uint64(2685821657736338717)) >> np.uint64(11)
                    if (float(val) * inv_2_53) < p_poi:
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
                        g[dst] += weights[e]
                        if (not use_dense) and in_active[dst] == 0:
                            in_active[dst] = 1
                            next_active[n_next] = dst
                            n_next += 1
            ring_cnt[cur_slot] = 0

            for s_i in range(n_spk_now):
                i = spiked_now[s_i]
                u[i] = 0.0
                g[i] = 0.0
                last_spk[i] = step

            if not use_dense:
                for a_i in range(n_next):
                    active_list[a_i] = next_active[a_i]
                n_act = n_next

        total_spikes[tr] = tr_spk

    return watch_counts, total_spikes


def _get_graph(npz_path: str, w_syn: float) -> tuple[dict, np.ndarray]:
    if npz_path not in _WORKER_CACHE:
        _WORKER_CACHE.clear()
        d = np.load(npz_path)
        _WORKER_CACHE[npz_path] = {
            "n_nodes": int(d["n_nodes"]),
            "indptr": d["indptr"],
            "post_s": d["post_s"],
            "w_raw_s": d["w_raw_s"],
            "node_class": d["node_class"],
            "ws": {},
        }
    g = _WORKER_CACHE[npz_path]
    wk = round(float(w_syn), 10)
    if wk not in g["ws"]:
        g["ws"][wk] = (g["w_raw_s"] * np.float32(w_syn)).astype(np.float64)
    return g, g["ws"][wk]


def _make_vbias(g: dict, mode: str, params: dict) -> tuple[np.ndarray, float]:
    n = g["n_nodes"]
    vb = np.zeros(n, dtype=np.float64)
    init_u_max = 0.0
    if mode == "global":
        v = float(params["v"])
        vb[:] = v
        init_u_max = 7.0 if v >= 7.0 else 0.0
    elif mode == "ei":
        ve = float(params["ve"])
        vi = float(params["vi"])
        cls = g["node_class"]
        vb[:] = ve
        vb[cls == 1] = vi
        init_u_max = 7.0 if (ve >= 7.0 or vi >= 7.0) else 0.0
    elif mode == "local_oracle":
        for idx_i, val in params["node_biases"]:
            vb[int(idx_i)] = float(val)
        init_u_max = 7.0
    return vb, init_u_max


def _paired_stats(on_counts: np.ndarray, bg_counts: np.ndarray) -> dict:
    """Compute paired trial-level Delta_r_MN9 stats (bilateral mean + L/R)."""
    on_m = on_counts[:, :2].mean(axis=1).astype(np.float64)
    bg_m = bg_counts[:, :2].mean(axis=1).astype(np.float64)
    diff = on_m - bg_m
    mean_d = float(diff.mean())
    se_d = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
    if np.all(diff == diff[0]):
        t_stat = 0.0 if mean_d == 0.0 else (1e9 if mean_d > 0 else -1e9)
        p_val = 1.0 if mean_d == 0.0 else 0.0
    else:
        res_t = stats.ttest_1samp(diff, 0.0)
        t_stat = float(res_t.statistic)
        p_val = float(res_t.pvalue)
    return {
        "on_mean_hz": float(on_m.mean()),
        "bg_mean_hz": float(bg_m.mean()),
        "on_l_hz": float(on_counts[:, 0].mean()),
        "on_r_hz": float(on_counts[:, 1].mean()),
        "delta_mean_hz": mean_d,
        "delta_se_hz": se_d,
        "delta_l_hz": float((on_counts[:, 0] - bg_counts[:, 0]).mean()),
        "delta_r_hz": float((on_counts[:, 1] - bg_counts[:, 1]).mean()),
        "paired_t": t_stat,
        "paired_p": p_val,
        "unlocked_ge1hz_p01": bool(mean_d >= 1.0 and p_val < 0.01),
    }


def _condition_slice_worker(job: dict) -> dict:
    """Run a slice of types under one background condition (paired with exact same bg_counts)."""
    ckpt = Path(job["ckpt_path"])
    if ckpt.exists():
        return json.load(open(ckpt))
    t0 = time.time()
    g, w_s = _get_graph(job["npz_path"], float(job["w_syn"]))
    vb, init_u_max = _make_vbias(g, job["bg_mode"], job["bg_params"])
    bg_exc_idx = np.array(job.get("bg_exc_idx", []), dtype=np.int32)
    bg_exc_rate = float(job.get("bg_exc_rate_hz", 0.0))
    watch_idx = np.array(job["watch_idx"], dtype=np.int32)
    n_trials = int(job["n_trials"])
    seed_bg = int(job["seed_bg"])
    bg_counts = np.array(job["bg_counts"], dtype=np.int32)

    results = {}
    for item in job["items"]:
        name = item["name"]
        exc_idx = np.array(item["exc_idx"], dtype=np.int32)
        rate_hz = float(item["rate_hz"])
        seed_opto = int(item["seed_opto"])
        if len(exc_idx) == 0:
            on_counts = bg_counts.copy()
        else:
            on_counts, _ = run_lif_dc_paired(
                g["indptr"], g["post_s"], w_s, float(job["w_syn"]), vb, init_u_max,
                exc_idx, rate_hz, bg_exc_idx, bg_exc_rate, watch_idx, n_trials, seed_bg, seed_opto,
            )
        results[name] = _paired_stats(on_counts, bg_counts)

    out = {"cond": job["cond"], "slice_id": job["slice_id"], "results": results, "elapsed_sec": float(time.time() - t0)}
    _save_json_atomic(ckpt, out)
    return out


def _single_bg_worker(job: dict) -> dict:
    """Compute baseline bg_counts for one condition."""
    ckpt = Path(job["ckpt_path"])
    if ckpt.exists():
        return json.load(open(ckpt))
    t0 = time.time()
    g, w_s = _get_graph(job["npz_path"], float(job["w_syn"]))
    vb, init_u_max = _make_vbias(g, job["bg_mode"], job["bg_params"])
    bg_exc_idx = np.array(job.get("bg_exc_idx", []), dtype=np.int32)
    bg_exc_rate = float(job.get("bg_exc_rate_hz", 0.0))
    watch_idx = np.array(job["watch_idx"], dtype=np.int32)
    empty = np.empty(0, dtype=np.int32)
    bg_counts, tot = run_lif_dc_paired(
        g["indptr"], g["post_s"], w_s, float(job["w_syn"]), vb, init_u_max,
        empty, 0.0, bg_exc_idx, bg_exc_rate, watch_idx, int(job["n_trials"]), int(job["seed_bg"]), 12345,
    )
    out = {
        "cond": job["cond"],
        "bg_counts": bg_counts.tolist(),
        "bg_mn9_l_hz": float(bg_counts[:, 0].mean()),
        "bg_mn9_r_hz": float(bg_counts[:, 1].mean()),
        "bg_mn9_mean_hz": float(bg_counts[:, :2].mean()),
        "brain_mean_hz": float(tot.mean() / g["n_nodes"]),
        "elapsed_sec": float(time.time() - t0),
    }
    _save_json_atomic(ckpt, out)
    return out


def _oracle_job_worker(job: dict) -> dict:
    """Run one local Oracle realizability check (baseline + stimulated paired trials)."""
    ckpt = Path(job["ckpt_path"])
    if ckpt.exists():
        return json.load(open(ckpt))
    t0 = time.time()
    g, w_s = _get_graph(job["npz_path"], float(job["w_syn"]))
    vb, init_u_max = _make_vbias(g, "local_oracle", {"node_biases": job["node_biases"]})
    watch_idx = np.array(job["watch_idx"], dtype=np.int32)
    exc_idx = np.array(job["exc_idx"], dtype=np.int32)
    empty = np.empty(0, dtype=np.int32)
    n_trials = int(job["n_trials"])
    seed_bg = int(job["seed_bg"])

    bg_counts, _ = run_lif_dc_paired(
        g["indptr"], g["post_s"], w_s, float(job["w_syn"]), vb, init_u_max,
        empty, 0.0, empty, 0.0, watch_idx, n_trials, seed_bg, 12345,
    )
    rates_out = {}
    for rate_hz in job["rates_hz"]:
        on_counts, _ = run_lif_dc_paired(
            g["indptr"], g["post_s"], w_s, float(job["w_syn"]), vb, init_u_max,
            exc_idx, float(rate_hz), empty, 0.0, watch_idx, n_trials, seed_bg, int(job["seed_opto"]) + int(rate_hz),
        )
        st = _paired_stats(on_counts, bg_counts)
        if watch_idx.shape[0] > 2:
            st["watched_intermed_bg_hz"] = bg_counts[:, 2:].mean(axis=0).tolist()
            st["watched_intermed_on_hz"] = on_counts[:, 2:].mean(axis=0).tolist()
        rates_out[f"{int(rate_hz)}hz"] = st

    out = {
        "tag": job["tag"],
        "bg_mn9_l_hz": float(bg_counts[:, 0].mean()),
        "bg_mn9_r_hz": float(bg_counts[:, 1].mean()),
        "bg_mn9_mean_hz": float(bg_counts[:, :2].mean()),
        "rates": rates_out,
        "elapsed_sec": float(time.time() - t0),
    }
    _save_json_atomic(ckpt, out)
    return out


def analyze_disinhibitory_anatomy(
    g: dict,
    node_meta: dict[int, dict],
    mn9_indices: list[int],
    inh_groups: dict[str, list[int]],
    selk_indices: list[int],
    is_vnc: np.ndarray | None = None,
) -> dict:
    """Module 1: Exact anatomical disinhibitory path audit for inhibitory candidates & SELK."""
    indptr = g["indptr"]
    post_s = g["post_s"]
    w_raw = g["w_raw_s"]
    node_class = g["node_class"]
    n_nodes = g["n_nodes"]
    mn9_set = set(mn9_indices)

    # Precompute 1-hop incoming weights into MN9_l + MN9_r per node
    w_to_mn9 = np.zeros(n_nodes, dtype=np.float64)
    for m in mn9_indices:
        # scan incoming edges via pre_s / post_s
        mask_m = post_s == m
        pre_m = g["pre_s"][mask_m]
        wm = w_raw[mask_m].astype(np.float64)
        np.add.at(w_to_mn9, pre_m, wm)

    inh_to_mn9 = np.maximum(0.0, -w_to_mn9)  # |w| for inhibitory 1-hop MN9 inputs (I_pre)
    exc_to_mn9 = np.maximum(0.0, w_to_mn9)   # w for excitatory 1-hop MN9 inputs (E_pre)

    # For fast 3-hop (X - I_1 - E_pre + MN9) and 4-hop (SELK + I_1 - I_2 - E_pre + MN9):
    # Precompute for every inhibitory node u (node_class == 1):
    #   - its inhibitory outgoing edges to I_pre (where inh_to_mn9[v] > 0)
    #   - its inhibitory outgoing edges to E_pre (where exc_to_mn9[v] > 0)
    # Specifically compute per-node u (inhibitory) the best & summed inhibition onto E_pre -> MN9
    e_pre_nodes = np.flatnonzero(exc_to_mn9 > 0)
    i_pre_nodes = np.flatnonzero(inh_to_mn9 > 0)

    # Map u (inhibitory) -> list of (e_pre, |w_{u, e_pre}|, w_{e_pre, mn9})
    inh_to_epre_prod = np.zeros(n_nodes, dtype=np.float64)
    inh_to_epre_max_bn = np.zeros(n_nodes, dtype=np.float64)
    inh_to_epre_best_e = np.full(n_nodes, -1, dtype=np.int32)
    inh_to_ipre_prod = np.zeros(n_nodes, dtype=np.float64)
    inh_to_ipre_max_bn = np.zeros(n_nodes, dtype=np.float64)
    inh_to_ipre_best_i = np.full(n_nodes, -1, dtype=np.int32)

    for u in np.flatnonzero(node_class == 1):
        p0, p1 = int(indptr[u]), int(indptr[u + 1])
        prod_e = 0.0
        mbn_e = 0.0
        best_e = -1
        prod_i = 0.0
        mbn_i = 0.0
        best_i = -1
        for e in range(p0, p1):
            v = int(post_s[e])
            w_abs = float(-w_raw[e])
            if w_abs <= 0:
                continue
            we = float(exc_to_mn9[v])
            if we > 0.0:
                prod_e += w_abs * we
                bn = min(w_abs, we)
                if bn > mbn_e:
                    mbn_e = bn
                    best_e = v
            wi = float(inh_to_mn9[v])
            if wi > 0.0:
                prod_i += w_abs * wi
                bn = min(w_abs, wi)
                if bn > mbn_i:
                    mbn_i = bn
                    best_i = v
        inh_to_epre_prod[u] = prod_e
        inh_to_epre_max_bn[u] = mbn_e
        inh_to_epre_best_e[u] = best_e
        inh_to_ipre_prod[u] = prod_i
        inh_to_ipre_max_bn[u] = mbn_i
        inh_to_ipre_best_i[u] = best_i

    def desc(u: int) -> dict:
        m = node_meta.get(int(u), {})
        return {
            "idx": int(u),
            "id": int(m.get("id", -1)),
            "cell_type": str(m.get("cell_type", "")),
            "fafb_cell_type": str(m.get("fafb_cell_type", m.get("cell_type", ""))),
            "side": str(m.get("side", "")),
            "region": str(m.get("region", "brain")),
            "nt": str(m.get("nt", "")),
        }

    # 1. Analyze every inhibitory group (phantom, tentacular, usnea, marge, and all 44 negative inhibitory controls)
    inh_results = {}
    for gname, gidxs in inh_groups.items():
        # Aggregate outgoing signed weights from group X
        out_w: dict[int, float] = defaultdict(float)
        for s in gidxs:
            p0, p1 = int(indptr[s]), int(indptr[s + 1])
            for e in range(p0, p1):
                out_w[int(post_s[e])] += float(w_raw[e])

        dir_inh_mn9 = float(sum(-out_w[m] for m in mn9_indices if out_w.get(m, 0.0) < 0.0))
        # 2-hop disinhibition: X -(-)--> I_pre -(-)--> MN9
        p2_disinh = []
        # Antagonistic 2-hop: X -(-)--> E_pre -(+)--> MN9
        p2_antag = []
        # 3-hop disinhibitory excitation: X -(-)--> I_1 -(-)--> E_pre -(+)--> MN9
        p3_disinh_e = []

        for v, w_xv in out_w.items():
            if w_xv >= 0.0 or v in mn9_set:
                continue
            w_abs = -w_xv
            if inh_to_mn9[v] > 0.0:
                wi = float(inh_to_mn9[v])
                p2_disinh.append((min(w_abs, wi), w_abs * wi, w_abs, wi, v))
            if exc_to_mn9[v] > 0.0:
                we = float(exc_to_mn9[v])
                p2_antag.append((min(w_abs, we), w_abs * we, w_abs, we, v))
            if node_class[v] == 1 and inh_to_epre_prod[v] > 0.0:
                # Enumerate exact (v=I_1, e_pre) pairs for top chains
                p0, p1 = int(indptr[v]), int(indptr[v + 1])
                for e in range(p0, p1):
                    ep = int(post_s[e])
                    we = float(exc_to_mn9[ep])
                    w_ve = float(-w_raw[e])
                    if we > 0.0 and w_ve > 0.0:
                        bn3 = min(w_abs, w_ve, we)
                        p3_disinh_e.append((bn3, w_abs * w_ve * we, w_abs, w_ve, we, v, ep))

        p2_disinh.sort(key=lambda x: (-x[0], -x[1]))
        p2_antag.sort(key=lambda x: (-x[0], -x[1]))
        p3_disinh_e.sort(key=lambda x: (-x[0], -x[1]))

        save_detail = gname in ("phantom", "tentacular", "usnea", "marge")
        rec = {
            "n_cells": len(gidxs),
            "total_out_inh_syn": float(sum(-w for w in out_w.values() if w < 0.0)),
            "direct_1hop_inh_to_mn9": dir_inh_mn9,
            "hop2_disinh_count": len(p2_disinh),
            "hop2_disinh_prod_sum": float(sum(x[1] for x in p2_disinh)),
            "hop2_disinh_bn_sum": float(sum(x[0] for x in p2_disinh)),
            "hop2_disinh_max_bn": float(p2_disinh[0][0]) if p2_disinh else 0.0,
            "hop2_antag_epre_count": len(p2_antag),
            "hop2_antag_epre_prod_sum": float(sum(x[1] for x in p2_antag)),
            "hop2_antag_epre_bn_sum": float(sum(x[0] for x in p2_antag)),
            "hop2_antag_epre_max_bn": float(p2_antag[0][0]) if p2_antag else 0.0,
            "hop2_net_disinh_minus_antag_prod": float(sum(x[1] for x in p2_disinh) - sum(x[1] for x in p2_antag)),
            "hop3_disinh_epre_count": len(p3_disinh_e),
            "hop3_disinh_epre_prod_sum": float(sum(x[1] for x in p3_disinh_e)),
            "hop3_disinh_epre_bn_sum": float(sum(x[0] for x in p3_disinh_e)),
            "hop3_disinh_epre_max_bn": float(p3_disinh_e[0][0]) if p3_disinh_e else 0.0,
        }
        if save_detail:
            rec["top15_hop2_ipre"] = [
                {"bn": float(x[0]), "prod": float(x[1]), "w_x_ipre": float(x[2]), "w_ipre_mn9": float(x[3]), "node": desc(x[4])}
                for x in p2_disinh[:15]
            ]
            rec["top15_hop2_antag_epre"] = [
                {"bn": float(x[0]), "prod": float(x[1]), "w_x_epre": float(x[2]), "w_epre_mn9": float(x[3]), "node": desc(x[4])}
                for x in p2_antag[:15]
            ]
            rec["top15_hop3_i1_epre"] = [
                {"bn": float(x[0]), "prod": float(x[1]), "w_x_i1": float(x[2]), "w_i1_epre": float(x[3]), "w_epre_mn9": float(x[4]),
                 "i1": desc(x[5]), "epre": desc(x[6])}
                for x in p3_disinh_e[:15]
            ]
        inh_results[gname] = rec

    # 2. Analyze cholinergic SELK (DNg68) 3-hop & 4-hop disinhibitory + antagonistic paths
    selk_out: dict[int, float] = defaultdict(float)
    for s in selk_indices:
        p0, p1 = int(indptr[s]), int(indptr[s + 1])
        for e in range(p0, p1):
            if w_raw[e] > 0:
                selk_out[int(post_s[e])] += float(w_raw[e])

    # Direct 2-hop antagonistic SELK + I_1 - MN9; 3-hop antagonistic SELK + I_1 - E_pre + MN9
    # 3-hop disinhibitory SELK + I_1 - I_2 - MN9; 4-hop disinhibitory SELK + I_1 - I_2 - E_pre + MN9
    selk_h2_antag = []
    selk_h3_antag = []
    selk_h3_disinh_all = []
    selk_h3_disinh_brain = []
    selk_h3_disinh_vnc = []
    selk_h4_disinh_all = []
    selk_h4_disinh_brain = []
    selk_h4_disinh_vnc = []

    for i1, w_si1 in selk_out.items():
        if node_class[i1] != 1:
            continue
        i1_vnc = bool(is_vnc[i1]) if is_vnc is not None else False
        if inh_to_mn9[i1] > 0.0:
            wi = float(inh_to_mn9[i1])
            selk_h2_antag.append((min(w_si1, wi), w_si1 * wi, w_si1, wi, i1))
        p0, p1 = int(indptr[i1]), int(indptr[i1 + 1])
        for e in range(p0, p1):
            v2 = int(post_s[e])
            w_12 = float(-w_raw[e])
            if w_12 <= 0.0:
                continue
            if exc_to_mn9[v2] > 0.0:
                we = float(exc_to_mn9[v2])
                selk_h3_antag.append((min(w_si1, w_12, we), w_si1 * w_12 * we, w_si1, w_12, we, i1, v2))
            if node_class[v2] == 1:
                v2_vnc = bool(is_vnc[v2]) if is_vnc is not None else False
                via_vnc = i1_vnc or v2_vnc
                if inh_to_mn9[v2] > 0.0:
                    w2m = float(inh_to_mn9[v2])
                    tup = (min(w_si1, w_12, w2m), w_si1 * w_12 * w2m, w_si1, w_12, w2m, i1, v2, via_vnc)
                    selk_h3_disinh_all.append(tup)
                    if via_vnc:
                        selk_h3_disinh_vnc.append(tup)
                    else:
                        selk_h3_disinh_brain.append(tup)
                # 4-hop: SELK + I_1 - I_2 - E_pre + MN9 (use best_e + aggregate for speed, exact top chains)
                if inh_to_epre_max_bn[v2] > 0.0:
                    ep = int(inh_to_epre_best_e[v2])
                    # lookup exact weight v2 -> ep
                    we_m = float(exc_to_mn9[ep])
                    bn_2e = float(inh_to_epre_max_bn[v2])
                    bn4 = min(w_si1, w_12, bn_2e)
                    prod4 = w_si1 * w_12 * float(inh_to_epre_prod[v2])
                    tup4 = (bn4, prod4, w_si1, w_12, bn_2e, we_m, i1, v2, ep, via_vnc)
                    selk_h4_disinh_all.append(tup4)
                    if via_vnc:
                        selk_h4_disinh_vnc.append(tup4)
                    else:
                        selk_h4_disinh_brain.append(tup4)

    for lst in (selk_h2_antag, selk_h3_antag, selk_h3_disinh_all, selk_h3_disinh_brain, selk_h3_disinh_vnc,
                selk_h4_disinh_all, selk_h4_disinh_brain, selk_h4_disinh_vnc):
        lst.sort(key=lambda x: (-x[0], -x[1]))

    def pack_h3(lst):
        return {
            "count": len(lst),
            "prod_sum": float(sum(x[1] for x in lst)),
            "bn_sum": float(sum(x[0] for x in lst)),
            "max_bn": float(lst[0][0]) if lst else 0.0,
            "top15": [
                {"bn": float(x[0]), "prod": float(x[1]), "w_s_i1": float(x[2]), "w_i1_i2": float(x[3]),
                 "w_i2_mn9": float(x[4]), "via_vnc": bool(x[7]), "i1": desc(x[5]), "i2": desc(x[6])}
                for x in lst[:15]
            ],
        }

    def pack_h4(lst):
        return {
            "count_i1_i2_pairs": len(lst),
            "prod_sum": float(sum(x[1] for x in lst)),
            "bn_sum": float(sum(x[0] for x in lst)),
            "max_bn": float(lst[0][0]) if lst else 0.0,
            "top15": [
                {"bn": float(x[0]), "prod_i1_i2_all_epre": float(x[1]), "w_s_i1": float(x[2]), "w_i1_i2": float(x[3]),
                 "bn_i2_epre_mn9": float(x[4]), "via_vnc": bool(x[9]), "i1": desc(x[6]), "i2": desc(x[7]), "best_epre": desc(x[8])}
                for x in lst[:15]
            ],
        }

    selk_anatomy = {
        "n_cells": len(selk_indices),
        "total_out_exc_syn": float(sum(selk_out.values())),
        "hop2_antag_i1_mn9": {
            "count": len(selk_h2_antag),
            "prod_sum": float(sum(x[1] for x in selk_h2_antag)),
            "max_bn": float(selk_h2_antag[0][0]) if selk_h2_antag else 0.0,
            "top10": [{"bn": float(x[0]), "prod": float(x[1]), "w_s_i1": float(x[2]), "w_i1_mn9": float(x[3]), "i1": desc(x[4])} for x in selk_h2_antag[:10]],
        },
        "hop3_antag_i1_epre_mn9": {
            "count": len(selk_h3_antag),
            "prod_sum": float(sum(x[1] for x in selk_h3_antag)),
            "max_bn": float(selk_h3_antag[0][0]) if selk_h3_antag else 0.0,
            "top10": [{"bn": float(x[0]), "prod": float(x[1]), "w_s_i1": float(x[2]), "w_i1_epre": float(x[3]), "w_epre_mn9": float(x[4]), "i1": desc(x[5]), "epre": desc(x[6])} for x in selk_h3_antag[:10]],
        },
        "hop3_disinh_all": pack_h3(selk_h3_disinh_all),
        "hop3_disinh_brain_only": pack_h3(selk_h3_disinh_brain),
        "hop3_disinh_via_vnc": pack_h3(selk_h3_disinh_vnc),
        "hop4_disinh_all": pack_h4(selk_h4_disinh_all),
        "hop4_disinh_brain_only": pack_h4(selk_h4_disinh_brain),
        "hop4_disinh_via_vnc": pack_h4(selk_h4_disinh_vnc),
    }

    return {"inhibitory_candidates_and_controls": inh_results, "selk_dng68": selk_anatomy}


def main() -> None:
    import os
    cache_root = Path(os.environ.get("PROJC_CACHE_DIR", str(Path.home() / ".cache/projc-flywire")))
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=cache_root / "data")
    parser.add_argument("--source02-dir", type=Path, default=cache_root / "source02")
    parser.add_argument("--banc-npz", type=Path, default=cache_root / "banc_ckpt/banc_main_graph.npz")
    parser.add_argument("--v783-results", type=Path, default=Path("mn9_transfer_results.json"))
    parser.add_argument("--banc-results", type=Path, default=Path("mn9_banc_results.json"))
    parser.add_argument("--ckpt-dir", type=Path, default=cache_root / "disinh_ckpt")
    parser.add_argument("--gpu-equiv-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("mn9_disinhibition_results.json"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("[1/5] Loading FlyWire v783 & BANC v888 graphs and metadata...", flush=True)
    g783 = mtb.load_graph_csr(args.data_dir / "Completeness_783.csv", args.data_dir / "Connectivity_783.parquet")
    v783_npz = args.ckpt_dir / "v783_graph.npz"
    if not v783_npz.exists():
        np.savez(
            v783_npz,
            n_nodes=np.int32(g783["n_nodes"]),
            indptr=g783["indptr"],
            pre_s=g783["pre_s"],
            post_s=g783["post_s"],
            w_raw_s=g783["w_raw_s"],
            node_class=g783["node_class"],
        )

    v783_res = json.load(open(args.v783_results))
    banc_res = json.load(open(args.banc_results))
    sez_dict = pickle.load(open(args.source02_dir / "sez_neurons.pickle", "rb"))
    st3_records = smb.load_sup_table_3(args.source02_dir / "shiu_moesm2.xlsx", sez_dict)

    v783_ann = {}
    with open(args.data_dir / "Supplemental_file1_neuron_annotations.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            v783_ann[int(row["root_id"])] = row
    idx783 = g783["flyid2idx"]
    meta_v783 = {}
    for i, fid in enumerate(g783["flyids"].tolist()):
        r = v783_ann.get(int(fid), {})
        meta_v783[i] = {
            "id": int(fid), "cell_type": r.get("cell_type", ""), "fafb_cell_type": r.get("cell_type", ""),
            "side": r.get("side", ""), "region": "brain", "nt": r.get("top_nt", ""),
        }

    # Load BANC main graph + node IDs + metadata
    meta_tbl = pf.read_table(args.data_dir / "banc/banc_888_meta.feather")
    b_rows = {}
    for b, pr, rpr, sc, ct, fct, side, reg in zip(
        meta_tbl["banc_888_id"].to_pylist(), meta_tbl["proofread"].to_pylist(), meta_tbl["roughly_proofread"].to_pylist(),
        meta_tbl["super_class"].to_pylist(), meta_tbl["cell_type"].to_pylist(), meta_tbl["fafb_cell_type"].to_pylist(),
        meta_tbl["side"].to_pylist(), meta_tbl["region"].to_pylist(),
    ):
        b_rows[int(b)] = {"pr": pr, "rpr": rpr, "sc": sc, "ct": ct or "", "fct": fct or "", "side": side or "", "reg": reg or ""}
    nt_csv = {}
    with open(args.data_dir / "banc/banc_888_neurotransmitter_prediction_v2.csv") as f:
        for row in csv.DictReader(f):
            if row["neurotransmitter_predicted"] not in ("", "NA"):
                nt_csv[int(row["root_id"])] = row["neurotransmitter_predicted"]

    # Reconstruct exact BANC node_ids order matching banc_main_graph.npz
    el = pf.read_table(args.data_dir / "banc/banc_888_edgelist_simple_v2.feather")
    raw_pre = el["pre"].to_numpy().astype(np.int64)
    raw_post = el["post"].to_numpy().astype(np.int64)
    ns = raw_pre != raw_post
    ns_pre, ns_post = raw_pre[ns], raw_post[ns]
    nt_ids = np.array(sorted(nt_csv.keys()), dtype=np.int64)
    nt_signs = np.array([mbb.nt_to_sign(nt_csv[int(b)]) for b in nt_ids], dtype=np.int8)
    main_node_set = np.array(sorted(b for b, d in b_rows.items() if (d["pr"] == "TRUE" or d["rpr"] == "TRUE") and d["sc"] != "glia"), dtype=np.int64)
    pos = np.minimum(np.searchsorted(nt_ids, ns_pre), len(nt_ids) - 1)
    vnt = np.where(nt_ids[pos] == ns_pre, nt_signs[pos], np.int8(0)) != 0
    mm = vnt & np.isin(ns_pre, main_node_set) & np.isin(ns_post, main_node_set)
    banc_node_ids = np.sort(np.union1d(ns_pre[mm], ns_post[mm]))
    del el, raw_pre, raw_post, ns_pre, ns_post

    bd = np.load(args.banc_npz)
    assert len(banc_node_ids) == int(bd["n_nodes"]), "BANC node_ids length mismatch with npz"
    g_banc = {
        "n_nodes": int(bd["n_nodes"]), "node_ids": banc_node_ids,
        "id2idx": {int(b): i for i, b in enumerate(banc_node_ids.tolist())},
        "indptr": bd["indptr"], "pre_s": bd["pre_s"], "post_s": bd["post_s"],
        "w_raw_s": bd["w_raw_s"], "node_class": bd["node_class"],
    }
    idx_banc = g_banc["id2idx"]
    meta_banc = {}
    is_vnc_banc = np.zeros(g_banc["n_nodes"], dtype=bool)
    for i, bid in enumerate(banc_node_ids.tolist()):
        r = b_rows[bid]
        is_vnc_banc[i] = (r["reg"] == "ventral_nerve_cord")
        meta_banc[i] = {
            "id": bid, "cell_type": r["ct"], "fafb_cell_type": r["fct"] or r["ct"],
            "side": r["side"], "region": r["reg"], "nt": nt_csv.get(bid, ""),
        }

    # Build target index dictionaries for v783 & BANC
    ma783 = v783_res["mapping_audit"]
    mn9_783 = [idx783[int(ma783["mn9_l"]["v783_id"])], idx783[int(ma783["mn9_r"]["v783_id"])]]
    mn9_banc = [idx_banc[720575941410682479], idx_banc[720575941623285450]]
    selk_783 = [idx783[f] for f in mtb.SELK_V783_FIDS]
    selk_banc = [idx_banc[b] for b in banc_res["selk_full_cns_diagnosis"]["selk_banc_ids"]]

    v783_sez_map = {r["name"]: [idx783[int(f)] for f in r["v783_ids"]] for r in ma783["per_type_sez106"]}
    banc_sez_map = {r["name"]: [idx_banc[int(b)] for b in r["m1_banc_ids"]] for r in banc_res["mapping_audit"]["per_type_sez106"]}

    sugar_783 = [idx783[720575940639259967 if f == 720575940620900446 else f] for f in smb.SUGAR_GRN_FIDS]
    water_783 = [idx783[f] for f in smb.WATER_GRN_FIDS]
    # BANC annotated right sugar LB3b (40) & right water LB3a (14)
    lb3b_r_banc = sorted(
        idx_banc[b] for b, d in b_rows.items()
        if b in idx_banc and d["pr"] == "TRUE" and d["ct"] == "LB3b" and d["side"] == "right"
    )
    # Match exact 14 right LB3a water GRNs from banc_meta
    lb3a_r_banc = []
    for b, pr, ct, cfd, side in zip(
        meta_tbl["banc_888_id"].to_pylist(), meta_tbl["proofread"].to_pylist(),
        meta_tbl["cell_type"].to_pylist(), meta_tbl["cell_function_detailed"].to_pylist(),
        meta_tbl["side"].to_pylist(),
    ):
        bi = int(b)
        if bi in idx_banc and pr == "TRUE" and ct == "LB3a" and cfd == "water, ppk28" and side == "right":
            lb3a_r_banc.append(idx_banc[bi])
    lb3a_r_banc.sort()

    y_max = np.array([r["y_max"] for r in st3_records], dtype=np.float64)
    y_bin = (y_max > 0.0).astype(np.int32)
    sez_names = [r["name"] for r in st3_records]
    w_syn_v783 = float(smb.W_SYN)  # 0.275
    w_syn_banc_anat = float(banc_res["graph_construction_and_anatomy"]["mn9_anatomy_and_wsyn_scale"]["w_syn_anat_mv"])

    # Identify all 44 purely-inhibitory negative SEZ types in v783 (+ 4 purely-inhibitory positives)
    inh_neg_names = [
        name for i, name in enumerate(sez_names)
        if y_bin[i] == 0 and all(int(g783["node_class"][u]) == 1 for u in v783_sez_map[name])
    ]
    inh_group_names = ["phantom", "tentacular", "usnea", "marge"] + inh_neg_names

    print(f"[2/5] Module 1: Cross-brain anatomical disinhibition audit ({len(inh_group_names)} inhibitory groups + SELK)...", flush=True)
    anat_v783 = analyze_disinhibitory_anatomy(
        g783, meta_v783, mn9_783, {k: v783_sez_map[k] for k in inh_group_names}, selk_783, is_vnc=None
    )
    anat_banc = analyze_disinhibitory_anatomy(
        g_banc, meta_banc, mn9_banc, {k: banc_sez_map[k] for k in inh_group_names}, selk_banc, is_vnc=is_vnc_banc
    )

    # Summarize percentiles of phantom/tentacular/usnea/marge vs 44 purely-inhibitory negative controls & cross-brain overlap
    def summarize_anat_percentiles(anat_obj: dict) -> dict:
        ic = anat_obj["inhibitory_candidates_and_controls"]
        ctrl_keys = inh_neg_names
        out = {}
        for metric in ("hop2_disinh_prod_sum", "hop2_disinh_max_bn", "hop2_disinh_bn_sum",
                       "hop2_antag_epre_prod_sum", "hop2_net_disinh_minus_antag_prod",
                       "hop3_disinh_epre_prod_sum", "hop3_disinh_epre_max_bn"):
            cvals = np.array([ic[k][metric] for k in ctrl_keys], dtype=np.float64)
            out[metric] = {
                "neg_ctrl_44_median": float(np.median(cvals)),
                "neg_ctrl_44_p75": float(np.percentile(cvals, 75)),
                "neg_ctrl_44_p90": float(np.percentile(cvals, 90)),
                "neg_ctrl_44_max": float(np.max(cvals)),
                "candidates": {
                    cand: {
                        "value": float(ic[cand][metric]),
                        "pct_among_44_neg_inh": float(100.0 * np.mean(cvals <= ic[cand][metric])),
                        "n_neg_ctrl_ge_cand": int(np.sum(cvals >= ic[cand][metric])),
                    }
                    for cand in ("phantom", "tentacular", "usnea", "marge")
                },
            }
        return out

    pct_v783 = summarize_anat_percentiles(anat_v783)
    pct_banc = summarize_anat_percentiles(anat_banc)

    # Cross-brain homologous intermediate cell_type overlap for phantom, tentacular, SELK
    def cross_brain_overlap(cand: str) -> dict:
        if cand == "SELK_DNg68":
            t7_i1 = {x["i1"]["fafb_cell_type"] for x in anat_v783["selk_dng68"]["hop3_disinh_all"]["top15"] if x["i1"]["fafb_cell_type"]}
            tb_i1 = {x["i1"]["fafb_cell_type"] for x in anat_banc["selk_dng68"]["hop3_disinh_all"]["top15"] if x["i1"]["fafb_cell_type"]}
            t7_i2 = {x["i2"]["fafb_cell_type"] for x in anat_v783["selk_dng68"]["hop3_disinh_all"]["top15"] if x["i2"]["fafb_cell_type"]}
            tb_i2 = {x["i2"]["fafb_cell_type"] for x in anat_banc["selk_dng68"]["hop3_disinh_all"]["top15"] if x["i2"]["fafb_cell_type"]}
            return {
                "shared_top15_h3_i1_types": sorted(t7_i1 & tb_i1),
                "shared_top15_h3_i2_types": sorted(t7_i2 & tb_i2),
            }
        r7 = anat_v783["inhibitory_candidates_and_controls"][cand]
        rb = anat_banc["inhibitory_candidates_and_controls"][cand]
        i2_7 = {x["node"]["fafb_cell_type"] for x in r7["top15_hop2_ipre"] if x["node"]["fafb_cell_type"]}
        i2_b = {x["node"]["fafb_cell_type"] for x in rb["top15_hop2_ipre"] if x["node"]["fafb_cell_type"]}
        e2_7 = {x["node"]["fafb_cell_type"] for x in r7["top15_hop2_antag_epre"] if x["node"]["fafb_cell_type"]}
        e2_b = {x["node"]["fafb_cell_type"] for x in rb["top15_hop2_antag_epre"] if x["node"]["fafb_cell_type"]}
        i3_7 = {x["i1"]["fafb_cell_type"] for x in r7["top15_hop3_i1_epre"] if x["i1"]["fafb_cell_type"]}
        i3_b = {x["i1"]["fafb_cell_type"] for x in rb["top15_hop3_i1_epre"] if x["i1"]["fafb_cell_type"]}
        return {
            "shared_top15_hop2_ipre_types": sorted(i2_7 & i2_b),
            "shared_top15_hop2_antag_epre_types": sorted(e2_7 & e2_b),
            "shared_top15_hop3_i1_types": sorted(i3_7 & i3_b),
        }

    cross_overlap = {c: cross_brain_overlap(c) for c in ("phantom", "tentacular", "usnea", "marge", "SELK_DNg68")}
    for c in ("phantom", "tentacular", "usnea", "marge"):
        r7 = anat_v783["inhibitory_candidates_and_controls"][c]
        rb = anat_banc["inhibitory_candidates_and_controls"][c]
        print(
            f"  [Anat {c:10s}] v783: 1h_inh={r7['direct_1hop_inh_to_mn9']:.0f}, 2h_dis_prod={r7['hop2_disinh_prod_sum']:.0f}(bn={r7['hop2_disinh_max_bn']:.0f}, pct={pct_v783['hop2_disinh_prod_sum']['candidates'][c]['pct_among_44_neg_inh']:.0f}%), "
            f"2h_antag={r7['hop2_antag_epre_prod_sum']:.0f}, 3h_dis_bn={r7['hop3_disinh_epre_max_bn']:.0f} | "
            f"BANC: 2h_dis_prod={rb['hop2_disinh_prod_sum']:.0f}(bn={rb['hop2_disinh_max_bn']:.0f}), 2h_antag={rb['hop2_antag_epre_prod_sum']:.0f}",
            flush=True,
        )
    print(
        f"  [Anat SELK_DNg68] v783: h2_antag={anat_v783['selk_dng68']['hop2_antag_i1_mn9']['prod_sum']:.0f}, "
        f"h3_antag={anat_v783['selk_dng68']['hop3_antag_i1_epre_mn9']['prod_sum']:.0f}, "
        f"h3_dis={anat_v783['selk_dng68']['hop3_disinh_all']['prod_sum']:.0f}(bn={anat_v783['selk_dng68']['hop3_disinh_all']['max_bn']:.0f}), "
        f"h4_dis_bn={anat_v783['selk_dng68']['hop4_disinh_all']['max_bn']:.0f} | "
        f"BANC: h3_dis_brain_bn={anat_banc['selk_dng68']['hop3_disinh_brain_only']['max_bn']:.0f}, "
        f"h3_dis_vnc_bn={anat_banc['selk_dng68']['hop3_disinh_via_vnc']['max_bn']:.0f}, "
        f"h4_dis_vnc_bn={anat_banc['selk_dng68']['hop4_disinh_via_vnc']['max_bn']:.0f}",
        flush=True,
    )

    print("[3/5] Module 2 (State Families 1, 2, 3a on FlyWire v783: 11 background regimes x 106 SEZ + SELK)...", flush=True)
    # Define all 11 locked v783 background conditions
    v783_conditions = [
        ("F1_global_sub_5.50", "global", {"v": 5.50}, [], 0.0, 20),
        ("F1_global_supra_7.10", "global", {"v": 7.10}, [], 0.0, 20),
        ("F1_global_supra_7.30", "global", {"v": 7.30}, [], 0.0, 20),
        ("F1_global_supra_7.40", "global", {"v": 7.40}, [], 0.0, 20),
        ("F2_ei_E5.50_I7.50", "ei", {"ve": 5.50, "vi": 7.50}, [], 0.0, 20),
        ("F2_ei_E7.30_I8.00", "ei", {"ve": 7.30, "vi": 8.00}, [], 0.0, 20),
        ("F2_ei_E7.45_I8.00", "ei", {"ve": 7.45, "vi": 8.00}, [], 0.0, 20),
        ("F2_ei_E7.55_I8.00", "ei", {"ve": 7.55, "vi": 8.00}, [], 0.0, 20),
        ("F3a_sugar_bg_40hz", "none", {}, sugar_783, 40.0, 30),
        ("F3a_sugar_bg_50hz", "none", {}, sugar_783, 50.0, 30),
        ("F3a_water_bg_160hz", "none", {}, water_783, 160.0, 30),
    ]
    focal_names = ["phantom", "tentacular", "SELK_DNg68", "marge", "usnea"]
    all_107_items_v783 = [
        {"name": name, "exc_idx": v783_sez_map[name], "rate_hz": 50.0, "seed_opto": 50000 + i * 97}
        for i, name in enumerate(sez_names)
    ] + [{"name": "SELK_DNg68", "exc_idx": selk_783, "rate_hz": 50.0, "seed_opto": 81050}]

    focal_30tr_items_v783 = []
    for fname in focal_names:
        eidx = selk_783 if fname == "SELK_DNg68" else v783_sez_map[fname]
        for f_hz in (50.0, 200.0):
            focal_30tr_items_v783.append({
                "name": f"{fname}_{int(f_hz)}hz", "exc_idx": eidx, "rate_hz": f_hz,
                "seed_opto": 90000 + focal_names.index(fname) * 1000 + int(f_hz),
            })

    ctx = mp.get_context("spawn")
    cond_bg_20: dict[str, dict] = {}
    cond_bg_30: dict[str, dict] = {}
    cond_full_res: dict[str, dict] = defaultdict(dict)
    cond_focal_30_res: dict[str, dict] = defaultdict(dict)

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        # Step 3A: Compute all baselines (n_trials for full scan AND 30 trials for focal) in parallel
        bg_futs = []
        for cname, bmode, bparams, bg_e, bg_r, ntr in v783_conditions:
            bg_futs.append(ex.submit(_single_bg_worker, {
                "cond": f"{cname}|full", "npz_path": str(v783_npz), "w_syn": w_syn_v783,
                "bg_mode": bmode, "bg_params": bparams, "bg_exc_idx": bg_e, "bg_exc_rate_hz": bg_r,
                "watch_idx": mn9_783, "n_trials": ntr, "seed_bg": 31001,
                "ckpt_path": str(args.ckpt_dir / f"bg_{cname}_{ntr}tr.json"),
            }))
            if ntr != 30:
                bg_futs.append(ex.submit(_single_bg_worker, {
                    "cond": f"{cname}|focal30", "npz_path": str(v783_npz), "w_syn": w_syn_v783,
                    "bg_mode": bmode, "bg_params": bparams, "bg_exc_idx": bg_e, "bg_exc_rate_hz": bg_r,
                    "watch_idx": mn9_783, "n_trials": 30, "seed_bg": 31001,
                    "ckpt_path": str(args.ckpt_dir / f"bg_{cname}_30tr.json"),
                }))
        for fut in concurrent.futures.as_completed(bg_futs):
            r = fut.result()
            cname, kind = r["cond"].split("|")
            if kind == "full":
                cond_bg_20[cname] = r
                if len(r["bg_counts"]) == 30:
                    cond_bg_30[cname] = r
                print(
                    f"  [bg {cname:22s}] MN9_l={r['bg_mn9_l_hz']:5.2f} MN9_r={r['bg_mn9_r_hz']:5.2f} "
                    f"mean={r['bg_mn9_mean_hz']:5.2f} brain={r['brain_mean_hz']:6.3f} Hz ({r['elapsed_sec']:.1f}s)",
                    flush=True,
                )
            else:
                cond_bg_30[cname] = r

        # Step 3B: Submit all 107-type slices + 30-trial focal 50/200 Hz jobs across all 11 v783 conditions!
        t3b = time.time()
        slice_futs = []
        for cname, bmode, bparams, bg_e, bg_r, ntr in v783_conditions:
            chunk_sz = 4 if bmode in ("global", "ei") else 11
            bg_c = cond_bg_20[cname]["bg_counts"]
            for s_i in range(0, len(all_107_items_v783), chunk_sz):
                sub_items = all_107_items_v783[s_i : s_i + chunk_sz]
                slice_futs.append(ex.submit(_condition_slice_worker, {
                    "cond": f"{cname}|full", "slice_id": s_i, "npz_path": str(v783_npz), "w_syn": w_syn_v783,
                    "bg_mode": bmode, "bg_params": bparams, "bg_exc_idx": bg_e, "bg_exc_rate_hz": bg_r,
                    "watch_idx": mn9_783, "n_trials": ntr, "seed_bg": 31001, "bg_counts": bg_c,
                    "items": sub_items, "ckpt_path": str(args.ckpt_dir / f"v783_{cname}_full_{s_i:03d}.json"),
                }))
            # 30-trial focal 50 & 200 Hz
            slice_futs.append(ex.submit(_condition_slice_worker, {
                "cond": f"{cname}|focal30", "slice_id": 0, "npz_path": str(v783_npz), "w_syn": w_syn_v783,
                "bg_mode": bmode, "bg_params": bparams, "bg_exc_idx": bg_e, "bg_exc_rate_hz": bg_r,
                "watch_idx": mn9_783, "n_trials": 30, "seed_bg": 31001, "bg_counts": cond_bg_30[cname]["bg_counts"],
                "items": focal_30tr_items_v783, "ckpt_path": str(args.ckpt_dir / f"v783_{cname}_focal30.json"),
            }))

        n_tot = len(slice_futs)
        n_done = 0
        for fut in concurrent.futures.as_completed(slice_futs):
            r = fut.result()
            n_done += 1
            cname, kind = r["cond"].split("|")
            if kind == "full":
                cond_full_res[cname].update(r["results"])
            else:
                cond_focal_30_res[cname].update(r["results"])
            if n_done % 40 == 0 or n_done == n_tot:
                print(f"  [v783 bg scans {n_done:3d}/{n_tot} elapsed={time.time()-t3b:.1f}s] last={cname}", flush=True)

        print("[4/5] Module 2 (State Family 3b: Local Oracle realizability bounds + BANC v888 confirmation)...", flush=True)
        oracle_jobs = []

        def add_oracle_suite(brain_tag: str, npz_p: Path, ws: float, mn9_ids: list[int], anat_obj: dict, cand_map: dict[str, list[int]]):
            # Inhibitory candidates (phantom, tentacular, usnea, marge)
            ic = anat_obj["inhibitory_candidates_and_controls"]
            for cname in ("phantom", "tentacular", "usnea", "marge"):
                rec = ic[cname]
                top_ipre = [x["node"]["idx"] for x in rec["top15_hop2_ipre"]]
                top_h3 = rec["top15_hop3_i1_epre"]
                for K in (5, 10):
                    # Oracle-Direct: MN9 held at V_mn9 in {7.5, 8.5} mV, top-K I_pre held at 8.5 mV
                    ip_k = top_ipre[:K]
                    if ip_k:
                        for v_mn9 in (7.5, 8.5):
                            nb = [(m, v_mn9) for m in mn9_ids] + [(u, 8.5) for u in ip_k]
                            oracle_jobs.append({
                                "tag": f"{brain_tag}|{cname}|oracle_direct_K{K}_vmn9_{v_mn9:.1f}",
                                "npz_path": str(npz_p), "w_syn": ws, "node_biases": nb,
                                "watch_idx": mn9_ids + ip_k[:5], "exc_idx": cand_map[cname],
                                "rates_hz": [50.0, 200.0], "n_trials": 30, "seed_bg": 41001, "seed_opto": 42000,
                                "ckpt_path": str(args.ckpt_dir / f"ora_{brain_tag}_{cname}_dir_K{K}_{int(v_mn9*10)}.json"),
                            })
                    # Oracle-Premotor (3-hop): top-K unique (I_1, E_pre) from top15_hop3_i1_epre
                    i1_set, ep_set = [], []
                    for x in top_h3:
                        if x["i1"]["idx"] not in i1_set and len(i1_set) < K:
                            i1_set.append(x["i1"]["idx"])
                        if x["epre"]["idx"] not in ep_set and len(ep_set) < K:
                            ep_set.append(x["epre"]["idx"])
                    if i1_set and ep_set:
                        nb = [(u, 8.5) for u in i1_set] + [(u, 7.8) for u in ep_set if u not in i1_set]
                        oracle_jobs.append({
                            "tag": f"{brain_tag}|{cname}|oracle_premotor_K{K}",
                            "npz_path": str(npz_p), "w_syn": ws, "node_biases": nb,
                            "watch_idx": mn9_ids + i1_set[:3] + ep_set[:3], "exc_idx": cand_map[cname],
                            "rates_hz": [50.0, 200.0], "n_trials": 30, "seed_bg": 41001, "seed_opto": 43000,
                            "ckpt_path": str(args.ckpt_dir / f"ora_{brain_tag}_{cname}_pre_K{K}.json"),
                        })

            # Cholinergic SELK (DNg68): 3-hop (SELK + I_1 - I_2 - MN9) & 4-hop (SELK + I_1 - I_2 - E_pre + MN9)
            sk = anat_obj["selk_dng68"]
            for branch_name, h3_key, h4_key in (
                [("all", "hop3_disinh_all", "hop4_disinh_all")]
                if brain_tag == "v783"
                else [("all", "hop3_disinh_all", "hop4_disinh_all"), ("via_vnc", "hop3_disinh_via_vnc", "hop4_disinh_via_vnc")]
            ):
                i2_list = []
                for x in sk[h3_key]["top15"]:
                    if x["i2"]["idx"] not in i2_list:
                        i2_list.append(x["i2"]["idx"])
                for K in (5, 10):
                    i2_k = i2_list[:K]
                    if i2_k:
                        for v_mn9 in (7.5, 8.5):
                            nb = [(m, v_mn9) for m in mn9_ids] + [(u, 8.5) for u in i2_k]
                            oracle_jobs.append({
                                "tag": f"{brain_tag}|SELK_DNg68|oracle_direct_{branch_name}_K{K}_vmn9_{v_mn9:.1f}",
                                "npz_path": str(npz_p), "w_syn": ws, "node_biases": nb,
                                "watch_idx": mn9_ids + i2_k[:5], "exc_idx": cand_map["SELK_DNg68"],
                                "rates_hz": [50.0, 200.0], "n_trials": 30, "seed_bg": 41001, "seed_opto": 44000,
                                "ckpt_path": str(args.ckpt_dir / f"ora_{brain_tag}_SELK_dir_{branch_name}_K{K}_{int(v_mn9*10)}.json"),
                            })
                    # 4-hop premotor: tonic I_2 (8.5 mV) + tonic E_pre (7.8 mV)
                    i2_4, ep_4 = [], []
                    for x in sk[h4_key]["top15"]:
                        if x["i2"]["idx"] not in i2_4 and len(i2_4) < K:
                            i2_4.append(x["i2"]["idx"])
                        if x["best_epre"]["idx"] not in ep_4 and len(ep_4) < K:
                            ep_4.append(x["best_epre"]["idx"])
                    if i2_4 and ep_4:
                        nb = [(u, 8.5) for u in i2_4] + [(u, 7.8) for u in ep_4 if u not in i2_4]
                        oracle_jobs.append({
                            "tag": f"{brain_tag}|SELK_DNg68|oracle_premotor_{branch_name}_K{K}",
                            "npz_path": str(npz_p), "w_syn": ws, "node_biases": nb,
                            "watch_idx": mn9_ids + i2_4[:3] + ep_4[:3], "exc_idx": cand_map["SELK_DNg68"],
                            "rates_hz": [50.0, 200.0], "n_trials": 30, "seed_bg": 41001, "seed_opto": 45000,
                            "ckpt_path": str(args.ckpt_dir / f"ora_{brain_tag}_SELK_pre_{branch_name}_K{K}.json"),
                        })

        v783_cand_map = {c: v783_sez_map[c] for c in ("phantom", "tentacular", "usnea", "marge")}
        v783_cand_map["SELK_DNg68"] = selk_783
        banc_cand_map = {c: banc_sez_map[c] for c in ("phantom", "tentacular", "usnea", "marge")}
        banc_cand_map["SELK_DNg68"] = selk_banc

        add_oracle_suite("v783", v783_npz, w_syn_v783, mn9_783, anat_v783, v783_cand_map)
        add_oracle_suite("banc", args.banc_npz, w_syn_banc_anat, mn9_banc, anat_banc, banc_cand_map)

        # Cross-individual BANC v888 representative background confirmations (focal + positive refs)
        banc_bg_specs = [
            ("banc_F1_global_sub_5.50", "global", {"v": 5.50}, [], 0.0),
            ("banc_F1_global_supra_7.30", "global", {"v": 7.30}, [], 0.0),
            ("banc_F2_ei_E7.45_I8.00", "ei", {"ve": 7.45, "vi": 8.00}, [], 0.0),
            ("banc_F3a_sugar_LB3b40_100hz", "none", {}, lb3b_r_banc, 100.0),
            ("banc_F3a_water_LB3a14_160hz", "none", {}, lb3a_r_banc, 160.0),
        ]
        banc_bg_futs = [
            ex.submit(_single_bg_worker, {
                "cond": bname, "npz_path": str(args.banc_npz), "w_syn": w_syn_banc_anat,
                "bg_mode": bm, "bg_params": bp, "bg_exc_idx": be, "bg_exc_rate_hz": br,
                "watch_idx": mn9_banc, "n_trials": 30, "seed_bg": 51001,
                "ckpt_path": str(args.ckpt_dir / f"bg_{bname}_30tr.json"),
            })
            for bname, bm, bp, be, br in banc_bg_specs
        ]
        ora_futs = [ex.submit(_oracle_job_worker, j) for j in oracle_jobs]

        banc_bg_out = {f.result()["cond"]: f.result() for f in concurrent.futures.as_completed(banc_bg_futs)}
        oracle_out = {f.result()["tag"]: f.result() for f in concurrent.futures.as_completed(ora_futs)}
        print(f"  [oracles] {len(oracle_out)} local realizability bounds finished", flush=True)

        # Submit BANC focal + ref confirmation slices under the 5 BANC background regimes
        banc_focal_names = ["phantom", "tentacular", "SELK_DNg68", "marge", "usnea", "roundup", "rattle", "Fdg"]
        banc_focal_items = []
        for fn in banc_focal_names:
            eidx = selk_banc if fn == "SELK_DNg68" else banc_sez_map[fn]
            for f_hz in (50.0, 200.0):
                banc_focal_items.append({
                    "name": f"{fn}_{int(f_hz)}hz", "exc_idx": eidx, "rate_hz": f_hz,
                    "seed_opto": 95000 + banc_focal_names.index(fn) * 1000 + int(f_hz),
                })
        banc_slice_futs = []
        for bname, bm, bp, be, br in banc_bg_specs:
            # Split into 4-item slices for fast concurrency
            for s_i in range(0, len(banc_focal_items), 4):
                banc_slice_futs.append(ex.submit(_condition_slice_worker, {
                    "cond": bname, "slice_id": s_i, "npz_path": str(args.banc_npz), "w_syn": w_syn_banc_anat,
                    "bg_mode": bm, "bg_params": bp, "bg_exc_idx": be, "bg_exc_rate_hz": br,
                    "watch_idx": mn9_banc, "n_trials": 30, "seed_bg": 51001,
                    "bg_counts": banc_bg_out[bname]["bg_counts"],
                    "items": banc_focal_items[s_i : s_i + 4],
                    "ckpt_path": str(args.ckpt_dir / f"banc_conf_{bname}_{s_i:02d}.json"),
                }))
        banc_conf_res: dict[str, dict] = defaultdict(dict)
        for fut in concurrent.futures.as_completed(banc_slice_futs):
            r = fut.result()
            banc_conf_res[r["cond"]].update(r["results"])
        print(f"  [BANC conf] 5 background regimes x {len(banc_focal_items)} focal/ref runs done", flush=True)

    print("[5/5] Aggregating metrics, specificity checks & writing JSON...", flush=True)
    # Zero-baseline v783 reference metrics (from mn9_transfer_results.json)
    v783_ref_50 = np.array([float(r["v783_real_lif_50hz_mean"]) for r in v783_res["per_type_task1"]], dtype=np.float64)
    v783_zero_ap = float(smb.average_precision_exact(y_bin, v783_ref_50))
    v783_zero_auc = float(smb.auroc_exact(y_bin, v783_ref_50))

    v783_cond_summary = {}
    for cname, bmode, bparams, bg_e, bg_r, ntr in v783_conditions:
        fmap = cond_full_res[cname]
        deltas = np.array([fmap[name]["delta_mean_hz"] for name in sez_names], dtype=np.float64)
        ap = float(smb.average_precision_exact(y_bin, deltas))
        auc = float(smb.auroc_exact(y_bin, deltas))
        cv = smb.evaluate_heldout_cv(y_bin, deltas)
        # Count significant net-positive FPs among 92 negatives (delta >= 1.0 Hz & paired_p < 0.01)
        sig_neg_fps = [
            {"name": name, **fmap[name]}
            for i, name in enumerate(sez_names)
            if y_bin[i] == 0 and fmap[name]["unlocked_ge1hz_p01"]
        ]
        sig_neg_fps.sort(key=lambda x: -x["delta_mean_hz"])
        sig_pos_tps = [
            {"name": name, **fmap[name]}
            for i, name in enumerate(sez_names)
            if y_bin[i] == 1 and fmap[name]["unlocked_ge1hz_p01"]
        ]
        specificity_broken = bool((v783_zero_ap - ap) > 0.05 or len(sig_neg_fps) >= 5)
        v783_cond_summary[cname] = {
            "bg_mode": bmode,
            "bg_params": bparams,
            "bg_exc_rate_hz": bg_r,
            "n_trials_full106": ntr,
            "bg_mn9_l_hz": cond_bg_20[cname]["bg_mn9_l_hz"],
            "bg_mn9_r_hz": cond_bg_20[cname]["bg_mn9_r_hz"],
            "bg_mn9_mean_hz": cond_bg_20[cname]["bg_mn9_mean_hz"],
            "brain_mean_hz": cond_bg_20[cname]["brain_mean_hz"],
            "all_106_delta_metrics": {
                "auprc": ap,
                "auprc_diff_vs_zero_bg": float(ap - v783_zero_ap),
                "auroc": auc,
                "loocv_f1": float(cv["loocv"]["f1"]),
                "loocv_mcc": float(cv["loocv"]["mcc"]),
                "spearman_max": smb.spearman_safe(y_max, deltas),
                "n_sig_pos_ge1hz_p01": len(sig_pos_tps),
                "n_sig_neg_fp_ge1hz_p01": len(sig_neg_fps),
                "specificity_broken": specificity_broken,
                "top10_sig_neg_fps": sig_neg_fps[:10],
            },
            "focal_30tr": cond_focal_30_res[cname],
            "per_type_50hz": fmap,
        }
        f30 = cond_focal_30_res[cname]
        print(
            f"  [{cname:22s}] bg={cond_bg_20[cname]['bg_mn9_mean_hz']:5.2f}Hz | "
            f"AP={ap:.4f} (d={ap-v783_zero_ap:+.4f}) AUC={auc:.4f} negFP={len(sig_neg_fps):2d} "
            f"| phan={f30['phantom_50hz']['delta_mean_hz']:+5.2f}/{f30['phantom_200hz']['delta_mean_hz']:+5.2f} "
            f"| tent={f30['tentacular_50hz']['delta_mean_hz']:+5.2f}/{f30['tentacular_200hz']['delta_mean_hz']:+5.2f} "
            f"| SELK={f30['SELK_DNg68_50hz']['delta_mean_hz']:+5.2f}/{f30['SELK_DNg68_200hz']['delta_mean_hz']:+5.2f}",
            flush=True,
        )

    # Record transparent local-CPU vs Colab-GPU checkpoint provenance & equivalence verification
    cpu_ckpts, gpu_ckpts = [], []
    for p in sorted(args.ckpt_dir.glob("*.json")):
        if p.name == "mn9_disinh_cloud_manifest.json":
            continue
        try:
            obj = json.load(open(p))
            if obj.get("compute_backend") == "colab_gpu_cuda_fp64":
                gpu_ckpts.append(p.name)
            else:
                cpu_ckpts.append(p.name)
        except Exception:
            continue
    gpu_equiv_data = None
    if args.gpu_equiv_json is not None and args.gpu_equiv_json.is_file():
        gpu_equiv_data = json.load(open(args.gpu_equiv_json))

    payload = {
        "protocol_commit": PROTOCOL_COMMIT,
        "compute_provenance": {
            "n_local_cpu_checkpoints": len(cpu_ckpts),
            "n_colab_gpu_checkpoints": len(gpu_ckpts),
            "local_cpu_checkpoints": cpu_ckpts,
            "colab_gpu_checkpoints": gpu_ckpts,
            "gpu_equivalence_check": gpu_equiv_data,
        },
        "v783_zero_bg_reference": {"auprc_all106": v783_zero_ap, "auroc_all106": v783_zero_auc, "n_purely_inh_neg_ctrl": len(inh_neg_names)},
        "module1_anatomy": {
            "v783": anat_v783,
            "banc": anat_banc,
            "percentiles_vs_44_inh_neg_v783": pct_v783,
            "percentiles_vs_44_inh_neg_banc": pct_banc,
            "cross_brain_intermediate_overlap": cross_overlap,
        },
        "module2_v783_background_states": v783_cond_summary,
        "module2_local_oracle_realizability": oracle_out,
        "module2_banc_confirmation": {
            bname: {
                "bg_mn9_l_hz": banc_bg_out[bname]["bg_mn9_l_hz"],
                "bg_mn9_r_hz": banc_bg_out[bname]["bg_mn9_r_hz"],
                "bg_mn9_mean_hz": banc_bg_out[bname]["bg_mn9_mean_hz"],
                "brain_mean_hz": banc_bg_out[bname]["brain_mean_hz"],
                "focal_and_refs_30tr": banc_conf_res[bname],
            }
            for bname, *_ in banc_bg_specs
        },
        "wall_time_sec": float(time.time() - t_start),
    }
    _save_json_atomic(args.output, payload)
    print(f"DONE in {payload['wall_time_sec']:.1f}s -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
