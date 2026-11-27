#!/usr/bin/env python3
"""Colab A100 PyTorch CUDA (FP64) companion runner for mn9-disinhibition-state-01.

Implements exact discrete Brian2 LIF dynamics matching `mn9_disinhibition_state.run_lif_dc_paired`
(bit-identical MT19937 initial u(0) + bg_exc Poisson stream and exact uint64 xorshift64* opto
Poisson stream, batched across trials/items with event-driven outgoing-CSR synaptic delivery).

Execution flow:
1. Load cached graphs (`v783_graph.npz`, `banc_main_graph.npz`) and `mn9_disinh_cloud_manifest.json`
   from `$RESEARCH_CACHE_DIR/projc-flywire/`.
2. Run CPU-vs-GPU equivalence verification FIRST on the 2 completed CPU reference slices
   (`F1_global_supra_7.30` dense supra + `F3a_sugar_bg_40hz` sparse gustatory) and write
   `$JOB_OUT_DIR/gpu_equivalence_check.json`. Fail fast if discrepancy exceeds strict tolerance.
3. Run all remaining `v783` slices (15), all `v783` + `BANC` local Oracle realizability bounds (66),
   and all 5 `BANC v888` background baselines + 20 focal/ref confirmation slices (25).
4. Bundle all 106 newly produced checkpoint JSONs into ONE tarball `$JOB_OUT_DIR/disinh_gpu_ckpts.tar.gz`
   in the exact schema expected by `mn9_disinhibition_state.py`.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats


# ---------------------------------------------------------------------------
# Fast exact uint64 xorshift64* opto Poisson kick generator (matches Numba CPU)
# ---------------------------------------------------------------------------
_XORSHIFT_C_SRC = r"""
#include <stdint.h>
int gen_xorshift_kicks(
    uint64_t seed_opto, int n_trials, int n_exc, double p_poi, int n_steps,
    int32_t *out_step, int32_t *out_k, int32_t *out_tr, int max_out
) {
    const double inv_2_53 = 1.0 / 9007199254740992.0;
    int cnt = 0;
    for (int tr = 0; tr < n_trials; tr++) {
        uint64_t s = seed_opto + (uint64_t)tr * 65537ULL + 0x9E3779B97F4A7C15ULL;
        if (s == 0ULL) s = 0x9E3779B97F4A7C15ULL;
        for (int step = 0; step < n_steps; step++) {
            for (int k = 0; k < n_exc; k++) {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                uint64_t val = (s * 2685821657736338717ULL) >> 11;
                if (((double)val * inv_2_53) < p_poi) {
                    if (cnt < max_out) {
                        out_step[cnt] = step;
                        out_k[cnt] = k;
                        out_tr[cnt] = tr;
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
_XORSHIFT_TMPDIR = None


def _get_xorshift_lib():
    global _XORSHIFT_LIB, _XORSHIFT_TMPDIR
    if _XORSHIFT_LIB is not None:
        return _XORSHIFT_LIB
    try:
        _XORSHIFT_TMPDIR = tempfile.TemporaryDirectory(prefix="xorshift_so_")
        c_path = os.path.join(_XORSHIFT_TMPDIR.name, "xorshift.c")
        so_path = os.path.join(_XORSHIFT_TMPDIR.name, "xorshift.so")
        with open(c_path, "w") as f:
            f.write(_XORSHIFT_C_SRC)
        subprocess.check_call(["gcc", "-O3", "-shared", "-fPIC", c_path, "-o", so_path])
        lib = ctypes.CDLL(so_path)
        lib.gen_xorshift_kicks.argtypes = [
            ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
        ]
        lib.gen_xorshift_kicks.restype = ctypes.c_int
        _XORSHIFT_LIB = lib
        return lib
    except Exception as e:
        print(f"[warn] gcc xorshift helper unavailable ({e}); using numpy fallback", flush=True)
        _XORSHIFT_LIB = False
        return False


def gen_opto_kicks(seed_opto: int, n_trials: int, exc_idx: list[int], rate_hz: float, n_steps: int = 10000):
    """Return (steps, nodes, trials) int32 arrays bit-identical to run_lif_dc_paired xorshift64*."""
    n_exc = len(exc_idx)
    p_poi = float(rate_hz) * 1e-3 * 0.1
    if n_exc == 0 or p_poi <= 0.0:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty
    exc_arr = np.asarray(exc_idx, dtype=np.int32)
    lib = _get_xorshift_lib()
    if lib:
        est = max(4096, int(n_trials * n_exc * n_steps * p_poi * 1.5) + 4096)
        while True:
            os_a = np.empty(est, dtype=np.int32)
            ok_a = np.empty(est, dtype=np.int32)
            ot_a = np.empty(est, dtype=np.int32)
            cnt = lib.gen_xorshift_kicks(
                int(seed_opto), int(n_trials), int(n_exc), float(p_poi), int(n_steps),
                os_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ok_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ot_a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                int(est),
            )
            if cnt <= est:
                return os_a[:cnt], exc_arr[ok_a[:cnt]], ot_a[:cnt]
            est = cnt + 1024

    # Pure-NumPy uint64 fallback (vectorized over n_trials)
    inv_2_53 = 1.0 / 9007199254740992.0
    s = np.array([int(seed_opto) + tr * 65537 + 0x9E3779B97F4A7C15 for tr in range(n_trials)], dtype=np.uint64)
    s[s == 0] = np.uint64(0x9E3779B97F4A7C15)
    c_mul = np.uint64(2685821657736338717)
    st_list, nd_list, tr_list = [], [], []
    for step in range(n_steps):
        for k in range(n_exc):
            s ^= s >> np.uint64(12)
            s ^= s << np.uint64(25)
            s ^= s >> np.uint64(27)
            val = (s * c_mul) >> np.uint64(11)
            hit = np.flatnonzero((val.astype(np.float64) * inv_2_53) < p_poi)
            if hit.size:
                st_list.append(np.full(hit.size, step, dtype=np.int32))
                nd_list.append(np.full(hit.size, exc_arr[k], dtype=np.int32))
                tr_list.append(hit.astype(np.int32))
    if not st_list:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty, empty
    return np.concatenate(st_list), np.concatenate(nd_list), np.concatenate(tr_list)


def gen_bg_init_and_kicks(
    n_nodes: int,
    v_bias_np: np.ndarray,
    init_u_max: float,
    bg_exc_idx: list[int],
    bg_exc_rate_hz: float,
    n_trials: int,
    seed_bg: int,
    n_steps: int = 10000,
):
    """Generate exact MT19937 u_init (N, n_trials) and bg Poisson kicks (steps, nodes, trials)."""
    pos_idx = np.flatnonzero(v_bias_np > 0.0) if init_u_max > 0.0 else np.empty(0, dtype=np.int64)
    u_init = np.zeros((n_nodes, n_trials), dtype=np.float64)
    p_bg = float(bg_exc_rate_hz) * 1e-3 * 0.1
    bg_arr = np.asarray(bg_exc_idx, dtype=np.int32)
    n_bg = len(bg_arr)
    st_list, nd_list, tr_list = [], [], []

    for tr in range(n_trials):
        np.random.seed(int(seed_bg) + tr * 1009)
        if pos_idx.size > 0:
            u_init[pos_idx, tr] = np.random.random(pos_idx.size) * float(init_u_max)
        if p_bg > 0.0 and n_bg > 0:
            hits = np.random.random((n_steps, n_bg)) < p_bg
            hs, hk = np.nonzero(hits)
            if hs.size > 0:
                st_list.append(hs.astype(np.int32))
                nd_list.append(bg_arr[hk])
                tr_list.append(np.full(hs.size, tr, dtype=np.int32))

    if st_list:
        bg_kicks = (np.concatenate(st_list), np.concatenate(nd_list), np.concatenate(tr_list))
    else:
        empty = np.empty(0, dtype=np.int32)
        bg_kicks = (empty, empty, empty)
    return u_init, bg_kicks


def make_vbias_np(n_nodes: int, node_class: np.ndarray, mode: str, params: dict) -> tuple[np.ndarray, float]:
    vb = np.zeros(n_nodes, dtype=np.float64)
    init_u_max = 0.0
    if mode == "global":
        v = float(params["v"])
        vb[:] = v
        init_u_max = 7.0 if v >= 7.0 else 0.0
    elif mode == "ei":
        ve = float(params["ve"])
        vi = float(params["vi"])
        vb[:] = ve
        vb[node_class == 1] = vi
        init_u_max = 7.0 if (ve >= 7.0 or vi >= 7.0) else 0.0
    elif mode == "local_oracle":
        for idx_i, val in params["node_biases"]:
            vb[int(idx_i)] = float(val)
        init_u_max = 7.0
    return vb, init_u_max


def _paired_stats(on_counts: np.ndarray, bg_counts: np.ndarray) -> dict:
    """Exact replica of mn9_disinhibition_state._paired_stats."""
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


def _save_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# GPU Graph & Exact Batched Event-Driven FP64 LIF Simulator
# ---------------------------------------------------------------------------
class GPUGraph:
    def __init__(self, npz_path: Path, w_syn: float, device: torch.device):
        d = np.load(npz_path)
        self.n_nodes = int(d["n_nodes"])
        self.node_class = d["node_class"].copy()
        self.w_syn = float(w_syn)
        indptr_np = d["indptr"].astype(np.int64)
        deg_np = indptr_np[1:] - indptr_np[:-1]
        post_np = d["post_s"].astype(np.int64)
        # Exact weight precision matching mn9_disinhibition_state._get_graph line 295
        w_np = (d["w_raw_s"] * np.float32(w_syn)).astype(np.float64)

        self.device = device
        self.indptr = torch.from_numpy(indptr_np).to(device)
        self.deg = torch.from_numpy(deg_np).to(device)
        self.post_s = torch.from_numpy(post_np).to(device)
        self.w_s = torch.from_numpy(w_np).to(device)


def _lif_step_with_vb(
    u: torch.Tensor, g: torch.Tensor, unlock_step: torch.Tensor, step_i32: torch.Tensor,
    vb_scaled: torch.Tensor, zero_f64: torch.Tensor, a_v: float, b_vg: float, a_g: float, v_th: float,
):
    elig = unlock_step <= step_i32
    u_new = (u * a_v + vb_scaled) + g * b_vg
    u_out = torch.where(elig, u_new, zero_f64)
    g_out = g * a_g
    spiked = u_out > v_th
    return u_out, g_out, elig, spiked


def _lif_step_no_vb(
    u: torch.Tensor, g: torch.Tensor, unlock_step: torch.Tensor, step_i32: torch.Tensor,
    zero_f64: torch.Tensor, a_v: float, b_vg: float, a_g: float, v_th: float,
):
    elig = unlock_step <= step_i32
    u_new = u * a_v + g * b_vg
    u_out = torch.where(elig, u_new, zero_f64)
    g_out = g * a_g
    spiked = u_out > v_th
    return u_out, g_out, elig, spiked


_COMPILED_STEP_VB = None
_COMPILED_STEP_NOVB = None


def _get_step_fns():
    global _COMPILED_STEP_VB, _COMPILED_STEP_NOVB
    if _COMPILED_STEP_VB is None:
        try:
            _COMPILED_STEP_VB = torch.compile(_lif_step_with_vb, dynamic=True)
            _COMPILED_STEP_NOVB = torch.compile(_lif_step_no_vb, dynamic=True)
        except Exception as e:
            print(f"[warn] torch.compile unavailable ({e}); using eager mode", flush=True)
            _COMPILED_STEP_VB = _lif_step_with_vb
            _COMPILED_STEP_NOVB = _lif_step_no_vb
    return _COMPILED_STEP_VB, _COMPILED_STEP_NOVB


@torch.no_grad()
def run_batched_lif_gpu(
    gg: GPUGraph,
    u_init_np: np.ndarray,           # (N, B) float64
    vb_np: np.ndarray | None,        # None, (N, 1), or (N, B) float64
    is_exc_np: np.ndarray,           # (N, B) bool
    kick_steps: np.ndarray,          # (K,) int32
    kick_nodes: np.ndarray,          # (K,) int32
    kick_cols: np.ndarray,           # (K,) int32
    watch_idx_mat_np: np.ndarray,    # (W, B) int64
    watch_valid_mat_np: np.ndarray,  # (W, B) bool
    track_total: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run B trials/items simultaneously on CUDA GPU with exact Brian2 FP64 LIF dynamics."""
    device = gg.device
    N, B = u_init_np.shape
    dt = 0.1
    n_steps = 10000
    dly_steps = 18
    rfc_steps = 22
    ring_len = dly_steps + 1
    t_mbr = 20.0
    tau = 5.0
    a_v = math.exp(-dt / t_mbr)
    a_g = math.exp(-dt / tau)
    b_vg = (tau / (t_mbr - tau)) * (a_v - a_g)
    one_minus_av = 1.0 - a_v
    v_th = 7.0
    w_poi = gg.w_syn * 250.0

    u = torch.from_numpy(np.ascontiguousarray(u_init_np)).to(device=device, dtype=torch.float64)
    g = torch.zeros((N, B), dtype=torch.float64, device=device)
    u_flat = u.view(-1)
    g_flat = g.view(-1)

    has_vb = vb_np is not None and bool(np.any(vb_np != 0.0))
    if has_vb:
        vb_scaled = torch.from_numpy(np.ascontiguousarray(vb_np * one_minus_av)).to(device=device, dtype=torch.float64)
    else:
        vb_scaled = None

    is_exc_flat = torch.from_numpy(np.ascontiguousarray(is_exc_np)).to(device=device, dtype=torch.bool).view(-1)
    unlock_step = torch.full((N, B), -100000, dtype=torch.int32, device=device)
    unlock_flat = unlock_step.view(-1)
    neg_inf_i32 = torch.tensor(-100000, dtype=torch.int32, device=device)
    zero_f64 = torch.tensor(0.0, dtype=torch.float64, device=device)
    step_cur_tbl = torch.arange(0, n_steps, dtype=torch.int32, device=device)
    step_unlock_tbl = torch.arange(rfc_steps, n_steps + rfc_steps, dtype=torch.int32, device=device)
    step_fn_vb, step_fn_novb = _get_step_fns()

    watch_idx_mat = torch.from_numpy(np.ascontiguousarray(watch_idx_mat_np)).to(device=device, dtype=torch.int64)
    watch_valid_mat = torch.from_numpy(np.ascontiguousarray(watch_valid_mat_np)).to(device=device, dtype=torch.bool)
    watch_counts = torch.zeros(watch_idx_mat.shape, dtype=torch.int32, device=device)
    total_spikes = torch.zeros((B,), dtype=torch.int64, device=device)

    # Pre-sort external Poisson kicks by step (zero host-device sync inside loop!)
    if kick_steps.size > 0:
        order = np.argsort(kick_steps, kind="mergesort")
        ks_s = kick_steps[order]
        kick_ptr = np.searchsorted(ks_s, np.arange(n_steps + 1, dtype=np.int32))
        lin_k_np = kick_nodes[order].astype(np.int64) * B + kick_cols[order].astype(np.int64)
        kick_lin = torch.from_numpy(lin_k_np).to(device=device, dtype=torch.int64)
        max_k_step = int(np.max(np.diff(kick_ptr)))
        kick_w_buf = torch.full((max(1, max_k_step),), w_poi, dtype=torch.float64, device=device)
    else:
        kick_ptr = np.zeros(n_steps + 1, dtype=np.int64)
        kick_lin = None
        kick_w_buf = None

    ring_src: list[torch.Tensor | None] = [None] * ring_len
    ring_col: list[torch.Tensor | None] = [None] * ring_len

    indptr = gg.indptr
    deg = gg.deg
    post_s = gg.post_s
    w_s = gg.w_s

    for step in range(n_steps):
        # 1-2-3. Fused eligibility + exact FP64 decay + threshold crossing
        try:
            if has_vb:
                u, g, elig, spiked = step_fn_vb(
                    u, g, unlock_step, step_cur_tbl[step], vb_scaled, zero_f64, a_v, b_vg, a_g, v_th
                )
            else:
                u, g, elig, spiked = step_fn_novb(
                    u, g, unlock_step, step_cur_tbl[step], zero_f64, a_v, b_vg, a_g, v_th
                )
        except Exception as e:
            print(f"[warn] compiled step failed ({e}); falling back to eager mode", flush=True)
            step_fn_vb, step_fn_novb = _lif_step_with_vb, _lif_step_no_vb
            if has_vb:
                u, g, elig, spiked = step_fn_vb(
                    u, g, unlock_step, step_cur_tbl[step], vb_scaled, zero_f64, a_v, b_vg, a_g, v_th
                )
            else:
                u, g, elig, spiked = step_fn_novb(
                    u, g, unlock_step, step_cur_tbl[step], zero_f64, a_v, b_vg, a_g, v_th
                )
        u_flat = u.view(-1)
        g_flat = g.view(-1)
        elig_flat = elig.view(-1)

        # Ring-buffer scheduling
        src_s, col_s = spiked.nonzero(as_tuple=True)
        fut_slot = (step + dly_steps) % ring_len
        ring_src[fut_slot] = src_s
        ring_col[fut_slot] = col_s

        if src_s.numel() > 0:
            watch_counts.add_((spiked.gather(0, watch_idx_mat) & watch_valid_mat).to(torch.int32))
            if track_total:
                total_spikes.add_(torch.bincount(col_s, minlength=B))

        # 4. External Poisson kicks (bg + opto)
        k0 = int(kick_ptr[step])
        k1 = int(kick_ptr[step + 1])
        if k0 < k1:
            u_flat.index_add_(0, kick_lin[k0:k1], kick_w_buf[: k1 - k0])

        # 5. Delayed synaptic delivery (event-driven outgoing CSR scatter)
        cur_slot = step % ring_len
        src_d = ring_src[cur_slot]
        col_d = ring_col[cur_slot]
        ring_src[cur_slot] = None
        ring_col[cur_slot] = None
        if src_d is not None and src_d.numel() > 0:
            d = deg[src_d]
            total_e = int(d.sum().item())
            if total_e > 0:
                starts = indptr[src_d]
                cum = torch.cumsum(d, 0)
                base = cum - d
                e_idx = torch.arange(total_e, device=device, dtype=torch.int64) + torch.repeat_interleave(
                    starts - base, d, output_size=total_e
                )
                dst = post_s[e_idx]
                w_e = w_s[e_idx]
                c_e = torch.repeat_interleave(col_d, d, output_size=total_e)
                lin = dst * B + c_e
                ok = elig_flat[lin]
                g_flat.index_add_(0, lin[ok], w_e[ok])

        # 6. Spike reset (AFTER Poisson kicks & synaptic delivery, matching CPU lines 264-268)
        if src_s.numel() > 0:
            lin_s = src_s * B + col_s
            u_flat[lin_s] = 0.0
            g_flat[lin_s] = 0.0
            unlock_flat[lin_s] = torch.where(is_exc_flat[lin_s], neg_inf_i32, step_unlock_tbl[step])

    return watch_counts.cpu().numpy(), total_spikes.cpu().numpy()


# ---------------------------------------------------------------------------
# High-level batching helpers for uniform-bg slices, bg-only runs, and oracles
# ---------------------------------------------------------------------------
def run_uniform_bg_group(
    gg: GPUGraph,
    bg_mode: str,
    bg_params: dict,
    bg_exc_idx: list[int],
    bg_exc_rate_hz: float,
    watch_idx: list[int],
    n_trials: int,
    seed_bg: int,
    items: list[dict],
    max_items_per_batch: int,
    include_bg_run: bool = False,
    track_total_for_bg: bool = False,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, np.ndarray]]:
    """Run optional baseline + a list of opto items sharing the same background regime."""
    N = gg.n_nodes
    vb_np, init_u_max = make_vbias_np(N, gg.node_class, bg_mode, bg_params)
    u_init_tr, (bg_st, bg_nd, bg_tr) = gen_bg_init_and_kicks(
        N, vb_np, init_u_max, bg_exc_idx, bg_exc_rate_hz, n_trials, seed_bg
    )

    run_specs = []
    if include_bg_run:
        run_specs.append({"_is_bg": True, "name": "__BG__", "exc_idx": [], "rate_hz": 0.0, "seed_opto": 12345})
    for it in items:
        run_specs.append({"_is_bg": False, **it})

    bg_counts_out = None
    bg_tot_out = None
    item_counts: dict[str, np.ndarray] = {}
    W = len(watch_idx)
    watch_arr = np.asarray(watch_idx, dtype=np.int64)[:, None]

    for start in range(0, len(run_specs), max_items_per_batch):
        chunk = run_specs[start : start + max_items_per_batch]
        n_it = len(chunk)
        B = n_it * n_trials
        u_init_b = np.empty((N, B), dtype=np.float64)
        is_exc_b = np.zeros((N, B), dtype=bool)
        ks_l, kn_l, kc_l = [], [], []

        for j, sp in enumerate(chunk):
            c0 = j * n_trials
            c1 = c0 + n_trials
            u_init_b[:, c0:c1] = u_init_tr
            if bg_exc_idx:
                is_exc_b[bg_exc_idx, c0:c1] = True
            if sp["exc_idx"]:
                is_exc_b[sp["exc_idx"], c0:c1] = True
            if bg_st.size > 0:
                ks_l.append(bg_st)
                kn_l.append(bg_nd)
                kc_l.append(bg_tr + c0)
            if sp["exc_idx"] and float(sp["rate_hz"]) > 0.0:
                os_s, os_n, os_t = gen_opto_kicks(int(sp["seed_opto"]), n_trials, sp["exc_idx"], float(sp["rate_hz"]))
                if os_s.size > 0:
                    ks_l.append(os_s)
                    kn_l.append(os_n)
                    kc_l.append(os_t + c0)

        if ks_l:
            k_steps = np.concatenate(ks_l)
            k_nodes = np.concatenate(kn_l)
            k_cols = np.concatenate(kc_l)
        else:
            empty = np.empty(0, dtype=np.int32)
            k_steps, k_nodes, k_cols = empty, empty, empty

        watch_mat = np.broadcast_to(watch_arr, (W, B)).copy()
        valid_mat = np.ones((W, B), dtype=bool)
        vb_arg = vb_np[:, None] if np.any(vb_np != 0.0) else None

        w_cnt, tot_spk = run_batched_lif_gpu(
            gg, u_init_b, vb_arg, is_exc_b, k_steps, k_nodes, k_cols,
            watch_mat, valid_mat, track_total=(track_total_for_bg and any(sp["_is_bg"] for sp in chunk)),
        )
        # w_cnt shape: (W, B) -> slice each item as (n_trials, W)
        for j, sp in enumerate(chunk):
            c0 = j * n_trials
            c1 = c0 + n_trials
            arr = w_cnt[:, c0:c1].T.copy()  # (n_trials, W)
            if sp["_is_bg"]:
                bg_counts_out = arr
                bg_tot_out = tot_spk[c0:c1].copy()
            else:
                item_counts[sp["name"]] = arr

    return bg_counts_out, bg_tot_out, item_counts


def run_oracle_batch_gpu(gg: GPUGraph, jobs: list[dict], out_dir: Path) -> None:
    """Batch multiple sparse local-oracle jobs (each: bg + 50Hz + 200Hz x 30tr = 90 cols) on GPU."""
    t0 = time.time()
    N = gg.n_nodes
    # All oracle jobs in a batch have n_trials=30 and 2 rates [50.0, 200.0] -> 3 runs per job
    col_records = []
    max_w = max(len(j["watch_idx"]) for j in jobs)
    total_cols = sum(int(j["n_trials"]) * (1 + len(j["rates_hz"])) for j in jobs)

    u_init_b = np.zeros((N, total_cols), dtype=np.float64)
    vb_b = np.zeros((N, total_cols), dtype=np.float64)
    is_exc_b = np.zeros((N, total_cols), dtype=bool)
    watch_mat = np.zeros((max_w, total_cols), dtype=np.int64)
    valid_mat = np.zeros((max_w, total_cols), dtype=bool)
    ks_l, kn_l, kc_l = [], [], []

    col_cursor = 0
    for j_idx, job in enumerate(jobs):
        n_tr = int(job["n_trials"])
        seed_bg = int(job["seed_bg"])
        vb_j, init_u_max = make_vbias_np(N, gg.node_class, "local_oracle", {"node_biases": job["node_biases"]})
        u_init_j, _ = gen_bg_init_and_kicks(N, vb_j, init_u_max, [], 0.0, n_tr, seed_bg)
        w_idx = job["watch_idx"]
        W_j = len(w_idx)
        exc_idx = job["exc_idx"]

        # Run 0: baseline (exc_rate = 0, exc_idx = empty)
        runs = [("bg", 0.0, [], 12345)]
        for r_hz in job["rates_hz"]:
            runs.append((f"{int(r_hz)}hz", float(r_hz), exc_idx, int(job["seed_opto"]) + int(r_hz)))

        job_slices = {}
        for r_key, r_hz, e_idx, s_opto in runs:
            c0 = col_cursor
            c1 = col_cursor + n_tr
            col_cursor = c1
            u_init_b[:, c0:c1] = u_init_j
            vb_b[:, c0:c1] = vb_j[:, None]
            if e_idx:
                is_exc_b[e_idx, c0:c1] = True
            watch_mat[:W_j, c0:c1] = np.asarray(w_idx, dtype=np.int64)[:, None]
            valid_mat[:W_j, c0:c1] = True
            if e_idx and r_hz > 0.0:
                os_s, os_n, os_t = gen_opto_kicks(s_opto, n_tr, e_idx, r_hz)
                if os_s.size > 0:
                    ks_l.append(os_s)
                    kn_l.append(os_n)
                    kc_l.append(os_t + c0)
            job_slices[r_key] = (c0, c1, W_j)
        col_records.append((job, job_slices))

    if ks_l:
        k_steps = np.concatenate(ks_l)
        k_nodes = np.concatenate(kn_l)
        k_cols = np.concatenate(kc_l)
    else:
        empty = np.empty(0, dtype=np.int32)
        k_steps, k_nodes, k_cols = empty, empty, empty

    w_cnt, _ = run_batched_lif_gpu(
        gg, u_init_b, vb_b, is_exc_b, k_steps, k_nodes, k_cols, watch_mat, valid_mat, track_total=False
    )
    elapsed_each = float((time.time() - t0) / len(jobs))

    for job, job_slices in col_records:
        c0_bg, c1_bg, W_j = job_slices["bg"]
        bg_counts = w_cnt[:W_j, c0_bg:c1_bg].T.copy()
        rates_out = {}
        for r_hz in job["rates_hz"]:
            r_key = f"{int(r_hz)}hz"
            c0, c1, _ = job_slices[r_key]
            on_counts = w_cnt[:W_j, c0:c1].T.copy()
            st = _paired_stats(on_counts, bg_counts)
            if W_j > 2:
                st["watched_intermed_bg_hz"] = bg_counts[:, 2:].mean(axis=0).tolist()
                st["watched_intermed_on_hz"] = on_counts[:, 2:].mean(axis=0).tolist()
            rates_out[r_key] = st
        out = {
            "tag": job["tag"],
            "bg_mn9_l_hz": float(bg_counts[:, 0].mean()),
            "bg_mn9_r_hz": float(bg_counts[:, 1].mean()),
            "bg_mn9_mean_hz": float(bg_counts[:, :2].mean()),
            "rates": rates_out,
            "elapsed_sec": elapsed_each,
            "compute_backend": "colab_gpu_cuda_fp64",
        }
        _save_json_atomic(out_dir / job["ckpt_name"], out)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-gpu-gb", type=float, default=0.0, help="Require GPU VRAM >= threshold (e.g. 30 for A100)")
    cli_args, _ = parser.parse_known_args()

    t_all = time.time()
    cache_dir = Path(os.environ.get("RESEARCH_CACHE_DIR", "/content/drive/MyDrive/ResearchCloud/cache")) / "projc-flywire"
    out_dir = Path(os.environ.get("JOB_OUT_DIR", "/tmp/mn9_disinh_gpu_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/tmp/disinh_gpu_work")
    work_dir.mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available(), "CUDA GPU is required on Colab runtime"
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[init] GPU={gpu_name} ({gpu_mem_gb:.1f} GB), torch={torch.__version__}, cache={cache_dir}", flush=True)
    if cli_args.min_gpu_gb > 0.0 and gpu_mem_gb < cli_args.min_gpu_gb:
        raise RuntimeError(f"GPU VRAM {gpu_mem_gb:.1f} GB < required {cli_args.min_gpu_gb:.1f} GB ({gpu_name}); aborting so A100 can run.")

    # Auto-tune batch sizes for A100 (40/80 GB) vs smaller GPUs (T4/L4), preserving exact FP64 parity
    is_a100_class = gpu_mem_gb >= 35.0
    dense_items_batch = 5 if is_a100_class else 2
    sparse_items_batch = 16 if is_a100_class else 11
    oracle_jobs_batch = 6 if is_a100_class else 3
    print(
        f"[init] batch profile: a100_class={is_a100_class} "
        f"(dense_items={dense_items_batch}, sparse_items={sparse_items_batch}, oracle_jobs={oracle_jobs_batch})",
        flush=True,
    )

    manifest = json.load(open(cache_dir / "mn9_disinh_cloud_manifest.json"))
    w_syn_v783 = float(manifest["w_syn_v783"])
    w_syn_banc = float(manifest["w_syn_banc_anat"])

    # =======================================================================
    # STAGE 1: FlyWire v783 (Equivalence check + 15 missing slices + 30 oracles)
    # =======================================================================
    print("\n[Stage 1] Loading FlyWire v783 graph onto CUDA...", flush=True)
    gg783 = GPUGraph(cache_dir / "v783_graph.npz", w_syn_v783, device)

    # 1A: Mandatory CPU vs GPU Equivalence Verification
    print("[Stage 1A] Running CPU-vs-GPU equivalence verification on 2 completed CPU slices...", flush=True)
    equiv_report = {
        "gpu_name": gpu_name,
        "gpu_mem_gb": round(gpu_mem_gb, 2),
        "torch_version": torch.__version__,
        "refs": {},
    }
    for ref in manifest["equiv_refs"]:
        t_eq = time.time()
        rname = ref["ref_name"]
        is_dense = ref["bg_mode"] in ("global", "ei")
        bg_gpu, _, items_gpu = run_uniform_bg_group(
            gg783, ref["bg_mode"], ref["bg_params"],
            ref["bg_exc_idx"], float(ref["bg_exc_rate_hz"]),
            ref["watch_idx"], int(ref["n_trials"]), int(ref["seed_bg"]),
            ref["items"], max_items_per_batch=(5 if is_dense else 12),
            include_bg_run=True, track_total_for_bg=False,
        )
        bg_cpu = np.asarray(ref["bg_counts"], dtype=np.int32)
        bg_abs_diff = np.abs(bg_gpu - bg_cpu)
        per_item_cmp = {}
        max_delta_diff = 0.0
        max_on_diff = 0.0
        for it in ref["items"]:
            nm = it["name"]
            gpu_st = _paired_stats(items_gpu[nm], bg_gpu)
            cpu_st = ref["cpu_expected"][nm]
            d_diff = abs(gpu_st["delta_mean_hz"] - cpu_st["delta_mean_hz"])
            o_diff = abs(gpu_st["on_mean_hz"] - cpu_st["on_mean_hz"])
            max_delta_diff = max(max_delta_diff, d_diff)
            max_on_diff = max(max_on_diff, o_diff)
            per_item_cmp[nm] = {
                "cpu_delta_hz": cpu_st["delta_mean_hz"],
                "gpu_delta_hz": gpu_st["delta_mean_hz"],
                "abs_diff_delta_hz": d_diff,
                "cpu_on_hz": cpu_st["on_mean_hz"],
                "gpu_on_hz": gpu_st["on_mean_hz"],
                "abs_diff_on_hz": o_diff,
            }
        rec = {
            "bg_exact_trial_match_frac": float(np.mean(np.all(bg_abs_diff == 0, axis=1))),
            "bg_max_abs_spike_diff": int(np.max(bg_abs_diff)),
            "bg_mean_hz_cpu": float(bg_cpu[:, :2].mean()),
            "bg_mean_hz_gpu": float(bg_gpu[:, :2].mean()),
            "max_abs_diff_delta_hz": float(max_delta_diff),
            "max_abs_diff_on_hz": float(max_on_diff),
            "elapsed_sec": float(time.time() - t_eq),
            "per_item": per_item_cmp,
        }
        equiv_report["refs"][rname] = rec
        print(
            f"  [equiv {rname}] bg_exact={rec['bg_exact_trial_match_frac']*100:.1f}% "
            f"(max_spk_diff={rec['bg_max_abs_spike_diff']}), "
            f"max_abs_diff_delta={max_delta_diff:.6f} Hz, max_abs_diff_on={max_on_diff:.6f} Hz "
            f"({rec['elapsed_sec']:.1f}s)",
            flush=True,
        )
        tol = 0.50 if is_dense else 0.05
        assert max_delta_diff <= tol, f"Equivalence check FAILED for {rname}: max_delta_diff={max_delta_diff} > {tol}"

    equiv_report["passed"] = True
    _save_json_atomic(out_dir / "gpu_equivalence_check.json", equiv_report)
    print("[Stage 1A] Equivalence check PASSED and saved!", flush=True)

    # 1B: 15 Missing v783 slices
    print(f"\n[Stage 1B] Running {len(manifest['missing_v783_slices'])} missing v783 slices...", flush=True)
    for idx_s, sl in enumerate(manifest["missing_v783_slices"], 1):
        t_s = time.time()
        is_dense = sl["bg_mode"] in ("global", "ei")
        bg_counts = np.asarray(sl["bg_counts"], dtype=np.int32)
        non_empty = [it for it in sl["items"] if len(it["exc_idx"]) > 0]
        _, _, item_counts = run_uniform_bg_group(
            gg783, sl["bg_mode"], sl["bg_params"],
            sl["bg_exc_idx"], float(sl["bg_exc_rate_hz"]),
            sl["watch_idx"], int(sl["n_trials"]), int(sl["seed_bg"]),
            non_empty, max_items_per_batch=(dense_items_batch if is_dense else sparse_items_batch),
            include_bg_run=False, track_total_for_bg=False,
        )
        results = {}
        for it in sl["items"]:
            nm = it["name"]
            on_c = item_counts[nm] if nm in item_counts else bg_counts.copy()
            results[nm] = _paired_stats(on_c, bg_counts)
        out_payload = {
            "cond": sl["cond"],
            "slice_id": sl["slice_id"],
            "results": results,
            "elapsed_sec": float(time.time() - t_s),
            "compute_backend": "colab_gpu_cuda_fp64",
        }
        _save_json_atomic(work_dir / sl["ckpt_name"], out_payload)
        print(f"  [v783 {idx_s:2d}/{len(manifest['missing_v783_slices'])}] {sl['ckpt_name']} ({out_payload['elapsed_sec']:.1f}s)", flush=True)

    # 1C: v783 Local Oracle jobs (30 jobs)
    v783_oracles = [j for j in manifest["oracle_jobs"] if j["graph"] == "v783"]
    print(f"\n[Stage 1C] Running {len(v783_oracles)} v783 local Oracle jobs (batch={oracle_jobs_batch})...", flush=True)
    t_o7 = time.time()
    for i in range(0, len(v783_oracles), oracle_jobs_batch):
        batch = v783_oracles[i : i + oracle_jobs_batch]
        run_oracle_batch_gpu(gg783, batch, work_dir)
        print(f"  [v783 oracles {min(i + oracle_jobs_batch, len(v783_oracles)):2d}/{len(v783_oracles)}] elapsed={time.time()-t_o7:.1f}s", flush=True)

    # Free v783 graph before loading BANC
    del gg783
    torch.cuda.empty_cache()

    # =======================================================================
    # STAGE 2: BANC v888 (36 local oracles + 5 backgrounds + 20 confirmation slices)
    # =======================================================================
    print("\n[Stage 2] Loading BANC v888 graph onto CUDA...", flush=True)
    gg_banc = GPUGraph(cache_dir / "banc_main_graph.npz", w_syn_banc, device)

    # 2A: BANC Local Oracle jobs (36 jobs)
    banc_oracles = [j for j in manifest["oracle_jobs"] if j["graph"] == "banc"]
    print(f"[Stage 2A] Running {len(banc_oracles)} BANC local Oracle jobs (batch={oracle_jobs_batch})...", flush=True)
    t_ob = time.time()
    for i in range(0, len(banc_oracles), oracle_jobs_batch):
        batch = banc_oracles[i : i + oracle_jobs_batch]
        run_oracle_batch_gpu(gg_banc, batch, work_dir)
        print(f"  [BANC oracles {min(i + oracle_jobs_batch, len(banc_oracles)):2d}/{len(banc_oracles)}] elapsed={time.time()-t_ob:.1f}s", flush=True)

    # 2B: BANC 5 background baselines + 20 focal/ref confirmation slices
    print(f"\n[Stage 2B] Running {len(manifest['banc_bg_specs'])} BANC backgrounds + 20 confirmation slices...", flush=True)
    mn9_banc = manifest["mn9_banc"]
    banc_focal_items = manifest["banc_focal_items"]
    for bspec in manifest["banc_bg_specs"]:
        t_b = time.time()
        bname = bspec["cond"]
        is_dense_supra = (
            (bspec["bg_mode"] == "global" and float(bspec["bg_params"].get("v", 0.0)) >= 7.0)
            or (bspec["bg_mode"] == "ei" and max(float(bspec["bg_params"].get("ve", 0.0)), float(bspec["bg_params"].get("vi", 0.0))) >= 7.0)
        )
        # 1. Baseline (n_trials=30, seed_bg=51001, track_total=True)
        bg_c, bg_tot, _ = run_uniform_bg_group(
            gg_banc, bspec["bg_mode"], bspec["bg_params"],
            bspec["bg_exc_idx"], float(bspec["bg_exc_rate_hz"]),
            mn9_banc, 30, 51001, [], max_items_per_batch=1,
            include_bg_run=True, track_total_for_bg=True,
        )
        bg_payload = {
            "cond": bname,
            "bg_counts": bg_c.tolist(),
            "bg_mn9_l_hz": float(bg_c[:, 0].mean()),
            "bg_mn9_r_hz": float(bg_c[:, 1].mean()),
            "bg_mn9_mean_hz": float(bg_c[:, :2].mean()),
            "brain_mean_hz": float(bg_tot.mean() / gg_banc.n_nodes),
            "elapsed_sec": float(time.time() - t_b),
            "compute_backend": "colab_gpu_cuda_fp64",
        }
        _save_json_atomic(work_dir / f"bg_{bname}_30tr.json", bg_payload)
        print(
            f"  [BANC bg {bname:28s}] MN9={bg_payload['bg_mn9_mean_hz']:5.2f} Hz "
            f"(L={bg_payload['bg_mn9_l_hz']:5.2f} R={bg_payload['bg_mn9_r_hz']:5.2f}) "
            f"brain={bg_payload['brain_mean_hz']:.3f} Hz ({bg_payload['elapsed_sec']:.1f}s)",
            flush=True,
        )

        # 2. All 16 focal/ref items, saved into 4 exact slices (00, 04, 08, 12)
        t_f = time.time()
        non_empty_b = [it for it in banc_focal_items if len(it["exc_idx"]) > 0]
        _, _, item_counts_b = run_uniform_bg_group(
            gg_banc, bspec["bg_mode"], bspec["bg_params"],
            bspec["bg_exc_idx"], float(bspec["bg_exc_rate_hz"]),
            mn9_banc, 30, 51001, non_empty_b,
            max_items_per_batch=(4 if (is_dense_supra and is_a100_class) else (2 if is_dense_supra else sparse_items_batch)),
            include_bg_run=False, track_total_for_bg=False,
        )
        slice_sec = float((time.time() - t_f) / 4.0)
        for s_i in range(0, len(banc_focal_items), 4):
            sub = banc_focal_items[s_i : s_i + 4]
            res_s = {}
            for it in sub:
                nm = it["name"]
                on_c = item_counts_b[nm] if nm in item_counts_b else bg_c.copy()
                res_s[nm] = _paired_stats(on_c, bg_c)
            _save_json_atomic(work_dir / f"banc_conf_{bname}_{s_i:02d}.json", {
                "cond": bname,
                "slice_id": s_i,
                "results": res_s,
                "elapsed_sec": slice_sec,
                "compute_backend": "colab_gpu_cuda_fp64",
            })
        print(f"  [BANC focal {bname:25s}] 16 items done ({time.time() - t_f:.1f}s)", flush=True)

    # =======================================================================
    # STAGE 3: Pack all 106 newly created JSON checkpoints into ONE tarball
    # =======================================================================
    json_files = sorted(work_dir.glob("*.json"))
    tar_path = out_dir / "disinh_gpu_ckpts.tar.gz"
    tar_tmp = out_dir / "disinh_gpu_ckpts.tar.gz.tmp"
    with tarfile.open(tar_tmp, "w:gz") as tf:
        for jf in json_files:
            tf.add(jf, arcname=jf.name)
    tar_tmp.replace(tar_path)

    total_sec = float(time.time() - t_all)
    print(
        f"\n[DONE] Packed {len(json_files)} GPU checkpoint JSONs -> {tar_path} "
        f"({tar_path.stat().st_size:,} B) in {total_sec:.1f}s!",
        flush=True,
    )


if __name__ == "__main__":
    main()
