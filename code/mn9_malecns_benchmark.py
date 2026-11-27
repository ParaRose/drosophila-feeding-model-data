#!/usr/bin/env python3
"""Third-connectome cross-modality, cross-sex zero-parameter transfer benchmark: MaleCNS v1.0.

Implements `MN9_MALECNS_PROTOCOL.md` (commit 8a01636 + pre-simulation QC addendum fd5d254):
  - Downloads official MaleCNS v1.0 flat-connectome feathers from GCS (with SHA-256 logging)
  - Builds Traced non-glial directed chemical-synapse graph (same E/I sign rules as FlyWire/BANC)
  - Documents 10.8x hemispheric MN9 reconstruction asymmetry (MN9_L=6,012 'Roughly traced' vs
    MN9_R=556 'RT Hard to trace') and evaluates all 3 zero-behavior-fitting weight scales:
      (1) frozen_0.275   : W_syn = 0.275 mV
      (2) anat_intact    : W_syn = 0.275 * (5732.5 / S_in_MN9_L)  [~0.2622 mV, intact arbor]
      (3) anat_bilat     : W_syn = 0.275 * (5732.5 / S_in_mean)   [~0.4800 mV, blind bilateral mean]
  - Pre-execution CPU-vs-GPU FP64 exact spike parity check (max_abs_spike_diff == 0)
  - Runs ALL real, K=10 full rewired, K=10 MN9-protected rewired, SELK, and 36+12 local Oracle
    simulations on A100 CUDA FP64 with OOM-proof chunked delivery + live progress logging.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import tempfile
import time
import urllib.request
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.feather as pf
import torch
from numba import njit
from scipy import sparse, stats

GCS_BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
MALECNS_FILES = {
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
}

INH_NTS = {"gaba", "glutamate"}
EXC_NTS = {"acetylcholine", "dopamine", "serotonin", "octopamine", "tyramine", "histamine"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_gcs_file(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or dest.stat().st_size == 0:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        print(f"[download] {url} -> {dest}", flush=True)
        t0 = time.time()
        try:
            subprocess.check_call(["curl", "-fSL", "--retry", "3", "-o", str(tmp), url])
        except Exception:
            urllib.request.urlretrieve(url, str(tmp))
        tmp.replace(dest)
        print(f"  downloaded {dest.stat().st_size / 1e6:.1f} MB in {time.time() - t0:.1f}s", flush=True)
    else:
        print(f"[cache hit] {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)", flush=True)
    return sha256_file(dest)


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


# ---------------------------------------------------------------------------
# Fast exact uint64 xorshift64* Poisson kick generator (bit-identical to CPU)
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
    except Exception:
        _XORSHIFT_LIB = False
        return False


def gen_opto_kicks(seed_opto: int, n_trials: int, exc_idx: list[int], rate_hz: float, n_steps: int = 10000):
    n_exc = len(exc_idx)
    p_poi = float(rate_hz) * 1e-3 * 0.1
    if n_exc == 0 or p_poi <= 0.0:
        e = np.empty(0, dtype=np.int32)
        return e, e, e
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
    inv_2_53 = 1.0 / 9007199254740992.0
    s = np.array([int(seed_opto) + tr * 65537 + 0x9E3779B97F4A7C15 for tr in range(n_trials)], dtype=np.uint64)
    s[s == 0] = np.uint64(0x9E3779B97F4A7C15)
    c_mul = np.uint64(2685821657736338717)
    st_l, nd_l, tr_l = [], [], []
    for step in range(n_steps):
        for k in range(n_exc):
            s ^= s >> np.uint64(12)
            s ^= s << np.uint64(25)
            s ^= s >> np.uint64(27)
            val = (s * c_mul) >> np.uint64(11)
            hit = np.flatnonzero((val.astype(np.float64) * inv_2_53) < p_poi)
            if hit.size:
                st_l.append(np.full(hit.size, step, dtype=np.int32))
                nd_l.append(np.full(hit.size, exc_arr[k], dtype=np.int32))
                tr_l.append(hit.astype(np.int32))
    if not st_l:
        e = np.empty(0, dtype=np.int32)
        return e, e, e
    return np.concatenate(st_l), np.concatenate(nd_l), np.concatenate(tr_l)


# ---------------------------------------------------------------------------
# Numba CPU reference FP64 LIF (for exact CPU-vs-GPU parity verification)
# ---------------------------------------------------------------------------
@njit
def run_lif_dc_cpu_ref(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    w_syn: float,
    v_bias: np.ndarray,
    init_u_max: float,
    exc_idx: np.ndarray,
    exc_rate_hz: float,
    watch_idx: np.ndarray,
    n_trials: int,
    seed_bg: int,
    seed_opto: int,
) -> np.ndarray:
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

    n_nodes = len(indptr) - 1
    n_watch = len(watch_idx)
    watch_counts = np.zeros((n_trials, n_watch), dtype=np.int32)

    is_exc = np.zeros(n_nodes, dtype=np.uint8)
    for k in range(len(exc_idx)):
        is_exc[exc_idx[k]] = 1

    vb_scaled = np.empty(n_nodes, dtype=np.float64)
    n_pos_vb = 0
    for i in range(n_nodes):
        vb_scaled[i] = one_minus_av * v_bias[i]
        if v_bias[i] > 0.0:
            n_pos_vb += 1
    pos_vb_idx = np.empty(n_pos_vb, dtype=np.int32)
    c_p = 0
    for i in range(n_nodes):
        if v_bias[i] > 0.0:
            pos_vb_idx[c_p] = i
            c_p += 1

    u = np.zeros(n_nodes, dtype=np.float64)
    g = np.zeros(n_nodes, dtype=np.float64)
    last_spk = np.full(n_nodes, -100000, dtype=np.int32)
    ring_len = dly_steps + 1
    ring_spk = np.empty((ring_len, n_nodes), dtype=np.int32)
    ring_cnt = np.zeros(ring_len, dtype=np.int32)
    spiked_now = np.empty(n_nodes, dtype=np.int32)
    inv_2_53 = 1.0 / 9007199254740992.0

    for tr in range(n_trials):
        for i in range(n_nodes):
            u[i] = 0.0
            g[i] = 0.0
            last_spk[i] = -100000
        for slot in range(ring_len):
            ring_cnt[slot] = 0

        np.random.seed(seed_bg + tr * 1009)
        if init_u_max > 0.0 and n_pos_vb > 0:
            for k in range(n_pos_vb):
                u[pos_vb_idx[k]] = np.random.random() * init_u_max

        s_opto = np.uint64(seed_opto) + np.uint64(tr) * np.uint64(65537) + np.uint64(0x9E3779B97F4A7C15)
        if s_opto == np.uint64(0):
            s_opto = np.uint64(0x9E3779B97F4A7C15)

        for step in range(n_steps):
            n_spk_now = 0
            for i in range(n_nodes):
                if is_exc[i] != 0 or (step - last_spk[i]) >= rfc_steps:
                    ui = (a_v * u[i] + vb_scaled[i]) + b_vg * g[i]
                    gi = a_g * g[i]
                    u[i] = ui
                    g[i] = gi
                    if ui > v_th:
                        spiked_now[n_spk_now] = i
                        n_spk_now += 1
                else:
                    u[i] = 0.0
                    g[i] = a_g * g[i]

            fut_slot = (step + dly_steps) % ring_len
            for s_i in range(n_spk_now):
                i = spiked_now[s_i]
                c_f = ring_cnt[fut_slot]
                ring_spk[fut_slot, c_f] = i
                ring_cnt[fut_slot] = c_f + 1
                for w_i in range(n_watch):
                    if watch_idx[w_i] == i:
                        watch_counts[tr, w_i] += 1

            if p_poi > 0.0 and len(exc_idx) > 0:
                for k in range(len(exc_idx)):
                    s_opto ^= s_opto >> np.uint64(12)
                    s_opto ^= s_opto << np.uint64(25)
                    s_opto ^= s_opto >> np.uint64(27)
                    val = (s_opto * np.uint64(2685821657736338717)) >> np.uint64(11)
                    if float(val) * inv_2_53 < p_poi:
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
            ring_cnt[cur_slot] = 0

            for s_i in range(n_spk_now):
                i = spiked_now[s_i]
                u[i] = 0.0
                g[i] = 0.0
                last_spk[i] = step

    return watch_counts


# ---------------------------------------------------------------------------
# OOM-Proof Batched FP64 CUDA Engine (event-driven CSR + dense SpMM fallback)
# ---------------------------------------------------------------------------
class GPUGraph:
    def __init__(self, n_nodes: int, indptr: np.ndarray, post_s: np.ndarray, w_raw_s: np.ndarray, node_class: np.ndarray, device: torch.device):
        self.n_nodes = int(n_nodes)
        self.node_class = node_class.copy()
        indptr_i64 = indptr.astype(np.int64)
        deg_np = indptr_i64[1:] - indptr_i64[:-1]
        self.device = device
        self.indptr = torch.from_numpy(indptr_i64).to(device)
        self.deg = torch.from_numpy(deg_np).to(device)
        self.post_s = torch.from_numpy(post_s.astype(np.int64)).to(device)
        self.w_raw_f64 = torch.from_numpy(w_raw_s.astype(np.float64)).to(device)
        # Also prepare transpose CSC->CSR for instant dense SpMM if a supercritical regime spikes >20k/step
        self._w_t_csr = None
        self._post_np = post_s
        self._indptr_np = indptr
        self._w_raw_np = w_raw_s

    def get_wt_csr(self) -> torch.Tensor:
        if self._w_t_csr is None:
            # Build W^T (post x pre) in CSR on GPU for fast dense SpMM: (W^T @ spk_f64)
            m_csc = sparse.csc_matrix(
                (self._w_raw_np.astype(np.float64), self._post_np, self._indptr_np),
                shape=(self.n_nodes, self.n_nodes),
            )
            wt_csr = m_csc.tocsr()  # shape (post, pre), exact M[dst, src] = w_{src -> dst}
            crow = torch.from_numpy(wt_csr.indptr.astype(np.int64)).to(self.device)
            col = torch.from_numpy(wt_csr.indices.astype(np.int64)).to(self.device)
            val = torch.from_numpy(wt_csr.data.astype(np.float64)).to(self.device)
            self._w_t_csr = torch.sparse_csr_tensor(crow, col, val, size=(self.n_nodes, self.n_nodes), device=self.device)
        return self._w_t_csr

    def update_post_indices(self, new_post: np.ndarray) -> None:
        self.post_s = torch.from_numpy(new_post.astype(np.int64)).to(self.device)
        self._post_np = new_post
        self._w_t_csr = None


def _lif_step_with_vb(u, g, unlock_step, step_i32, vb_scaled, zero_f64, a_v: float, b_vg: float, a_g: float, v_th: float):
    elig = unlock_step <= step_i32
    u_new = (u * a_v + vb_scaled) + g * b_vg
    u_out = torch.where(elig, u_new, zero_f64)
    g_out = g * a_g
    spiked = u_out > v_th
    return u_out, g_out, elig, spiked


def _lif_step_no_vb(u, g, unlock_step, step_i32, zero_f64, a_v: float, b_vg: float, a_g: float, v_th: float):
    elig = unlock_step <= step_i32
    u_new = u * a_v + g * b_vg
    u_out = torch.where(elig, u_new, zero_f64)
    g_out = g * a_g
    spiked = u_out > v_th
    return u_out, g_out, elig, spiked


_COMPILED_VB = None
_COMPILED_NOVB = None


def _get_step_fns():
    global _COMPILED_VB, _COMPILED_NOVB
    if _COMPILED_VB is None:
        try:
            _COMPILED_VB = torch.compile(_lif_step_with_vb, dynamic=True)
            _COMPILED_NOVB = torch.compile(_lif_step_no_vb, dynamic=True)
        except Exception:
            _COMPILED_VB = _lif_step_with_vb
            _COMPILED_NOVB = _lif_step_no_vb
    return _COMPILED_VB, _COMPILED_NOVB


@torch.no_grad()
def run_batched_lif_gpu(
    gg: GPUGraph,
    w_syn: float,
    u_init_np: np.ndarray,
    vb_np: np.ndarray | None,
    is_exc_np: np.ndarray,
    kick_steps: np.ndarray,
    kick_nodes: np.ndarray,
    kick_cols: np.ndarray,
    watch_idx_mat_np: np.ndarray,
    watch_valid_mat_np: np.ndarray,
    track_total: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run B columns on A100 FP64. Uses event-driven CSR scatter when sparse and SpMM if dense."""
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
    w_poi = float(w_syn) * 250.0
    w_s = gg.w_raw_f64 * float(w_syn)

    u = torch.from_numpy(np.ascontiguousarray(u_init_np)).to(device=device, dtype=torch.float64)
    g = torch.zeros((N, B), dtype=torch.float64, device=device)
    u_flat = u.view(-1)
    g_flat = g.view(-1)

    has_vb = vb_np is not None and bool(np.any(vb_np != 0.0))
    vb_scaled = (
        torch.from_numpy(np.ascontiguousarray(vb_np * one_minus_av)).to(device=device, dtype=torch.float64)
        if has_vb
        else None
    )

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
    spk_dense_buf: torch.Tensor | None = None

    for step in range(n_steps):
        try:
            if has_vb:
                u, g, elig, spiked = step_fn_vb(u, g, unlock_step, step_cur_tbl[step], vb_scaled, zero_f64, a_v, b_vg, a_g, v_th)
            else:
                u, g, elig, spiked = step_fn_novb(u, g, unlock_step, step_cur_tbl[step], zero_f64, a_v, b_vg, a_g, v_th)
        except Exception:
            step_fn_vb, step_fn_novb = _lif_step_with_vb, _lif_step_no_vb
            if has_vb:
                u, g, elig, spiked = step_fn_vb(u, g, unlock_step, step_cur_tbl[step], vb_scaled, zero_f64, a_v, b_vg, a_g, v_th)
            else:
                u, g, elig, spiked = step_fn_novb(u, g, unlock_step, step_cur_tbl[step], zero_f64, a_v, b_vg, a_g, v_th)

        u_flat = u.view(-1)
        g_flat = g.view(-1)
        elig_flat = elig.view(-1)

        src_s, col_s = spiked.nonzero(as_tuple=True)
        fut_slot = (step + dly_steps) % ring_len
        ring_src[fut_slot] = src_s
        ring_col[fut_slot] = col_s

        if src_s.numel() > 0:
            watch_counts.add_((spiked.gather(0, watch_idx_mat) & watch_valid_mat).to(torch.int32))
            if track_total:
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
                    e_idx = torch.arange(total_e, device=device, dtype=torch.int64) + torch.repeat_interleave(
                        starts - base, d, output_size=total_e
                    )
                    dst = post_s[e_idx]
                    w_e = w_s[e_idx]
                    c_e = torch.repeat_interleave(col_d, d, output_size=total_e)
                    lin = dst * B + c_e
                    ok = elig_flat[lin]
                    g_flat.index_add_(0, lin[ok], w_e[ok])
            else:
                # Dense supercritical step: O(1)-memory cuSPARSE SpMM in FP64 with chunked fallback
                try:
                    if spk_dense_buf is None:
                        spk_dense_buf = torch.zeros((N, B), dtype=torch.float64, device=device)
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
                            ei_c = torch.arange(tot_c, device=device, dtype=torch.int64) + torch.repeat_interleave(st_c - (cm_c - d_c), d_c, output_size=tot_c)
                            lin_c = post_s[ei_c] * B + torch.repeat_interleave(cd_c, d_c, output_size=tot_c)
                            ok_c = elig_flat[lin_c]
                            g_flat.index_add_(0, lin_c[ok_c], w_s[ei_c][ok_c])

        if src_s.numel() > 0:
            lin_s = src_s * B + col_s
            u_flat[lin_s] = 0.0
            g_flat[lin_s] = 0.0
            unlock_flat[lin_s] = torch.where(is_exc_flat[lin_s], neg_inf_i32, step_unlock_tbl[step])

    return watch_counts.cpu().numpy(), total_spikes.cpu().numpy()


def run_task1_sweep_gpu(
    gg: GPUGraph,
    w_syn: float,
    type_indices: list[list[int]],
    rate_hz: float,
    n_trials: int,
    seed_base: int,
    mn9_indices: list[int],
    batch_types: int = 27,
    track_total: bool = False,
) -> dict:
    """Evaluate all 106 SEZ types on A100 GPU in OOM-safe batches (works equally fast on sparse or supercritical)."""
    n_types = len(type_indices)
    N = gg.n_nodes
    means = np.zeros(n_types, dtype=np.float64)
    l_rates = np.zeros(n_types, dtype=np.float64)
    r_rates = np.zeros(n_types, dtype=np.float64)
    brain_hz_per_type = np.zeros(n_types, dtype=np.float64)

    active_types = [i for i, idxs in enumerate(type_indices) if len(idxs) > 0]
    for b0 in range(0, len(active_types), batch_types):
        sub_i = active_types[b0 : b0 + batch_types]
        B = len(sub_i) * n_trials
        u_init = np.zeros((N, B), dtype=np.float64)
        is_exc = np.zeros((N, B), dtype=bool)
        watch_mat = np.repeat(np.array(mn9_indices, dtype=np.int64)[:, None], B, axis=1)
        valid_mat = np.ones((2, B), dtype=bool)
        ks_l, kn_l, kc_l = [], [], []
        for slot, ti in enumerate(sub_i):
            c0, c1 = slot * n_trials, (slot + 1) * n_trials
            e_idx = type_indices[ti]
            is_exc[e_idx, c0:c1] = True
            os_s, os_n, os_t = gen_opto_kicks(seed_base + ti, n_trials, e_idx, rate_hz)
            if os_s.size > 0:
                ks_l.append(os_s)
                kn_l.append(os_n)
                kc_l.append(os_t + c0)
        if ks_l:
            k_steps, k_nodes, k_cols = np.concatenate(ks_l), np.concatenate(kn_l), np.concatenate(kc_l)
        else:
            e = np.empty(0, dtype=np.int32)
            k_steps, k_nodes, k_cols = e, e, e

        w_cnt, tot_spk = run_batched_lif_gpu(gg, w_syn, u_init, None, is_exc, k_steps, k_nodes, k_cols, watch_mat, valid_mat, track_total=track_total)
        for slot, ti in enumerate(sub_i):
            c0, c1 = slot * n_trials, (slot + 1) * n_trials
            l_rates[ti] = float(w_cnt[0, c0:c1].mean())
            r_rates[ti] = float(w_cnt[1, c0:c1].mean())
            means[ti] = float(w_cnt[:, c0:c1].mean())
            if track_total:
                brain_hz_per_type[ti] = float(tot_spk[c0:c1].mean() / N)

    return {
        "means": means,
        "l_rates": l_rates,
        "r_rates": r_rates,
        "mean_whole_brain_hz": float(brain_hz_per_type.mean()) if track_total else None,
        "max_whole_brain_hz": float(brain_hz_per_type.max()) if track_total else None,
    }


# ---------------------------------------------------------------------------
# Fast Rewiring (CPU Numba parallel, graph generation ONLY), Baselines & Metrics
# ---------------------------------------------------------------------------
@njit
def rewire_block_swaps(indptr, indices, block_edge_ids, edge_src, n_attempts, seed) -> int:
    np.random.seed(seed)
    m = len(block_edge_ids)
    if m < 2:
        return 0
    accepted = 0
    for _ in range(n_attempts):
        a = np.random.randint(0, m)
        b = np.random.randint(0, m)
        if a == b:
            continue
        e1 = block_edge_ids[a]
        e2 = block_edge_ids[b]
        u1 = edge_src[e1]
        u2 = edge_src[e2]
        if u1 == u2:
            continue
        v1 = indices[e1]
        v2 = indices[e2]
        if v1 == v2 or u1 == v2 or u2 == v1:
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


@njit
def verify_rewire_numba(indptr, orig_indices, new_indices, weights, n_nodes):
    self_loops = 0
    dup_edges = 0
    kept_edges = 0
    in_deg_pos = np.zeros(n_nodes, dtype=np.int32)
    in_deg_neg = np.zeros(n_nodes, dtype=np.int32)
    for u in range(n_nodes):
        p0 = indptr[u]
        p1 = indptr[u + 1]
        if p1 <= p0:
            continue
        row_orig = np.sort(orig_indices[p0:p1])
        row_new = np.sort(new_indices[p0:p1])
        for k in range(p0, p1):
            v = new_indices[k]
            if v == u:
                self_loops += 1
            if weights[k] > 0.0:
                in_deg_pos[v] += 1
            else:
                in_deg_neg[v] += 1
        for k in range(1, len(row_new)):
            if row_new[k] == row_new[k - 1]:
                dup_edges += 1
        i = 0
        j = 0
        rn = len(row_orig)
        while i < rn and j < rn:
            a = row_orig[i]
            b = row_new[j]
            if a == b:
                kept_edges += 1
                i += 1
                j += 1
            elif a < b:
                i += 1
            else:
                j += 1
    return self_loops, dup_edges, kept_edges, in_deg_pos, in_deg_neg


_RW_NPZ_CACHE: dict = {}


def _build_one_rewire_worker(job: dict) -> dict:
    t0 = time.time()
    npz_path = job["npz_path"]
    if npz_path not in _RW_NPZ_CACHE:
        d = np.load(npz_path)
        _RW_NPZ_CACHE[npz_path] = {k: d[k] for k in d.files}
    g = _RW_NPZ_CACHE[npz_path]
    indptr, orig_indices, w_raw, edge_src, node_class = g["indptr"], g["post_s"], g["w_raw_s"], g["pre_s"], g["node_class"]
    orig_in_pos, orig_in_neg = g["orig_in_pos"], g["orig_in_neg"]
    protect_targets = set(job["protect_targets"])
    seed = int(job["seed"])

    n_nodes = len(indptr) - 1
    new_indices = orig_indices.copy()
    movable_mask = np.ones(len(orig_indices), dtype=bool)
    for t in protect_targets:
        movable_mask &= orig_indices != t
    pre_cls = node_class[edge_src]
    post_cls = node_class[orig_indices]
    total_attempts, total_accepted = 0, 0
    for pc in (0, 1):
        for qc in (0, 1, 2):
            eids = np.flatnonzero(movable_mask & (pre_cls == pc) & (post_cls == qc)).astype(np.int32)
            if len(eids) < 2:
                continue
            n_att = 2 * len(eids)
            acc = int(rewire_block_swaps(indptr, new_indices, eids, edge_src, n_att, seed * 31 + pc * 7 + qc))
            total_attempts += n_att
            total_accepted += acc

    self_loops, dup_edges, kept_edges, in_pos, in_neg = verify_rewire_numba(indptr, orig_indices, new_indices, w_raw, n_nodes)
    assert self_loops == 0 and dup_edges == 0
    assert np.array_equal(in_pos, orig_in_pos) and np.array_equal(in_neg, orig_in_neg)
    for t in protect_targets:
        assert np.array_equal(orig_indices == t, new_indices == t)

    out_npy = Path(job["out_npy"])
    np.save(out_npy, new_indices)
    n_movable = int(movable_mask.sum())
    n_protected = len(orig_indices) - n_movable
    replaced_movable = n_movable - (kept_edges - n_protected)
    return {
        "family": job["family"],
        "idx": int(job["idx"]),
        "out_npy": str(out_npy),
        "elapsed_sec": float(time.time() - t0),
        "meta": {
            "seed": seed,
            "n_movable": n_movable,
            "n_protected": n_protected,
            "total_attempts": int(total_attempts),
            "total_accepted": int(total_accepted),
            "replaced_movable": int(replaced_movable),
            "replaced_fraction_movable": float(replaced_movable / max(1, n_movable)),
            "self_loops": 0,
            "dup_edges": 0,
            "signed_indegree_conserved": True,
            "hash": hashlib.sha256(new_indices.tobytes()).hexdigest()[:16],
        },
    }


def compute_static_baselines(pre, post, w_raw, n_nodes, type_indices, mn9_indices):
    rev_all = sparse.csr_matrix((np.ones(len(pre), dtype=np.int8), (post, pre)), shape=(n_nodes, n_nodes))
    exc_m = w_raw > 0
    rev_exc = sparse.csr_matrix((np.ones(int(exc_m.sum()), dtype=np.int8), (post[exc_m], pre[exc_m])), shape=(n_nodes, n_nodes))

    def bfs_dist(adj_rev, targets):
        dist = np.full(n_nodes, -1, dtype=np.int32)
        q = list(targets)
        for t in targets:
            dist[t] = 0
        head = 0
        while head < len(q):
            cur = q[head]
            head += 1
            d_next = dist[cur] + 1
            row = adj_rev.indices[adj_rev.indptr[cur] : adj_rev.indptr[cur + 1]]
            for nb in row:
                if dist[nb] == -1:
                    dist[nb] = d_next
                    q.append(int(nb))
        return dist

    dist_all = bfs_dist(rev_all, mn9_indices)
    dist_exc = bfs_dist(rev_exc, mn9_indices)
    W = sparse.csr_matrix((w_raw.astype(np.float64), (pre, post)), shape=(n_nodes, n_nodes))
    W_abs = sparse.csr_matrix((np.abs(w_raw).astype(np.float64), (pre, post)), shape=(n_nodes, n_nodes))
    e_mn9 = np.zeros(n_nodes, dtype=np.float64)
    for m in mn9_indices:
        e_mn9[m] = 1.0
    w1_vec = W.dot(e_mn9)
    w1_abs_vec = W_abs.dot(e_mn9)
    w2_vec = W.dot(w1_vec)
    w2_abs_vec = W_abs.dot(w1_abs_vec)

    n_types = len(type_indices)
    s_neg_sp = np.zeros(n_types, dtype=np.float64)
    s_exc_sp = np.zeros(n_types, dtype=np.float64)
    s_1hop = np.zeros(n_types, dtype=np.float64)
    s_2hop_lin = np.zeros(n_types, dtype=np.float64)
    s_2hop_relu = np.zeros(n_types, dtype=np.float64)
    s_2hop_uns = np.zeros(n_types, dtype=np.float64)
    is_direct = np.zeros(n_types, dtype=bool)

    for i, idxs in enumerate(type_indices):
        if len(idxs) == 0:
            continue
        ds = dist_all[idxs]
        pos_d = ds[ds > 0]
        if len(pos_d) > 0:
            md = int(pos_d.min())
            s_neg_sp[i] = 1.0 / (1.0 + md)
            is_direct[i] = md == 1
        dse = dist_exc[idxs]
        pos_de = dse[dse > 0]
        if len(pos_de) > 0:
            s_exc_sp[i] = 1.0 / (1.0 + int(pos_de.min()))
        h1 = float(w1_vec[idxs].sum())
        h2 = float(w2_vec[idxs].sum())
        h1_abs = float(w1_abs_vec[idxs].sum())
        h2_abs = float(w2_abs_vec[idxs].sum())
        out_k = np.asarray(W[idxs, :].sum(axis=0)).ravel()
        h2_relu = float(np.dot(np.maximum(0.0, out_k), w1_vec))
        s_1hop[i] = h1
        s_2hop_lin[i] = h1 + 0.01 * h2
        s_2hop_relu[i] = h1 + 0.01 * h2_relu
        s_2hop_uns[i] = h1_abs + 0.01 * h2_abs

    return (
        {
            "static_neg_shortest_path": s_neg_sp,
            "static_exc_shortest_path": s_exc_sp,
            "static_1hop_signed": s_1hop,
            "static_2hop_signed_fixed": s_2hop_lin,
            "static_2hop_relu_signed": s_2hop_relu,
            "static_2hop_unsigned_fixed": s_2hop_uns,
        },
        is_direct,
    )


def average_precision_exact(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-y_score, kind="mergesort")
    y_s, s_s = y_true[order], y_score[order]
    distinct_idx = np.where(np.diff(s_s) != 0)[0]
    thr_idxs = np.r_[distinct_idx, len(y_s) - 1]
    tps = np.cumsum(y_s)[thr_idxs]
    fps = (1 + thr_idxs) - tps
    precision = tps / (tps + fps)
    recall = tps / n_pos
    rec_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - rec_prev) * precision))


def auroc_exact(y_true, y_score) -> float:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = stats.rankdata(y_score)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def spearman_safe(y_true, y_score) -> float:
    if len(y_score) < 2 or np.all(y_score == y_score[0]) or np.all(y_true == y_true[0]):
        return 0.0
    r = stats.spearmanr(y_true, y_score).statistic
    return 0.0 if np.isnan(r) else float(r)


def pearson_safe(a, b) -> float:
    if len(a) < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return 0.0
    r = stats.pearsonr(a, b).statistic
    return 0.0 if np.isnan(r) else float(r)


def binary_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec, "f1": f1, "balanced_accuracy": 0.5 * (rec + spec), "mcc": mcc}


def fit_best_f1_threshold(y_train, s_train) -> float:
    uniq = np.unique(s_train)
    cands = [float(uniq[0] - 1.0), float(uniq[-1] + 1.0)] + [0.5 * float(uniq[j] + uniq[j + 1]) for j in range(len(uniq) - 1)]
    cands.sort()
    best_f1, best_thr = -1.0, cands[-1]
    for thr in cands:
        f1 = binary_metrics(y_train, (s_train > thr).astype(np.int32))["f1"]
        if f1 > best_f1 + 1e-12:
            best_f1, best_thr = f1, thr
    return best_thr


def evaluate_heldout_cv(y_bin, scores, seed: int = 20260888) -> dict:
    n = len(y_bin)
    loo_pred = np.zeros(n, dtype=np.int32)
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        loo_pred[i] = int(scores[i] > fit_best_f1_threshold(y_bin[m], scores[m]))
    rng = np.random.default_rng(seed)
    pos_idx, neg_idx = np.flatnonzero(y_bin == 1), np.flatnonzero(y_bin == 0)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    folds = [[] for _ in range(5)]
    for k, idx in enumerate(pos_idx):
        folds[k % 5].append(int(idx))
    for k, idx in enumerate(neg_idx):
        folds[k % 5].append(int(idx))
    cv5_pred = np.zeros(n, dtype=np.int32)
    for f in range(5):
        ti = np.array(folds[f], dtype=np.int32)
        tm = np.ones(n, dtype=bool)
        tm[ti] = False
        cv5_pred[ti] = (scores[ti] > fit_best_f1_threshold(y_bin[tm], scores[tm])).astype(np.int32)
    return {"loocv": binary_metrics(y_bin, loo_pred), "cv5": binary_metrics(y_bin, cv5_pred), "zero_threshold": binary_metrics(y_bin, (scores > 0.0).astype(np.int32))}


def stratified_bootstrap_summary(y_bin, y_max, y_mean, scores_by_model, ref_model, n_boot=2000, seed=20260888):
    rng = np.random.default_rng(seed)
    pos_idx, neg_idx = np.flatnonzero(y_bin == 1), np.flatnonzero(y_bin == 0)
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    boot_idx = np.empty((n_boot, n_pos + n_neg), dtype=np.int32)
    for b in range(n_boot):
        boot_idx[b, :n_pos] = rng.choice(pos_idx, size=n_pos, replace=True)
        boot_idx[b, n_pos:] = rng.choice(neg_idx, size=n_neg, replace=True)
    raw = {m: {k: np.empty(n_boot) for k in ("auprc", "auroc", "rho_max", "rho_mean")} for m in scores_by_model}
    for b in range(n_boot):
        idx = boot_idx[b]
        yb, ymx, ymn = y_bin[idx], y_max[idx], y_mean[idx]
        for m, sc in scores_by_model.items():
            sb = sc[idx]
            raw[m]["auprc"][b] = average_precision_exact(yb, sb)
            raw[m]["auroc"][b] = auroc_exact(yb, sb)
            raw[m]["rho_max"][b] = spearman_safe(ymx, sb)
            raw[m]["rho_mean"][b] = spearman_safe(ymn, sb)
    out = {}
    for m, sc in scores_by_model.items():
        pt = {"auprc": average_precision_exact(y_bin, sc), "auroc": auroc_exact(y_bin, sc), "rho_max": spearman_safe(y_max, sc), "rho_mean": spearman_safe(y_mean, sc)}
        ci, diff = {}, {}
        for k in ("auprc", "auroc", "rho_max", "rho_mean"):
            arr = raw[m][k]
            ci[k] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
            if m != ref_model:
                d = raw[ref_model][k] - arr
                p_two = float(2.0 * min(np.mean(d <= 0.0), np.mean(d >= 0.0)))
                ref_pt = (
                    average_precision_exact(y_bin, scores_by_model[ref_model]) if k == "auprc"
                    else auroc_exact(y_bin, scores_by_model[ref_model]) if k == "auroc"
                    else spearman_safe(y_max, scores_by_model[ref_model]) if k == "rho_max"
                    else spearman_safe(y_mean, scores_by_model[ref_model])
                )
                diff[k] = {"point_diff_ref_minus_model": float(ref_pt - pt[k]), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_two_sided_boot": min(1.0, p_two)}
        out[m] = {"point": pt, "ci95": ci, "ref_minus_this": diff}
    return out


def evaluate_subset_suite(y_bin, y_max, y_mean, subset_mask, real_scores, primary_key, static_scores, rw_full_list, rw_prot_list):
    idx = np.flatnonzero(subset_mask)
    yb, ymax, ymean = y_bin[idx].astype(np.int32), y_max[idx], y_mean[idx]
    n_pos, n_neg = int(yb.sum()), int(len(yb) - yb.sum())
    scores_boot = {k: arr[idx] for k, arr in real_scores.items()}
    for k, arr in static_scores.items():
        scores_boot[k] = arr[idx]
    has_rw = bool(rw_full_list and rw_prot_list)
    if has_rw:
        ens_full = np.mean([arr[idx] for arr in rw_full_list], axis=0)
        ens_prot = np.mean([arr[idx] for arr in rw_prot_list], axis=0)
        scores_boot["rewire_full_ensemble_mean"] = ens_full
        scores_boot["rewire_mn9_prot_ensemble_mean"] = ens_prot
    boot_all = stratified_bootstrap_summary(yb, ymax, ymean, scores_boot, ref_model=primary_key, n_boot=2000, seed=20260888)

    def one_m(s, bkey=None):
        out = {
            "auprc": average_precision_exact(yb, s),
            "auroc": auroc_exact(yb, s),
            "spearman_max": spearman_safe(ymax, s),
            "spearman_mean": spearman_safe(ymean, s),
            "heldout_cv": evaluate_heldout_cv(yb, s),
            "n_nonzero": int((s > 0.0).sum()),
        }
        if bkey is not None and bkey in boot_all:
            out["ci95"] = boot_all[bkey]["ci95"]
            if bkey != primary_key:
                out["celltype_boot_primary_minus_this"] = boot_all[bkey]["ref_minus_this"]
        return out

    models_out = {k: one_m(arr[idx], k) for k, arr in real_scores.items()}
    for k, arr in static_scores.items():
        models_out[k] = one_m(arr[idx], k)

    res = {
        "n_total": int(len(idx)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "random_auprc_baseline": float(n_pos / len(idx)) if len(idx) else 0.0,
        "primary_model_key": primary_key,
        "models": models_out,
    }
    if has_rw:
        def eval_rw(rw_list, ens, ens_key):
            pg = [one_m(arr[idx]) for arr in rw_list]
            aps = [m["auprc"] for m in pg]
            aucs = [m["auroc"] for m in pg]
            f1s = [m["heldout_cv"]["loocv"]["f1"] for m in pg]
            real_ap = models_out[primary_key]["auprc"]
            wins = int(sum(real_ap > a + 1e-12 for a in aps))
            ties = int(sum(abs(real_ap - a) <= 1e-12 for a in aps))
            ng = len(rw_list)
            return {
                "n_graphs": ng,
                "graph_level_rank_vs_primary": {
                    "wins": wins, "ties": ties, "losses": ng - wins - ties,
                    "empirical_p_resolution": float(1.0 / (ng + 1)),
                    "empirical_p_upper": float((ng - wins + 1) / (ng + 1)),
                },
                "per_graph_auprc_mean": float(np.mean(aps)),
                "per_graph_auprc_min": float(np.min(aps)),
                "per_graph_auprc_max": float(np.max(aps)),
                "per_graph_auprc_list": [float(a) for a in aps],
                "per_graph_auroc_mean": float(np.mean(aucs)),
                "per_graph_loocv_f1_mean": float(np.mean(f1s)),
                "ensemble_mean_metrics": one_m(ens, ens_key),
            }
        res["rewire_full"] = eval_rw(rw_full_list, ens_full, "rewire_full_ensemble_mean")
        res["rewire_mn9_protected"] = eval_rw(rw_prot_list, ens_prot, "rewire_mn9_prot_ensemble_mean")
    return res


def shortest_exc_path_with_constraint(
    indptr, post_s, w_raw_s, node_class, sources, targets, allowed_intermediate=None, init_bottlenecks=None
):
    """BFS min-hop purely-excitatory path from sources to targets (maximizing bottleneck within the minimum hop count)."""
    n_nodes = len(indptr) - 1
    dist = np.full(n_nodes, -1, dtype=np.int32)
    best_bn = np.zeros(n_nodes, dtype=np.float64)
    parent = np.full(n_nodes, -1, dtype=np.int32)
    q = deque()
    for s in sources:
        dist[s] = 0
        best_bn[s] = float(init_bottlenecks[s]) if init_bottlenecks is not None else 1e9
        q.append(s)
    found_dist = None
    while q:
        u = q.popleft()
        du = int(dist[u])
        if found_dist is not None and du >= found_dist:
            continue
        if node_class[u] != 0:
            continue
        for e in range(int(indptr[u]), int(indptr[u + 1])):
            w = float(w_raw_s[e])
            if w <= 0:
                continue
            v = int(post_s[e])
            if v not in targets and allowed_intermediate is not None and not allowed_intermediate[v]:
                continue
            cand_bn = min(float(best_bn[u]), w)
            if dist[v] == -1:
                dist[v] = du + 1
                best_bn[v] = cand_bn
                parent[v] = u
                if v in targets:
                    found_dist = du + 1
                else:
                    q.append(v)
            elif dist[v] == du + 1 and cand_bn > best_bn[v]:
                best_bn[v] = cand_bn
                parent[v] = u
    reached = [t for t in sorted(targets) if dist[t] > 0]
    if not reached:
        return {"reachable": False, "min_hops": None, "best_target": None, "bottleneck_weight": 0.0, "path_indices": []}
    best_t = min(reached, key=lambda t: (int(dist[t]), -float(best_bn[t])))
    path, cur = [], best_t
    while cur != -1:
        path.append(int(cur))
        cur = int(parent[cur])
    path.reverse()
    return {"reachable": True, "min_hops": int(dist[best_t]), "best_target": int(best_t), "bottleneck_weight": float(best_bn[best_t]), "path_indices": path}


def analyze_disinhibitory_anatomy(g, node_meta, mn9_indices, inh_groups, selk_indices, is_vnc):
    indptr, pre_s, post_s, w_raw, node_class, n_nodes = g["indptr"], g["pre_s"], g["post_s"], g["w_raw_s"], g["node_class"], g["n_nodes"]
    mn9_set = set(mn9_indices)
    w_to_mn9 = np.zeros(n_nodes, dtype=np.float64)
    for m in mn9_indices:
        mask_m = post_s == m
        np.add.at(w_to_mn9, pre_s[mask_m], w_raw[mask_m].astype(np.float64))
    inh_to_mn9 = np.maximum(0.0, -w_to_mn9)
    exc_to_mn9 = np.maximum(0.0, w_to_mn9)

    inh_to_epre_prod = np.zeros(n_nodes, dtype=np.float64)
    inh_to_epre_max_bn = np.zeros(n_nodes, dtype=np.float64)
    inh_to_epre_best_e = np.full(n_nodes, -1, dtype=np.int32)
    for u in np.flatnonzero(node_class == 1):
        prod_e, mbn_e, best_e = 0.0, 0.0, -1
        for e in range(int(indptr[u]), int(indptr[u + 1])):
            w_abs = float(-w_raw[e])
            if w_abs <= 0:
                continue
            v = int(post_s[e])
            we = float(exc_to_mn9[v])
            if we > 0.0:
                prod_e += w_abs * we
                bn = min(w_abs, we)
                if bn > mbn_e:
                    mbn_e, best_e = bn, v
        inh_to_epre_prod[u] = prod_e
        inh_to_epre_max_bn[u] = mbn_e
        inh_to_epre_best_e[u] = best_e

    def desc(u: int) -> dict:
        m = node_meta.get(int(u), {})
        return {
            "idx": int(u), "id": int(m.get("id", -1)), "cell_type": str(m.get("cell_type", "")),
            "flywire_type": str(m.get("flywire_type", "")), "side": str(m.get("side", "")),
            "region": str(m.get("region", "")), "superclass": str(m.get("superclass", "")), "nt": str(m.get("nt", "")),
        }

    inh_results = {}
    for gname, gidxs in inh_groups.items():
        out_w = defaultdict(float)
        for s in gidxs:
            for e in range(int(indptr[s]), int(indptr[s + 1])):
                out_w[int(post_s[e])] += float(w_raw[e])
        dir_inh_mn9 = float(sum(-out_w[m] for m in mn9_indices if out_w.get(m, 0.0) < 0.0))
        p2_disinh, p2_antag, p3_disinh_e = [], [], []
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
                for e in range(int(indptr[v]), int(indptr[v + 1])):
                    ep = int(post_s[e])
                    we = float(exc_to_mn9[ep])
                    w_ve = float(-w_raw[e])
                    if we > 0.0 and w_ve > 0.0:
                        p3_disinh_e.append((min(w_abs, w_ve, we), w_abs * w_ve * we, w_abs, w_ve, we, v, ep))
        p2_disinh.sort(key=lambda x: (-x[0], -x[1]))
        p2_antag.sort(key=lambda x: (-x[0], -x[1]))
        p3_disinh_e.sort(key=lambda x: (-x[0], -x[1]))
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
        if gname in ("phantom", "tentacular", "usnea", "marge"):
            rec["top15_hop2_ipre"] = [{"bn": float(x[0]), "prod": float(x[1]), "w_x_ipre": float(x[2]), "w_ipre_mn9": float(x[3]), "node": desc(x[4])} for x in p2_disinh[:15]]
            rec["top15_hop2_antag_epre"] = [{"bn": float(x[0]), "prod": float(x[1]), "w_x_epre": float(x[2]), "w_epre_mn9": float(x[3]), "node": desc(x[4])} for x in p2_antag[:15]]
            rec["top15_hop3_i1_epre"] = [{"bn": float(x[0]), "prod": float(x[1]), "w_x_i1": float(x[2]), "w_i1_epre": float(x[3]), "w_epre_mn9": float(x[4]), "i1": desc(x[5]), "epre": desc(x[6])} for x in p3_disinh_e[:15]]
        inh_results[gname] = rec

    selk_out = defaultdict(float)
    for s in selk_indices:
        for e in range(int(indptr[s]), int(indptr[s + 1])):
            if w_raw[e] > 0:
                selk_out[int(post_s[e])] += float(w_raw[e])
    selk_h2_antag, selk_h3_antag = [], []
    selk_h3_all, selk_h3_brain, selk_h3_vnc = [], [], []
    selk_h4_all, selk_h4_brain, selk_h4_vnc = [], [], []
    for i1, w_si1 in selk_out.items():
        if node_class[i1] != 1:
            continue
        i1_vnc = bool(is_vnc[i1])
        if inh_to_mn9[i1] > 0.0:
            wi = float(inh_to_mn9[i1])
            selk_h2_antag.append((min(w_si1, wi), w_si1 * wi, w_si1, wi, i1))
        for e in range(int(indptr[i1]), int(indptr[i1 + 1])):
            v2 = int(post_s[e])
            w_12 = float(-w_raw[e])
            if w_12 <= 0.0:
                continue
            if exc_to_mn9[v2] > 0.0:
                we = float(exc_to_mn9[v2])
                selk_h3_antag.append((min(w_si1, w_12, we), w_si1 * w_12 * we, w_si1, w_12, we, i1, v2))
            if node_class[v2] == 1:
                via_vnc = i1_vnc or bool(is_vnc[v2])
                if inh_to_mn9[v2] > 0.0:
                    w2m = float(inh_to_mn9[v2])
                    tup = (min(w_si1, w_12, w2m), w_si1 * w_12 * w2m, w_si1, w_12, w2m, i1, v2, via_vnc)
                    selk_h3_all.append(tup)
                    (selk_h3_vnc if via_vnc else selk_h3_brain).append(tup)
                if inh_to_epre_max_bn[v2] > 0.0:
                    ep = int(inh_to_epre_best_e[v2])
                    bn_2e = float(inh_to_epre_max_bn[v2])
                    tup4 = (min(w_si1, w_12, bn_2e), w_si1 * w_12 * float(inh_to_epre_prod[v2]), w_si1, w_12, bn_2e, float(exc_to_mn9[ep]), i1, v2, ep, via_vnc)
                    selk_h4_all.append(tup4)
                    (selk_h4_vnc if via_vnc else selk_h4_brain).append(tup4)
    for lst in (selk_h2_antag, selk_h3_antag, selk_h3_all, selk_h3_brain, selk_h3_vnc, selk_h4_all, selk_h4_brain, selk_h4_vnc):
        lst.sort(key=lambda x: (-x[0], -x[1]))

    def pack_h3(lst):
        return {
            "count": len(lst), "prod_sum": float(sum(x[1] for x in lst)), "bn_sum": float(sum(x[0] for x in lst)),
            "max_bn": float(lst[0][0]) if lst else 0.0,
            "top15": [{"bn": float(x[0]), "prod": float(x[1]), "w_s_i1": float(x[2]), "w_i1_i2": float(x[3]), "w_i2_mn9": float(x[4]), "via_vnc": bool(x[7]), "i1": desc(x[5]), "i2": desc(x[6])} for x in lst[:15]],
        }

    def pack_h4(lst):
        return {
            "count_i1_i2_pairs": len(lst), "prod_sum": float(sum(x[1] for x in lst)), "bn_sum": float(sum(x[0] for x in lst)),
            "max_bn": float(lst[0][0]) if lst else 0.0,
            "top15": [{"bn": float(x[0]), "prod_i1_i2_all_epre": float(x[1]), "w_s_i1": float(x[2]), "w_i1_i2": float(x[3]), "bn_i2_epre_mn9": float(x[4]), "via_vnc": bool(x[9]), "i1": desc(x[6]), "i2": desc(x[7]), "best_epre": desc(x[8])} for x in lst[:15]],
        }

    return {
        "inhibitory_candidates_and_controls": inh_results,
        "selk_dng68": {
            "n_cells": len(selk_indices), "total_out_exc_syn": float(sum(selk_out.values())),
            "hop2_antag_i1_mn9": {"count": len(selk_h2_antag), "prod_sum": float(sum(x[1] for x in selk_h2_antag)), "max_bn": float(selk_h2_antag[0][0]) if selk_h2_antag else 0.0},
            "hop3_antag_i1_epre_mn9": {"count": len(selk_h3_antag), "prod_sum": float(sum(x[1] for x in selk_h3_antag)), "max_bn": float(selk_h3_antag[0][0]) if selk_h3_antag else 0.0},
            "hop3_disinh_all": pack_h3(selk_h3_all), "hop3_disinh_brain_only": pack_h3(selk_h3_brain), "hop3_disinh_via_vnc": pack_h3(selk_h3_vnc),
            "hop4_disinh_all": pack_h4(selk_h4_all), "hop4_disinh_brain_only": pack_h4(selk_h4_brain), "hop4_disinh_via_vnc": pack_h4(selk_h4_vnc),
        },
    }


def _paired_stats(on_counts: np.ndarray, bg_counts: np.ndarray) -> dict:
    on_m = on_counts[:, :2].mean(axis=1).astype(np.float64)
    bg_m = bg_counts[:, :2].mean(axis=1).astype(np.float64)
    diff = on_m - bg_m
    mean_d = float(diff.mean())
    se_d = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else 0.0
    if np.all(diff == diff[0]):
        t_stat = 0.0 if mean_d == 0.0 else (1e9 if mean_d > 0 else -1e9)
        p_val = 1.0 if mean_d == 0.0 else 0.0
    else:
        rt = stats.ttest_1samp(diff, 0.0)
        t_stat, p_val = float(rt.statistic), float(rt.pvalue)
    # Also report intact left MN9_L paired stats!
    diff_l = (on_counts[:, 0] - bg_counts[:, 0]).astype(np.float64)
    mean_dl = float(diff_l.mean())
    if np.all(diff_l == diff_l[0]):
        p_l = 1.0 if mean_dl == 0.0 else 0.0
    else:
        p_l = float(stats.ttest_1samp(diff_l, 0.0).pvalue)
    return {
        "on_mean_hz": float(on_m.mean()),
        "bg_mean_hz": float(bg_m.mean()),
        "on_l_hz": float(on_counts[:, 0].mean()),
        "on_r_hz": float(on_counts[:, 1].mean()),
        "delta_mean_hz": mean_d,
        "delta_se_hz": se_d,
        "delta_l_hz": mean_dl,
        "delta_r_hz": float((on_counts[:, 1] - bg_counts[:, 1]).mean()),
        "paired_t": t_stat,
        "paired_p": p_val,
        "paired_p_l": p_l,
        "unlocked_ge1hz_p01": bool(mean_d >= 1.0 and p_val < 0.01),
        "unlocked_l_ge1hz_p01": bool(mean_dl >= 1.0 and p_l < 0.01),
    }


def classify_region(superclass: str, cls: str, soma_nm: str, typ: str) -> tuple[str, bool, bool]:
    sc = superclass.strip().lower()
    nm = soma_nm.strip().lower()
    is_asc = "ascending" in sc
    if sc.startswith("ol_"):
        return "optic_lobe", False, False
    if is_asc:
        return "ventral_nerve_cord", True, True
    if sc.startswith("vnc_"):
        return "ventral_nerve_cord", True, False
    if sc.startswith(("cb_", "visual_", "descending", "sensory_descending", "efferent_descending")) or sc == "ens":
        return "central_brain", False, False
    if nm and any(nm.startswith(p) for p in ("t1", "t2", "t3", "anm", "ab", "pro", "meso", "meta", "vnc")):
        return "ventral_nerve_cord", True, False
    return "central_brain", False, is_asc


def run_oracle_batch_gpu(gg: GPUGraph, w_syn: float, jobs: list[dict]) -> dict[str, dict]:
    t0 = time.time()
    N = gg.n_nodes
    total_B = sum(int(j["n_trials"]) * (1 + len(j["rates_hz"])) for j in jobs)
    max_W = max(len(j["watch_idx"]) for j in jobs)

    u_init_b = np.zeros((N, total_B), dtype=np.float64)
    vb_b = np.zeros((N, total_B), dtype=np.float64)
    is_exc_b = np.zeros((N, total_B), dtype=bool)
    watch_mat = np.zeros((max_W, total_B), dtype=np.int64)
    valid_mat = np.zeros((max_W, total_B), dtype=bool)
    ks_l, kn_l, kc_l = [], [], []
    col_records = []
    col_cursor = 0

    for job in jobs:
        n_tr = int(job["n_trials"])
        seed_bg = int(job["seed_bg"])
        vb_j = np.zeros(N, dtype=np.float64)
        for idx_i, val in job["node_biases"]:
            vb_j[int(idx_i)] = float(val)
        pos_idx = np.flatnonzero(vb_j > 0.0)
        u_init_j = np.zeros((N, n_tr), dtype=np.float64)
        for tr in range(n_tr):
            np.random.seed(seed_bg + tr * 1009)
            if pos_idx.size > 0:
                u_init_j[pos_idx, tr] = np.random.random(pos_idx.size) * 7.0
        w_idx = job["watch_idx"]
        W_j = len(w_idx)
        exc_idx = job["exc_idx"]
        runs = [("bg", 0.0, [], 12345)] + [(f"{int(rh)}hz", float(rh), exc_idx, int(job["seed_opto"]) + int(rh)) for rh in job["rates_hz"]]
        job_slices = {}
        for r_key, r_hz, e_idx, s_opto in runs:
            c0, c1 = col_cursor, col_cursor + n_tr
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
        k_steps, k_nodes, k_cols = np.concatenate(ks_l), np.concatenate(kn_l), np.concatenate(kc_l)
    else:
        e = np.empty(0, dtype=np.int32)
        k_steps, k_nodes, k_cols = e, e, e

    w_cnt, _ = run_batched_lif_gpu(gg, w_syn, u_init_b, vb_b, is_exc_b, k_steps, k_nodes, k_cols, watch_mat, valid_mat, track_total=False)
    elapsed_each = float((time.time() - t0) / len(jobs))
    results = {}
    for job, job_slices in col_records:
        c0_bg, c1_bg, W_j = job_slices["bg"]
        bg_counts = w_cnt[:W_j, c0_bg:c1_bg].T.copy()
        rates_out = {}
        for r_hz in job["rates_hz"]:
            rk = f"{int(r_hz)}hz"
            c0, c1, _ = job_slices[rk]
            on_counts = w_cnt[:W_j, c0:c1].T.copy()
            st = _paired_stats(on_counts, bg_counts)
            if W_j > 2:
                st["watched_intermed_bg_hz"] = bg_counts[:, 2:].mean(axis=0).tolist()
                st["watched_intermed_on_hz"] = on_counts[:, 2:].mean(axis=0).tolist()
            rates_out[rk] = st
        results[job["tag"]] = {
            "tag": job["tag"],
            "bg_mn9_l_hz": float(bg_counts[:, 0].mean()),
            "bg_mn9_r_hz": float(bg_counts[:, 1].mean()),
            "bg_mn9_mean_hz": float(bg_counts[:, :2].mean()),
            "rates": rates_out,
            "elapsed_sec": elapsed_each,
            "compute_backend": "colab_gpu_cuda_fp64",
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="MaleCNS v1.0 A100 GPU-accelerated benchmark.")
    parser.add_argument("--workers", type=int, default=min(10, max(2, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--min-gpu-gb", type=float, default=0.0)
    args, _ = parser.parse_known_args()

    t_all = time.time()
    cache_root = Path(os.environ.get("RESEARCH_CACHE_DIR", "/content/drive/MyDrive/ResearchCloud/cache")) / "projc-flywire"
    malecns_dir = cache_root / "malecns"
    out_dir = Path(os.environ.get("JOB_OUT_DIR", "/tmp/mn9_malecns_out"))
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path("/tmp/mn9_malecns_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    prog_path = out_dir / "progress.log"

    def log(msg: str) -> None:
        line = f"[{time.time() - t_all:6.1f}s] {msg}"
        print(line, flush=True)
        with prog_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    assert torch.cuda.is_available(), "CUDA GPU required for fast MaleCNS execution"
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log(f"init: GPU={gpu_name} ({vram_gb:.1f} GB), torch={torch.__version__}, cpus={os.cpu_count()}, workers={args.workers}")
    assert vram_gb >= args.min_gpu_gb

    manifest_path = cache_root / "mn9_malecns_input_manifest.json"
    if not manifest_path.is_file():
        manifest_path = Path("/tmp/mn9_malecns_input_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_hashes = {"manifest_sha256": sha256_file(manifest_path)}
    local_files = {}
    for k, fname in MALECNS_FILES.items():
        fpath = malecns_dir / fname
        input_hashes[fname] = ensure_gcs_file(f"{GCS_BASE}/{fname}", fpath)
        local_files[k] = fpath

    # 1. Build MaleCNS graph
    log("Stage 1: loading annotations, neurotransmitters, and weights...")
    ann_tbl = pf.read_table(local_files["annotations"])
    nt_tbl = pf.read_table(local_files["neurotransmitters"])
    w_tbl = pf.read_table(local_files["weights"])

    c_bid_a = pick_col(ann_tbl.column_names, ["bodyId", "body_id", "body"])
    c_status = pick_col(ann_tbl.column_names, ["status", "statusLabel"])
    c_stlabel = pick_col(ann_tbl.column_names, ["statusLabel"], required=False)
    c_type = pick_col(ann_tbl.column_names, ["type", "cell_type"])
    c_fwtype = pick_col(ann_tbl.column_names, ["flywireType", "flywire_type"], required=False)
    c_super = pick_col(ann_tbl.column_names, ["superclass", "super_class"], required=False)
    c_class = pick_col(ann_tbl.column_names, ["class", "cell_class"], required=False)
    c_side = pick_col(ann_tbl.column_names, ["somaSide", "rootSide", "side"], required=False)
    c_nm = pick_col(ann_tbl.column_names, ["somaNeuromere", "neuromere", "region"], required=False)
    c_dim = pick_col(ann_tbl.column_names, ["dimorphism", "sexualDimorphism"], required=False)
    c_bid_nt = pick_col(nt_tbl.column_names, ["bodyId", "body_id", "body"])
    c_cons_nt = pick_col(nt_tbl.column_names, ["consensus_nt", "consensusNt", "predicted_nt"])
    c_pre = pick_col(w_tbl.column_names, ["body_pre", "bodyPre", "pre"])
    c_post = pick_col(w_tbl.column_names, ["body_post", "bodyPost", "post"])
    c_w = pick_col(w_tbl.column_names, ["weight", "syn_count", "count"])

    bid_to_nt = {int(b): ("" if nv is None else str(nv).strip()) for b, nv in zip(nt_tbl[c_bid_nt].to_numpy(zero_copy_only=False), nt_tbl[c_cons_nt].to_pylist()) if b is not None}

    a_bids = ann_tbl[c_bid_a].to_numpy(zero_copy_only=False)
    a_status = ann_tbl[c_status].to_pylist()
    a_stlabel = ann_tbl[c_stlabel].to_pylist() if c_stlabel else [None] * ann_tbl.num_rows
    a_type = ann_tbl[c_type].to_pylist()
    a_fwtype = ann_tbl[c_fwtype].to_pylist() if c_fwtype else [None] * ann_tbl.num_rows
    a_super = ann_tbl[c_super].to_pylist() if c_super else [None] * ann_tbl.num_rows
    a_class = ann_tbl[c_class].to_pylist() if c_class else [None] * ann_tbl.num_rows
    a_side = ann_tbl[c_side].to_pylist() if c_side else [None] * ann_tbl.num_rows
    a_nm = ann_tbl[c_nm].to_pylist() if c_nm else [None] * ann_tbl.num_rows
    a_dim = ann_tbl[c_dim].to_pylist() if c_dim else [None] * ann_tbl.num_rows

    traced_bids, bid_meta = [], {}
    superclass_counts = Counter()
    glia_excluded, traced_with_nt = 0, 0
    for i in range(ann_tbl.num_rows):
        if ("" if a_status[i] is None else str(a_status[i]).strip()).lower() != "traced":
            continue
        sc = "" if a_super[i] is None else str(a_super[i]).strip()
        cl = "" if a_class[i] is None else str(a_class[i]).strip()
        if "glia" in sc.lower() or "glia" in cl.lower():
            glia_excluded += 1
            continue
        bid = int(a_bids[i])
        tp = "" if a_type[i] is None else str(a_type[i]).strip()
        fwt = "" if a_fwtype[i] is None else str(a_fwtype[i]).strip()
        sd = "" if a_side[i] is None else str(a_side[i]).strip()
        nm = "" if a_nm[i] is None else str(a_nm[i]).strip()
        dm = "" if a_dim[i] is None else str(a_dim[i]).strip()
        stl = "" if a_stlabel[i] is None else str(a_stlabel[i]).strip()
        nt_c = bid_to_nt.get(bid, "")
        sign_c = nt_to_sign(nt_c)
        if sign_c != 0:
            traced_with_nt += 1
        reg, is_v, is_a = classify_region(sc, cl, nm, tp)
        superclass_counts[sc or "(empty)"] += 1
        traced_bids.append(bid)
        bid_meta[bid] = {
            "id": bid, "cell_type": tp, "flywire_type": fwt, "status_label": stl,
            "superclass": sc, "class": cl, "side": sd, "soma_neuromere": nm,
            "dimorphism": dm, "nt": nt_c, "sign": sign_c, "region": reg, "is_vnc": is_v, "is_ascending": is_a,
        }

    traced_bids.sort()
    bid_to_idx = {b: idx for idx, b in enumerate(traced_bids)}
    n_nodes = len(traced_bids)
    node_class = np.full(n_nodes, 2, dtype=np.int8)
    is_vnc_arr = np.zeros(n_nodes, dtype=bool)
    node_meta = {}
    region_counts = Counter()
    for idx, bid in enumerate(traced_bids):
        m = bid_meta[bid]
        node_class[idx] = 0 if m["sign"] > 0 else (1 if m["sign"] < 0 else 2)
        is_vnc_arr[idx] = m["is_vnc"]
        node_meta[idx] = m
        region_counts[m["region"]] += 1

    mn9_l_bid = int(manifest["mn9_malecns_body_ids"]["left"])
    mn9_r_bid = int(manifest["mn9_malecns_body_ids"]["right"])
    mn9_l_idx, mn9_r_idx = bid_to_idx[mn9_l_bid], bid_to_idx[mn9_r_bid]
    mn9_indices = [mn9_l_idx, mn9_r_idx]

    raw_pre = w_tbl[c_pre].to_numpy(zero_copy_only=False).astype(np.int64)
    raw_post = w_tbl[c_post].to_numpy(zero_copy_only=False).astype(np.int64)
    raw_w = w_tbl[c_w].to_numpy(zero_copy_only=False).astype(np.float64)
    traced_np = np.asarray(traced_bids, dtype=np.int64)
    pos_pre = np.searchsorted(traced_np, raw_pre)
    pos_post = np.searchsorted(traced_np, raw_post)
    in_pre = (pos_pre < n_nodes) & (traced_np[np.minimum(pos_pre, n_nodes - 1)] == raw_pre)
    in_post = (pos_post < n_nodes) & (traced_np[np.minimum(pos_post, n_nodes - 1)] == raw_post)
    valid_pair = in_pre & in_post & (raw_w >= 1.0)
    self_loops_removed = int((valid_pair & (pos_pre == pos_post)).sum())
    keep_non_self = valid_pair & (pos_pre != pos_post)

    W_unsigned = sparse.coo_matrix((raw_w[keep_non_self], (pos_pre[keep_non_self].astype(np.int32), pos_post[keep_non_self].astype(np.int32))), shape=(n_nodes, n_nodes)).tocsr()
    W_unsigned.sum_duplicates()
    n_non_self_unique = int(W_unsigned.nnz)
    u_u, v_u = W_unsigned.nonzero()
    u_u, v_u, w_u = u_u.astype(np.int32), v_u.astype(np.int32), W_unsigned.data.astype(np.float64)

    pre_signs = np.where(node_class[u_u] == 0, 1.0, np.where(node_class[u_u] == 1, -1.0, 0.0))
    valid_nt_mask = pre_signs != 0.0
    nt_edge_coverage = float(valid_nt_mask.mean())
    pre_s = u_u[valid_nt_mask]
    post_s = v_u[valid_nt_mask]
    w_raw_s = (w_u[valid_nt_mask] * pre_signs[valid_nt_mask]).astype(np.float32)

    order_e = np.lexsort((post_s, pre_s))
    pre_s, post_s, w_raw_s = pre_s[order_e].astype(np.int32), post_s[order_e].astype(np.int32), w_raw_s[order_e].astype(np.float32)
    indptr = np.zeros(n_nodes + 1, dtype=np.int32)
    np.cumsum(np.bincount(pre_s, minlength=n_nodes), out=indptr[1:])
    orig_in_pos = np.bincount(post_s[w_raw_s > 0], minlength=n_nodes).astype(np.int32)
    orig_in_neg = np.bincount(post_s[w_raw_s < 0], minlength=n_nodes).astype(np.int32)

    npz_path = work_dir / "malecns_main_graph.npz"
    np.savez(npz_path, n_nodes=np.int32(n_nodes), indptr=indptr, pre_s=pre_s, post_s=post_s, w_raw_s=w_raw_s, node_class=node_class, orig_in_pos=orig_in_pos, orig_in_neg=orig_in_neg)
    g_dict = {"n_nodes": n_nodes, "indptr": indptr, "pre_s": pre_s, "post_s": post_s, "w_raw_s": w_raw_s, "node_class": node_class, "orig_in_pos": orig_in_pos, "orig_in_neg": orig_in_neg}

    # MN9 incoming stats & 3 zero-free-parameter scales
    mn9_l_mask, mn9_r_mask = (post_s == mn9_l_idx), (post_s == mn9_r_idx)
    s_in_l = float(np.abs(w_raw_s[mn9_l_mask]).sum())
    s_in_r = float(np.abs(w_raw_s[mn9_r_mask]).sum())
    s_in_mean = 0.5 * (s_in_l + s_in_r)
    mn9_u_mask = (v_u == mn9_l_idx) | (v_u == mn9_r_idx)
    mn9_nt_cov = float((pre_signs[mn9_u_mask] != 0.0).mean())

    s_in_v783_mean = float(manifest["s_in_mn9_v783_mean"])
    w_syn_frozen = float(manifest["w_syn_frozen_mv"])
    anat_bilat_ratio = s_in_v783_mean / s_in_mean
    w_syn_bilat = w_syn_frozen * anat_bilat_ratio
    anat_intact_ratio = s_in_v783_mean / s_in_l
    w_syn_intact = w_syn_frozen * anat_intact_ratio

    v783_pre_set = set(manifest["v783_mn9_presyn_types"])
    shared_l = sum(float(wv) for pu, wv in zip(pre_s[mn9_l_mask], np.abs(w_raw_s[mn9_l_mask])) if (split_tokens(node_meta[int(pu)]["cell_type"]) | split_tokens(node_meta[int(pu)]["flywire_type"])) & v783_pre_set)
    shared_r = sum(float(wv) for pu, wv in zip(pre_s[mn9_r_mask], np.abs(w_raw_s[mn9_r_mask])) if (split_tokens(node_meta[int(pu)]["cell_type"]) | split_tokens(node_meta[int(pu)]["flywire_type"])) & v783_pre_set)

    log(f"Graph ready: N={n_nodes:,}, E={len(pre_s):,}, NT_cov={nt_edge_coverage:.5f}; MN9_L={s_in_l:.1f} ({bid_meta[mn9_l_bid]['status_label']}), MN9_R={s_in_r:.1f} ({bid_meta[mn9_r_bid]['status_label']})")
    log(f"Scales: frozen={w_syn_frozen:.6f}, intact(L)={w_syn_intact:.6f} ({anat_intact_ratio:.4f}x), bilat={w_syn_bilat:.6f} ({anat_bilat_ratio:.4f}x)")

    # Deterministic mapping
    tok_to_indices = defaultdict(list)
    for idx in range(n_nodes):
        m = node_meta[idx]
        for tok in (split_tokens(m["cell_type"]) | split_tokens(m["flywire_type"])):
            tok_to_indices[tok].append(idx)
    out_syn_per_node = np.bincount(pre_s, weights=np.abs(w_raw_s).astype(np.float64), minlength=n_nodes)

    sez_list = manifest["sez106"]
    type_indices, mapping_records, missing_types, dimorphic_annotated = [], [], [], []
    for rec in sez_list:
        nm = rec["name"]
        hits = set()
        for t in rec["v783_cell_types"]:
            hits.update(tok_to_indices.get(t, []))
        if not hits and rec["v783_hemibrain_types"]:
            for t in rec["v783_hemibrain_types"]:
                hits.update(tok_to_indices.get(t, []))
        hlist = sorted(hits)
        type_indices.append(hlist)
        if not hlist:
            missing_types.append(nm)
        dims = sorted({node_meta[u]["dimorphism"] for u in hlist if node_meta[u]["dimorphism"]})
        if dims and any(d.lower() not in ("isomorphic", "none", "nan") for d in dims):
            dimorphic_annotated.append({"name": nm, "dimorphism": dims, "y_bin": rec["y_bin"]})
        cls_cnt = Counter(int(node_class[u]) for u in hlist)
        mapping_records.append({
            "name": nm, "y_bin": int(rec["y_bin"]), "v783_cell_types": rec["v783_cell_types"],
            "n_malecns": len(hlist), "malecns_body_ids": [int(node_meta[u]["id"]) for u in hlist],
            "malecns_out_syn": float(sum(out_syn_per_node[u] for u in hlist)),
            "malecns_cls_counts": {str(k): int(v) for k, v in cls_cnt.items()}, "dimorphism_tags": dims,
        })
    sez_idx_by_name = {sez_list[i]["name"]: type_indices[i] for i in range(len(sez_list))}

    extra_map = {}
    for ek, einfo in manifest["extra_crosslab_types"].items():
        hits = set()
        for t in einfo["v783_cell_types"]:
            hits.update(tok_to_indices.get(t, []))
        extra_map[ek] = sorted(hits)
    selk_idxs = extra_map["SELK_DNg68"]
    log(f"Mapping: {106 - len(missing_types)}/106 SEZ (missing={missing_types}); SELK={len(selk_idxs)}, Fdg={len(extra_map['Fdg_CB0038'])}, Amulet={len(extra_map['Amulet_CB0626'])}")

    # Disinhibition anatomy (pure graph, ~5s)
    inh_groups = {c: sez_idx_by_name[c] for c in ("phantom", "tentacular", "usnea", "marge") if sez_idx_by_name[c]}
    mapped_neg_inh_ctrls = [c for c in manifest["neg_inh_ctrl44"] if sez_idx_by_name[c]]
    for c in mapped_neg_inh_ctrls:
        inh_groups[c] = sez_idx_by_name[c]
    anat_male = analyze_disinhibitory_anatomy(g_dict, node_meta, mn9_indices, inh_groups, selk_idxs, is_vnc_arr)

    # 2. Generate 20 rewired graphs FIRST on 8 CPU cores BEFORE CUDA init (100% fork-safe & fast!)
    log("Stage 2: generating K=10 full + K=10 MN9-protected rewired graphs on CPU workers...")
    rw_jobs = []
    for k in range(10):
        rw_jobs.append({"family": "rewire_full", "idx": k, "seed": 9501 + k, "protect_targets": [], "npz_path": str(npz_path), "out_npy": str(work_dir / f"rw_full_{k}.npy")})
        rw_jobs.append({"family": "rewire_mn9_protected", "idx": k, "seed": 9601 + k, "protect_targets": mn9_indices, "npz_path": str(npz_path), "out_npy": str(work_dir / f"rw_prot_{k}.npy")})
    rw_full_paths, rw_prot_paths = [None] * 10, [None] * 10
    rw_full_meta, rw_prot_meta = [None] * 10, [None] * 10
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_build_one_rewire_worker, j) for j in rw_jobs]
        done_rw = 0
        for fut in as_completed(futs):
            r = fut.result()
            done_rw += 1
            if r["family"] == "rewire_full":
                rw_full_paths[r["idx"]] = r["out_npy"]
                rw_full_meta[r["idx"]] = r["meta"]
            else:
                rw_prot_paths[r["idx"]] = r["out_npy"]
                rw_prot_meta[r["idx"]] = r["meta"]
            if done_rw % 5 == 0 or done_rw == 20:
                log(f"  rewired graphs generated: {done_rw}/20 (last {r['family']}#{r['idx']} in {r['elapsed_sec']:.1f}s)")

    # 3. Initialize A100 CUDA graph & run exact CPU-vs-GPU FP64 parity checks
    log("Stage 3: CPU-vs-GPU FP64 exact spike equivalence verification on A100...")
    device = torch.device("cuda:0")
    gg = GPUGraph(n_nodes, indptr, post_s, w_raw_s, node_class, device)
    equiv_report = {"gpu_name": gpu_name, "checks": {}}
    n_eq = 10
    weights_intact_f64 = (w_raw_s.astype(np.float64) * w_syn_intact).astype(np.float64)

    # Check A: zero-bg roundup 50 Hz under intact anatomical scale
    ru_idx = sez_idx_by_name["roundup"]
    w_cpu_a = run_lif_dc_cpu_ref(indptr, post_s, weights_intact_f64, w_syn_intact, np.zeros(n_nodes, dtype=np.float64), 0.0, np.array(ru_idx, dtype=np.int32), 50.0, np.array(mn9_indices, dtype=np.int32), n_eq, 1001, 50000)
    u0_a = np.zeros((n_nodes, n_eq), dtype=np.float64)
    is_exc_a = np.zeros((n_nodes, n_eq), dtype=bool)
    is_exc_a[ru_idx, :] = True
    ks_a, kn_a, kc_a = gen_opto_kicks(50000, n_eq, ru_idx, 50.0)
    wm_a = np.repeat(np.array(mn9_indices, dtype=np.int64)[:, None], n_eq, axis=1)
    vm_a = np.ones_like(wm_a, dtype=bool)
    w_gpu_a, _ = run_batched_lif_gpu(gg, w_syn_intact, u0_a, None, is_exc_a, ks_a, kn_a, kc_a, wm_a, vm_a)
    w_gpu_a = w_gpu_a.T
    diff_a = int(np.max(np.abs(w_cpu_a - w_gpu_a)))
    equiv_report["checks"]["zero_bg_roundup_intact_50hz"] = {"n_trials": n_eq, "cpu_mean_hz": float(w_cpu_a.mean()), "gpu_mean_hz": float(w_gpu_a.mean()), "max_abs_spike_diff": diff_a, "abs_rate_diff_hz": float(abs(w_cpu_a.mean() - w_gpu_a.mean()))}
    log(f"  [equiv A] roundup 50Hz: cpu={w_cpu_a.mean():.3f} Hz, gpu={w_gpu_a.mean():.3f} Hz, max_spk_diff={diff_a}")
    assert diff_a == 0

    # Check B: tonic-DC Oracle (tentacular Direct K=5, 200 Hz)
    t_idx = sez_idx_by_name["tentacular"]
    tent_ip5 = [x["node"]["idx"] for x in anat_male["inhibitory_candidates_and_controls"]["tentacular"]["top15_hop2_ipre"][:5]]
    vb_tent = np.zeros(n_nodes, dtype=np.float64)
    for m in mn9_indices:
        vb_tent[m] = 7.5
    for u in tent_ip5:
        vb_tent[u] = 8.5
    w_cpu_b = run_lif_dc_cpu_ref(indptr, post_s, weights_intact_f64, w_syn_intact, vb_tent, 7.0, np.array(t_idx, dtype=np.int32), 200.0, np.array(mn9_indices, dtype=np.int32), n_eq, 41001, 42200)
    pos_vb = np.flatnonzero(vb_tent > 0.0)
    u0_b = np.zeros((n_nodes, n_eq), dtype=np.float64)
    for tr in range(n_eq):
        np.random.seed(41001 + tr * 1009)
        u0_b[pos_vb, tr] = np.random.random(pos_vb.size) * 7.0
    is_exc_b = np.zeros((n_nodes, n_eq), dtype=bool)
    is_exc_b[t_idx, :] = True
    ks_b, kn_b, kc_b = gen_opto_kicks(42200, n_eq, t_idx, 200.0)
    w_gpu_b, _ = run_batched_lif_gpu(gg, w_syn_intact, u0_b, vb_tent[:, None], is_exc_b, ks_b, kn_b, kc_b, wm_a, vm_a)
    w_gpu_b = w_gpu_b.T
    diff_b = int(np.max(np.abs(w_cpu_b - w_gpu_b)))
    equiv_report["checks"]["tonic_oracle_tentacular_intact_200hz"] = {"n_trials": n_eq, "cpu_mean_hz": float(w_cpu_b.mean()), "gpu_mean_hz": float(w_gpu_b.mean()), "max_abs_spike_diff": diff_b, "abs_rate_diff_hz": float(abs(w_cpu_b.mean() - w_gpu_b.mean()))}
    log(f"  [equiv B] tentacular Oracle 200Hz: cpu={w_cpu_b.mean():.3f} Hz, gpu={w_gpu_b.mean():.3f} Hz, max_spk_diff={diff_b}")
    assert diff_b == 0

    # 4. Module 1: Real graph sweeps across all 3 scales (20/50/100/200 Hz) + Static baselines
    log("Stage 4: Module 1 real graph sweeps on A100 GPU (3 scales x 4 frequencies)...")
    static_scores, is_direct_male = compute_static_baselines(pre_s, post_s, w_raw_s, n_nodes, [np.array(x, dtype=np.int32) for x in type_indices], mn9_indices)
    scales = [("anat_intact", w_syn_intact), ("frozen_0.275", w_syn_frozen), ("anat_bilat", w_syn_bilat)]
    real_runs = {s: {} for s, _ in scales}
    for sname, ws in scales:
        bt = 14 if sname == "anat_bilat" else 27
        for hz, sbase in ((20, 20000), (50, 50000), (100, 100000), (200, 200000)):
            t_s = time.time()
            res_s = run_task1_sweep_gpu(gg, ws, type_indices, float(hz), 30, sbase, mn9_indices, batch_types=bt, track_total=True)
            real_runs[sname][f"{hz}hz"] = res_s
            log(f"  [real GPU] {sname:12s} {hz:3d}Hz: nz_mean={(res_s['means']>0).sum():3d}/106, nz_L={(res_s['l_rates']>0).sum():3d}/106, brain_hz={res_s['mean_whole_brain_hz']:.3f} (max={res_s['max_whole_brain_hz']:.2f}) [{time.time()-t_s:.1f}s]")

    # Module 3: SELK / Fdg / Amulet 20-200 Hz on A100 GPU (watching MN9 + top-20 SELK targets)
    log("Stage 5: Module 3 full-CNS SELK (DNg68), Fdg, and Amulet sweep on A100 GPU...")
    selk_out_by_post = defaultdict(float)
    for s in selk_idxs:
        for e in range(int(indptr[s]), int(indptr[s + 1])):
            selk_out_by_post[int(post_s[e])] += abs(float(w_raw_s[e]))
    selk_sorted_targets = sorted(selk_out_by_post.items(), key=lambda x: -x[1])
    top20_selk_idxs = [u for u, _ in selk_sorted_targets[:20]]
    selk_reg_ei = defaultdict(lambda: {"n_targets": 0, "sum_syn": 0.0})
    for u, syn in selk_sorted_targets:
        m = node_meta[u]
        ei = "E" if node_class[u] == 0 else ("I" if node_class[u] == 1 else "U")
        k = f"{m['region']}_{ei}"
        selk_reg_ei[k]["n_targets"] += 1
        selk_reg_ei[k]["sum_syn"] += float(syn)

    selk_lif = {s: {} for s, _ in scales}
    selk_top20_rates_200hz = {}
    for sname, ws in scales:
        for cname in ("SELK_DNg68", "Fdg_CB0038", "Amulet_CB0626"):
            selk_lif[sname][cname] = {}
            e_idx = extra_map[cname]
            w_list = mn9_indices + (top20_selk_idxs if cname == "SELK_DNg68" else [])
            W_n = len(w_list)
            for hz in (20, 50, 100, 200):
                if not e_idx:
                    selk_lif[sname][cname][f"{hz}hz"] = {"mn9_l_hz": 0.0, "mn9_r_hz": 0.0, "mn9_mean_hz": 0.0}
                    continue
                u0 = np.zeros((n_nodes, 30), dtype=np.float64)
                ie = np.zeros((n_nodes, 30), dtype=bool)
                ie[e_idx, :] = True
                ks, kn, kc = gen_opto_kicks(80000 + hz, 30, e_idx, float(hz))
                wm = np.repeat(np.array(w_list, dtype=np.int64)[:, None], 30, axis=1)
                vm = np.ones((W_n, 30), dtype=bool)
                wc, _ = run_batched_lif_gpu(gg, ws, u0, None, ie, ks, kn, kc, wm, vm)
                selk_lif[sname][cname][f"{hz}hz"] = {"mn9_l_hz": float(wc[0].mean()), "mn9_r_hz": float(wc[1].mean()), "mn9_mean_hz": float(wc[:2].mean())}
                if sname == "anat_intact" and cname == "SELK_DNg68" and hz == 200:
                    for ti, tu in enumerate(top20_selk_idxs):
                        selk_top20_rates_200hz[tu] = float(wc[2 + ti].mean())

    top20_records = [
        {
            "idx": int(u), "body_id": int(node_meta[u]["id"]), "cell_type": node_meta[u]["cell_type"],
            "flywire_type": node_meta[u]["flywire_type"], "superclass": node_meta[u]["superclass"],
            "class": node_meta[u]["class"], "region": node_meta[u]["region"],
            "is_vnc": bool(node_meta[u]["is_vnc"]), "is_ascending": bool(node_meta[u]["is_ascending"]),
            "side": node_meta[u]["side"], "nt_predicted": node_meta[u]["nt"],
            "ei_class": "E" if node_class[u] == 0 else ("I" if node_class[u] == 1 else "U"),
            "syn_from_selk": float(syn), "rate_200hz_intact": float(selk_top20_rates_200hz.get(u, 0.0)),
        }
        for u, syn in selk_sorted_targets[:20]
    ]
    sp_unconstrained = shortest_exc_path_with_constraint(indptr, post_s, w_raw_s, node_class, selk_idxs, set(mn9_indices))
    vnc_exc_step1 = [u for u, _ in selk_sorted_targets if is_vnc_arr[u] and node_class[u] == 0]
    sp_via_vnc = shortest_exc_path_with_constraint(
        indptr, post_s, w_raw_s, node_class, vnc_exc_step1, set(mn9_indices), init_bottlenecks=selk_out_by_post
    )
    if sp_via_vnc["reachable"] and sp_via_vnc["path_indices"]:
        first_vnc = sp_via_vnc["path_indices"][0]
        sp_via_vnc["min_hops_from_selk"] = sp_via_vnc["min_hops"] + 1
        sp_via_vnc["bottleneck_weight_from_selk"] = float(min(selk_out_by_post[first_vnc], sp_via_vnc["bottleneck_weight"]))
        sp_via_vnc["first_vnc_node"] = node_meta[first_vnc]
    else:
        sp_via_vnc["min_hops_from_selk"] = None
        sp_via_vnc["bottleneck_weight_from_selk"] = 0.0

    def annotate_path(pinfo):
        o = dict(pinfo)
        o["path_nodes"] = [{"id": node_meta[u]["id"], "cell_type": node_meta[u]["cell_type"], "region": node_meta[u]["region"], "nt": node_meta[u]["nt"]} for u in pinfo.get("path_indices", [])]
        return o

    selk_diagnosis = {
        "selk_n_cells": len(selk_idxs), "selk_cells": [node_meta[u] for u in selk_idxs],
        "selk_total_outgoing_synapses": float(sum(selk_out_by_post.values())),
        "selk_total_postsynaptic_targets": len(selk_sorted_targets),
        "selk_postsynaptic_region_ei_summary": dict(selk_reg_ei),
        "selk_top20_postsynaptic_targets": top20_records,
        "selk_exc_shortest_paths_to_mn9": {"unconstrained": annotate_path(sp_unconstrained), "forced_via_vnc": annotate_path(sp_via_vnc)},
        "lif_responses": selk_lif,
    }
    log(f"  SELK done: {len(selk_idxs)} cells, syn={selk_diagnosis['selk_total_outgoing_synapses']:.0f}, intact 50/200={selk_lif['anat_intact']['SELK_DNg68']['50hz']['mn9_mean_hz']:.3f}/{selk_lif['anat_intact']['SELK_DNg68']['200hz']['mn9_mean_hz']:.3f} Hz")

    # 5. Module 4: 36 Local Oracles on REAL graph BEFORE swapping gg to rewired graphs!
    log("Stage 6: Module 4 local premotor Oracles on A100 GPU (36 jobs at intact 0.262 mV)...")
    ic_male = anat_male["inhibitory_candidates_and_controls"]
    oracle_jobs = []
    for cname in ("phantom", "tentacular", "usnea", "marge"):
        rec = ic_male[cname]
        top_ipre = [x["node"]["idx"] for x in rec["top15_hop2_ipre"]]
        top_h3 = rec["top15_hop3_i1_epre"]
        for K in (5, 10):
            ip_k = top_ipre[:K]
            if ip_k:
                for v_mn9 in (7.5, 8.5):
                    oracle_jobs.append({
                        "tag": f"malecns|{cname}|oracle_direct_K{K}_vmn9_{v_mn9:.1f}",
                        "node_biases": [(m, v_mn9) for m in mn9_indices] + [(u, 8.5) for u in ip_k],
                        "watch_idx": mn9_indices + ip_k[:5], "exc_idx": sez_idx_by_name[cname],
                        "rates_hz": [50.0, 200.0], "n_trials": 30, "seed_bg": 41001, "seed_opto": 42000,
                    })
            i1_set, ep_set = [], []
            for x in top_h3:
                if x["i1"]["idx"] not in i1_set and len(i1_set) < K:
                    i1_set.append(x["i1"]["idx"])
                if x["epre"]["idx"] not in ep_set and len(ep_set) < K:
                    ep_set.append(x["epre"]["idx"])
            if i1_set and ep_set:
                oracle_jobs.append({
                    "tag": f"malecns|{cname}|oracle_premotor_K{K}",
                    "node_biases": [(u, 8.5) for u in i1_set] + [(u, 7.8) for u in ep_set if u not in i1_set],
                    "watch_idx": mn9_indices + i1_set[:3] + ep_set[:3], "exc_idx": sez_idx_by_name[cname],
                    "rates_hz": [50.0, 200.0], "n_trials": 30, "seed_bg": 41001, "seed_opto": 43000,
                })
    sk = anat_male["selk_dng68"]
    for branch_name, h3_key, h4_key in (("all", "hop3_disinh_all", "hop4_disinh_all"), ("via_vnc", "hop3_disinh_via_vnc", "hop4_disinh_via_vnc")):
        i2_list = []
        for x in sk[h3_key]["top15"]:
            if x["i2"]["idx"] not in i2_list:
                i2_list.append(x["i2"]["idx"])
        for K in (5, 10):
            i2_k = i2_list[:K]
            if i2_k:
                for v_mn9 in (7.5, 8.5):
                    oracle_jobs.append({
                        "tag": f"malecns|SELK_DNg68|oracle_direct_{branch_name}_K{K}_vmn9_{v_mn9:.1f}",
                        "node_biases": [(m, v_mn9) for m in mn9_indices] + [(u, 8.5) for u in i2_k],
                        "watch_idx": mn9_indices + i2_k[:5], "exc_idx": selk_idxs,
                        "rates_hz": [50.0, 200.0], "n_trials": 30, "seed_bg": 41001, "seed_opto": 44000,
                    })
            i2_4, ep_4 = [], []
            for x in sk[h4_key]["top15"]:
                if x["i2"]["idx"] not in i2_4 and len(i2_4) < K:
                    i2_4.append(x["i2"]["idx"])
                if x["best_epre"]["idx"] not in ep_4 and len(ep_4) < K:
                    ep_4.append(x["best_epre"]["idx"])
            if i2_4 and ep_4:
                oracle_jobs.append({
                    "tag": f"malecns|SELK_DNg68|oracle_premotor_{branch_name}_K{K}",
                    "node_biases": [(u, 8.5) for u in i2_4] + [(u, 7.8) for u in ep_4 if u not in i2_4],
                    "watch_idx": mn9_indices + i2_4[:3] + ep_4[:3], "exc_idx": selk_idxs,
                    "rates_hz": [50.0, 200.0], "n_trials": 30, "seed_bg": 41001, "seed_opto": 45000,
                })

    oracle_results_intact = {}
    for b0 in range(0, len(oracle_jobs), 6):
        sub = oracle_jobs[b0 : b0 + 6]
        t_b = time.time()
        oracle_results_intact.update(run_oracle_batch_gpu(gg, w_syn_intact, sub))
        log(f"  [oracle intact {min(b0+6, len(oracle_jobs)):2d}/{len(oracle_jobs)}] done in {time.time()-t_b:.1f}s")

    # Sensitivity check: tentacular & SELK Oracles under frozen 0.275 mV
    sens_jobs = [j for j in oracle_jobs if ("|tentacular|" in j["tag"] or "|SELK_DNg68|" in j["tag"])]
    oracle_results_frozen275 = {}
    for b0 in range(0, len(sens_jobs), 6):
        sub = sens_jobs[b0 : b0 + 6]
        oracle_results_frozen275.update(run_oracle_batch_gpu(gg, w_syn_frozen, sub))
    log(f"  [oracle frozen_0.275 sensitivity] {len(sens_jobs)} jobs done")

    # 6. Evaluate K=10 full + K=10 MN9-protected rewired graphs on A100 GPU!
    log("Stage 7: evaluating all 20 rewired graphs on A100 GPU (50 Hz, 30 trials)...")
    rw_eval = {s: {"full_mean": [], "full_l": [], "prot_mean": [], "prot_l": []} for s, _ in scales}
    bilat_brain_hz = real_runs["anat_bilat"]["50hz"]["mean_whole_brain_hz"] or 0.0
    run_bilat_rw = bilat_brain_hz < 5.0  # run full 30-tr rewires for bilat if non-epileptic, else 10-tr GPU SpMM!
    bilat_rw_trials = 30 if run_bilat_rw else 10
    log(f"  anat_bilat 50 Hz mean brain rate = {bilat_brain_hz:.3f} Hz -> bilat rewire trials = {bilat_rw_trials}")

    for fam, p_list in (("full", rw_full_paths), ("prot", rw_prot_paths)):
        for k, npy_p in enumerate(p_list):
            t_r = time.time()
            gg.update_post_indices(np.load(npy_p))
            for sname, ws in scales:
                ntr = bilat_rw_trials if sname == "anat_bilat" else 30
                bt = 14 if sname == "anat_bilat" else 27
                res_k = run_task1_sweep_gpu(gg, ws, type_indices, 50.0, ntr, 50000, mn9_indices, batch_types=bt, track_total=False)
                rw_eval[sname][f"{fam}_mean"].append(res_k["means"])
                rw_eval[sname][f"{fam}_l"].append(res_k["l_rates"])
            log(f"  [rewire GPU] {fam} #{k} (3 scales) done in {time.time()-t_r:.1f}s")

    # 7. Compute subsets, two-tier inference, 3-brain table, and disinhibition percentiles
    log("Stage 8: computing 7 subsets, B=2000 bootstraps, and 3-brain tie-breaker summaries...")
    y_bin = np.array([r["y_bin"] for r in sez_list], dtype=np.int32)
    y_bin_02 = (np.array([r["y_max"] for r in sez_list], dtype=np.float64) >= 0.2).astype(np.int32)
    y_max = np.array([r["y_max"] for r in sez_list], dtype=np.float64)
    y_mean = np.array([r["y_mean"] for r in sez_list], dtype=np.float64)
    prior_pub = np.array([r["prior_published"] for r in sez_list], dtype=bool)
    is_mapped = np.array([len(x) > 0 for x in type_indices], dtype=bool)
    is_iso102 = is_mapped & np.array([r["name"] not in ("horn", "shark") for r in sez_list], dtype=bool)
    is_dir_v783 = np.array([r["is_direct_1hop_v783"] for r in sez_list], dtype=bool)
    is_dir_banc = np.array([r["is_direct_1hop_banc_m1"] for r in sez_list], dtype=bool)

    subsets = {
        "all_106": np.ones(len(sez_list), dtype=bool),
        "mapped_104": is_mapped,
        "isomorphic_102": is_iso102,
        "per_ge_0_2": np.ones(len(sez_list), dtype=bool),
        "nondirect_malecns": ~is_direct_male,
        "nondirect_common_3brains": (~is_dir_v783) & (~is_dir_banc) & (~is_direct_male),
        "novel_95": ~prior_pub,
    }

    task1_eval = {}
    for sname, _ in scales:
        task1_eval[sname] = {"bilateral_mean": {}, "intact_left_mn9": {}}
        scores_m = {f"{hz}hz": real_runs[sname][f"{hz}hz"]["means"] for hz in (20, 50, 100, 200)}
        scores_l = {f"{hz}hz": real_runs[sname][f"{hz}hz"]["l_rates"] for hz in (20, 50, 100, 200)}
        for sub_name, smask in subsets.items():
            yb_use = y_bin_02 if sub_name == "per_ge_0_2" else y_bin
            task1_eval[sname]["bilateral_mean"][sub_name] = evaluate_subset_suite(
                yb_use, y_max, y_mean, smask, scores_m, "50hz", static_scores, rw_eval[sname]["full_mean"], rw_eval[sname]["prot_mean"]
            )
            task1_eval[sname]["intact_left_mn9"][sub_name] = evaluate_subset_suite(
                yb_use, y_max, y_mean, smask, scores_l, "50hz", static_scores, rw_eval[sname]["full_l"], rw_eval[sname]["prot_l"]
            )

    v783_50 = np.array([r["v783_50hz"] for r in sez_list], dtype=np.float64)
    banc_50 = np.array([r["banc_m1_anat_50hz"] for r in sez_list], dtype=np.float64)
    no_ru = np.array([r["name"] != "roundup" for r in sez_list], dtype=bool)

    def corr_pack(ref_arr, test_arr):
        return {
            "pearson_r_all106": pearson_safe(ref_arr, test_arr),
            "spearman_rho_all106": spearman_safe(ref_arr, test_arr),
            "pearson_r_excl_roundup": pearson_safe(ref_arr[no_ru], test_arr[no_ru]),
            "spearman_rho_excl_roundup": spearman_safe(ref_arr[no_ru], test_arr[no_ru]),
            "gt0_agreement_count": int(((ref_arr > 0.0) == (test_arr > 0.0)).sum()),
            "n_test_gt0": int((test_arr > 0.0).sum()),
            "mae_hz": float(np.mean(np.abs(ref_arr - test_arr))),
        }

    cross_brain_corr = {}
    for sname, _ in scales:
        cross_brain_corr[f"v783_vs_{sname}_50hz_mean"] = corr_pack(v783_50, real_runs[sname]["50hz"]["means"])
        cross_brain_corr[f"v783_vs_{sname}_50hz_left"] = corr_pack(v783_50, real_runs[sname]["50hz"]["l_rates"])
        cross_brain_corr[f"banc_vs_{sname}_50hz_mean"] = corr_pack(banc_50, real_runs[sname]["50hz"]["means"])
        cross_brain_corr[f"banc_vs_{sname}_50hz_left"] = corr_pack(banc_50, real_runs[sname]["50hz"]["l_rates"])

    per_type_3brains = []
    for i, rec in enumerate(sez_list):
        per_type_3brains.append({
            "name": rec["name"], "y_bin": int(rec["y_bin"]), "y_max": float(rec["y_max"]),
            "y_mean": float(rec["y_mean"]), "prior_published": bool(rec["prior_published"]),
            "n_v783": int(rec["n_v783"]), "n_banc_m1": int(rec["n_m1_banc"]), "n_banc_m2": int(rec["n_m2_banc"]),
            "n_malecns": int(mapping_records[i]["n_malecns"]),
            "v783_out_syn": float(rec["v783_out_syn"]), "banc_m1_out_syn": float(rec["banc_m1_out_syn"]),
            "malecns_out_syn": float(mapping_records[i]["malecns_out_syn"]),
            "is_direct_v783": bool(rec["is_direct_1hop_v783"]), "is_direct_banc_m1": bool(rec["is_direct_1hop_banc_m1"]),
            "is_direct_malecns": bool(is_direct_male[i]),
            "v783_static_1hop": float(rec["v783_static_1hop"]), "banc_static_1hop": float(rec["banc_m1_static_1hop"]),
            "malecns_static_1hop": float(static_scores["static_1hop_signed"][i]),
            "v783_static_2hop_relu": float(rec["v783_static_2hop_relu"]), "banc_static_2hop_relu": float(rec["banc_m1_static_2hop_relu"]),
            "malecns_static_2hop_relu": float(static_scores["static_2hop_relu_signed"][i]),
            "v783_50hz": float(rec["v783_50hz"]), "v783_100hz": float(rec["v783_100hz"]), "v783_200hz": float(rec["v783_200hz"]),
            "banc_frozen_50hz": float(rec["banc_m1_frozen_50hz"]), "banc_anat_50hz": float(rec["banc_m1_anat_50hz"]),
            "banc_anat_100hz": float(rec["banc_m1_anat_100hz"]), "banc_anat_200hz": float(rec["banc_m1_anat_200hz"]),
            "malecns_intact_50hz_mean": float(real_runs["anat_intact"]["50hz"]["means"][i]),
            "malecns_intact_50hz_l": float(real_runs["anat_intact"]["50hz"]["l_rates"][i]),
            "malecns_intact_50hz_r": float(real_runs["anat_intact"]["50hz"]["r_rates"][i]),
            "malecns_intact_100hz_mean": float(real_runs["anat_intact"]["100hz"]["means"][i]),
            "malecns_intact_200hz_mean": float(real_runs["anat_intact"]["200hz"]["means"][i]),
            "malecns_intact_200hz_l": float(real_runs["anat_intact"]["200hz"]["l_rates"][i]),
            "malecns_frozen_50hz_mean": float(real_runs["frozen_0.275"]["50hz"]["means"][i]),
            "malecns_frozen_50hz_l": float(real_runs["frozen_0.275"]["50hz"]["l_rates"][i]),
            "malecns_frozen_200hz_mean": float(real_runs["frozen_0.275"]["200hz"]["means"][i]),
            "malecns_bilat_50hz_mean": float(real_runs["anat_bilat"]["50hz"]["means"][i]),
            "malecns_bilat_50hz_l": float(real_runs["anat_bilat"]["50hz"]["l_rates"][i]),
        })

    # Disinhibition percentiles & 3-brain shared top-15 intermediate overlap
    metrics_p = ["hop2_disinh_prod_sum", "hop2_disinh_max_bn", "hop2_disinh_bn_sum", "hop2_antag_epre_prod_sum", "hop2_net_disinh_minus_antag_prod", "hop3_disinh_epre_prod_sum", "hop3_disinh_epre_max_bn"]
    pct_male = {}
    for mk in metrics_p:
        ctrl_vals = np.array([ic_male[c][mk] for c in mapped_neg_inh_ctrls], dtype=np.float64)
        cand_out = {}
        for c in ("tentacular", "phantom", "marge", "usnea"):
            v = float(ic_male[c][mk])
            cand_out[c] = {"value": v, "pct_among_mapped_neg_inh": float(100.0 * np.mean(ctrl_vals < v)), "rank_above_ctrls": int(np.sum(ctrl_vals < v)), "n_ctrls": len(mapped_neg_inh_ctrls)}
        pct_male[mk] = {"n_ctrls": len(mapped_neg_inh_ctrls), "ctrl_median": float(np.median(ctrl_vals)), "ctrl_p75": float(np.percentile(ctrl_vals, 75)), "ctrl_p90": float(np.percentile(ctrl_vals, 90)), "ctrl_max": float(np.max(ctrl_vals)), "candidates": cand_out}

    overlap_3brains = {}
    for cand in ("tentacular", "phantom", "marge", "usnea"):
        rm = ic_male[cand]
        m_h2_ip = {tok for x in rm["top15_hop2_ipre"] for tok in (split_tokens(x["node"]["cell_type"]) | split_tokens(x["node"]["flywire_type"]))}
        m_h2_an = {tok for x in rm["top15_hop2_antag_epre"] for tok in (split_tokens(x["node"]["cell_type"]) | split_tokens(x["node"]["flywire_type"]))}
        m_h3_i1 = {tok for x in rm["top15_hop3_i1_epre"] for tok in (split_tokens(x["i1"]["cell_type"]) | split_tokens(x["i1"]["flywire_type"]))}
        v_s, b_s = manifest["v783_top15_types"][cand], manifest["banc_top15_types"][cand]
        overlap_3brains[cand] = {
            "shared_v783_malecns_hop2_ipre": sorted(set(v_s["hop2_ipre"]) & m_h2_ip),
            "shared_v783_malecns_hop3_i1": sorted(set(v_s["hop3_i1"]) & m_h3_i1),
            "shared_all3_hop2_ipre": sorted(set(v_s["hop2_ipre"]) & set(b_s["hop2_ipre"]) & m_h2_ip),
            "shared_all3_hop2_antag_epre": sorted(set(v_s["hop2_antag_epre"]) & set(b_s["hop2_antag_epre"]) & m_h2_an),
            "shared_all3_hop3_i1": sorted(set(v_s["hop3_i1"]) & set(b_s["hop3_i1"]) & m_h3_i1),
        }

    whole_brain_rates = {
        sname: {f"{hz}hz": {"mean_hz": real_runs[sname][f"{hz}hz"]["mean_whole_brain_hz"], "max_hz": real_runs[sname][f"{hz}hz"]["max_whole_brain_hz"]} for hz in (20, 50, 100, 200)}
        for sname, _ in scales
    }

    final = {
        "protocol_commit": "fd5d254bcb20d5b092d07cbb679dfc65a9cc4f61",
        "initial_protocol_commit": manifest["protocol_commit"],
        "input_hashes": input_hashes,
        "compute_provenance": {"gpu_name": gpu_name, "cpu_count": os.cpu_count(), "workers": args.workers, "bilat_rewire_trials": bilat_rw_trials, "equivalence_check": equiv_report},
        "graph_construction_and_anatomy": {
            "schema_columns": {"annotations": ann_tbl.column_names, "neurotransmitters": nt_tbl.column_names, "weights": w_tbl.column_names},
            "main_graph_malecns_v1": {
                "n_nodes": n_nodes, "glia_excluded": glia_excluded, "traced_with_valid_nt": traced_with_nt,
                "traced_nt_coverage": float(traced_with_nt / n_nodes), "self_loops_removed": self_loops_removed,
                "n_unique_non_self_edges": n_non_self_unique, "n_edges_valid_nt": int(len(pre_s)),
                "nt_coverage_on_non_self_edges": nt_edge_coverage, "sum_abs_weights": float(np.abs(w_raw_s).sum()),
                "region_counts": dict(region_counts), "superclass_counts": dict(superclass_counts),
            },
            "mn9_anatomy_and_wsyn_scale": {
                "mn9_l_body_id": mn9_l_bid, "mn9_r_body_id": mn9_r_bid,
                "mn9_l_meta": bid_meta[mn9_l_bid], "mn9_r_meta": bid_meta[mn9_r_bid],
                "mn9_incoming_nt_coverage": mn9_nt_cov,
                "n_edges_mn9_l": int(mn9_l_mask.sum()), "n_edges_mn9_r": int(mn9_r_mask.sum()),
                "n_edges_mn9_total": int(mn9_l_mask.sum() + mn9_r_mask.sum()),
                "s_in_mn9_l": s_in_l, "s_in_mn9_r": s_in_r, "s_in_mn9_malecns_mean": s_in_mean,
                "mn9_lr_asymmetry_ratio": float(s_in_l / max(1.0, s_in_r)),
                "s_in_mn9_v783_mean": s_in_v783_mean,
                "w_syn_frozen_mv": w_syn_frozen,
                "anat_intact_ratio": anat_intact_ratio, "w_syn_intact_mv": w_syn_intact,
                "anat_bilat_ratio": anat_bilat_ratio, "w_syn_bilat_mv": w_syn_bilat,
                "shared_presyn_type_syn_frac_l": float(shared_l / max(1.0, s_in_l)),
                "shared_presyn_type_syn_frac_r": float(shared_r / max(1.0, s_in_r)),
            },
        },
        "mapping_audit": {
            "sez106_mapped_count": 106 - len(missing_types), "sez106_missing_types": missing_types,
            "dimorphic_annotated_types": dimorphic_annotated,
            "n_nondirect_malecns": int((~is_direct_male).sum()),
            "n_nondirect_common_3brains": int(subsets["nondirect_common_3brains"].sum()),
            "per_type_sez106": mapping_records,
        },
        "whole_brain_firing_diagnostics": whole_brain_rates,
        "rewire_full_meta": rw_full_meta,
        "rewire_mn9_protected_meta": rw_prot_meta,
        "task1_activation_malecns": task1_eval,
        "cross_brain_correlations": cross_brain_corr,
        "per_type_3brains": per_type_3brains,
        "selk_full_cns_diagnosis": selk_diagnosis,
        "disinhibition_module4": {
            "n_mapped_neg_inh_ctrls": len(mapped_neg_inh_ctrls),
            "mapped_neg_inh_ctrls": mapped_neg_inh_ctrls,
            "anatomy_malecns": anat_male,
            "percentiles_vs_mapped_neg_inh": pct_male,
            "three_brain_top15_intermediate_overlap": overlap_3brains,
            "local_oracle_realizability_intact": oracle_results_intact,
            "local_oracle_sensitivity_frozen275": oracle_results_frozen275,
        },
        "wall_time_sec": float(time.time() - t_all),
    }

    out_path = out_dir / "mn9_malecns_results.json"
    out_path.write_text(json.dumps(final, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    log(f"DONE! Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB) in {final['wall_time_sec']:.1f}s")

    for sname in ("anat_intact", "frozen_0.275", "anat_bilat"):
        a106 = task1_eval[sname]["bilateral_mean"]["all_106"]
        a106l = task1_eval[sname]["intact_left_mn9"]["all_106"]
        ndm = task1_eval[sname]["bilateral_mean"]["nondirect_malecns"]
        nd3 = task1_eval[sname]["bilateral_mean"]["nondirect_common_3brains"]
        log(
            f"SUMMARY [{sname:12s}] all106_mean 50/100/200={a106['models']['50hz']['auprc']:.4f}/{a106['models']['100hz']['auprc']:.4f}/{a106['models']['200hz']['auprc']:.4f} "
            f"(L={a106l['models']['50hz']['auprc']:.4f}, 2hReLU={a106['models']['static_2hop_relu_signed']['auprc']:.4f}, rewMN9={a106['rewire_mn9_protected']['per_graph_auprc_mean']:.4f} [{a106['rewire_mn9_protected']['graph_level_rank_vs_primary']['wins']}/10]) | "
            f"nd_male={ndm['models']['50hz']['auprc']:.4f} (rew={ndm['rewire_mn9_protected']['per_graph_auprc_mean']:.4f}) | nd_3br={nd3['models']['50hz']['auprc']:.4f}"
        )


if __name__ == "__main__":
    main()
