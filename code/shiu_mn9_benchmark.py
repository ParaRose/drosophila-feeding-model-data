#!/usr/bin/env python3
"""FlyWire SEZ -> MN9 -> Rostrum Extension source-task benchmark.

Implements the locked protocol in SHIU_MN9_PROTOCOL.md:
1. Exact Brian2-verified sparse active-set Numba LIF reproduction of Shiu et al. (2024).
2. Gate 1 (100 Hz sugar GRN vs ST 1D) and Gate 2 (106 SEZ 50 Hz vs Sup Table 3).
3. Task 1: 106 SEZ cell types optogenetic activation vs rostrum extension (Sup Table 3),
   comparing real FlyWire LIF against:
   - K=10 full signed-degree-preserving rewired connectomes (rewire_full)
   - K=10 MN9-direct-input-protected signed-degree-preserving rewired connectomes (rewire_mn9_protected)
   - Static graph baselines (shortest path, excitatory shortest path, 1-hop signed,
     2-hop signed linear, 2-hop signed ReLU, 2-hop unsigned).
4. Strict held-out evaluation (threshold-free AUPRC/AUROC/Spearman with stratified bootstrap,
   LOOCV and 5-fold stratified CV threshold selection, novel-95 & non-direct >=2-hop subgroups,
   max-line vs mean-line sensitivity, frequency sensitivity 20/50/100/200 Hz).
5. Task 2: 11 SEZ cell types acute water-silencing (Supp Table 9B, 6-arm paired controls)
   comparing real FlyWire LIF against K=10 rewire_full and K=10 rewire_mn9_protected.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numba import njit
from scipy import sparse, stats

# Fixed FlyWire v630 neuron IDs from Shiu et al. (2024) figures.ipynb & ST 1A/ST 6C
MN9_L_FID = 720575940645521262
MN9_R_FID = 720575940660219265

SUGAR_GRN_FIDS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]

WATER_GRN_FIDS = [
    720575940612950568, 720575940631898285, 720575940606002609, 720575940612579053,
    720575940622902535, 720575940616177458, 720575940660292225, 720575940622486922,
    720575940613786774, 720575940629852866, 720575940625861168, 720575940613996959,
    720575940617857694, 720575940644965399, 720575940625203504, 720575940630553415,
    720575940635172191, 720575940634796536,
]

ZORRO_FIDS = [720575940629888530, 720575940611015122]  # Zorro_l, Zorro_r from ST 1A & ST 6C

ST1D_REF = [
    ("MN9_l", 720575940645521262, 49.0333333333333),
    ("MN9_r", 720575940660219265, 68.0),
    ("Zorro_l", 720575940629888530, 102.2),
    ("G2N_1_l", 720575940620874757, 69.4),
    ("Fdg_l", 720575940631997032, 38.3),
    ("roundup_l", 720575940623211725, 46.3333333333333),
]

WATER_SILENCING_ORDER = [
    ("Bract", "bract"),
    ("Clavicle", "clavicle"),
    ("Fudog", "fudog"),
    ("G2N-1", "G2N_1"),
    ("Phantom", "phantom"),
    ("Rattle", "rattle"),
    ("Roundup", "roundup"),
    ("Tophat", "tophat"),
    ("Tulip", "tulip"),
    ("Usnea", "usnea"),
    ("Zorro", "Zorro"),
]

W_SYN = 0.275


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_xlsx_sheet(path: Path, sheet_name: str) -> list[dict[str, str]]:
    ns = {
        "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with zipfile.ZipFile(path) as z:
        ss = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("m:si", ns):
                ss.append("".join(t.text or "" for t in si.findall(".//m:t", ns)))
        rels = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        }
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        for s in wb.findall(".//m:sheet", ns):
            if s.attrib["name"] != sheet_name:
                continue
            rid = s.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            target = rels[rid]
            if not target.startswith("xl/"):
                target = "xl/" + target.lstrip("/")
            sroot = ET.fromstring(z.read(target))
            out = []
            for r in sroot.findall(".//m:sheetData/m:row", ns):
                row_vals: dict[str, str] = {}
                for c in r.findall("m:c", ns):
                    ref = c.attrib["r"]
                    col = "".join(ch for ch in ref if ch.isalpha())
                    t = c.attrib.get("t")
                    v = c.find("m:v", ns)
                    if v is None:
                        is_el = c.find("m:is", ns)
                        val = (
                            "".join(x.text or "" for x in is_el.findall(".//m:t", ns))
                            if is_el is not None
                            else ""
                        )
                    elif t == "s":
                        val = ss[int(v.text)]
                    else:
                        val = v.text or ""
                    row_vals[col] = val
                if any(str(x).strip() for x in row_vals.values()):
                    out.append(row_vals)
            return out
    raise KeyError(f"Sheet not found: {sheet_name}")


def parse_line_rates(info_str: str, fallback_rate: float) -> list[float]:
    """Parse per-split-GAL4 rostrum extension rates from Sup Table 3 Col C."""
    # Remove citation suffix starting with 'See also'
    clean = re.split(r"[;,]\s*See also", info_str, flags=re.I)[0]
    # Handle formats like "SS47730: .9, SS47731: .8", "SS:50091: .9", "SS44928, SS44937: 0"
    rates: list[float] = []
    # Split into chunks by ';' or ',' when followed by SS... or after a number
    # Match patterns of one or more SS IDs followed by ':' and a numeric value
    pattern = re.compile(r"((?:SS[: ]?\d+(?:\s*,\s*)?)+)\s*:\s*([0-9]*\.?[0-9]+)")
    for m in pattern.finditer(clean):
        ss_ids = re.findall(r"SS[: ]?\d+", m.group(1))
        val = float(m.group(2))
        rates.extend([val] * max(1, len(ss_ids)))
    if not rates:
        return [fallback_rate]
    return rates


def load_sup_table_3(xlsx_path: Path, sez_dict: dict[str, list[int]]) -> list[dict]:
    rows = parse_xlsx_sheet(xlsx_path, "Sup Table 3 Predicted MN9 vs. o")
    sez_low = {k.lower(): k for k in sez_dict}
    records = []
    for r in rows[1:]:
        name = str(r.get("A", "")).strip()
        if not name:
            continue
        sez_key = name if name in sez_dict else sez_low[name.lower()]
        y_max = float(r.get("B", "0") or 0.0)
        info = str(r.get("C", "")).strip()
        line_rates = parse_line_rates(info, y_max)
        y_mean = float(np.mean(line_rates))
        prior_pub = bool(re.search(r"See also", info, flags=re.I))
        pub_50_l = float(r.get("M", "0") or 0.0)
        pub_50_r = float(r.get("N", "0") or 0.0)
        pub_200_l = float(r.get("S", "0") or 0.0)
        pub_200_r = float(r.get("T", "0") or 0.0)
        records.append(
            {
                "name": name,
                "sez_key": sez_key,
                "fids": [int(x) for x in sez_dict[sez_key]],
                "y_max": y_max,
                "y_mean": y_mean,
                "line_rates": line_rates,
                "prior_published": prior_pub,
                "info": info,
                "pub_shortest_path": str(r.get("D", "")).strip(),
                "pub_50hz_l": pub_50_l,
                "pub_50hz_r": pub_50_r,
                "pub_50hz_mean": 0.5 * (pub_50_l + pub_50_r),
                "pub_200hz_mean": 0.5 * (pub_200_l + pub_200_r),
            }
        )
    assert len(records) == 106, f"Expected 106 rows in Sup Table 3, got {len(records)}"
    return records


def load_supp_table_9b(xlsx_path: Path, sez_dict: dict[str, list[int]]) -> list[dict]:
    rows = parse_xlsx_sheet(xlsx_path, "Supp Table 9B Water silencing")
    records = []
    i = 1
    while i + 2 < len(rows):
        hdr = rows[i]
        ext = rows[i + 1]
        tot = rows[i + 2]
        label = str(hdr.get("A", "")).strip()
        matched_target = None
        for disp, skey in WATER_SILENCING_ORDER:
            if label.lower().startswith(disp.lower()):
                matched_target = (disp, skey)
                break
        if matched_target is not None:
            disp, skey = matched_target
            cols = ["B", "C", "D", "E", "F", "G"]
            k = [int(round(float(ext[c]))) for c in cols]
            n = [int(round(float(tot[c]))) for c in cols]
            p = [k[j] / n[j] for j in range(6)]
            # Arms: 0=Empty Off, 1=Empty On, 2=GAL4/+ Off, 3=GAL4/+ On, 4=Exp Off, 5=Exp On
            d_empty = p[1] - p[0]
            d_gal4 = p[3] - p[2]
            d_exp = p[5] - p[4]
            d_did = d_exp - 0.5 * (d_empty + d_gal4)
            # Two-sided Fisher exact tests: Exp On vs Exp Off, Exp On vs Empty On, Exp On vs GAL4/+ On
            _, p_on_off = stats.fisher_exact([[k[5], n[5] - k[5]], [k[4], n[4] - k[4]]])
            _, p_vs_empty = stats.fisher_exact([[k[5], n[5] - k[5]], [k[1], n[1] - k[1]]])
            _, p_vs_gal4 = stats.fisher_exact([[k[5], n[5] - k[5]], [k[3], n[3] - k[3]]])
            sig_impaired = bool(
                (p[5] < p[4])
                and (p[5] < p[1])
                and (p[5] < p[3])
                and (p_on_off < 0.05)
                and (p_vs_empty < 0.05)
                and (p_vs_gal4 < 0.05)
            )
            fids = ZORRO_FIDS if skey == "Zorro" else [int(x) for x in sez_dict[skey]]
            records.append(
                {
                    "name": disp,
                    "sez_key": skey,
                    "label": label,
                    "fids": fids,
                    "counts_extended": k,
                    "counts_total": n,
                    "rates": p,
                    "delta_empty": d_empty,
                    "delta_gal4": d_gal4,
                    "delta_exp": d_exp,
                    "delta_did": d_did,
                    "p_fisher_on_off": float(p_on_off),
                    "p_fisher_vs_empty_on": float(p_vs_empty),
                    "p_fisher_vs_gal4_on": float(p_vs_gal4),
                    "sig_impaired": sig_impaired,
                }
            )
            i += 3
        else:
            i += 1
    assert len(records) == 11, f"Expected 11 water silencing types, got {len(records)}"
    return records


@njit
def run_lif_brian2_exact(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    exc_idx: np.ndarray,
    exc_rate_hz: float,
    slnc_mask: np.ndarray,
    watch_idx: np.ndarray,
    n_trials: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Exact Brian2 discrete-time LIF schedule for Shiu et al. (2024)."""
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
    w_poi = W_SYN * 250.0
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
            # 1. groups (stateupdater) + 2. thresholds
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

            # 3. synapses (PoissonInput + delayed synaptic on_pre)
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

            # 4. resets
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


@njit
def rewire_block_swaps(
    indptr: np.ndarray,
    indices: np.ndarray,
    block_edge_ids: np.ndarray,
    edge_src: np.ndarray,
    n_attempts: int,
    seed: int,
) -> int:
    """In-place directed double-edge target swaps within one sign block."""
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
        # Reject parallel edges by scanning CSR rows of u1 and u2
        dup = False
        p0 = indptr[u1]
        p1 = indptr[u1 + 1]
        for k in range(p0, p1):
            if indices[k] == v2:
                dup = True
                break
        if dup:
            continue
        q0 = indptr[u2]
        q1 = indptr[u2 + 1]
        for k in range(q0, q1):
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
def verify_rewire_numba(
    indptr: np.ndarray,
    orig_indices: np.ndarray,
    new_indices: np.ndarray,
    weights: np.ndarray,
    n_nodes: int,
) -> tuple[int, int, int, np.ndarray, np.ndarray]:
    """Check self-loops, duplicate targets per row, exact replaced-edge count, and signed in-degrees."""
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
        # Check self-loops, dups, signed in-deg
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
        # Set intersection size on sorted rows
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


def build_rewired_graph(
    indptr: np.ndarray,
    orig_indices: np.ndarray,
    weights: np.ndarray,
    edge_src: np.ndarray,
    node_class: np.ndarray,
    orig_in_pos: np.ndarray,
    orig_in_neg: np.ndarray,
    protect_targets: set[int],
    seed: int,
) -> tuple[np.ndarray, dict]:
    n_nodes = len(indptr) - 1
    new_indices = orig_indices.copy()

    movable_mask = np.ones(len(orig_indices), dtype=bool)
    for t in protect_targets:
        movable_mask &= orig_indices != t

    pre_cls = node_class[edge_src]
    post_cls = node_class[orig_indices]

    total_attempts = 0
    total_accepted = 0
    block_stats = {}
    for pc in (0, 1):
        for qc in (0, 1, 2):
            eids = np.flatnonzero(movable_mask & (pre_cls == pc) & (post_cls == qc)).astype(np.int32)
            if len(eids) < 2:
                continue
            n_att = 2 * len(eids)
            acc = int(
                rewire_block_swaps(
                    indptr,
                    new_indices,
                    eids,
                    edge_src,
                    n_att,
                    seed * 31 + pc * 7 + qc,
                )
            )
            total_attempts += n_att
            total_accepted += acc
            block_stats[f"{pc}->{qc}"] = {"edges": int(len(eids)), "accepted": acc}

    self_loops, dup_edges, kept_edges, in_pos, in_neg = verify_rewire_numba(
        indptr, orig_indices, new_indices, weights, n_nodes
    )
    assert self_loops == 0, f"Rewire created {self_loops} self-loops"
    assert dup_edges == 0, f"Rewire created {dup_edges} duplicate edges"
    assert np.array_equal(in_pos, orig_in_pos), "Excitatory in-degree not conserved"
    assert np.array_equal(in_neg, orig_in_neg), "Inhibitory in-degree not conserved"

    if protect_targets:
        for t in protect_targets:
            orig_mask = orig_indices == t
            new_mask = new_indices == t
            assert np.array_equal(orig_mask, new_mask), f"Protected target {t} edges mutated"

    n_movable = int(movable_mask.sum())
    n_protected = len(orig_indices) - n_movable
    replaced_movable = n_movable - (kept_edges - n_protected)
    stats_out = {
        "seed": seed,
        "n_movable": n_movable,
        "n_protected": n_protected,
        "total_attempts": total_attempts,
        "total_accepted": total_accepted,
        "replaced_movable": int(replaced_movable),
        "replaced_fraction_movable": float(replaced_movable / max(1, n_movable)),
        "self_loops": int(self_loops),
        "dup_edges": int(dup_edges),
        "signed_indegree_conserved": True,
        "hash": hashlib.sha256(new_indices.tobytes()).hexdigest()[:16],
    }
    return new_indices, stats_out


def compute_static_baselines(
    pre: np.ndarray,
    post: np.ndarray,
    w_raw: np.ndarray,
    n_nodes: int,
    type_indices: list[np.ndarray],
    mn9_indices: list[int],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Compute zero-simulation static graph baselines and exact 1-hop direct flag."""
    # Reverse adjacency for shortest paths to {MN9_l, MN9_r}
    rev_all = sparse.csr_matrix((np.ones(len(pre), dtype=np.int8), (post, pre)), shape=(n_nodes, n_nodes))
    exc_m = w_raw > 0
    rev_exc = sparse.csr_matrix(
        (np.ones(int(exc_m.sum()), dtype=np.int8), (post[exc_m], pre[exc_m])),
        shape=(n_nodes, n_nodes),
    )

    def bfs_dist(adj_rev: sparse.csr_matrix, targets: list[int]) -> np.ndarray:
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

    # Target vector summing incoming weights into MN9_l + MN9_r
    e_mn9 = np.zeros(n_nodes, dtype=np.float64)
    for m in mn9_indices:
        e_mn9[m] = 1.0

    # 1-hop vector: (W @ e_mn9)[u] = sum_{m in MN9} w_{u, m}
    w1_vec = W.dot(e_mn9)
    w1_abs_vec = W_abs.dot(e_mn9)
    # 2-hop linear vector: (W @ (W @ e_mn9))[u] = sum_k w_{u,k} * w_{k, MN9}
    w2_vec = W.dot(w1_vec)
    w2_abs_vec = W_abs.dot(w1_abs_vec)

    n_types = len(type_indices)
    s_neg_sp = np.zeros(n_types, dtype=np.float64)
    s_exc_sp = np.zeros(n_types, dtype=np.float64)
    s_1hop = np.zeros(n_types, dtype=np.float64)
    s_2hop_lin = np.zeros(n_types, dtype=np.float64)
    s_2hop_relu = np.zeros(n_types, dtype=np.float64)
    s_2hop_uns = np.zeros(n_types, dtype=np.float64)
    is_direct_1hop = np.zeros(n_types, dtype=bool)

    for i, idxs in enumerate(type_indices):
        ds = dist_all[idxs]
        pos_d = ds[ds > 0]
        if len(pos_d) > 0:
            md = int(pos_d.min())
            s_neg_sp[i] = 1.0 / (1.0 + md)
            is_direct_1hop[i] = md == 1
        dse = dist_exc[idxs]
        pos_de = dse[dse > 0]
        if len(pos_de) > 0:
            s_exc_sp[i] = 1.0 / (1.0 + int(pos_de.min()))

        h1 = float(w1_vec[idxs].sum())
        h2 = float(w2_vec[idxs].sum())
        h1_abs = float(w1_abs_vec[idxs].sum())
        h2_abs = float(w2_abs_vec[idxs].sum())

        # ReLU 2-hop: sum_k max(0, sum_{u in c} w_{u,k}) * w_{k, MN9}
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
        is_direct_1hop,
    )


def average_precision_exact(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact tie-aware Average Precision matching sklearn.metrics.average_precision_score."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    y_s = y_true[order]
    s_s = y_score[order]
    # Group ties at distinct thresholds (end of each tie group)
    distinct_idx = np.where(np.diff(s_s) != 0)[0]
    thr_idxs = np.r_[distinct_idx, len(y_s) - 1]
    tps = np.cumsum(y_s)[thr_idxs]
    fps = (1 + thr_idxs) - tps
    precision = tps / (tps + fps)
    recall = tps / n_pos
    rec_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - rec_prev) * precision))


def auroc_exact(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Exact tie-aware AUROC via mid-ranks (Mann-Whitney U)."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(y_score)
    u = float(ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0)
    return u / (n_pos * n_neg)


def spearman_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.all(y_score == y_score[0]) or np.all(y_true == y_true[0]):
        return 0.0
    r = stats.spearmanr(y_true, y_score).statistic
    return 0.0 if np.isnan(r) else float(r)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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
    bacc = 0.5 * (rec + spec)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "balanced_accuracy": bacc,
        "mcc": mcc,
    }


def fit_best_f1_threshold(y_train: np.ndarray, s_train: np.ndarray) -> float:
    """Choose scalar threshold strictly on training fold maximizing F1 (tie-break: lowest positive)."""
    uniq = np.unique(s_train)
    # Candidate thresholds: midpoints between consecutive unique values, plus above max and at 0
    cands = [float(uniq[0] - 1.0), float(uniq[-1] + 1.0)]
    for j in range(len(uniq) - 1):
        cands.append(0.5 * float(uniq[j] + uniq[j + 1]))
    cands.sort()
    best_f1 = -1.0
    best_thr = cands[-1]
    for thr in cands:
        pred = (s_train > thr).astype(np.int32)
        f1 = binary_metrics(y_train, pred)["f1"]
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_thr = thr
    return best_thr


def evaluate_heldout_cv(y_bin: np.ndarray, scores: np.ndarray, seed: int = 20261122) -> dict:
    n = len(y_bin)
    # 1. LOOCV
    loo_pred = np.zeros(n, dtype=np.int32)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        thr = fit_best_f1_threshold(y_bin[mask], scores[mask])
        loo_pred[i] = int(scores[i] > thr)
    # 2. 5-fold stratified CV
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_bin == 1)
    neg_idx = np.flatnonzero(y_bin == 0)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    folds = [[] for _ in range(5)]
    for k, idx in enumerate(pos_idx):
        folds[k % 5].append(int(idx))
    for k, idx in enumerate(neg_idx):
        folds[k % 5].append(int(idx))
    cv5_pred = np.zeros(n, dtype=np.int32)
    for f in range(5):
        test_i = np.array(folds[f], dtype=np.int32)
        train_m = np.ones(n, dtype=bool)
        train_m[test_i] = False
        thr = fit_best_f1_threshold(y_bin[train_m], scores[train_m])
        cv5_pred[test_i] = (scores[test_i] > thr).astype(np.int32)
    return {
        "loocv": binary_metrics(y_bin, loo_pred),
        "cv5": binary_metrics(y_bin, cv5_pred),
        "zero_threshold": binary_metrics(y_bin, (scores > 0.0).astype(np.int32)),
    }


def stratified_bootstrap_summary(
    y_bin: np.ndarray,
    y_max: np.ndarray,
    y_mean: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    ref_model: str,
    n_boot: int = 2000,
    seed: int = 20261122,
) -> dict:
    rng = np.random.default_rng(seed)
    pos_idx = np.flatnonzero(y_bin == 1)
    neg_idx = np.flatnonzero(y_bin == 0)
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    boot_idx = np.empty((n_boot, n_pos + n_neg), dtype=np.int32)
    for b in range(n_boot):
        boot_idx[b, :n_pos] = rng.choice(pos_idx, size=n_pos, replace=True)
        boot_idx[b, n_pos:] = rng.choice(neg_idx, size=n_neg, replace=True)

    raw_boot = {m: {"auprc": np.empty(n_boot), "auroc": np.empty(n_boot), "rho_max": np.empty(n_boot), "rho_mean": np.empty(n_boot)} for m in scores_by_model}
    for b in range(n_boot):
        idx = boot_idx[b]
        yb = y_bin[idx]
        ymx = y_max[idx]
        ymn = y_mean[idx]
        for m, sc in scores_by_model.items():
            sb = sc[idx]
            raw_boot[m]["auprc"][b] = average_precision_exact(yb, sb)
            raw_boot[m]["auroc"][b] = auroc_exact(yb, sb)
            raw_boot[m]["rho_max"][b] = spearman_safe(ymx, sb)
            raw_boot[m]["rho_mean"][b] = spearman_safe(ymn, sb)

    out = {}
    for m, sc in scores_by_model.items():
        pt = {
            "auprc": average_precision_exact(y_bin, sc),
            "auroc": auroc_exact(y_bin, sc),
            "rho_max": spearman_safe(y_max, sc),
            "rho_mean": spearman_safe(y_mean, sc),
        }
        ci = {}
        diff_vs_ref = {}
        for k in ("auprc", "auroc", "rho_max", "rho_mean"):
            arr = raw_boot[m][k]
            ci[k] = [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
            if m != ref_model:
                d = raw_boot[ref_model][k] - arr
                p_two = float(2.0 * min(np.mean(d <= 0.0), np.mean(d >= 0.0)))
                diff_vs_ref[k] = {
                    "point_diff_ref_minus_model": float(
                        (
                            average_precision_exact(y_bin, scores_by_model[ref_model])
                            if k == "auprc"
                            else auroc_exact(y_bin, scores_by_model[ref_model])
                            if k == "auroc"
                            else spearman_safe(y_max, scores_by_model[ref_model])
                            if k == "rho_max"
                            else spearman_safe(y_mean, scores_by_model[ref_model])
                        )
                        - pt[k]
                    ),
                    "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                    "p_two_sided_boot": min(1.0, p_two),
                }
        out[m] = {"point": pt, "ci95": ci, "ref_minus_this": diff_vs_ref}
    return out


def run_all_sez_activation(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
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
        cnts, occ = run_lif_brian2_exact(
            indptr, indices, weights, idxs, rate_hz, slnc_none, mn9_watch, n_trials, seed_base + i * 97
        )
        means[i] = cnts.mean(axis=0)
        stds[i] = cnts.std(axis=0)
        if occ > max_occ:
            max_occ = occ
    return means, stds, max_occ


def run_water_silencing(
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
    water_grn_idx: np.ndarray,
    silencing_type_indices: list[np.ndarray],
    mn9_watch: np.ndarray,
    rate_hz: float,
    n_trials: int,
    seed_base: int,
) -> dict:
    n_nodes = len(indptr) - 1
    slnc_mask = np.zeros(n_nodes, dtype=np.uint8)
    base_cnts, _ = run_lif_brian2_exact(
        indptr, indices, weights, water_grn_idx, rate_hz, slnc_mask, mn9_watch, n_trials, seed_base
    )
    base_l = float(base_cnts[:, 0].mean())
    base_r = float(base_cnts[:, 1].mean())
    base_mean = 0.5 * (base_l + base_r)

    sil_l = []
    sil_r = []
    rel_r = []
    rel_mean = []
    for i, idxs in enumerate(silencing_type_indices):
        slnc_mask[:] = 0
        for idx in idxs:
            slnc_mask[idx] = 1
        cnts, _ = run_lif_brian2_exact(
            indptr, indices, weights, water_grn_idx, rate_hz, slnc_mask, mn9_watch, n_trials, seed_base + (i + 1) * 131
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
        "base_mn9_l": base_l,
        "base_mn9_r": base_r,
        "base_mn9_mean": base_mean,
        "sil_mn9_l": sil_l,
        "sil_mn9_r": sil_r,
        "rel_change_mn9_r": rel_r,
        "rel_change_mn9_mean": rel_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completeness", type=Path, default=Path("/tmp/projc-flywire-cache/completeness_630.csv"))
    parser.add_argument("--connectivity", type=Path, default=Path("/tmp/projc-flywire-cache/connectivity_630.parquet"))
    parser.add_argument("--sez-pickle", type=Path, default=Path("/tmp/projc-source02-probe/sez_neurons.pickle"))
    parser.add_argument("--xlsx", type=Path, default=Path("/tmp/projc-source02-probe/shiu_moesm2.xlsx"))
    parser.add_argument("--output", type=Path, default=Path("shiu_mn9_results.json"))
    parser.add_argument("--n-rewires", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=30)
    args = parser.parse_args()

    t_start = time.time()
    hashes = {
        "completeness_630_csv": sha256_file(args.completeness),
        "connectivity_630_parquet": sha256_file(args.connectivity),
        "sez_neurons_pickle": sha256_file(args.sez_pickle),
        "shiu_moesm2_xlsx": sha256_file(args.xlsx),
    }

    with open(args.completeness) as f:
        r = csv.reader(f)
        next(r)
        flyids = np.array([int(x[0]) for x in r], dtype=np.int64)
    flyid2idx = {int(fid): i for i, fid in enumerate(flyids)}
    n_nodes = len(flyids)

    tbl = pq.read_table(
        args.connectivity,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = tbl["Presynaptic_Index"].to_numpy().astype(np.int32)
    post = tbl["Postsynaptic_Index"].to_numpy().astype(np.int32)
    w_raw = tbl["Excitatory x Connectivity"].to_numpy().astype(np.float32)

    order = np.argsort(pre, kind="mergesort")
    pre_s = pre[order]
    post_s = post[order]
    w_s = (w_raw[order] * np.float32(W_SYN)).astype(np.float64)
    indptr = np.empty(n_nodes + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(np.bincount(pre_s, minlength=n_nodes), out=indptr[1:])

    # Precompute node sign class & exact signed in-degrees for rewiring verification
    node_class = np.full(n_nodes, 2, dtype=np.int8)  # 2 = zero out-degree sink
    has_pos = np.zeros(n_nodes, dtype=bool)
    has_neg = np.zeros(n_nodes, dtype=bool)
    has_pos[pre_s[w_s > 0]] = True
    has_neg[pre_s[w_s < 0]] = True
    assert not np.any(has_pos & has_neg), "Mixed-sign presynaptic node found"
    node_class[has_pos] = 0
    node_class[has_neg] = 1

    orig_in_pos = np.bincount(post_s[w_s > 0], minlength=n_nodes).astype(np.int32)
    orig_in_neg = np.bincount(post_s[w_s < 0], minlength=n_nodes).astype(np.int32)

    sez_dict = pickle.load(open(args.sez_pickle, "rb"))
    st3_records = load_sup_table_3(args.xlsx, sez_dict)
    st9b_records = load_supp_table_9b(args.xlsx, sez_dict)

    mn9_l_idx = flyid2idx[MN9_L_FID]
    mn9_r_idx = flyid2idx[MN9_R_FID]
    mn9_watch = np.array([mn9_l_idx, mn9_r_idx], dtype=np.int32)
    sugar_idx = np.array([flyid2idx[fid] for fid in SUGAR_GRN_FIDS], dtype=np.int32)
    water_idx = np.array([flyid2idx[fid] for fid in WATER_GRN_FIDS], dtype=np.int32)

    type_indices = [np.array([flyid2idx[fid] for fid in rec["fids"]], dtype=np.int32) for rec in st3_records]
    water_sil_indices = [np.array([flyid2idx[fid] for fid in rec["fids"]], dtype=np.int32) for rec in st9b_records]
    slnc_none = np.zeros(n_nodes, dtype=np.uint8)

    # Warmup JIT
    _ = run_lif_brian2_exact(indptr, post_s, w_s, sugar_idx[:1], 50.0, slnc_none, mn9_watch, 1, 0)

    # === REPRODUCTION GATE 1: 100 Hz Sugar GRN vs ST 1D ===
    gate1_watch = np.array([flyid2idx[fid] for _, fid, _ in ST1D_REF], dtype=np.int32)
    g1_cnts, g1_occ = run_lif_brian2_exact(
        indptr, post_s, w_s, sugar_idx, 100.0, slnc_none, gate1_watch, args.n_trials, 42
    )
    gate1_rows = []
    gate1_pass = True
    for j, (nm, fid, pub) in enumerate(ST1D_REF):
        sim_m = float(g1_cnts[:, j].mean())
        sim_sd = float(g1_cnts[:, j].std())
        z = (sim_m - pub) / (sim_sd / math.sqrt(args.n_trials) + 1e-9)
        if abs(z) >= 3.0:
            gate1_pass = False
        gate1_rows.append({"neuron": nm, "fid": fid, "sim_mean_hz": sim_m, "sim_sd_hz": sim_sd, "pub_st1d_hz": pub, "diff_hz": sim_m - pub, "z": z})
    print(f"[Gate 1] Sugar 100Hz pass={gate1_pass}, max|z|={max(abs(x['z']) for x in gate1_rows):.2f}, max_ring_occ={g1_occ}")

    # === REPRODUCTION GATE 2 + PRIMARY REAL LIF: 106 SEZ types at 50 Hz (plus 20/100/200 Hz) ===
    t_real = time.time()
    real_50_means, real_50_stds, real_50_occ = run_all_sez_activation(
        indptr, post_s, w_s, type_indices, mn9_watch, slnc_none, 50.0, args.n_trials, 50000
    )
    real_50_score = real_50_means.mean(axis=1)
    pub_50_score = np.array([rec["pub_50hz_mean"] for rec in st3_records], dtype=np.float64)
    g2_r = float(stats.pearsonr(real_50_score, pub_50_score).statistic)
    g2_mae = float(np.mean(np.abs(real_50_score - pub_50_score)))
    g2_max_abs = float(np.max(np.abs(real_50_score - pub_50_score)))
    gate2_pass = (g2_r > 0.995) and (g2_mae < 0.5)
    print(
        f"[Gate 2] 106 SEZ 50Hz ({time.time()-t_real:.1f}s) pass={gate2_pass}, "
        f"r={g2_r:.6f}, MAE={g2_mae:.4f} Hz, max|diff|={g2_max_abs:.3f} Hz, max_ring_occ={real_50_occ}"
    )
    if not (gate1_pass and gate2_pass):
        raise RuntimeError("Simulator reproduction gate failed; aborting comparison per protocol.")

    # Frequency sensitivity runs on real graph (20, 100, 200 Hz)
    real_freq_scores = {"real_lif_50hz": real_50_score}
    for f_hz, s_base in [(20.0, 20000), (100.0, 100000), (200.0, 200000)]:
        m_f, _, _ = run_all_sez_activation(
            indptr, post_s, w_s, type_indices, mn9_watch, slnc_none, f_hz, args.n_trials, s_base
        )
        real_freq_scores[f"real_lif_{int(f_hz)}hz"] = m_f.mean(axis=1)
    print(f"[Real LIF freq sensitivity 20/100/200 Hz done in {time.time()-t_real:.1f}s]")

    # === STATIC GRAPH BASELINES ===
    static_scores, is_direct_1hop = compute_static_baselines(
        pre_s, post_s, w_raw[order], n_nodes, type_indices, [mn9_l_idx, mn9_r_idx]
    )

    # === TASK 2 REAL LIF WATER SILENCING (160 Hz primary + 180/200 Hz sensitivity) ===
    water_real_160 = run_water_silencing(
        indptr, post_s, w_s, water_idx, water_sil_indices, mn9_watch, 160.0, args.n_trials, 70000
    )
    water_real_180 = run_water_silencing(
        indptr, post_s, w_s, water_idx, water_sil_indices, mn9_watch, 180.0, args.n_trials, 72000
    )
    water_real_200 = run_water_silencing(
        indptr, post_s, w_s, water_idx, water_sil_indices, mn9_watch, 200.0, args.n_trials, 74000
    )
    print(
        f"[Task 2 Real Water 160Hz] base MN9_r={water_real_160['base_mn9_r']:.2f} Hz, "
        f"base MN9_l={water_real_160['base_mn9_l']:.2f} Hz"
    )

    # === NULL FAMILIES: K=10 rewire_full & K=10 rewire_mn9_protected ===
    rewire_full_scores = []
    rewire_full_meta = []
    rewire_full_water = []
    t_rw = time.time()
    for k in range(args.n_rewires):
        seed_rw = 8101 + k
        rw_indices, meta = build_rewired_graph(
            indptr, post_s, w_s, pre_s, node_class, orig_in_pos, orig_in_neg, set(), seed_rw
        )
        rw_m, _, _ = run_all_sez_activation(
            indptr, rw_indices, w_s, type_indices, mn9_watch, slnc_none, 50.0, args.n_trials, 50000
        )
        rw_w = run_water_silencing(
            indptr, rw_indices, w_s, water_idx, water_sil_indices, mn9_watch, 160.0, args.n_trials, 70000
        )
        rewire_full_scores.append(rw_m.mean(axis=1))
        rewire_full_meta.append(meta)
        rewire_full_water.append(rw_w)
        print(
            f"  [rewire_full {k+1}/{args.n_rewires}] seed={seed_rw} replaced={meta['replaced_fraction_movable']:.4f} "
            f"nonzero_types={int((rw_m.mean(axis=1) > 0).sum())} water_base_r={rw_w['base_mn9_r']:.2f}Hz"
        )

    rewire_prot_scores = []
    rewire_prot_meta = []
    rewire_prot_water = []
    prot_set = {mn9_l_idx, mn9_r_idx}
    for k in range(args.n_rewires):
        seed_rw = 8201 + k
        rw_indices, meta = build_rewired_graph(
            indptr, post_s, w_s, pre_s, node_class, orig_in_pos, orig_in_neg, prot_set, seed_rw
        )
        rw_m, _, _ = run_all_sez_activation(
            indptr, rw_indices, w_s, type_indices, mn9_watch, slnc_none, 50.0, args.n_trials, 50000
        )
        rw_w = run_water_silencing(
            indptr, rw_indices, w_s, water_idx, water_sil_indices, mn9_watch, 160.0, args.n_trials, 70000
        )
        rewire_prot_scores.append(rw_m.mean(axis=1))
        rewire_prot_meta.append(meta)
        rewire_prot_water.append(rw_w)
        print(
            f"  [rewire_mn9_prot {k+1}/{args.n_rewires}] seed={seed_rw} replaced={meta['replaced_fraction_movable']:.4f} "
            f"nonzero_types={int((rw_m.mean(axis=1) > 0).sum())} water_base_r={rw_w['base_mn9_r']:.2f}Hz"
        )
    print(f"[All {2*args.n_rewires} rewired graphs completed in {time.time()-t_rw:.1f}s]")

    # === TASK 1 EVALUATION ===
    y_max = np.array([rec["y_max"] for rec in st3_records], dtype=np.float64)
    y_mean = np.array([rec["y_mean"] for rec in st3_records], dtype=np.float64)
    y_bin = (y_max > 0.0).astype(np.int32)
    y_bin_ge02 = (y_max >= 0.2).astype(np.int32)
    novel_mask = np.array([not rec["prior_published"] for rec in st3_records], dtype=bool)
    nondirect_mask = ~is_direct_1hop

    rewire_full_arr = np.vstack(rewire_full_scores)
    rewire_prot_arr = np.vstack(rewire_prot_scores)

    all_model_scores = {
        **real_freq_scores,
        "rewire_full_ensemble_mean": rewire_full_arr.mean(axis=0),
        "rewire_mn9_prot_ensemble_mean": rewire_prot_arr.mean(axis=0),
        **static_scores,
    }

    def eval_subset(mask: np.ndarray, label_bin: np.ndarray, subset_name: str) -> dict:
        ymx = y_max[mask]
        ymn = y_mean[mask]
        yb = label_bin[mask]
        sc_sub = {m: sc[mask] for m, sc in all_model_scores.items()}
        boot = stratified_bootstrap_summary(yb, ymx, ymn, sc_sub, "real_lif_50hz", n_boot=2000, seed=20261122)
        cv_res = {m: evaluate_heldout_cv(yb, sc_sub[m], seed=20261122) for m in sc_sub}

        # Per-rewired-graph metrics & empirical rank of real_lif_50hz among 1+K graphs
        def null_family_summary(arr: np.ndarray) -> dict:
            auprcs = [average_precision_exact(yb, arr[k][mask]) for k in range(len(arr))]
            aurocs = [auroc_exact(yb, arr[k][mask]) for k in range(len(arr))]
            rhos = [spearman_safe(ymx, arr[k][mask]) for k in range(len(arr))]
            f1_loos = [evaluate_heldout_cv(yb, arr[k][mask], seed=20261122)["loocv"]["f1"] for k in range(len(arr))]
            f1_z = [binary_metrics(yb, (arr[k][mask] > 0.0).astype(np.int32))["f1"] for k in range(len(arr))]
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
        "all_106_per_ge_02": eval_subset(np.ones(106, dtype=bool), y_bin_ge02, "all_106_per_ge_02"),
        "novel_95": eval_subset(novel_mask, y_bin, "novel_95"),
        "nondirect_ge2hop": eval_subset(nondirect_mask, y_bin, "nondirect_ge2hop"),
        "novel_and_nondirect": eval_subset(novel_mask & nondirect_mask, y_bin, "novel_and_nondirect"),
    }

    # === TASK 2 EVALUATION (11 Water Silencing types) ===
    w_did = np.array([rec["delta_did"] for rec in st9b_records], dtype=np.float64)
    w_exp = np.array([rec["delta_exp"] for rec in st9b_records], dtype=np.float64)
    w_sig = np.array([1 if rec["sig_impaired"] else 0 for rec in st9b_records], dtype=np.int32)

    def eval_water_vec(rel_change: list[float]) -> dict:
        v = np.asarray(rel_change, dtype=np.float64)
        # More negative rel_change predicts more negative delta_did (impairment -> -v predicts sig_impaired)
        return {
            "spearman_with_did": spearman_safe(w_did, v),
            "spearman_with_exp": spearman_safe(w_exp, v),
            "auroc_impaired": auroc_exact(w_sig, -v),
            "auprc_impaired": average_precision_exact(w_sig, -v),
        }

    task2_eval = {
        "n_types": len(st9b_records),
        "n_sig_impaired": int(w_sig.sum()),
        "real_160hz_mn9r": {**water_real_160, "metrics": eval_water_vec(water_real_160["rel_change_mn9_r"])},
        "real_160hz_mn9mean": {"metrics": eval_water_vec(water_real_160["rel_change_mn9_mean"])},
        "real_180hz_mn9r": {**water_real_180, "metrics": eval_water_vec(water_real_180["rel_change_mn9_r"])},
        "real_200hz_mn9r": {**water_real_200, "metrics": eval_water_vec(water_real_200["rel_change_mn9_r"])},
        "rewire_full_water": [
            {"seed": m["seed"], "base_mn9_r": w["base_mn9_r"], "metrics": eval_water_vec(w["rel_change_mn9_r"])}
            for m, w in zip(rewire_full_meta, rewire_full_water)
        ],
        "rewire_mn9_prot_water": [
            {"seed": m["seed"], "base_mn9_r": w["base_mn9_r"], "metrics": eval_water_vec(w["rel_change_mn9_r"])}
            for m, w in zip(rewire_prot_meta, rewire_prot_water)
        ],
        "per_type_records": [
            {
                **rec,
                "sim_160hz_mn9_r": water_real_160["sil_mn9_r"][i],
                "sim_160hz_rel_change_mn9_r": water_real_160["rel_change_mn9_r"][i],
                "sim_160hz_rel_change_mn9_mean": water_real_160["rel_change_mn9_mean"][i],
            }
            for i, rec in enumerate(st9b_records)
        ],
    }

    # Detailed per-cell-type table for Task 1
    per_type_task1 = []
    for i, rec in enumerate(st3_records):
        per_type_task1.append(
            {
                "name": rec["name"],
                "n_fids": len(rec["fids"]),
                "y_max": rec["y_max"],
                "y_mean": rec["y_mean"],
                "prior_published": rec["prior_published"],
                "is_direct_1hop_mn9": bool(is_direct_1hop[i]),
                "pub_50hz_mean": rec["pub_50hz_mean"],
                "real_lif_50hz_mean": float(real_50_score[i]),
                "real_lif_50hz_l": float(real_50_means[i, 0]),
                "real_lif_50hz_r": float(real_50_means[i, 1]),
                "real_lif_20hz_mean": float(real_freq_scores["real_lif_20hz"][i]),
                "real_lif_100hz_mean": float(real_freq_scores["real_lif_100hz"][i]),
                "real_lif_200hz_mean": float(real_freq_scores["real_lif_200hz"][i]),
                "rewire_full_mean": float(rewire_full_arr[:, i].mean()),
                "rewire_mn9_prot_mean": float(rewire_prot_arr[:, i].mean()),
                "static_1hop_signed": float(static_scores["static_1hop_signed"][i]),
                "static_2hop_signed_fixed": float(static_scores["static_2hop_signed_fixed"][i]),
                "static_2hop_relu_signed": float(static_scores["static_2hop_relu_signed"][i]),
                "static_2hop_unsigned_fixed": float(static_scores["static_2hop_unsigned_fixed"][i]),
                "static_neg_shortest_path": float(static_scores["static_neg_shortest_path"][i]),
            }
        )

    out_payload = {
        "input_hashes": hashes,
        "graph_summary": {
            "n_nodes": n_nodes,
            "n_edges": int(len(pre_s)),
            "n_exc_nodes": int((node_class == 0).sum()),
            "n_inh_nodes": int((node_class == 1).sum()),
            "n_sink_nodes": int((node_class == 2).sum()),
            "mn9_l_fid": MN9_L_FID,
            "mn9_r_fid": MN9_R_FID,
            "mn9_l_indeg_pos_neg": [int(orig_in_pos[mn9_l_idx]), int(orig_in_neg[mn9_l_idx])],
            "mn9_r_indeg_pos_neg": [int(orig_in_pos[mn9_r_idx]), int(orig_in_neg[mn9_r_idx])],
        },
        "reproduction_gates": {
            "gate1_sugar_100hz": {"passed": gate1_pass, "rows": gate1_rows, "max_ring_occ": int(g1_occ)},
            "gate2_sez106_50hz": {
                "passed": gate2_pass,
                "pearson_r": g2_r,
                "mae_hz": g2_mae,
                "max_abs_diff_hz": g2_max_abs,
                "max_ring_occ": int(real_50_occ),
            },
        },
        "rewire_full_meta": rewire_full_meta,
        "rewire_mn9_prot_meta": rewire_prot_meta,
        "task1_activation": task1_eval,
        "task2_water_silencing": task2_eval,
        "per_type_task1": per_type_task1,
        "wall_time_sec": float(time.time() - t_start),
    }

    with open(args.output, "w") as f:
        json.dump(out_payload, f, indent=2)
    print(f"\nSaved results to {args.output} in {out_payload['wall_time_sec']:.1f}s")

    # Compact console summary
    print("\n=== TASK 1 SUMMARY (ALL 106 SEZ CELL TYPES) ===")
    s106 = task1_eval["all_106"]
    for m in [
        "real_lif_50hz",
        "real_lif_20hz",
        "real_lif_100hz",
        "real_lif_200hz",
        "static_2hop_relu_signed",
        "static_2hop_signed_fixed",
        "static_2hop_unsigned_fixed",
        "static_1hop_signed",
        "static_exc_shortest_path",
        "static_neg_shortest_path",
    ]:
        pt = s106["bootstrap"][m]["point"]
        cv = s106["cv"][m]["loocv"]
        print(
            f"  {m:28s} | AUPRC={pt['auprc']:.4f} | AUROC={pt['auroc']:.4f} | "
            f"rho={pt['rho_max']:.4f} | LOOCV_F1={cv['f1']:.4f} (P={cv['precision']:.3f}, R={cv['recall']:.3f}, MCC={cv['mcc']:.4f})"
        )
    for nf_name in ("rewire_full_graphs", "rewire_mn9_prot_graphs"):
        nf = s106[nf_name]
        print(
            f"  {nf_name:28s} | AUPRC mean={nf['auprc_mean_sd_min_max'][0]:.4f} [{nf['auprc_mean_sd_min_max'][2]:.4f},{nf['auprc_mean_sd_min_max'][3]:.4f}] | "
            f"AUROC mean={nf['auroc_mean_sd_min_max'][0]:.4f} | real>null: {nf['real_exceeds_count_auprc']}/{args.n_rewires} (p={nf['empirical_upper_p_auprc']:.4f})"
        )

    print("\n=== TASK 2 SUMMARY (11 WATER SILENCING TYPES) ===")
    print(
        f"  real_160hz_mn9r: rho_did={task2_eval['real_160hz_mn9r']['metrics']['spearman_with_did']:.4f}, "
        f"AUROC={task2_eval['real_160hz_mn9r']['metrics']['auroc_impaired']:.4f}, "
        f"AUPRC={task2_eval['real_160hz_mn9r']['metrics']['auprc_impaired']:.4f}"
    )


if __name__ == "__main__":
    main()
