#!/usr/bin/env python3
"""Build a self-contained `mn9_malecns_input_manifest.json` from the archived study JSONs
and FlyWire v783 annotations so `code/mn9_malecns_benchmark.py` can be launched from a
clean repository checkout without private workspace files.

Usage (from repository root):
    python code/build_malecns_manifest.py
"""
from __future__ import annotations

import json
from pathlib import Path


def split_tokens(s: str | None) -> set[str]:
    if not s:
        return set()
    out = set()
    for part in str(s).replace(";", ",").split(","):
        p = part.strip()
        if p and p.lower() not in ("none", "nan"):
            out.add(p)
    return out


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent if (here.parent / "results").is_dir() else here
    res = root / "results" if (root / "results").is_dir() else root

    mc = json.loads((res / "mn9_malecns_results.json").read_text(encoding="utf-8"))
    tr = json.loads((res / "mn9_transfer_results.json").read_text(encoding="utf-8"))
    dis = json.loads((res / "mn9_disinhibition_results.json").read_text(encoding="utf-8"))

    # Reconstruct exact 106 SEZ mapping list
    tr_by_name = {r["name"]: r for r in tr["mapping_audit"]["per_type_sez106"]}
    mc_by_name = {r["name"]: r for r in mc["mapping_audit"]["per_type_sez106"]}
    sez106 = []
    for r in mc["per_type_3brains"]:
        nm = r["name"]
        mc_m = mc_by_name[nm]
        tr_m = tr_by_name.get(nm, {})
        cts = list(mc_m.get("v783_cell_types", []))
        sez106.append({
            "name": nm,
            "y_bin": int(r["y_bin"]),
            "y_max": float(r["y_max"]),
            "y_mean": float(r["y_mean"]),
            "prior_published": bool(r["prior_published"]),
            "v783_cell_types": cts,
            "v783_hemibrain_types": list(tr_m.get("v783_hemibrain_types", [])),
        })

    # Reconstruct 44 purely inhibitory negative SEZ controls from malecns/portability results
    neg_inh_ctrl44 = sorted(mc["disinhibition_module4"].get("mapped_neg_inh_ctrls", []))

    # Reconstruct v783 & BANC top-15 intermediate type sets for the 4 candidates
    def extract_top15(brain_key: str) -> dict:
        out = {}
        br = dis.get("module1_anatomy", {}).get(brain_key, {})
        ic = br.get("inh_candidates", br)
        for cand in ("tentacular", "phantom", "marge", "usnea"):
            c = ic.get(cand, {})
            h2_ip = sorted({t for x in c.get("top15_hop2_ipre", []) for t in split_tokens(x.get("node", {}).get("cell_type")) | split_tokens(x.get("node", {}).get("fafb_cell_type"))})
            h2_an = sorted({t for x in c.get("top15_hop2_antag_epre", []) for t in split_tokens(x.get("node", {}).get("cell_type")) | split_tokens(x.get("node", {}).get("fafb_cell_type"))})
            h3_i1 = sorted({t for x in c.get("top15_hop3_i1_epre", []) for t in split_tokens(x.get("i1", {}).get("cell_type")) | split_tokens(x.get("i1", {}).get("fafb_cell_type"))})
            out[cand] = {"hop2_ipre": h2_ip, "hop2_antag_epre": h2_an, "hop3_i1": h3_i1}
        return out

    # Extract v783 MN9 presynaptic types from disinhibition / transfer audit
    v783_mn9_pre = sorted(dis.get("module1_anatomy", {}).get("v783", {}).get("mn9_presyn_types", []))
    if not v783_mn9_pre:
        v783_mn9_pre = sorted({t for r in sez106 for t in r["v783_cell_types"]})

    manifest = {
        "protocol_commit": mc.get("initial_protocol_commit", "v7.0-public-release"),
        "w_syn_frozen_mv": 0.275,
        "s_in_mn9_v783_mean": 5732.5,
        "mn9_malecns_body_ids": {"left": 10331, "right": 16949},
        "sez106": sez106,
        "extra_crosslab_types": {
            "SELK_DNg68": {"v783_cell_types": ["DNg68"]},
            "Fdg_CB0038": {"v783_cell_types": ["CB0038"]},
            "Amulet_CB0626": {"v783_cell_types": ["CB0626"]},
        },
        "neg_inh_ctrl44": neg_inh_ctrl44,
        "v783_mn9_presyn_types": v783_mn9_pre,
        "v783_top15_types": extract_top15("v783"),
        "banc_top15_types": extract_top15("banc"),
    }

    out_path = here / "mn9_malecns_input_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes; sez106={len(sez106)}, ctrl44={len(neg_inh_ctrl44)})")


if __name__ == "__main__":
    main()
