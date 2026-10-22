"""
stage_data_v911.py — DATA 階段：普查（census_v911）＋ 逐顆特徵表（frames）落地。

流程（PLAN §1、§6）：
  1. 讀 dataset_v9（40 顆）與老師電芯容量表（GA1–GA7）→ census_v911.csv（欄位同 census_v97，含 GA 列）。
  2. 依 §6 每來源之主設定（視窗／濾波／特徵族／正規化）建立逐顆特徵表：
       build_cell_frame_v911 → （own）gaps_v911.apply_rule（cyc_A／cyc_c／drop_flag）→ dropna
       → norm（none／REL；REL 參考＝該顆前 3 循環中位數，屬訓練段）
     建置順序與 SEARCH_SOH 階段完全相同（沿用 ox P-B 檢查點之等值前提）。
  3. 每顆落地 results_v911/frames/<src>__<cell>.csv（own 另含 cyc_A／cyc_c／drop_flag 之完整表）。
     GA 顆為容量表（無 IC）→ 落地 cyc／cap_Ah／soh／maxT。
"""
import os, json
import numpy as np, pandas as pd

import data_v911 as D, soh_v911 as S, gaps_v911 as G
import config_v911 as C


def build_frames(src, log=print):
    """回傳 (modeling_frames, full_frames, info)：modeling_frames 為下游建模用（own 已去斷點、cyc=cyc_c）。"""
    sel = C.soh_sel(src); cfg = C.soh_cfg_of(sel); win = tuple(sel["window"])   # v9.11：每來源前處理選定（SOH_SELECTED_V911）
    gate = sel["label_gate"]                                                     # A2 開關（§6 選定）
    cells = {c: d for c, d in D.load_ica_cells().items()
             if d["src"] == src and not (src == "nasa" and c not in D.JOURNAL_NASA_CELLS)}
    mod, full, info = {}, {}, {}
    for c, d in sorted(cells.items()):
        fr, raw, ncl = S.build_cell_frame_v911(d, cfg, win, sel["family"])
        # A2（v9.11）：標籤一致性閘規則對全來源計算；是否生效依該來源選定之 label_gate
        keep, fl, gi = G.apply_rule(fr, **C.DATA["gap_params"])
        fl = fl.copy(); fl["label_gate"] = gate            # 落地之完整表：cyc_A／cyc_c／drop_flag／label_gate
        full[c] = fl
        info[c] = dict(label_gate=gate, n_before=gi["n_before"], n_after=gi["n_after"], n_dropped=gi["n_dropped"],
                       drop_cyc_A=gi["drop_cyc_A"], n_gaps=len(gi["gaps"]))
        fr = S.apply_label_gate(keep, fl, gate)            # 建模用：on→cyc_c 且剔除；off→cyc_A 且不剔除
        for dk in ("d0", "d50", "d100"):                   # A1：逐顆無效循環計數（於 A2 之後、dropna 之前）
            info[c][f"n_seg_invalid_{dk}"] = int((fr[f"seg_valid_{dk}"] == 0).sum())
        f2 = fr.dropna().reset_index(drop=True)
        info[c]["n_dropna"] = int(len(fr) - len(f2))
        info[c]["n_model_rows"] = int(len(f2))
        info[c]["n_clean_total"] = int(sum(ncl.values()))
        info[c]["q_ref_Ah"] = float(d["qref"])
        mod[c] = f2
    if sel["rel_nref"]:                                    # v9.11：norm ∈ {none, REL3, REL5}
        mod = {c: S.rel_normalize(f, nref=sel["rel_nref"]) for c, f in mod.items()}
    log(f"  DATA {src}: {len(mod)} 顆，組態 {sel['config_id']}，正規化 {sel['norm']}，A2 label_gate={gate}，"
        f"濾波 {sel['filter_key']}（{sel['filter']}，σ={sel['sigma_mV']} mV／SG win={sel['sg_window_pts']}）")
    return mod, full, info


def main(RES, log=print):
    os.makedirs(RES, exist_ok=True)
    FR = os.path.join(RES, "frames"); os.makedirs(FR, exist_ok=True)
    ica = D.load_ica_cells(); cap = D.load_own_capacity_cells()
    cen = D.census(ica, cap)
    log(f"  census_v911（基礎）：{len(cen)} 列（ICA {len(ica)} 顆＋容量表 {len(cap)} 顆）")

    meta = dict(generated_by="stage_data_v911",
                soh_selected=C.SOH_SELECTED, soh_selected_v911=C.SOH_SELECTED_V911,
                soh_resolved={s: C.soh_sel(s) for s in C.DATA["sources_soh"]},
                gap_params=C.DATA["gap_params"], sources={})
    qrefs = {}
    for src in C.DATA["sources_soh"]:
        mod, full, info = build_frames(src, log)
        for c, f in full.items():
            f.to_csv(os.path.join(FR, f"{src}__{c}.csv"), index=False, float_format="%.17g")
        meta["sources"][src] = dict(config=C.soh_sel(src), cells=sorted(mod), info=info,
                                    n_features=int(len([k for k in mod[sorted(mod)[0]].columns
                                                        if k not in S.BASE_COLS])))
        qrefs.update({c: info[c]["q_ref_Ah"] for c in info})
    # GA：容量表（只進 B2）
    ga_info = {}
    for c, d in sorted(cap.items()):
        df = pd.DataFrame(dict(cyc=d["cyc"], cap_Ah=d["cap_Ah"], soh=d["soh"], maxT=d["maxT"]))
        df.to_csv(os.path.join(FR, f"GA__{c}.csv"), index=False, float_format="%.17g")
        ga_info[c] = dict(n_cycles=int(d["n_cycles"]), q_ref_Ah=float(d["qref"]), file=d["file"])
        qrefs[c] = float(d["qref"])
    meta["sources"]["GA"] = dict(kind="capacity-only", cells=sorted(cap), info=ga_info)
    meta["q_ref_Ah"] = qrefs

    # ── v9.11 census 擴充欄（A1／A2 逐顆揭露）───────────────────────────────────
    add = {}
    for src in C.DATA["sources_soh"]:
        for c, i in meta["sources"][src]["info"].items():
            add[(src, c)] = dict(label_gate=i.get("label_gate"),
                                 n_seg_invalid_d0=i.get("n_seg_invalid_d0"), n_seg_invalid_d50=i.get("n_seg_invalid_d50"),
                                 n_seg_invalid_d100=i.get("n_seg_invalid_d100"), n_label_dropped=i.get("n_dropped"),
                                 dropped_cycA=";".join(f"{x:g}" for x in i.get("drop_cyc_A", [])),
                                 n_model_rows=i.get("n_model_rows"))
    for k in ["label_gate", "n_seg_invalid_d0", "n_seg_invalid_d50", "n_seg_invalid_d100",
              "n_label_dropped", "dropped_cycA", "n_model_rows"]:
        cen[k] = [add.get((r.source, r.cell), {}).get(k) for r in cen.itertuples()]
    cen.to_csv(os.path.join(RES, "census_v911.csv"), index=False)
    log(f"  census_v911.csv：{len(cen)} 列（含 A1 seg_invalid／A2 label_dropped 欄）")
    with open(os.path.join(RES, "frames_meta_v911.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=lambda o: (o.item() if hasattr(o, "item") else str(o)))
    log(f"  frames/：{len(os.listdir(FR))} 個檔案")
    return meta


_FRCACHE = {}


def frames_for(RES, src):
    """建模用逐顆表（記憶體內重建；與 DATA 階段落地之 frames 同一計算路徑，避免 CSV 往返之末位差）。"""
    if src not in _FRCACHE:
        _FRCACHE[src] = build_frames(src, log=lambda m: None)[0]
    return _FRCACHE[src]


def load_frames(RES, src):
    """由落地之 frames 還原建模用表（與 build_frames 之 modeling_frames 相同）。"""
    FR = os.path.join(RES, "frames"); sel = C.soh_sel(src)
    out = {}
    for fn in sorted(os.listdir(FR)):
        if not fn.startswith(src + "__") or not fn.endswith(".csv"):
            continue
        c = fn[len(src) + 2:-4]
        f = pd.read_csv(os.path.join(FR, fn))
        gate = str(f["label_gate"].iloc[0])                # A2 開關（由 frames csv 之 label_gate 欄還原）
        if gate == "on":
            f = f[f.drop_flag == 0].copy(); f["cyc"] = f["cyc_c"]
        else:
            f = f.copy(); f["cyc"] = f["cyc_A"]; f["cyc_c"] = f["cyc_A"]
        f = f.dropna().reset_index(drop=True)
        out[c] = f
    if sel["rel_nref"]:
        out = {c: S.rel_normalize(f, nref=sel["rel_nref"]) for c, f in out.items()}
    return out


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_v911"))
