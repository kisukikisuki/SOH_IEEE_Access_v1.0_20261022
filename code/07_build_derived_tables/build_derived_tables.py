#!/usr/bin/env python3
"""build_derived_tables.py — 34 顆電芯逐列衍生資料表（data/derived_per_row/*.csv）、cells_summary.csv 與驗證紀錄。

新撰（公開程式包）。本程式不改動任何封存檔；只讀封存 frames 原文，於每列右側附加 ann_* 標註欄。
邏輯引用（逐條；行號以作者封存樹 5版\\02_程式 為準）：
  ann_measured_label / ann_n_meas_ev
      逐事件實測旗標：03_審查後處理/addendum_C2.py L19–22（seg、cs=sort(cap_seg)、nxt=append(seg[1:],inf)、
      meas_i = any(seg_i < cs < nxt_i)）；同式複製於 05_審稿補跑/s5_common.py L288–292（measured_events）。
      逐事件→逐循環列：05_審稿補跑/s5_common.py L295–307（rows_flag_from_events：該循環 cyc_A 之任一事件為實測即實測）、
      05_審稿補跑/s7_meas_flag.py L14–23（n_meas_ev＝該循環實測事件數、is_measured＝n_meas_ev>0）。
  ann_A2_drop / ann_in_model / ann_eval_idx / ann_cyc_axis
      02_SOH重跑/pv911_run_SOH/02_programs/stage_data_v911.py L111–129（load_frames：label_gate=on → 取 drop_flag==0 且
      cyc=cyc_c；off → cyc=cyc_A、cyc_c=cyc_A；之後 dropna）＋ soh_v911.py L196–201（apply_delta：δ=inf 不過濾，
      否則取 seg_valid_<δ>==1）；呼叫順序見 02_SOH重跑/run_soh_v3.py L56–65（get_frames）。
  ann_A1_pass
      soh_v911.py L196–201（apply_delta 之列條件；δ=inf 恆為 1）；seg_valid_* 欄之定義見 ic_v911.py L159–168、soh_v911.py L230–232。
  ann_ref_M10 / ann_ref_M20
      stage_soh_v911.py L205（idx=arange(n)、is_ref = idx % Mref == 0；n＝建模列數）；殘差 GPR 之輸入為 Xte.cyc（L206、L224）。
  cluster（cells_summary）
      05_審稿補跑/s6_cluster_bootstrap.py L44–54（NASA_<cell>；OWN_<campaign>，campaign 取 post_review/c2_own_campaign_split.csv；OX_<組>）。
  label_stratum
      05_審稿補跑/s5_common.py L240（實測標籤子集＝非 Oxford 或 Oxford G7–G10；其餘 Oxford G1/G3/G5 為內插子集）。
鎖定組態（label_gate、delta_key）取自封存 frames_meta_v911.json 之 soh_resolved（＝config_v911.py L410–432 SOH_SELECTED_V911），
並與本檔 LOCKED 常數比對；不一致即中止。
用法（輸入路徑全部為參數）：
  python build_derived_tables.py --frames <results_R3_dedup/frames> --meta <results_R3_dedup/frames_meta_v911.json>
      --npz <dataset_v9_npz_R3dedup> --camp <post_review/c2_own_campaign_split.csv> --out-data <公開程式/data>
      [--records <soh_records_v911.csv>] [--predictions <soh_predictions_v911.json>] [--c2 <c2_measured_fraction_per_cell.csv>]
      [--measflag <frames_meas_flag_R3.csv>] [--census <census_v911.csv>] [--core-dir <pv911_run_SOH/02_programs> --core-res <results_R3_dedup>]
      [--log <build_log.json>]
--npz 為作者端資料（npz 不公開）；公開使用者可由衍生表之 ann_n_meas_ev 與 n_ev 欄直接重算實測參考比例。
"""
import argparse, os, sys, json, hashlib, platform, datetime
import numpy as np, pandas as pd

SOURCES = ("nasa", "own", "ox")
LOCKED = {"nasa": dict(label_gate="off", delta_key="inf"),     # config_v911.py L411–412
          "own": dict(label_gate="on", delta_key="d0"),        # config_v911.py L418–419
          "ox": dict(label_gate="on", delta_key="d0")}         # config_v911.py L425–426
M_LIST = (10, 20)
MEAS_OX_GROUPS = ("G7", "G8", "G9", "G10")                     # s5_common.py L240；s6_cluster_bootstrap.py MEAS_GROUPS
ANN_COLS = ["ann_measured_label", "ann_n_meas_ev", "ann_A1_pass", "ann_A2_drop", "ann_in_model", "ann_eval_idx",
            "ann_cyc_axis", "ann_ref_M10", "ann_ref_M20"]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def measured_events(seg, cap_seg):
    """複製自 addendum_C2.py L19–22（＝s5_common.measured_events）。"""
    seg = np.asarray(seg, float); cs = np.sort(np.asarray(cap_seg, float)); n = len(seg)
    nxt = np.append(seg[1:], np.inf)
    return np.array([np.any((cs > seg[i]) & (cs < nxt[i])) for i in range(n)])


def rows_meas_from_events(row_cycA, ev_cycA, ev_flag):
    """s5_common.rows_flag_from_events（L295–307）之規則，並回傳每循環實測事件數（s7_meas_flag.py L21–22 n_meas_ev）。"""
    cnt, anym = {}, {}
    for a, fl in zip(np.asarray(ev_cycA, float), np.asarray(ev_flag, bool)):
        cnt[a] = cnt.get(a, 0) + int(fl); anym[a] = anym.get(a, False) or bool(fl)
    flag = np.zeros(len(row_cycA), int); nme = np.zeros(len(row_cycA), int); unm = 0
    for i, a in enumerate(np.asarray(row_cycA, float)):
        if a in anym:
            flag[i] = int(anym[a]); nme[i] = cnt[a]
        else:
            unm += 1
    return flag, nme, unm


def model_rows(f, gate, dkey):
    """stage_data_v911.load_frames L120–126 ＋ soh_v911.apply_delta L196–201（以原列號追蹤）。"""
    g = f.copy(); g["_row"] = np.arange(len(f))
    if gate == "on":
        g = g[g.drop_flag == 0].copy(); g["cyc"] = g["cyc_c"]
    else:
        g = g.copy(); g["cyc"] = g["cyc_A"]; g["cyc_c"] = g["cyc_A"]
    g = g.dropna().reset_index(drop=True)
    if dkey != "inf":
        g = g[g[f"seg_valid_{dkey}"] == 1].reset_index(drop=True)
    return g


def group_of(src, cell):
    return cell.split("__")[0] if src == "ox" else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True); ap.add_argument("--meta", required=True); ap.add_argument("--npz", required=True)
    ap.add_argument("--camp", required=True); ap.add_argument("--out-data", required=True)
    ap.add_argument("--records"); ap.add_argument("--predictions"); ap.add_argument("--c2"); ap.add_argument("--measflag"); ap.add_argument("--census")
    ap.add_argument("--core-dir"); ap.add_argument("--core-res"); ap.add_argument("--log")
    a = ap.parse_args()
    t0 = utc()
    OUTD = os.path.join(a.out_data, "derived_per_row"); os.makedirs(OUTD, exist_ok=True)
    meta = json.load(open(a.meta, encoding="utf-8"))
    for s in SOURCES:
        r = meta["soh_resolved"][s]
        assert r["label_gate"] == LOCKED[s]["label_gate"] and r["delta_key"] == LOCKED[s]["delta_key"], (s, r)
    qref = {k: float(v) for k, v in meta["q_ref_Ah"].items()}
    camp = pd.read_csv(a.camp).drop_duplicates("cell").set_index("cell").campaign.to_dict()
    files = sorted(fn for fn in os.listdir(a.frames) if fn.endswith(".csv"))
    log = dict(program="build_derived_tables.py", start_utc=t0, python=sys.version, platform=platform.platform(),
               numpy=np.__version__, pandas=pd.__version__, inputs={}, outputs={}, checks={}, per_cell=[])
    for k in ("meta", "camp", "records", "predictions", "c2", "measflag", "census"):
        p = getattr(a, k)
        if p:
            log["inputs"][k] = dict(path=os.path.basename(p), sha256=sha256(p))
    summ, cols_by_src, mf_rows = [], {}, []
    for fn in files:
        src, cell = fn[:-4].split("__", 1)
        assert src in SOURCES, fn
        gate, dkey = LOCKED[src]["label_gate"], LOCKED[src]["delta_key"]
        p = os.path.join(a.frames, fn)
        raw = open(p, "rb").read().decode("utf-8")
        lines = raw.splitlines()
        f = pd.read_csv(p)
        assert len(lines) == len(f) + 1, fn
        hdr = lines[0].split(",")
        assert hdr == list(f.columns), fn
        cols_by_src.setdefault(src, hdr)
        assert cols_by_src[src] == hdr, ("columns differ within source", fn)
        n = len(f)
        # 實測旗標（npz 事件 → 循環列）
        z = np.load(os.path.join(a.npz, f"{src}__{cell}.npz"), allow_pickle=True)
        mev = measured_events(z["seg"], z["cap_seg"])
        meas, nme, unm = rows_meas_from_events(f.cyc_A.to_numpy(float), z["cyc_A"], mev)
        # 閘門與建模列
        A1 = np.ones(n, int) if dkey == "inf" else (f[f"seg_valid_{dkey}"].to_numpy() == 1).astype(int)
        A2 = ((f.drop_flag.to_numpy() == 1) & (gate == "on")).astype(int)
        g = model_rows(f, gate, dkey)
        inm = np.zeros(n, int); inm[g["_row"].to_numpy()] = 1
        eidx = np.full(n, -1); eidx[g["_row"].to_numpy()] = np.arange(len(g))
        cyc_axis_col = "cyc_c" if gate == "on" else "cyc_A"
        refs = {M: ((inm == 1) & (eidx % M == 0)).astype(int) for M in M_LIST}
        # 原 frames 中 cyc_axis 欄之原字串（不重新格式化）
        jcol = hdr.index(cyc_axis_col)
        out_lines = [lines[0] + "," + ",".join(ANN_COLS)]
        for i in range(n):
            fields = lines[i + 1].split(",")
            assert len(fields) == len(hdr), (fn, i)
            ca = fields[jcol]
            out_lines.append(lines[i + 1] + "," + ",".join([str(meas[i]), str(nme[i]), str(A1[i]), str(A2[i]), str(inm[i]),
                                                            (str(eidx[i]) if eidx[i] >= 0 else ""), ca, str(refs[10][i]), str(refs[20][i])]))
        op = os.path.join(OUTD, fn)
        with open(op, "w", encoding="utf-8", newline="\n") as fo:
            fo.write("\n".join(out_lines) + "\n")
        # 回讀比對：原欄位逐值等於封存 frames
        o = pd.read_csv(op)
        orig_equal = bool(o[hdr].equals(f))
        # 建模列之 cyc 軸（數值）與 g.cyc 一致
        cyc_ok = bool(np.array_equal(o.loc[inm == 1, "ann_cyc_axis"].to_numpy(float), g["cyc"].to_numpy(float)))
        a2kept = (A2 == 0)
        rec = dict(source=src, cell=cell, file=fn, n_rows=n, n_events=int(f.n_ev.sum()), n_events_npz=int(len(z["seg"])),
                   n_ref_discharges=int(len(z["cap_seg"])), n_measured_events=int(mev.sum()),
                   measured_ref_fraction_event=float(mev.mean()), n_rows_measured=int(meas.sum()), measured_row_fraction=float(meas.mean()),
                   n_rows_unmapped_meas=int(unm), label_gate=gate, delta_key=dkey,
                   n_A2_removed=int(A2.sum()), n_A1_removed=int(((A1 == 0) & a2kept).sum()), n_A1_fail_all_rows=int((A1 == 0).sum()),
                   n_dropna_removed=int(a2kept.sum() - len(model_rows(f, gate, "inf"))),
                   n_in_model=int(inm.sum()), n_ref_M10=int(refs[10].sum()), n_ref_M20=int(refs[20].sum()),
                   n_eval_M10=int(inm.sum() - refs[10].sum()), n_eval_M20=int(inm.sum() - refs[20].sum()),
                   q_ref_Ah=qref[cell], orig_columns_equal_archived=orig_equal, cyc_axis_equals_model_cyc=cyc_ok,
                   out_sha256=sha256(op))
        log["per_cell"].append(rec)
        grp = group_of(src, cell)
        cluster = ("NASA_" + cell) if src == "nasa" else ("OWN_" + camp[cell]) if src == "own" else ("OX_" + grp)
        stratum = "measured" if (src != "ox" or grp in MEAS_OX_GROUPS) else "interpolated"
        summ.append(dict(source=src, cell=cell, group=(grp if src == "ox" else ""), own_campaign=(camp[cell] if src == "own" else ""),
                         cluster=cluster, label_stratum=stratum, q_ref_Ah=qref[cell], n_rows=n, n_events_pre_gate=int(f.n_ev.sum()),
                         n_A1_removed=rec["n_A1_removed"], n_A2_removed=rec["n_A2_removed"], n_dropna_removed=rec["n_dropna_removed"],
                         n_in_model=rec["n_in_model"], n_measured_events=rec["n_measured_events"],
                         measured_ref_fraction=rec["measured_ref_fraction_event"], n_rows_measured=rec["n_rows_measured"],
                         n_ref_M10=rec["n_ref_M10"], n_ref_M20=rec["n_ref_M20"], n_eval_M10=rec["n_eval_M10"], n_eval_M20=rec["n_eval_M20"]))
        mf_rows.append(pd.DataFrame(dict(source=src, cell=cell, cyc=f.cyc_A.to_numpy(float), n_meas_ev=nme, is_measured=meas)))
        print(f"{fn}: rows {n} in_model {rec['n_in_model']} A1 {rec['n_A1_removed']} A2 {rec['n_A2_removed']} "
              f"nref10 {rec['n_ref_M10']} nref20 {rec['n_ref_M20']} meas_frac {rec['measured_ref_fraction_event']:.4f} orig_eq {orig_equal}", flush=True)
    S = pd.DataFrame(summ)
    sp = os.path.join(a.out_data, "cells_summary.csv"); S.to_csv(sp, index=False, float_format="%.17g")
    P = pd.DataFrame(log["per_cell"])
    C = log["checks"]
    C["n_cells"] = int(len(P)); C["n_cells_by_source"] = P.groupby("source").size().to_dict()
    C["all_orig_columns_equal_archived"] = bool(P.orig_columns_equal_archived.all())
    C["all_cyc_axis_equals_model_cyc"] = bool(P.cyc_axis_equals_model_cyc.all())
    C["frames_n_ev_sum_equals_npz_events"] = bool((P.n_events == P.n_events_npz).all())
    C["n_rows_unmapped_meas_total"] = int(P.n_rows_unmapped_meas.sum())
    C["rows_total"] = int(P.n_rows.sum()); C["rows_by_source"] = P.groupby("source").n_rows.sum().astype(int).to_dict()
    C["rows_per_cell_range_by_source"] = {s: [int(d.n_rows.min()), int(d.n_rows.max())] for s, d in P.groupby("source")}
    C["events_pre_gate_by_source"] = P.groupby("source").n_events.sum().astype(int).to_dict()
    C["q_ref_range_by_source"] = {s: [float(d.q_ref_Ah.min()), float(d.q_ref_Ah.max())] for s, d in P.groupby("source")}
    C["A1_removed_by_cell_nonzero"] = {r.cell: int(r.n_A1_removed) for r in P.itertuples() if r.n_A1_removed}
    C["A1_removed_by_source"] = P.groupby("source").n_A1_removed.sum().astype(int).to_dict()
    C["A1_fail_all_rows_by_source"] = P.groupby("source").n_A1_fail_all_rows.sum().astype(int).to_dict()
    C["A2_removed_by_source"] = P.groupby("source").n_A2_removed.sum().astype(int).to_dict()
    own = P[P.source == "own"]
    C["A2_removed_own_GA"] = int(own[own.cell.str.startswith("GA")].n_A2_removed.sum())
    C["A2_removed_own_cell2_cell3"] = int(own[own.cell.isin(["cell2", "cell3"])].n_A2_removed.sum())
    C["dropna_removed_by_source"] = P.groupby("source").n_dropna_removed.sum().astype(int).to_dict()
    C["in_model_by_source"] = P.groupby("source").n_in_model.sum().astype(int).to_dict()
    for M in M_LIST:
        C[f"n_ref_M{M}_by_source"] = {s: dict(mean=float(d[f"n_ref_M{M}"].mean()), min=int(d[f"n_ref_M{M}"].min()), max=int(d[f"n_ref_M{M}"].max()))
                                      for s, d in P.groupby("source")}
        g135 = P[(P.source == "ox") & P.cell.str.split("__").str[0].isin(["G1", "G3", "G5"])]
        C[f"n_ref_M{M}_ox_G1G3G5_range"] = [int(g135[f"n_ref_M{M}"].min()), int(g135[f"n_ref_M{M}"].max())]
    mf = P.set_index("cell").measured_ref_fraction_event
    C["measured_fraction_event"] = {
        "nasa": [float(mf[P[P.source == "nasa"].cell].min()), float(mf[P[P.source == "nasa"].cell].max())],
        "own": [float(mf[own.cell].min()), float(mf[own.cell].max())],
        "own_cell2": float(mf["cell2"]), "own_cell3": float(mf["cell3"]),
        "own_GA": [float(mf[own[own.cell.str.startswith("GA")].cell].min()), float(mf[own[own.cell.str.startswith("GA")].cell].max())],
        "ox_G1G3G5": [float(mf[g135.cell].min()), float(mf[g135.cell].max())],
        "ox_G7_G10": [float(mf[P[(P.source == "ox") & ~P.cell.isin(g135.cell)].cell].min()), float(mf[P[(P.source == "ox") & ~P.cell.isin(g135.cell)].cell].max())]}
    C["measured_fraction_row"] = {s: [float(d.measured_row_fraction.min()), float(d.measured_row_fraction.max())] for s, d in P.groupby("source")}
    b5 = P[P.cell == "B0005"].iloc[0]
    C["B0005_events_and_ref_discharges"] = [int(b5.n_events_npz), int(b5.n_ref_discharges)]
    C["label_stratum_counts"] = S.label_stratum.value_counts().to_dict()
    C["clusters"] = sorted(S.cluster.unique().tolist())
    # 與封存檔比對
    if a.c2:
        c2 = pd.read_csv(a.c2).set_index("file")
        d = [dict(file=f"{r.source}__{r.cell}", n_meas=int(r.n_measured_events), c2_n_meas=int(c2.loc[f"{r.source}__{r.cell}", "n_measured_events"]),
                  frac=float(r.measured_ref_fraction_event), c2_frac=float(c2.loc[f"{r.source}__{r.cell}", "ref_fraction_event"]),
                  n_events=int(r.n_events_npz), c2_n_events=int(c2.loc[f"{r.source}__{r.cell}", "n_events"])) for r in P.itertuples()]
        ok = lambda x: x["n_meas"] == x["c2_n_meas"] and x["n_events"] == x["c2_n_events"] and abs(x["frac"] - x["c2_frac"]) <= 1e-9
        C["vs_c2_measured_fraction"] = dict(n=len(d), counts_all_equal=all(x["n_meas"] == x["c2_n_meas"] and x["n_events"] == x["c2_n_events"] for x in d),
                                            n_frac_bit_identical=sum(x["frac"] == x["c2_frac"] for x in d),
                                            max_frac_abs_diff=max(abs(x["frac"] - x["c2_frac"]) for x in d),
                                            all_equal_tol_1e9=all(ok(x) for x in d), mismatches_tol_1e9=[x for x in d if not ok(x)])
    if a.measflag:
        A = pd.read_csv(a.measflag); B = pd.concat(mf_rows, ignore_index=True)
        A = A[A.in_frames == 1] if "in_frames" in A.columns else A
        Mg = B.merge(A[["source", "cell", "cyc", "n_meas_ev", "is_measured"]], on=["source", "cell", "cyc"], how="outer", suffixes=("", "_arch"), indicator=True)
        C["vs_frames_meas_flag_archived"] = dict(n_rows_ours=int(len(B)), n_rows_archived_in_frames=int(len(A)), n_unmatched=int((Mg._merge != "both").sum()),
                                                 n_is_measured_diff=int((Mg[Mg._merge == "both"].is_measured != Mg[Mg._merge == "both"].is_measured_arch).sum()),
                                                 n_n_meas_ev_diff=int((Mg[Mg._merge == "both"].n_meas_ev != Mg[Mg._merge == "both"].n_meas_ev_arch).sum()))
    if a.census:
        ce = pd.read_csv(a.census); ce = ce[ce.cell.isin(P.cell)].set_index("cell")
        d = []
        for r in P.itertuples():
            c = ce.loc[r.cell]
            exp_a1 = 0 if r.delta_key == "inf" else int(c.n_seg_invalid_d0)
            exp_a2 = int(c.n_label_dropped) if r.label_gate == "on" else 0
            d.append(dict(cell=r.cell, A1=int(r.n_A1_removed), census_A1=exp_a1, A2=int(r.n_A2_removed), census_A2=exp_a2,
                          in_model=int(r.n_in_model), census_n_model_rows_minus_A1=int(c.n_model_rows) - exp_a1, n_rows=int(r.n_rows), census_n_cycles=int(c.n_cycles)))
        C["vs_census_v911"] = dict(all_equal=all(x["A1"] == x["census_A1"] and x["A2"] == x["census_A2"] and x["n_rows"] == x["census_n_cycles"] for x in d),
                                   in_model_equals_n_model_rows_minus_A1=all(x["in_model"] == x["census_n_model_rows_minus_A1"] for x in d),
                                   mismatches=[x for x in d if not (x["A1"] == x["census_A1"] and x["A2"] == x["census_A2"] and x["n_rows"] == x["census_n_cycles"]
                                                                    and x["in_model"] == x["census_n_model_rows_minus_A1"])])
    if a.records:
        R = pd.read_csv(a.records); R = R[R.protocol == "P-B+"]
        d = []
        for r in P.itertuples():
            for M in M_LIST:
                rr = R[(R.source == r.source) & (R.cell == r.cell) & (R.M == M)]
                for x in rr.itertuples():
                    d.append(dict(cell=r.cell, M=M, model=x.model, n_ref=int(x.n_ref), n_eval=int(x.n_eval),
                                  ours_ref=int(getattr(r, f"n_ref_M{M}")), ours_eval=int(getattr(r, f"n_eval_M{M}"))))
        D = pd.DataFrame(d)
        C["vs_soh_records_PBplus"] = dict(n_record_rows=int(len(D)), n_cells=int(D.cell.nunique()),
                                          all_equal=bool(((D.n_ref == D.ours_ref) & (D.n_eval == D.ours_eval)).all()),
                                          mismatches=D[(D.n_ref != D.ours_ref) | (D.n_eval != D.ours_eval)].to_dict("records"))
    if a.predictions:
        pr = json.load(open(a.predictions, encoding="utf-8"))
        d = []
        for r in P.itertuples():
            o = pd.read_csv(os.path.join(OUTD, r.file))
            for M in M_LIST:
                k = f"{r.source}|P-B+|{r.cell}|M={M}|GPR-LSP"
                if k not in pr:
                    d.append(dict(key=k, found=False)); continue
                v = pr[k]; ref = o[o[f"ann_ref_M{M}"] == 1]; ev = o[(o.ann_in_model == 1) & (o[f"ann_ref_M{M}"] == 0)]
                d.append(dict(key=k, found=True,
                              ref_cyc_equal=bool(np.array_equal(np.asarray(v["ref_cyc"], float), ref.ann_cyc_axis.to_numpy(float))),
                              eval_cyc_equal=bool(np.array_equal(np.asarray(v["cyc"], float), ev.ann_cyc_axis.to_numpy(float))),
                              eval_y_equal=bool(np.array_equal(np.asarray(v["y"], float), ev.soh.to_numpy(float))),
                              eval_y_maxabs=float(np.max(np.abs(np.asarray(v["y"], float) - ev.soh.to_numpy(float)))) if len(v["y"]) == len(ev) else None))
        C["vs_soh_predictions_PBplus_GPR_LSP"] = dict(n_keys=len(d), all_found=all(x["found"] for x in d),
                                                      all_ref_cyc_equal=all(x.get("ref_cyc_equal") for x in d),
                                                      all_eval_cyc_equal=all(x.get("eval_cyc_equal") for x in d),
                                                      n_eval_y_bit_identical=sum(bool(x.get("eval_y_equal")) for x in d),
                                                      max_eval_y_abs_diff=max((x.get("eval_y_maxabs") if x.get("eval_y_maxabs") is not None else float("inf")) for x in d),
                                                      all_eval_y_equal_tol_1e9=all(x.get("eval_y_maxabs") is not None and x["eval_y_maxabs"] <= 1e-9 for x in d),
                                                      failures_tol_1e9=[x for x in d if not (x.get("ref_cyc_equal") and x.get("eval_cyc_equal")
                                                                                            and x.get("eval_y_maxabs") is not None and x["eval_y_maxabs"] <= 1e-9)])
    if a.core_dir and a.core_res:
        sys.path.insert(0, a.core_dir)
        import data_v911 as Dm
        Dm.DATASET_V9 = a.npz
        import config_v911 as Cm, soh_v911 as Sm, stage_data_v911 as SDm
        d = []
        for s in SOURCES:
            fr = Sm.apply_delta(SDm.load_frames(a.core_res, s), Cm.soh_sel(s)["delta_key"])
            for c, ff in fr.items():
                o = pd.read_csv(os.path.join(OUTD, f"{s}__{c}.csv")); om = o[o.ann_in_model == 1].reset_index(drop=True)
                d.append(dict(cell=c, n_core=int(len(ff)), n_ours=int(len(om)),
                              cycA_equal=bool(len(ff) == len(om) and np.array_equal(ff.cyc_A.to_numpy(float), om.cyc_A.to_numpy(float))),
                              cyc_equal=bool(len(ff) == len(om) and np.array_equal(ff.cyc.to_numpy(float), om.ann_cyc_axis.to_numpy(float)))))
        C["vs_core_load_frames_apply_delta"] = dict(core_dir=os.path.basename(os.path.normpath(a.core_dir)), n_cells=len(d),
                                                    all_equal=all(x["cycA_equal"] and x["cyc_equal"] for x in d),
                                                    failures=[x for x in d if not (x["cycA_equal"] and x["cyc_equal"])])
    log["columns_by_source"] = cols_by_src
    log["outputs"]["cells_summary.csv"] = sha256(sp)
    log["outputs"]["derived_per_row"] = {r["file"]: r["out_sha256"] for r in log["per_cell"]}
    log["end_utc"] = utc()
    if a.log:
        with open(a.log, "w", encoding="utf-8") as fo:
            json.dump(log, fo, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(C, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
