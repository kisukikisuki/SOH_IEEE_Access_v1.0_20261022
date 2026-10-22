#!/usr/bin/env python3
"""
修訂 v9a（先驗登記 v9 之修訂，揭露）：A3 之 Ah_int 於 (a) time_s 缺值（無法積分）或 (b) 段內 dt 曾補接且 |Ah_int−Ah_col|/Ah_col > 1e-2 時，
以 Q 欄極差 Ah_col 代入並旗標 a_flag_ah_from_col；其後 cyc_A／Ah_cum／EFC／事件軸對照全部重算。動機：G1__Cell9 有 35 個放電段 time_s 為 NaN、
30 段 dt 異常（Ah_int 達 Ah_col 之 1.99 倍）；G9__Cell1 1 段。此為資料層時間戳缺陷之處置，不涉任何 SOH／RUL 結果。
"""
import os, glob, json, numpy as np, pandas as pd
AL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_v94", "a_layer"); DS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset_v8")
log = []
for f in sorted(glob.glob(os.path.join(AL, "*_A_layer_discharge.csv"))):
    cell = os.path.basename(f).split("_A_layer")[0]; dis = pd.read_csv(f)
    if len(dis) == 0: continue
    rel = ((dis.Ah_int - dis.Ah_col) / dis.Ah_col.replace(0, np.nan)).abs()
    flag = dis.Ah_int.isna() | ((dis.n_dt_fixed > 0) & (rel > 1e-2))
    if not flag.any(): continue
    dis["Ah_int_raw"] = dis.Ah_int; dis["a_flag_ah_from_col"] = flag; dis.loc[flag, "Ah_int"] = dis.loc[flag, "Ah_col"]
    dis["a_flag_shallow"] = dis.Ah_int < 0.3 * dis.Ah_int.max(); dis["cyc_A"] = np.cumsum(~dis.a_flag_shallow)
    dis["Ah_cum"] = dis.Ah_int.cumsum(); dis["Ah_cum_shallow"] = (dis.Ah_int * dis.a_flag_shallow).cumsum()
    fs = sorted(glob.glob(os.path.join(DS, f"*__{cell}.npz")))
    sp = os.path.join(AL, f"{cell}_summary.json"); summ = json.load(open(sp))
    if fs:
        d = np.load(fs[0], allow_pickle=True); q_ref = float(d["q_ref"]); dis["EFC"] = dis.Ah_cum / q_ref
        ev = pd.read_csv(os.path.join(AL, f"{cell}_event_axis_map.csv"))
        idx = np.searchsorted(dis.seg.to_numpy(), ev.seg.to_numpy(), side="left") - 1; idx = np.clip(idx, 0, len(dis) - 1)
        ev["cyc_A"] = dis.cyc_A.to_numpy()[idx]; ev["EFC"] = dis.EFC.to_numpy()[idx]; ev["Ah_cum"] = dis.Ah_cum.to_numpy()[idx]
        eol = np.flatnonzero(ev.soh_B.to_numpy() <= 0.80); drift = (ev.cyc_A - ev.event_idx) - (ev.cyc_A.iloc[0] - ev.event_idx.iloc[0])
        ev.to_csv(os.path.join(AL, f"{cell}_event_axis_map.csv"), index=False)
        B = pd.DataFrame(dict(seg=d["cap_seg"].astype(int), cap_B=d["cap_soh"] * q_ref)); Mg = dis.merge(B, on="seg", how="inner")
        summ.update(EFC_total=round(float(dis.EFC.iloc[-1]), 2), event_axis_drift_max=int(np.abs(drift).max()), Ah_A_vs_capB_relmax=round(float(((Mg.Ah_int - Mg.cap_B) / Mg.cap_B).abs().max()), 6),
                    eol_cyc_A=int(ev.cyc_A.iloc[eol[0]]) if eol.size else None, eol_EFC=round(float(ev.EFC.iloc[eol[0]]), 2) if eol.size else None)
    else:
        capmax = dis.loc[dis.b_ucs5c_n50, "Ah_col"].max(); dis["EFC"] = dis.Ah_cum / capmax; summ.update(EFC_total=round(float(dis.EFC.iloc[-1]), 2))
    full = dis[dis.b_ucs5c_pass]; rel2 = ((full.Ah_int - full.Ah_col) / full.Ah_col).abs()
    summ.update(n_cycles_A=int(dis.cyc_A.max()), n_shallow=int(dis.a_flag_shallow.sum()), shallow_Ah_share=round(float(dis.Ah_cum_shallow.iloc[-1] / dis.Ah_cum.iloc[-1]), 4),
                Ah_int_vs_col_relmax=round(float(rel2.max()), 5), Ah_int_vs_col_relmed=round(float(rel2.median()), 5), n_ah_from_col_v9a=int(flag.sum()), Ah_total=round(float(dis.Ah_int.sum()), 3))
    dis.to_csv(f, index=False); json.dump(summ, open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    log.append(dict(cell=cell, n_flagged=int(flag.sum()), n_nan_time=int(dis.Ah_int_raw.isna().sum()), EFC_total=summ.get("EFC_total"), cyc_A=summ["n_cycles_A"]))
    print(log[-1])
rows = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(AL, "*_summary.json")))]; pd.DataFrame(rows).to_csv(os.path.join(AL, "..", "summary_A_layer_all_v94.csv"), index=False)
json.dump(log, open(os.path.join(AL, "..", "a_layer_fix_v9a_log.json"), "w"), indent=1)
