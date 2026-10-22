"""
gaps_v911.py — own_gap_rule_v98（prereg v14 附錄 §1 規則 1–3）之程式實作。
規則 1（斷點定義）：
  (a) 循環索引缺口：cyc 不連續處。
  (b) 階梯：相鄰循環 |ΔSOH| > gap_dsoh_thr，且其後 gap_med_win 循環之 SOH 中位數
      與其前 gap_med_win 循環之中位數（均含該相鄰對之對應端點）差之絕對值 > gap_med_thr
      （排除單點離群）。
規則 2（處理）：斷點 (b) 之過渡循環（階梯落差發生之循環＝相鄰對之後一循環）直接拉掉；
  保留之循環依原順序賦予連續索引 cyc_c = 1..N，原值存 cyc_A。不做任何內插或平移。
規則 3：cell2／cell3 同規則，程式判定，不得手選。
"""
import numpy as np


def detect(cyc, soh, dsoh_thr=0.010, med_win=5, med_thr=0.010):
    """回傳 dict：gaps（規則 a）、steps（規則 b 全部候選，含是否通過中位數檢定）、drop_cyc_A（拉掉之原循環）。"""
    cyc = np.asarray(cyc, float); soh = np.asarray(soh, float); n = len(soh)
    gaps = [dict(prev_cyc_A=float(cyc[i]), next_cyc_A=float(cyc[i + 1]), missing=int(cyc[i + 1] - cyc[i] - 1))
            for i in range(n - 1) if cyc[i + 1] - cyc[i] > 1]
    steps = []
    for i in range(n - 1):
        d = soh[i + 1] - soh[i]
        if abs(d) <= dsoh_thr:
            continue
        pre = soh[max(0, i - (med_win - 1)):i + 1]
        post = soh[i + 1:min(n, i + 1 + med_win)]
        md = float(np.median(post) - np.median(pre))
        ok = abs(md) > med_thr
        steps.append(dict(prev_cyc_A=float(cyc[i]), step_cyc_A=float(cyc[i + 1]), dSOH=float(d),
                          med_pre=float(np.median(pre)), med_post=float(np.median(post)), med_diff=md,
                          n_pre=int(len(pre)), n_post=int(len(post)), is_breakpoint=bool(ok)))
    drop = sorted({s["step_cyc_A"] for s in steps if s["is_breakpoint"]})
    return dict(gaps=gaps, steps=steps, drop_cyc_A=drop)


def apply_rule(frame, dsoh_thr=0.010, med_win=5, med_thr=0.010):
    """對逐循環表套用規則：回傳 (新 frame, info)。
    新 frame：欄 cyc 換為 cyc_c（1..N 連續），原值存 cyc_A，另加 drop_flag=0（保留列）。
    info 另含 renumber（cyc_A → cyc_c 對照）與被拉掉之列（drop_flag=1）之 cyc_A。"""
    f = frame.copy()
    info = detect(f.cyc.to_numpy(float), f.soh.to_numpy(float), dsoh_thr, med_win, med_thr)
    drop = set(info["drop_cyc_A"])
    f["cyc_A"] = f["cyc"].astype(float)
    f["drop_flag"] = f["cyc_A"].isin(drop).astype(int)
    keep = f[f.drop_flag == 0].copy().reset_index(drop=True)
    keep["cyc_c"] = np.arange(1, len(keep) + 1, dtype=float)
    keep["cyc"] = keep["cyc_c"]                       # 下游以 cyc_c 為循環軸
    info["n_before"] = int(len(f)); info["n_after"] = int(len(keep))
    info["n_dropped"] = int(len(f) - len(keep))
    info["renumber"] = [dict(cyc_A=float(a), cyc_c=float(c)) for a, c in zip(keep.cyc_A, keep.cyc_c)]
    info["dropped_rows"] = [dict(cyc_A=float(r.cyc_A), soh=float(r.soh)) for r in f[f.drop_flag == 1].itertuples()]
    full = f.copy()                                    # 含被拉掉列之完整表（供 frames/own__cellX.csv 落地）
    m = dict(zip(keep.cyc_A.tolist(), keep.cyc_c.tolist()))
    full["cyc_c"] = full["cyc_A"].map(m)
    return keep, full, info
