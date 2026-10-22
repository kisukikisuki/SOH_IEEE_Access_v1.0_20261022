"""reproduce_v34_numbers.py — 由重跑輸出重算 v34 稿件之表 5（P-B、P-B+ 列）、表 6、表 8 與第 IV-B 節 cluster bootstrap 數字，逐一列出 v34 值與重算值。

新撰（公開程式包）。v34 值為稿件 IEEE_Access_SOH_中文_v34_20261022 之表列數字（逐字抄錄於下方 V34_* 常數）；
重算值依稿件之定義由程式輸出計算，再依 v34 之小數位數以 Python format（round-half-even，半數取偶）取位後比對字串。
  表 5：soh_records 之 GPR-LSP 列：RMSE_mean＝逐顆 RMSE_pct 平均、RMSE_max、MAE_mean、R²_min；最佳模型＝該來源×協定下逐顆平均 RMSE 最低之模型。
        P-C 列屬範圍外，不重算。
  表 6：A04 逐顆檔（相同非參考評估列）：P-B＝PB_same_rows_RMSE_pct 平均、ZOH、LIN2、P-B+＝PBplus_RMSE_pct 平均、
        「P-B+ 低於 ZOH」＝PBplus_RMSE_pct < ZOH_RMSE_pct 之顆數。子集：NASA、自有、Oxford G7–G10、Oxford G1/G3/G5、
        實測標籤子集（NASA＋自有＋G7–G10，25 顆）、全部 34 顆。Wilcoxon／Holm 欄屬範圍外，不重算。
  表 8：n_ref＝soh_records 之 P-B+ GPR-LSP 列 n_ref 之平均 [最小–最大]；ΔRMSE＝P-B+ RMSE_mean − A04 PB_same_rows 平均。
  第 IV-B 節：c7_cluster_bootstrap.csv 之 M＝20「Measured-label subset (25)」列。
用法：python reproduce_v34_numbers.py --records <soh_records_v911.csv> --a04 <A04 csv> --c7 <c7_cluster_bootstrap.csv> --out <報告.json>
"""
import json, argparse
import numpy as np, pandas as pd

SRC_NAME = {"nasa": "NASA PCoE", "own": "自有 NCR18650GA", "ox": "Oxford"}
# 表 5（P-B、P-B+ 列）：RMSE_mean, RMSE_max, MAE_mean, R2_min, 最佳模型, 最佳模型 RMSE_mean
V34_T5 = {
    ("nasa", "P-B"): ("0.8662", "1.1032", "0.7003", "0.9878", "Ridge", "0.8611"),
    ("nasa", "P-B+|10"): ("0.7213", "0.8870", "0.5092", "0.9891", "GPR-LSP", "0.7213"),
    ("nasa", "P-B+|20"): ("0.7267", "0.9505", "0.4964", "0.9878", "GPR-LSP", "0.7267"),
    ("own", "P-B"): ("3.0492", "7.9371", "2.5365", "-0.5011", "MLP", "2.2099"),
    ("own", "P-B+|10"): ("0.9547", "1.8723", "0.6705", "0.8882", "SVR", "0.9304"),
    ("own", "P-B+|20"): ("1.2322", "2.9211", "0.8880", "0.7280", "MLP", "1.2048"),
    ("ox", "P-B"): ("1.0106", "3.1323", "0.7853", "0.7893", "GPR-LSP", "1.0106"),
    ("ox", "P-B+|10"): ("0.6249", "1.8745", "0.4047", "0.9271", "SVR", "0.5166"),
    ("ox", "P-B+|20"): ("0.6833", "1.9663", "0.4662", "0.9117", "SVR", "0.5867"),
}
# 表 6：電芯數, P-B, ZOH, LIN2, P-B+, P-B+ 低於 ZOH
V34_T6 = {
    (20, "NASA PCoE"): ("4", "0.872", "2.616", "1.956", "0.727", "4/4"),
    (20, "自有 NCR18650GA"): ("9", "3.049", "1.352", "1.171", "1.232", "6/9"),
    (20, "Oxford G7–G10"): ("12", "0.850", "0.809", "0.798", "0.593", "10/12"),
    (20, "Oxford G1/G3/G5"): ("9", "1.219", "0.657", "0.581", "0.804", "1/9"),
    (20, "實測標籤子集"): ("25", "1.645", "1.294", "1.118", "0.844", "20/25"),
    (20, "全部電芯"): ("34", "1.532", "1.125", "0.976", "0.834", "21/34"),
    (10, "NASA PCoE"): ("4", "0.874", "1.886", "1.990", "0.721", "4/4"),
    (10, "自有 NCR18650GA"): ("9", "3.053", "0.905", "0.913", "0.955", "6/9"),
    (10, "Oxford G7–G10"): ("12", "0.850", "0.610", "0.682", "0.527", "9/12"),
    (10, "Oxford G1/G3/G5"): ("9", "1.221", "0.488", "0.465", "0.756", "0/9"),
    (10, "實測標籤子集"): ("25", "1.647", "0.920", "0.974", "0.712", "19/25"),
    (10, "全部電芯"): ("34", "1.534", "0.806", "0.839", "0.724", "19/34"),
}
# 表 8：n_ref 平均 [範圍], ΔRMSE
V34_T8 = {
    ("nasa", 10): ("16.2 [14–17]", "-0.1528"), ("nasa", 20): ("8.5 [7–9]", "-0.1450"),
    ("own", 10): ("30.8 [19–58]", "-2.0988"), ("own", 20): ("15.7 [10–29]", "-1.8169"),
    ("ox", 10): ("62.3 [23–129]", "-0.3841"), ("ox", 20): ("31.4 [12–65]", "-0.3250"),
}
# 第 IV-B 節（M = 20，實測標籤子集）
V34_IVB = dict(n_clusters="10", mean_diff="-0.449", pct_lo="-0.923", pct_hi="-0.033", bca_lo="-0.850", bca_hi="0.041",
               mixed="-0.725", mixed_lo="-1.410", mixed_hi="-0.041")


def f(x, d):
    s = format(float(x), f".{d}f")
    return "0." + "0" * d if s == "-0." + "0" * d else s


def subsets(a):
    g = np.array([c.split("__")[0] if s == "ox" else "" for s, c in zip(a.source, a.cell)])
    return {"NASA PCoE": a[a.source == "nasa"], "自有 NCR18650GA": a[a.source == "own"],
            "Oxford G7–G10": a[(a.source == "ox") & np.isin(g, ["G7", "G8", "G9", "G10"])],
            "Oxford G1/G3/G5": a[(a.source == "ox") & np.isin(g, ["G1", "G3", "G5"])],
            "實測標籤子集": a[(a.source != "ox") | np.isin(g, ["G7", "G8", "G9", "G10"])], "全部電芯": a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True); ap.add_argument("--a04", required=True); ap.add_argument("--c7", required=True)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()
    R = pd.read_csv(A.records); a04 = pd.read_csv(A.a04); c7 = pd.read_csv(A.c7)
    rows = []

    def add(item, v34, val):
        rows.append(dict(item=item, v34=v34, rerun=val, match=(v34 == val)))

    # 表 5
    for (s, p), v in V34_T5.items():
        proto, M = (p, None) if p == "P-B" else ("P-B+", int(p.split("|")[1]))
        r = R[(R.source == s) & (R.protocol == proto)]
        if M is not None:
            r = r[pd.to_numeric(r.M) == M]
        g = r[r.model == "GPR-LSP"]
        mm = r.groupby("model").RMSE_pct.mean(); best = mm.idxmin()
        lab = f"表5 {SRC_NAME[s]} {proto}{'' if M is None else f'，M = {M}'}"
        add(lab + " RMSE_mean", v[0], f(g.RMSE_pct.mean(), 4)); add(lab + " RMSE_max", v[1], f(g.RMSE_pct.max(), 4))
        add(lab + " MAE_mean", v[2], f(g.MAE_pct.mean(), 4)); add(lab + " R2_min", v[3], f(g.R2.min(), 4))
        add(lab + " 最佳模型", f"{v[4]} ({v[5]})", f"{best} ({f(mm[best], 4)})")
    # 表 6
    for (M, sn), v in V34_T6.items():
        d = subsets(a04[a04.M == M])[sn]
        lab = f"表6 M = {M} {sn}"
        add(lab + " 電芯數", v[0], str(len(d)))
        add(lab + " P-B", v[1], f(d.PB_same_rows_RMSE_pct.mean(), 3)); add(lab + " ZOH", v[2], f(d.ZOH_RMSE_pct.mean(), 3))
        add(lab + " LIN2", v[3], f(d.LIN2_RMSE_pct.mean(), 3)); add(lab + " P-B+", v[4], f(d.PBplus_RMSE_pct.mean(), 3))
        add(lab + " P-B+ 低於 ZOH", v[5], f"{int((d.PBplus_RMSE_pct < d.ZOH_RMSE_pct).sum())}/{len(d)}")
    # 表 8
    for (s, M), v in V34_T8.items():
        g = R[(R.source == s) & (R.protocol == "P-B+") & (pd.to_numeric(R.M) == M) & (R.model == "GPR-LSP")]
        nr = pd.to_numeric(g.n_ref)
        same = a04[(a04.source == s) & (a04.M == M)].PB_same_rows_RMSE_pct.mean()
        lab = f"表8 {SRC_NAME[s]} P-B+，M = {M}"
        add(lab + " n_ref 平均 [範圍]", v[0], f"{f(nr.mean(), 1)} [{int(nr.min())}–{int(nr.max())}]")
        add(lab + " 相對 P-B 之 ΔRMSE", v[1], f(g.RMSE_pct.mean() - same, 4))
    # 第 IV-B 節
    x = c7[(c7.M == 20) & (c7.subset == "Measured-label subset (25)")].iloc[0]
    lab = "IV-B M = 20 實測標籤子集"
    add(lab + " cluster 數", V34_IVB["n_clusters"], str(int(x.n_clusters)))
    add(lab + " P-B+ 減 ZOH 平均配對差", V34_IVB["mean_diff"], f(x.mean_diff_PBplus_minus_ZOH, 3))
    add(lab + " percentile 下限", V34_IVB["pct_lo"], f(x.ci95_percentile_lo, 3)); add(lab + " percentile 上限", V34_IVB["pct_hi"], f(x.ci95_percentile_hi, 3))
    add(lab + " BCa 下限", V34_IVB["bca_lo"], f(x.ci95_bca_lo, 3)); add(lab + " BCa 上限", V34_IVB["bca_hi"], f(x.ci95_bca_hi, 3))
    add(lab + " 混合效應截距", V34_IVB["mixed"], f(x.mixedlm_intercept, 3))
    add(lab + " 混合效應 CI 下限", V34_IVB["mixed_lo"], f(x.mixedlm_ci95_lo, 3)); add(lab + " 混合效應 CI 上限", V34_IVB["mixed_hi"], f(x.mixedlm_ci95_hi, 3))
    # 第 IV-B 節之區間判讀敘述（v34：全部 34 顆以 13 個 cluster 之區間於 M = 10、20 皆含 0；NASA 區間不含 0 偏向 P-B+；內插 9 顆區間不含 0 偏向 ZOH）
    def verdict(r):
        p0 = r.ci95_percentile_lo <= 0 <= r.ci95_percentile_hi; b0 = r.ci95_bca_lo <= 0 <= r.ci95_bca_hi
        return "區間含 0" if (p0 and b0) else ("區間不含 0" if (not p0 and not b0) else "兩種區間結論不一致")
    for M in (10, 20):
        r = c7[(c7.M == M) & (c7.subset == "All cells (34)")].iloc[0]
        add(f"IV-B M = {M} 全部電芯 cluster 數", "13", str(int(r.n_clusters)))
        add(f"IV-B M = {M} 全部電芯 區間判讀", "區間含 0", verdict(r))
    r = c7[(c7.M == 20) & (c7.subset == "NASA")].iloc[0]
    add("IV-B M = 20 NASA 區間判讀與方向", "區間不含 0；偏向 P-B+", f"{verdict(r)}；{'偏向 P-B+' if r.mean_diff_PBplus_minus_ZOH < 0 else '偏向 ZOH'}")
    r = c7[(c7.M == 20) & (c7.subset == "Oxford G1/G3/G5 (interpolated, 9)")].iloc[0]
    add("IV-B M = 20 內插 9 顆區間判讀與方向", "區間不含 0；偏向 ZOH", f"{verdict(r)}；{'偏向 P-B+' if r.mean_diff_PBplus_minus_ZOH < 0 else '偏向 ZOH'}")
    add("IV-B M = 20 實測標籤子集區間判讀", "兩種區間結論不一致", verdict(x))

    for r in rows:   # v34 以 U+2212 表示負號；此處統一以 ASCII '-' 比對
        r["v34"] = r["v34"].replace("−", "-"); r["match"] = (r["v34"] == r["rerun"])
    rep = dict(n_items=len(rows), n_match=int(sum(r["match"] for r in rows)), items=rows)
    json.dump(rep, open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.DataFrame(rows).to_csv(A.out[:-5] + ".csv" if A.out.endswith(".json") else A.out + ".csv", index=False, encoding="utf-8-sig")
    for r in rows:
        print(f"{'OK ' if r['match'] else 'DIFF'} {r['item']}: v34={r['v34']}  rerun={r['rerun']}")
    print(f"相符 {rep['n_match']}/{rep['n_items']}")


if __name__ == "__main__":
    main()
