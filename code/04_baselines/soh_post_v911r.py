"""審稿項 A04／A09／A12／B05／B07：SOH 既有預測之後處理。
A04：reference-only 基線（ZOH、LIN2、LINALL）——僅使用 P-B+ 已取用之同一批參考列，
     與 P-B+ 相同之因果規則（區塊 B_j 只用索引 ≤ b_j 之參考列），評估列完全相同。
A09：逐顆配對 Wilcoxon、bootstrap 95% CI、median/IQR。
A12：P-A／P-B／P-C 之 CI95 覆蓋率與半寬（取自 soh_records）。
B05：n_ref 平均與範圍。
不重新訓練任何模型。"""
import csv, math, numpy as np
from scipy.stats import wilcoxon
from common_v911r import *

P = load_json("soh_predictions_v911.json")
SRC = ["nasa", "own", "ox"]

def baselines(full_cyc, full_y, M):
    n = len(full_cyc); c = np.asarray(full_cyc, float); y = np.asarray(full_y, float)
    zoh, lin2, linall, idx = [], [], [], []
    for i in range(n):
        if i % M == 0: continue
        b = (i // M) * M
        refs = list(range(0, b + 1, M))
        z = y[b]
        if len(refs) >= 2:
            a = refs[-2]; l2 = y[b] + (y[b] - y[a]) / (c[b] - c[a]) * (c[i] - c[b])
            A = np.vstack([c[refs], np.ones(len(refs))]).T
            coef = np.linalg.lstsq(A, y[refs], rcond=None)[0]; la = coef[0] * c[i] + coef[1]
        else:
            l2 = z; la = z
        zoh.append(z); lin2.append(l2); linall.append(la); idx.append(i)
    return np.array(idx), np.array(zoh), np.array(lin2), np.array(linall)

rows = []
for M in (10, 20):
    for k in sorted(P):
        s, p, cell, m, mod = k.split("|")
        if p != "P-B+" or m != f"M={M}" or mod != "GPR-LSP": continue
        pb = P[f"{s}|P-B|{cell}|loo|GPR-LSP"]
        e = P[k]
        idx, z, l2, la = baselines(pb["cyc"], pb["y"], M)
        assert [pb["cyc"][i] for i in idx] == e["cyc"], k
        y = np.asarray(e["y"], float)
        rec = dict(source=s, cell=cell, M=M, n_ref=len(e["ref_cyc"]), n_eval=len(y))
        for name, yh in (("PBplus", e["yhat"]), ("PB_same_rows", e["base_yhat"]),
                         ("ZOH", z), ("LIN2", l2), ("LINALL", la)):
            mm = metrics(y, yh)
            for kk, vv in mm.items():
                if kk != "n": rec[f"{name}_{kk}"] = vv
            rec[f"{name}_pass"] = passes(mm)
        rows.append(rec)

with open(os.path.join(OUT, "A04_reference_only_baselines_percell.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

summ = {}
for M in (10, 20):
    for s in SRC + ["all"]:
        R = [r for r in rows if r["M"] == M and (s == "all" or r["source"] == s)]
        d = dict(n_cells=len(R), n_ref_mean=float(np.mean([r["n_ref"] for r in R])),
                 n_ref_min=int(min(r["n_ref"] for r in R)), n_ref_max=int(max(r["n_ref"] for r in R)),
                 n_ref_median=float(np.median([r["n_ref"] for r in R])))
        for name in ("PBplus", "PB_same_rows", "ZOH", "LIN2", "LINALL"):
            v = np.array([r[f"{name}_RMSE_pct"] for r in R])
            d[name] = dict(RMSE_mean=float(v.mean()), RMSE_median=float(np.median(v)),
                           RMSE_IQR=[float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                           RMSE_max=float(v.max()), RMSE_boot95=boot_ci(v),
                           MAE_mean=float(np.mean([r[f"{name}_MAE_pct"] for r in R])),
                           R2_min=float(min(r[f"{name}_R2"] for r in R)),
                           n_pass=int(sum(r[f"{name}_pass"] for r in R)))
        # 配對檢定：P-B+ vs 各基線（逐顆 RMSE）
        tests = {}
        for name in ("PB_same_rows", "ZOH", "LIN2", "LINALL"):
            a = np.array([r["PBplus_RMSE_pct"] for r in R]); b = np.array([r[f"{name}_RMSE_pct"] for r in R])
            n_better = int(np.sum(a < b))
            if len(R) >= 6:
                st = wilcoxon(a, b, alternative="two-sided", method="exact" if len(R) <= 50 else "auto")
                tests[name] = dict(n=len(R), n_PBplus_better=n_better, W=float(st.statistic), p_two_sided=float(st.pvalue))
            else:
                tests[name] = dict(n=len(R), n_PBplus_better=n_better, W=None, p_two_sided=None,
                                   note="n<6：雙尾 exact Wilcoxon 最小可達 p ≥ 0.0625，不做檢定")
        d["paired_tests_vs_PBplus"] = tests
        summ[f"M{M}_{s}"] = d
dump("A04_A09_B05_soh_post_summary.json", summ)

# A12：既有 CI95 覆蓋率（P-A、P-B、P-C），GPR-LSP 與 GPR-SE
cov = {}
with open(os.path.join(BASE, "soh_records_v911.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["model"] not in ("GPR-LSP", "GPR-SE") or r["CI95_cov"] in ("", "None"): continue
        key = f'{r["source"]}|{r["protocol"]}|{r["s0"] or r["M"] or ""}|{r["model"]}'
        cov.setdefault(key, []).append((float(r["CI95_cov"]), float(r["CI95_halfwidth_soh"])))
A12 = {k: dict(n_cells=len(v), cov_mean=float(np.mean([a for a, _ in v])),
               halfwidth_mean_pct=float(100 * np.mean([b for _, b in v]))) for k, v in sorted(cov.items())}
dump("A12_ci95_coverage.json", A12)
print(json.dumps({k: {n: summ[k][n]["RMSE_mean"] for n in ("PBplus","PB_same_rows","ZOH","LIN2","LINALL")} for k in summ}, indent=1))
print(json.dumps({k: {kk: summ[k]["paired_tests_vs_PBplus"][kk] for kk in ("ZOH","PB_same_rows")} for k in summ if k.endswith(("all","ox"))}, indent=1, ensure_ascii=False))
print({k: (summ[k]["n_ref_mean"], summ[k]["n_ref_min"], summ[k]["n_ref_max"]) for k in summ})
for k,v in A12.items(): print(k, v)
