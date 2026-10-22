"""run_baselines.py — 三種僅用參考容量之基線 ZOH／LIN2／LINALL（v34 第 III-B 節；表 6、表 7 之基線欄、表 8 之 n_ref 與 ΔRMSE）。

新撰（公開程式包之最小包裝）；只以子程序呼叫同夾之原程式 soh_post_v911r.py（逐位元複製自
ROOT\\5版\\02_程式\\02_SOH重跑\\post_SOH_v3\\soh_post_v911r.py；共用模組 common_v911r.py 同），不改原檔、不 import 原檔
（原檔於模組頂層即執行全部計算）。
  環境變數 PV911_RESULTS ＝ run_from_derived_tables.py 之輸出夾（讀 soh_predictions_v911.json 與 soh_records_v911.csv）
  環境變數 PV911R_OUT    ＝ 本程式 --out
保留之輸出（範圍內）：
  A04_reference_only_baselines_percell.csv —— 逐顆 × M ∈ {10, 20}：n_ref、n_eval 與
      {PBplus, PB_same_rows, ZOH, LIN2, LINALL}_{RMSE_pct, MAE_pct, R2, pass}（GPR-LSP；相同非參考評估列）。
      表 6、表 8 之數字由 ../06_verify/reproduce_v34_numbers.py 自此檔（及 soh_records）重算。
本程式移除之附帶輸出（原程式於同一次執行一併寫出；原檔不改，故由本包裝於原程式成功結束後刪除）：
  A04_A09_B05_soh_post_summary.json —— 含 Wilcoxon 成對檢定（paired_tests_vs_PBplus）與逐顆 RMSE bootstrap 區間（RMSE_boot95），
      屬 v34 公開範圍外之分析；其範圍內之平均 RMSE 與 n_ref 摘要可由 A04 csv 直接重算，故整檔移除。
  A12_ci95_coverage.json —— CI95 涵蓋率，屬範圍外。
原程式 stdout 含範圍外之 Wilcoxon／CI 涵蓋率輸出，故不保存：stdout 只留在記憶體，失敗時印出末段供除錯，成功時丟棄。
<out>/log_run_baselines.txt 只記錄原程式之回傳碼、stderr 與本程式刪除了哪些附帶檔（刪除清單同時印到 stdout）。
用法：python run_baselines.py --results <runner 輸出夾> --out <輸出夾>
"""
import os, sys, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
# 原程式一併寫出之範圍外附帶檔（見檔頭），於原程式成功結束後刪除
OUT_OF_SCOPE_OUTPUTS = ("A04_A09_B05_soh_post_summary.json", "A12_ci95_coverage.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    env = dict(os.environ)
    env["PV911_RESULTS"] = os.path.abspath(a.results); env["PV911R_OUT"] = os.path.abspath(a.out)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(v, "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")   # 不在程式夾內產生 __pycache__
    os.makedirs(a.out, exist_ok=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "soh_post_v911r.py")], env=env, cwd=HERE,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    log = os.path.join(a.out, "log_run_baselines.txt")
    note = "原程式 stdout 含範圍外之 Wilcoxon／CI 涵蓋率輸出，故不保存。\n"
    with open(log, "w", encoding="utf-8") as fo:
        fo.write(note + f"soh_post_v911r.py returncode={r.returncode}\n--- stderr ---\n{r.stderr}\n")
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-4000:])          # 失敗時才印出 stdout 末段供除錯（不落地）
        sys.exit(f"soh_post_v911r.py 結束碼 {r.returncode}；stderr 見 {log}")
    del r                                           # 成功：丟棄原程式 stdout
    removed = []
    for fn in OUT_OF_SCOPE_OUTPUTS:
        fp = os.path.join(a.out, fn)
        if os.path.exists(fp):
            os.remove(fp); removed.append(fn)
    msg = ("已移除範圍外附帶輸出：" + "、".join(removed)) if removed else "無範圍外附帶輸出需移除"
    with open(log, "a", encoding="utf-8") as fo:
        fo.write(msg + "\n")
    print(msg)
    print("A04 →", os.path.join(os.path.abspath(a.out), "A04_reference_only_baselines_percell.csv"))


if __name__ == "__main__":
    main()
