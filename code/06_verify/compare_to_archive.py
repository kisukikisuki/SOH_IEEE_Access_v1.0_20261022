"""compare_to_archive.py — 重跑輸出與 expected/（封存結果之逐顆指標）逐值比對；容差固定 1e-9（絕對差）。

新撰（公開程式包）。比對三組檔案：
  1. soh_records（runner 輸出 soh_records_v911.csv） vs expected/soh_records_v911_PB_PBplus.csv
     鍵：source, protocol, cell, model, M；主要欄：RMSE_pct, MAE_pct, R2, n_ref, n_eval（其餘共同欄一併列出，標為補充）。
  2. A04（run_baselines.py 輸出）vs expected/A04_reference_only_baselines_percell.csv；鍵：source, cell, M；全欄。
  3. c7_cluster_bootstrap.csv、c7_S14_table_data.csv（s6_cluster_bootstrap.py 輸出）vs expected/ 同名檔；鍵：M, subset；全欄。
數值欄：|重跑 − 封存| ≤ 1e-9 為相等（兩邊皆 NaN 視為相等）；非數值欄：字串完全相同為相等。
不相等者逐一列出（鍵、欄、封存值、重跑值、差），不改判準。
用法：python compare_to_archive.py --records <..> --a04 <..> --c7 <..> --s14 <..> [--expected <本夾>/expected] --out <報告.json>
"""
import os, json, argparse
import numpy as np, pandas as pd

TOL = 1e-9
HERE = os.path.dirname(os.path.abspath(__file__))


def _num(s):
    return pd.to_numeric(s, errors="coerce") if s.dtype == object or str(s.dtype).startswith(("str", "string")) else s


def compare(exp, got, keys, name, main_cols=None):
    exp = exp.copy(); got = got.copy()
    for k in keys:
        if k == "M":
            exp[k] = pd.to_numeric(exp[k], errors="coerce").fillna(-1).astype(int)
            got[k] = pd.to_numeric(got[k], errors="coerce").fillna(-1).astype(int)
        else:
            exp[k] = exp[k].astype(str); got[k] = got[k].astype(str)
    m = exp.merge(got, on=keys, how="outer", suffixes=("__exp", "__got"), indicator=True)
    out = dict(file=name, n_rows_expected=int(len(exp)), n_rows_rerun=int(len(got)),
               n_only_expected=int((m._merge == "left_only").sum()), n_only_rerun=int((m._merge == "right_only").sum()),
               only_expected=m.loc[m._merge == "left_only", keys].astype(str).values.tolist(),
               only_rerun=m.loc[m._merge == "right_only", keys].astype(str).values.tolist(), columns={}, mismatches=[])
    b = m[m._merge == "both"]
    cols = [c for c in exp.columns if c not in keys and c in got.columns]
    for c in cols:
        e = b[c + "__exp"]; g = b[c + "__got"]
        en, gn = pd.to_numeric(e, errors="coerce"), pd.to_numeric(g, errors="coerce")
        numeric = bool(en.notna().sum() >= e.notna().sum() and gn.notna().sum() >= g.notna().sum() and e.notna().sum() > 0) \
            or (e.isna().all() and g.isna().all())
        if e.dtype == bool or g.dtype == bool or set(map(str, e.dropna().unique())) <= {"True", "False"}:
            numeric = False
        if numeric:
            ea, ga = en.to_numpy(float), gn.to_numpy(float)
            both_nan = np.isnan(ea) & np.isnan(ga)
            d = np.where(both_nan, 0.0, np.abs(ea - ga))
            d = np.where(np.isnan(d), np.inf, d)
            bad = d > TOL
            out["columns"][c] = dict(kind="numeric", main=(main_cols is None or c in main_cols),
                                     max_abs_diff=float(d.max()) if len(d) else 0.0, n_exceed=int(bad.sum()))
            for i in np.where(bad)[0]:
                r = b.iloc[i]
                out["mismatches"].append(dict(key=[str(r[k]) for k in keys], column=c, expected=None if np.isnan(ea[i]) else float(ea[i]),
                                              rerun=None if np.isnan(ga[i]) else float(ga[i]), abs_diff=float(d[i])))
        else:
            es = e.astype(str).where(e.notna(), "<NA>"); gs = g.astype(str).where(g.notna(), "<NA>")
            bad = (es.to_numpy() != gs.to_numpy())
            out["columns"][c] = dict(kind="text", main=(main_cols is None or c in main_cols), n_differ=int(bad.sum()))
            for i in np.where(bad)[0]:
                r = b.iloc[i]
                out["mismatches"].append(dict(key=[str(r[k]) for k in keys], column=c, expected=es.iloc[i], rerun=gs.iloc[i]))
    main_bad = [x for x in out["mismatches"] if (main_cols is None or x["column"] in main_cols)]
    out["n_mismatch_all_columns"] = len(out["mismatches"])
    out["n_mismatch_main_columns"] = len(main_bad)
    out["pass_main"] = bool(len(main_bad) == 0 and out["n_only_expected"] == 0 and out["n_only_rerun"] == 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True); ap.add_argument("--a04", required=True)
    ap.add_argument("--c7", required=True); ap.add_argument("--s14", required=True)
    ap.add_argument("--expected", default=os.path.join(HERE, "expected")); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    X = a.expected
    rep = dict(tolerance_abs=TOL, results=[])
    got = pd.read_csv(a.records); got = got[got.protocol.isin(["P-B", "P-B+"])]
    rep["results"].append(compare(pd.read_csv(os.path.join(X, "soh_records_v911_PB_PBplus.csv")), got,
                                  ["source", "protocol", "cell", "model", "M"], "soh_records（P-B、P-B+）",
                                  main_cols=["RMSE_pct", "MAE_pct", "R2", "n_ref", "n_eval"]))
    rep["results"].append(compare(pd.read_csv(os.path.join(X, "A04_reference_only_baselines_percell.csv")), pd.read_csv(a.a04),
                                  ["source", "cell", "M"], "A04_reference_only_baselines_percell.csv"))
    rep["results"].append(compare(pd.read_csv(os.path.join(X, "c7_cluster_bootstrap.csv")), pd.read_csv(a.c7),
                                  ["M", "subset"], "c7_cluster_bootstrap.csv"))
    rep["results"].append(compare(pd.read_csv(os.path.join(X, "c7_S14_table_data.csv")), pd.read_csv(a.s14),
                                  ["M", "subset"], "c7_S14_table_data.csv"))
    rep["all_pass_main"] = all(r["pass_main"] for r in rep["results"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(rep, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for r in rep["results"]:
        mx = {c: v.get("max_abs_diff", v.get("n_differ")) for c, v in r["columns"].items()}
        print(f"[{'PASS' if r['pass_main'] else 'FAIL'}] {r['file']}: 列 {r['n_rows_expected']}/{r['n_rows_rerun']}，"
              f"缺 {r['n_only_expected']}／多 {r['n_only_rerun']}，主要欄不符 {r['n_mismatch_main_columns']}，全欄不符 {r['n_mismatch_all_columns']}")
        print("   各欄最大絕對差（文字欄為不同列數）：", json.dumps(mx, ensure_ascii=False))
    print("ALL_PASS_MAIN", rep["all_pass_main"])


if __name__ == "__main__":
    main()
