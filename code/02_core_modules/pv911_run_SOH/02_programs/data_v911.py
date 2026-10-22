"""
data_v911.py — v9.11 資料層（沿 v9.8；僅改資料路徑）。
讀 dataset_v9（40 顆 npz；Q(V) 5 mV 網格、SOH 標籤、cyc_A／EFC 軸）與 own 老師電芯逐循環容量表（7 顆 xlsx），
依先驟登記 v13 之納入規則建立「逐循環資料表」（每顆一列一循環），並產出可用電池普查（census）。
不回原始 CSV；不動任何標籤（R24：右設限不補 RUL 標籤）。
"""
import os, glob, json, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                          # pv99_run
V99 = os.path.dirname(ROOT)                           # 工作根目錄 /home/claude/v99
PV95 = os.path.join(V99, "pv95_run")
DATASET_V9 = os.path.join(PV95, "dataset_v9")
OWN_CAP_DIR = os.path.join(V99, "pv98_run", "data_own_capacity")   # v9.11：沿用 v9.8 之老師電芯容量表（唯讀）

# ── 登記常數（v13）──────────────────────────────────────────────────────────
RATED_AH = {"nasa": 2.0, "own": 3.35, "ox": float("nan")}        # 額定容量：NASA 2 Ah（05 Table 1；21 之 1.4 Ah＝70 %）、NCR18650GA 3.35 Ah；Oxford 未用額定
EOL_DEFS = {                                                     # EOL 門檻（多定義並列；主＝EOL_A）
    "EOL_A": dict(kind="frac_qref", value=0.80, label="80 % 初始容量 Q_ref（20：80 % of initial；v9.5 標籤鏈）"),
    "EOL_B": dict(kind="frac_rated", value=0.70, label="70 % 額定容量（21／05：NASA 1.4 Ah）"),
    "EOL_C": dict(kind="frac_rated", value=0.80, label="80 % 額定容量（24：NASA 1.6 Ah）"),
}
IC_WINDOW = {"nasa": (3.80, 4.10), "ox": (3.60, 4.00), "own": (3.80, 4.10)}   # 部分 IC 區間（22／23／21：NASA 3.8–4.1 V；ox／own 沿 v9.5 HI 視窗）
JOURNAL_NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
MIN_CYCLES_FOR_START = 100          # 起算點 60／80／100 循環（21／24）→ 至少 100 循環＋測試段
MIN_SOH_RANGE = 0.05                # SOH 變化量不足 5 % 之顆不進 SOH 架構（無法建立退化映射；沿 v8 對 C/4 群之排除理由）


def _first_cross(cyc, soh, thr):
    idx = np.where(soh <= thr)[0]
    return (float(cyc[idx[0]]), int(idx[0])) if len(idx) else (None, None)


def load_ica_cells():
    """讀 dataset_v9 全部 40 顆；回傳 dict cell → dict(src, grid, Q, soh, cyc, efc, qref, T, I_max, cap_seg, cap_soh)."""
    cells = {}
    for f in sorted(glob.glob(os.path.join(DATASET_V9, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        c = str(z["cell_id"]); src = str(z["src_arm"])
        cells[c] = dict(src=src, sub_arm=str(z["sub_arm"]), grid=np.asarray(z["grid"], float), Q=np.asarray(z["Q"], float),
                        support=np.asarray(z["support"]), soh=np.asarray(z["soh"], float), cyc=np.asarray(z["cyc_A"], float),
                        efc=np.asarray(z["efc"], float), qref=float(z["q_ref"]), T=np.asarray(z["T"], float), I_max=np.asarray(z["I_max"], float),
                        cap_seg=np.asarray(z["cap_seg"], float), cap_soh=np.asarray(z["cap_soh"], float), n_events=int(len(z["soh"])),
                        F_cond=np.asarray(z["F_cond"], float), proto_CCCV=int(round(float(np.asarray(z["F_cond"], float)[:, 3].mean()))))
    return cells


def load_own_capacity_cells():
    """老師電芯 NCR18650GA 1–7：逐循環放電容量（mAh）＋ MAX-T；GA5／GA7 取 20240115 延長版。回傳 dict cell → dict(cyc, cap_Ah, maxT, qref)."""
    files = {"GA1": "NCR18650GA-1cyclelife.xlsx", "GA2": "NCR18650GA-2cyclelife.xlsx", "GA3": "NCR18650GA-3cyclelife.xlsx",
             "GA4": "NCR18650GA-4cyclelife.xlsx", "GA5": "NCR18650GA5-Cyclecount.xlsx", "GA6": "NCR18650GA-6cyclelife.xlsx", "GA7": "NCR18650GA7-Cyclecount.xlsx"}
    out = {}
    for c, fn in files.items():
        p = os.path.join(OWN_CAP_DIR, fn)
        pcsv = p[:-5] + ".csv"   # 同名 CSV（由同一 xlsx 前三欄以 %.17g 匯出，數值逐一相等；供無 openpyxl 之 soc_env 使用）
        if os.path.exists(pcsv):
            df = pd.read_csv(pcsv).iloc[:, :3]
        else:
            df = pd.read_excel(p, header=0).iloc[:, :3]
        df.columns = ["cycle", "cap_mAh", "maxT"]
        cyc = pd.to_numeric(df.cycle, errors="coerce").to_numpy(float); cap = pd.to_numeric(df.cap_mAh, errors="coerce").to_numpy(float)
        mt = pd.to_numeric(df.maxT, errors="coerce").to_numpy(float)
        ok = np.isfinite(cyc) & np.isfinite(cap); cyc, cap, mt = cyc[ok], cap[ok], mt[ok]
        order = np.argsort(cyc); cyc, cap, mt = cyc[order], cap[order], mt[order]
        qref = float(np.median(cap[:5]) / 1000.0)      # 初始容量＝前 5 循環中位（Ah）
        out[c] = dict(src="own", cyc=cyc, cap_Ah=cap / 1000.0, soh=cap / 1000.0 / qref, maxT=mt, qref=qref, file=fn, n_cycles=int(len(cyc)))
    return out


def eol_threshold_soh(src, qref, key):
    d = EOL_DEFS[key]
    if d["kind"] == "frac_qref":
        return d["value"]
    r = RATED_AH.get(src, float("nan"))
    return (d["value"] * r / qref) if np.isfinite(r) else None


def per_cycle_table(cell):
    """把逐充電事件（可能一循環多事件，ox G5）聚合為逐循環：Q(V) 取事件平均、SOH／EFC／T 取平均。"""
    cyc = cell["cyc"]; u, inv = np.unique(cyc, return_inverse=True)
    n = len(u); Q = np.full((n, cell["Q"].shape[1]), np.nan); soh = np.zeros(n); efc = np.zeros(n); T = np.zeros(n); imax = np.zeros(n); cnt = np.zeros(n)
    for i in range(n):
        m = inv == i; cnt[i] = m.sum()
        with np.errstate(all="ignore"):
            Q[i] = np.nanmean(cell["Q"][m], axis=0)
        soh[i] = cell["soh"][m].mean(); efc[i] = cell["efc"][m].mean(); T[i] = cell["T"][m].mean(); imax[i] = cell["I_max"][m].mean()
    return dict(cyc=u, Q=Q, soh=soh, efc=efc, T=T, I_max=imax, n_ev_per_cyc=cnt)


def census(ica_cells, cap_cells):
    """可用電池普查：每顆之來源、循環數、SOH 範圍、IC 視窗覆蓋、各 EOL 定義之交點循環、納入判定與理由。"""
    rows = []
    for c, d in ica_cells.items():
        src = d["src"]; lo, hi = IC_WINDOW[src]; m = (d["grid"] >= lo - 1e-9) & (d["grid"] <= hi + 1e-9)
        pc = per_cycle_table(d)
        cov_full = float(np.isfinite(pc["Q"][:, m]).all(axis=1).mean())
        cov_any = float(np.isfinite(pc["Q"][:, m]).mean())
        rng = float(d["soh"].max() - d["soh"].min())
        r = dict(source=src, cell=c, data_kind="ICA+capacity", n_events=d["n_events"], n_cycles=int(len(pc["cyc"])), cyc_max=int(pc["cyc"].max()),
                 efc_max=float(d["efc"].max()), q_ref_Ah=d["qref"], soh_min=float(d["soh"].min()), soh_max=float(d["soh"].max()), soh_range=rng,
                 ic_window=f"{lo:.2f}–{hi:.2f} V", win_cov_allbins=cov_full, win_cov_bins=cov_any, T_mean=float(np.mean(d["T"])),
                 I_chg_A=float(np.median(d["I_max"])), I_dis_A=float(np.median(d["F_cond"][:, 2]) * d["qref"]) if d.get("F_cond") is not None else None, proto_CCCV=int(d["proto_CCCV"]) if d.get("proto_CCCV") is not None else None)
        for k in EOL_DEFS:
            thr = eol_threshold_soh(src, d["qref"], k)
            if thr is None:
                r[f"{k}_thr_soh"] = None; r[f"{k}_cyc"] = None
            else:
                r[f"{k}_thr_soh"] = float(thr); r[f"{k}_cyc"] = _first_cross(pc["cyc"], pc["soh"], thr)[0]
        # 納入判定（登記 v13 ①）
        reasons = []
        if src == "nasa" and c not in JOURNAL_NASA_CELLS:
            reasons.append("非期刊 4 顆（B0005／06／07／18）；工況不同（倍率／溫度）")
        if r["n_cycles"] < MIN_CYCLES_FOR_START:
            reasons.append(f"循環數 {r['n_cycles']} < {MIN_CYCLES_FOR_START}（起算點 60／80／100 不適用）")
        if rng < MIN_SOH_RANGE:
            reasons.append(f"SOH 變化 {rng:.3f} < {MIN_SOH_RANGE}")
        if cov_any < 0.90:
            reasons.append(f"IC 視窗覆蓋 {cov_any:.2f} < 0.90")
        r["soh_arch"] = "納入" if not reasons else "不納入"
        r["soh_arch_reason"] = "；".join(reasons) if reasons else "通過 ①–④"
        has_eol = r["EOL_A_cyc"] is not None
        r["rul_arch"] = ("納入（有 EOL）" if has_eol else "只報預測 EOL（右設限，無誤差）") if not reasons else "不納入"
        r["rul_arch_reason"] = ("" if not reasons else r["soh_arch_reason"]) if not reasons else r["soh_arch_reason"]
        if not reasons:
            r["rul_arch_reason"] = "EOL_A 已達" if has_eol else "EOL_A 未達（R24 不補標籤）"
        rows.append(r)
    for c, d in cap_cells.items():
        rng = float(d["soh"].max() - d["soh"].min())
        r = dict(source="own", cell=c, data_kind="capacity-only", n_events=None, n_cycles=d["n_cycles"], cyc_max=int(d["cyc"].max()), efc_max=None,
                 q_ref_Ah=d["qref"], soh_min=float(d["soh"].min()), soh_max=float(d["soh"].max()), soh_range=rng, ic_window="—（無 CC 充電 V/I 時序）",
                 win_cov_allbins=None, win_cov_bins=None, T_mean=float(np.nanmean(d["maxT"])))
        for k in EOL_DEFS:
            thr = eol_threshold_soh("own", d["qref"], k)
            r[f"{k}_thr_soh"] = float(thr); r[f"{k}_cyc"] = _first_cross(d["cyc"], d["soh"], thr)[0]
        r["soh_arch"] = "不納入"; r["soh_arch_reason"] = "只有逐循環容量表，無 CC 充電 V/I 時序 → 無法計算 IC（架構一不適用）"
        has_eol = r["EOL_A_cyc"] is not None
        short = d["n_cycles"] < MIN_CYCLES_FOR_START
        r["rul_arch"] = ("納入（容量軌跡法 B2；有 EOL）" if has_eol else "只報預測 EOL（右設限）") if not short else "不納入"
        r["rul_arch_reason"] = (f"循環數 {d['n_cycles']} < {MIN_CYCLES_FOR_START}" if short else ("EOL_A 已達" if has_eol else "EOL_A 未達（R24）"))
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ica = load_ica_cells(); cap = load_own_capacity_cells()
    df = census(ica, cap)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print(df[["source", "cell", "data_kind", "n_cycles", "q_ref_Ah", "soh_min", "win_cov_bins", "EOL_A_cyc", "EOL_B_cyc", "EOL_C_cyc", "soh_arch", "rul_arch", "soh_arch_reason"]].to_string(index=False))
