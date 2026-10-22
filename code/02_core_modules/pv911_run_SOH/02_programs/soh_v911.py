"""
soh_v911.py — 架構一（SOH 估測）：逐循環特徵表、相關係數篩選、三種期刊驗證協定。
  P-A 同顆前 N 循環訓練（22／23：N=30；起點 s0∈{1,40}）→ 其餘循環測試
  P-B 留一顆 LOOCV（05：N−1 顆訓練、1 顆測試；主設定含循環數輔助特徵，消融不含）
  P-C 同顆隨機 80/20（25）
指標：MAE／RMSE（SOH 與 Ah）、最大相對誤差、95 % CI 涵蓋率（GPR）。
"""
import os, json, numpy as np, pandas as pd
import data_v911 as D, ic_v911 as IC, models_v911 as M

SEED = M.SEED
CKPT = None   # run_v98 設定：逐顆檢查點目錄（可續跑）
KERNELS = None   # v9.8 附加：若設為 list，GPR 之最佳化 kernel 以 dict 追加（E4 之 gpr_kernel_params_v98.json）


def kern_record(tag, name, model, feats):
    """v9.8 附加：記錄 GPR 之最佳化 kernel 參數（僅在 KERNELS 為 list 時動作；不影響 v9.7 路徑）。"""
    if KERNELS is None or not name.startswith("GPR"):
        return
    try:
        gp = model.named_steps["m"]; k = gp.kernel_
        prm = k.get_params(deep=True)
        ls = prm.get("k1__k2__length_scale", None)
        const = prm.get("k1__k1__constant_value", None)
        noise = prm.get("k2__noise_level", None)
        ls = (np.asarray(ls, float).ravel().tolist() if ls is not None else None)
        KERNELS.append(dict(tag=tag, model=name, features=list(feats), kernel=str(k),
                            length_scale=ls, constant_value=(float(const) if const is not None else None),
                            noise_level=(float(noise) if noise is not None else None),
                            log_marginal_likelihood=float(gp.log_marginal_likelihood_value_)))
    except Exception as e:      # 不讓記錄失敗影響主結果
        KERNELS.append(dict(tag=tag, model=name, error=str(e)))


def _ck(key):
    return None if CKPT is None else os.path.join(CKPT, key.replace("|", "__") + ".json")


def _load(key):
    p = _ck(key)
    if p and os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return None


def _save(key, obj):
    p = _ck(key)
    if p:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)


def build_cell_frame(cell, cfg, window=None):
    """單顆逐循環表：cyc, efc, soh, cap_Ah, T, I_max ＋ 特徵（清洗後）；另回傳原始特徵與清洗計數。"""
    src = cell["src"]; pc = D.per_cycle_table(cell); g = cell["grid"]; win = tuple(window) if window is not None else D.IC_WINDOW[src]
    raw = []
    for i in range(len(pc["cyc"])):
        q = IC.fill_Q(pc["Q"][i]); ic = IC.ic_raw(q, g)
        icf = IC.gs_filter(ic, cfg["sigma_mV"]) if cfg["filter"] == "GS" else IC.sg_filter(ic, cfg["sg_window_pts"], cfg["sg_order"])
        raw.append(IC.features(icf, q, g, win, pc["I_max"][i]))
    R = pd.DataFrame(raw); F = R.copy(); ncl = {}
    for c in F.columns:
        F[c], ncl[c] = IC.robust_z_clean(F[c].to_numpy(), cfg["clean_window"], cfg["clean_z"], cfg["clean_edge_ratio"])
    base = pd.DataFrame(dict(cyc=pc["cyc"], efc=pc["efc"], soh=pc["soh"], cap_Ah=pc["soh"] * cell["qref"], T=pc["T"], I_max=pc["I_max"], n_ev=pc["n_ev_per_cyc"]))
    return pd.concat([base, F], axis=1), R, ncl


def correlation_table(frames, cells):
    rows = []
    for c in cells:
        f = frames[c]; feats = [k for k in f.columns if k not in BASE_COLS]
        cr = IC.correlations(f[feats], f["soh"].to_numpy())
        for k, (r, rho) in cr.items():
            rows.append(dict(cell=c, feature=k, pearson=r, spearman=rho))
    return pd.DataFrame(rows)


def select_features(corr_df, cells, min_abs_rho, top_k):
    """登記規則：min over cells |ρ_S| ≥ min_abs_rho（22 之跨顆一致性）→ 依 mean |ρ_S| 取前 top_k（05 之 top ranking）。"""
    sub = corr_df[corr_df.cell.isin(cells)]
    piv = sub.pivot(index="feature", columns="cell", values="spearman").abs()
    piv = piv.dropna(how="any")
    ok = piv[piv.min(axis=1) >= min_abs_rho]
    if len(ok) < top_k:                                      # 不足 top_k 時放寬為全部候選之前 top_k（登記備援）
        ok = piv
    order = ok.mean(axis=1).sort_values(ascending=False)
    return list(order.index[:top_k]), piv.min(axis=1).to_dict(), piv.mean(axis=1).to_dict()


def metrics(y, yhat, qref, sd=None):
    e = yhat - y; out = dict(n=int(len(y)), MAE_soh=float(np.mean(np.abs(e))), RMSE_soh=float(np.sqrt(np.mean(e ** 2))),
                             MAE_Ah=float(np.mean(np.abs(e)) * qref), RMSE_Ah=float(np.sqrt(np.mean(e ** 2)) * qref),
                             MaxRE_pct=float(np.max(np.abs(e) / y) * 100), MAPE_pct=float(np.mean(np.abs(e) / y) * 100))
    if sd is not None:
        out["CI95_cov"] = float(np.mean(np.abs(e) <= 1.96 * sd)); out["CI95_halfwidth_soh"] = float(np.mean(1.96 * sd))
    return out


def protocol_A(frames, cells, feats, cfg, qrefs, log):
    """P-A：同顆前 N 循環訓練；s0 起點。回傳 (records, predictions)."""
    recs, preds = [], {}
    for c in cells:
        ck = _load(f"PA|{c}")
        if ck:
            recs += ck["recs"]; preds.update({(k.split("|")[0], int(float(k.split("|")[1])), k.split("|")[2]): v for k, v in ck["preds"].items()}); log(f"  P-A {c} (checkpoint)"); continue
        f = frames[c]; recs_c, preds_c = [], {}
        for s0 in cfg["PA_starts"]:
            fs = f[f.cyc >= s0]
            if len(fs) < cfg["PA_ntrain"] + 10:
                recs_c.append(dict(protocol="P-A", cell=c, s0=s0, model="—", note="循環數不足", n_train=None)); continue
            tr = fs.iloc[:cfg["PA_ntrain"]]; te = fs.iloc[cfg["PA_ntrain"]:]
            for name in M.MODEL_NAMES:
                mu, sd, _m = M.fit_predict(name, tr[feats].to_numpy(), tr.soh.to_numpy(), te[feats].to_numpy())
                kern_record(f"P-A|{c}|s0={s0}", name, _m, feats)
                r = dict(protocol="P-A", cell=c, s0=s0, model=name, n_train=len(tr), n_test=len(te), **metrics(te.soh.to_numpy(), mu, qrefs[c], sd))
                recs_c.append(r); preds_c[(c, s0, name)] = dict(cyc=te.cyc.to_numpy().tolist(), y=te.soh.to_numpy().tolist(), yhat=mu.tolist(), sd=(sd.tolist() if sd is not None else None))
                log(f"  P-A {c} s0={s0} {name}: RMSE {r['RMSE_soh']*100:.2f} % MAE {r['MAE_soh']*100:.2f} %")
        _save(f"PA|{c}", dict(recs=recs_c, preds={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in preds_c.items()})); recs += recs_c; preds.update(preds_c)
    return recs, preds


def protocol_B(frames, cells, corr_df, cfg, qrefs, log, src):
    """P-B：LOOCV；特徵選擇只用訓練顆；含／不含循環數（05 主設定／Group 2）。"""
    recs, preds, sel_log = [], {}, []
    if len(cells) < 2:
        return recs, preds, sel_log
    for c in cells:
        ck = _load(f"PB|{c}")
        if ck:
            recs += ck["recs"]; sel_log.append(ck["sel"]); preds.update({(k.split("|")[0], k.split("|")[1] == "True", k.split("|")[2]): v for k, v in ck["preds"].items()}); log(f"  P-B {c} (checkpoint)"); continue
        tr_cells = [x for x in cells if x != c]
        feats, _, _ = select_features(corr_df, tr_cells, cfg["sel_min_rho"], cfg["sel_top_k"])
        sel_c = dict(test_cell=c, train_cells=tr_cells, selected=feats); sel_log.append(sel_c); recs_c, preds_c = [], {}
        Xtr = pd.concat([frames[x] for x in tr_cells]); Xte = frames[c]
        for with_cyc in (True, False):
            cols = feats + (["cyc"] if with_cyc else [])
            for name in M.MODEL_NAMES:
                mu, sd, _m = M.fit_predict(name, Xtr[cols].to_numpy(), Xtr.soh.to_numpy(), Xte[cols].to_numpy(), sub=cfg["PB_gpr_subsample"])
                kern_record(f"P-B|test={c}|with_cycle={with_cyc}", name, _m, cols)
                r = dict(protocol="P-B", cell=c, with_cycle=with_cyc, model=name, n_train=len(Xtr), n_test=len(Xte), n_train_cells=len(tr_cells), **metrics(Xte.soh.to_numpy(), mu, qrefs[c], sd))
                recs_c.append(r); preds_c[(c, with_cyc, name)] = dict(cyc=Xte.cyc.to_numpy().tolist(), y=Xte.soh.to_numpy().tolist(), yhat=mu.tolist(), sd=(sd.tolist() if sd is not None else None))
                log(f"  P-B {src} test={c} cyc={with_cyc} {name}: RMSE {r['RMSE_soh']*100:.2f} %")
        _save(f"PB|{c}", dict(recs=recs_c, sel=sel_c, preds={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in preds_c.items()})); recs += recs_c; preds.update(preds_c)
    return recs, preds, sel_log


def protocol_C(frames, cells, feats, cfg, qrefs, log):
    """P-C：同顆隨機 80/20（25）。"""
    recs, preds = [], {}
    rng = np.random.RandomState(SEED)
    for c in cells:
        f = frames[c]; n = len(f); idx = rng.permutation(n); nte = int(round(0.2 * n)); te_i, tr_i = np.sort(idx[:nte]), np.sort(idx[nte:])
        ck = _load(f"PC|{c}")
        if ck:
            recs += ck["recs"]; preds.update({(k.split("|")[0], k.split("|")[1]): v for k, v in ck["preds"].items()}); log(f"  P-C {c} (checkpoint)"); continue
        tr, te = f.iloc[tr_i], f.iloc[te_i]; recs_c, preds_c = [], {}
        for name in M.MODEL_NAMES:
            mu, sd, _m = M.fit_predict(name, tr[feats].to_numpy(), tr.soh.to_numpy(), te[feats].to_numpy(), sub=cfg["PB_gpr_subsample"])
            kern_record(f"P-C|{c}", name, _m, feats)
            r = dict(protocol="P-C", cell=c, model=name, n_train=len(tr), n_test=len(te), **metrics(te.soh.to_numpy(), mu, qrefs[c], sd))
            recs_c.append(r); preds_c[(c, name)] = dict(cyc=te.cyc.to_numpy().tolist(), y=te.soh.to_numpy().tolist(), yhat=mu.tolist(), sd=(sd.tolist() if sd is not None else None))
            log(f"  P-C {c} {name}: RMSE {r['RMSE_soh']*100:.2f} %")
        _save(f"PC|{c}", dict(recs=recs_c, preds={f"{k[0]}|{k[1]}": v for k, v in preds_c.items()})); recs += recs_c; preds.update(preds_c)
    return recs, preds


# ── v9.11 搜尋擴充：特徵族（F_IC／F_dQ／F_both）與 REL 正規化 ──────────────────
# A1（v9.11）：充電段有效性閘之欄位一併列為基礎欄（不進特徵篩選、不進 robust_z_clean）
SEG_COLS = ("seg_first_V", "seg_last_V", "seg_valid_d0", "seg_valid_d50", "seg_valid_d100")
GAP_COLS = ("cyc_A", "cyc_c", "drop_flag", "label_gate")
BASE_COLS = ("cyc", "efc", "soh", "cap_Ah", "T", "I_max", "n_ev") + SEG_COLS + GAP_COLS


def apply_label_gate(keep, full, gate):
    """A2 開關（v9.11 追加之搜尋維度）。
      gate="on"  ：拉掉過渡循環，循環軸＝cyc_c（連續重編）— 即 v9.8 own_gap_rule 行為
      gate="off" ：不剔除任何循環，循環軸＝cyc_A（原物理循環）
    兩者皆保留 cyc_A／drop_flag 欄，並加註 label_gate 欄（基礎欄，不進特徵）。"""
    if gate == "on":
        f = keep.copy()
        f["cyc"] = f["cyc_c"].astype(float)
    elif gate == "off":
        f = full.copy()
        f["cyc"] = f["cyc_A"].astype(float)
        f["cyc_c"] = f["cyc_A"].astype(float)       # off：不重編，cyc_c 欄等同 cyc_A（避免被拉掉列之 NaN 影響 dropna）
    else:
        raise ValueError(gate)
    f["label_gate"] = gate
    return f.reset_index(drop=True)

# A1 搜尋維度 δ（V）：d0／d50／d100 對應 seg_valid_d* 欄；inf ＝ 不設閘（v9.9 行為）
DELTAS = {"d0": 0.0, "d50": 0.05, "d100": 0.10, "inf": float("inf")}


def apply_delta(frames, dkey):
    """A1：依選定 δ 過濾逐顆表（訓練與評估皆剔除無效循環）。dkey ∈ DELTAS。"""
    if dkey == "inf":
        return {c: f.reset_index(drop=True) for c, f in frames.items()}
    col = f"seg_valid_{dkey}"
    return {c: f[f[col] == 1].reset_index(drop=True) for c, f in frames.items()}


def build_cell_frame_v911(cell, cfg, window, family="F_IC"):
    """v9.11：單顆逐循環表，特徵族可選。
      F_IC  ＝ v9.7 之部分 IC 區間特徵（FV_k／peak_h／V_pk／area／intercept／slope／t_pk_h）
      F_dQ  ＝ 部分充電容量增量 ΔQ(V)=Q(V)−Q(V_lo)（每 50 mV）
      F_both＝ 兩者並用
    全部特徵欄一律套 robust_z_clean（與 v9.7 同參數）。回傳 (frame, raw, n_clean)。"""
    src = cell["src"]; pc = D.per_cycle_table(cell); g = cell["grid"]; win = tuple(window)
    raw = []; seg_lo = []; seg_hi = []
    for i in range(len(pc["cyc"])):
        a, b = IC.seg_bounds(pc["Q"][i], g)                # A1：未補值之原始 Q(V) 列
        seg_lo.append(a); seg_hi.append(b)
        q = IC.fill_Q(pc["Q"][i]); d = {}
        if family in ("F_IC", "F_both"):
            ic = IC.ic_raw(q, g)
            icf = IC.gs_filter(ic, cfg["sigma_mV"]) if cfg["filter"] == "GS" else IC.sg_filter(ic, cfg["sg_window_pts"], cfg["sg_order"])
            d.update(IC.features(icf, q, g, win, pc["I_max"][i]))
        if family in ("F_dQ", "F_both"):
            d.update(IC.dq_features(q, g, win, cfg.get("dq_step_V", 0.05)))
        raw.append(d)
    R = pd.DataFrame(raw); F = R.copy(); ncl = {}
    for c in F.columns:
        F[c], ncl[c] = IC.robust_z_clean(F[c].to_numpy(), cfg["clean_window"], cfg["clean_z"], cfg["clean_edge_ratio"])
    seg_lo = np.asarray(seg_lo, float); seg_hi = np.asarray(seg_hi, float)
    base = pd.DataFrame(dict(cyc=pc["cyc"], efc=pc["efc"], soh=pc["soh"], cap_Ah=pc["soh"] * cell["qref"],
                             T=pc["T"], I_max=pc["I_max"], n_ev=pc["n_ev_per_cyc"],
                             seg_first_V=seg_lo, seg_last_V=seg_hi))
    for dk in ("d0", "d50", "d100"):                       # A1：三個有限 δ 之有效性旗標（δ=inf 不設閘，不落欄）
        base[f"seg_valid_{dk}"] = np.array([int(IC.seg_valid(a, b, win, DELTAS[dk]))
                                            for a, b in zip(seg_lo, seg_hi)], dtype=int)
    return pd.concat([base, F], axis=1), R, ncl


def rel_normalize(frame, nref=3):
    """REL 正規化：每顆之每個特徵除以「該顆前 nref 個循環之中位數」（參考值 |ref| < 1e-9 者跳過不除）。
    只用該顆初始循環，與 Q_ref 定義一致；不動 cyc／efc／soh 等基礎欄。"""
    f = frame.copy()
    feats = [k for k in f.columns if k not in BASE_COLS]
    ref = f[feats].iloc[:nref].median()
    for k in feats:
        if abs(float(ref[k])) > 1e-9:
            f[k] = f[k] / float(ref[k])
    return f
