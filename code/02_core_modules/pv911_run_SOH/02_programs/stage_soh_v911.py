"""
stage_soh_v911.py — SOH 階段（架構一正式跑）。

每來源以 §6 之主設定跑三協定 × 五模型，逐顆記錄：
  P-A  同顆前 N=30 循環訓練（s0 ∈ {1, 40}）；特徵篩選只用該 30 個訓練循環（§5.3）
  P-B  留一顆 LOOCV；特徵篩選只用訓練顆（§5.3）；循環軸依該來源組態（cyc／efc／none）
  P-C  同顆隨機 80/20（seed 20260830）；特徵篩選只用訓練之 80 %（§5.3）
指標：RMSE／MAE（% 與 Ah）、R²、MaxRE、MAPE、GPR 之 CI95 涵蓋率與平均半寬。
逐顆檢查點：results_v911/_ckpt_soh/<src>__<protocol>__<cell>.json（可續跑）。

v9.11：前處理（視窗／濾波／特徵族／正規化／label_gate／δ）與主模型 GPR-LSP 之核組態（週期核／ARD／子抽樣）
一律取自 config.SOH_SELECTED_V911（每來源獨立，由 search_soh_v911 依 §3.1 選定）；
其餘四模型（GPR-SE／Ridge／SVR／MLP）沿 v9.x 基線設定全報；P-B+ 之 M ∈ {10, 20}（M=50 取消，登記）。

計算成本例外（登記於 prereg v17）：ox P-B 之五模型 × 21 折直接沿用 SEARCH_SOH 之檢查點；
ox 之 GPR-SE 於其餘協定（P-A／P-C／P-B+）一律略過（EXEC.soh_stage_reduction）
（同組態／協定／模型／種子／篩選規則之確定性計算）；沿用列 reused=1，無 CI95 與逐點預測。
"""
import os, json, time
import numpy as np, pandas as pd

import data_v911 as D, ic_v911 as IC, soh_v911 as S, models_v911 as M
import config_v911 as C
import stage_data_v911 as SD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as CK, RBF, WhiteKernel

SEED = C.SEED
MODELS = C.SOH_FIXED["models"]
MODELS_BY_SRC = C.EXEC["soh_models_by_source"]   # ox 略過 GPR-SE（EXEC.soh_stage_reduction）


def models_of(src):
    return list(MODELS_BY_SRC.get(src, MODELS))
SEARCH_CKPT = None   # 由 main 設定
ONLY = None          # v3 平行化：None＝全部測試電芯；set＝只跑其中之電芯（其餘略過，不改任何計算）
_BASE_PRED = {}      # (src, cell, model) → P-B 之 LOOCV 預測 mu0（P-B 與 P-B+ 之預訓練完全相同，
                     # 同一折同一模型同一資料 → 確定性相等；快取只避免重複擬合，不改任何數值）
M_LIST = C.SEARCH_SPACE["soh"]["M"]          # v9.11：P-B+ 之 M ∈ {10, 20}，主 M=20（M=50 取消，見 SEARCH_SPACE.soh.M_cancelled）


# ── 指標 ──────────────────────────────────────────────────────────────────────
def score(y, yhat, qref, sd=None):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float); e = yhat - y
    sst = float(np.sum((y - y.mean()) ** 2))
    out = dict(n_test=int(len(y)), RMSE_pct=float(np.sqrt(np.mean(e ** 2)) * 100),
               MAE_pct=float(np.mean(np.abs(e)) * 100),
               R2=(float(1 - np.sum(e ** 2) / sst) if sst > 0 else float("nan")),
               RMSE_Ah=float(np.sqrt(np.mean(e ** 2)) * qref), MAE_Ah=float(np.mean(np.abs(e)) * qref),
               MaxRE_pct=float(np.max(np.abs(e) / y) * 100), MAPE_pct=float(np.mean(np.abs(e) / y) * 100))
    if sd is not None:
        out["CI95_cov"] = float(np.mean(np.abs(e) <= 1.96 * np.asarray(sd)))
        out["CI95_halfwidth_soh"] = float(np.mean(1.96 * np.asarray(sd)))
    else:
        out["CI95_cov"] = None; out["CI95_halfwidth_soh"] = None
    return out


def judge(r):
    T = C.TARGETS["T_SOH"]
    ok = (r["RMSE_pct"] < T["RMSE_pct"]) and (r["MAE_pct"] < T["MAE_pct"]) and (r["R2"] > T["R2"])
    return int(bool(ok))


def mk(name, sel):
    """（v9.11）主模型 GPR-LSP 之核組態 kwargs；其餘四模型沿 v9.x 基線設定（不帶 kwargs）。"""
    return C.gpr_kwargs(sel) if name == "GPR-LSP" else None


def cfg_cols(sel):
    """（v9.11）records 之前處理組態欄。"""
    return dict(window=f"{sel['window'][0]:.2f}-{sel['window'][1]:.2f}", filter=sel["filter_key"],
                norm=sel["norm"], family=sel["family"], top_k=int(sel["top_k"]),
                periodic=int(bool(sel["periodic"])), ard=int(bool(sel["ard"])), sub=int(sel["sub"]),
                config_id=sel["config_id"])


def cols_of(feats, cyc_axis):
    return list(feats) + ([] if cyc_axis == "none" else [cyc_axis])


def select_on(rows_frame, cell, min_rho, top_k):
    """在給定之訓練資料列上做特徵篩選（規則同 soh_v911.select_features）。回傳 (feats, corr_df)。"""
    feats_all = [k for k in rows_frame.columns if k not in S.BASE_COLS]
    cr = IC.correlations(rows_frame[feats_all], rows_frame["soh"].to_numpy(float))
    corr = pd.DataFrame([dict(cell=cell, feature=k, pearson=v[0], spearman=v[1]) for k, v in cr.items()])
    feats, _mn, _mean = S.select_features(corr, [cell], min_rho, top_k)
    return feats, corr


# ── 三協定 ────────────────────────────────────────────────────────────────────
def run_PA(frames, src, qrefs, log):
    sel = C.soh_sel(src); recs, preds, sel_log = [], {}, []
    for c in sorted(frames):
        if ONLY is not None and c not in ONLY:
            continue
        f = frames[c]
        for s0 in C.SOH_FIXED["PA_starts"]:
            fs = f[f.cyc >= s0]
            if len(fs) < C.SOH_FIXED["PA_ntrain"] + 10:
                recs.append(dict(source=src, protocol="P-A", cell=c, s0=s0, model="—", note="循環數不足"))
                continue
            tr = fs.iloc[:C.SOH_FIXED["PA_ntrain"]]; te = fs.iloc[C.SOH_FIXED["PA_ntrain"]:]
            feats, _ = select_on(tr, c, sel["min_rho"], sel["top_k"])
            cols = cols_of(feats, sel["cyc_axis"])
            sel_log.append(dict(source=src, protocol="P-A", cell=c, s0=s0, selected=feats,
                                sel_rows_cyc=tr.cyc.tolist(), train_rows_cyc=tr.cyc.tolist(),
                                test_rows_cyc=te.cyc.tolist()))
            for name in models_of(src):
                mu, sd, _m = M.fit_predict(name, tr[cols].to_numpy(), tr.soh.to_numpy(), te[cols].to_numpy(),
                                           sub=sel["sub"], model_kwargs=mk(name, sel))
                r = dict(source=src, protocol="P-A", cell=c, s0=s0, model=name, cyc_axis=sel["cyc_axis"],
                         n_train=int(len(tr)), feats=",".join(feats), reused=0, **cfg_cols(sel),
                         **score(te.soh.to_numpy(), mu, qrefs[c], sd))
                r["pass_T_SOH"] = judge(r); recs.append(r)
                preds[f"{src}|P-A|{c}|s0={s0}|{name}"] = dict(cyc=te.cyc.tolist(), y=te.soh.tolist(),
                                                              yhat=np.asarray(mu).tolist(),
                                                              sd=(np.asarray(sd).tolist() if sd is not None else None))
                log(f"  P-A {src}/{c} s0={s0} {name}: RMSE {r['RMSE_pct']:.3f} % R² {r['R2']:.4f}")
    return recs, preds, sel_log


def run_PB(frames, src, qrefs, log):
    sel = C.soh_sel(src); recs, preds, sel_log = [], {}, []
    cl = sorted(frames)
    if len(cl) < 2:
        return recs, preds, sel_log
    corr = S.correlation_table(frames, cl)
    for c in cl:
        if ONLY is not None and c not in ONLY:
            continue
        tr_cells = [x for x in cl if x != c]
        feats, _mn, _mean = S.select_features(corr, tr_cells, sel["min_rho"], sel["top_k"])
        cols = cols_of(feats, sel["cyc_axis"])
        Xtr = pd.concat([frames[x] for x in tr_cells], ignore_index=True); Xte = frames[c]
        sel_log.append(dict(source=src, protocol="P-B", test_cell=c, train_cells=tr_cells, selected=feats))
        for name in models_of(src):
            mu, sd, _m = M.fit_predict(name, Xtr[cols].to_numpy(), Xtr.soh.to_numpy(), Xte[cols].to_numpy(),
                                       sub=sel["sub"], model_kwargs=mk(name, sel))
            _BASE_PRED[(src, c, name)] = np.asarray(mu, float)
            r = dict(source=src, protocol="P-B", cell=c, model=name, cyc_axis=sel["cyc_axis"],
                     n_train=int(len(Xtr)), n_train_cells=len(tr_cells), feats=",".join(feats), reused=0, **cfg_cols(sel),
                     **score(Xte.soh.to_numpy(), mu, qrefs[c], sd))
            r["pass_T_SOH"] = judge(r); recs.append(r)
            preds[f"{src}|P-B|{c}|loo|{name}"] = dict(cyc=Xte.cyc.tolist(), y=Xte.soh.tolist(),
                                                      yhat=np.asarray(mu).tolist(),
                                                      sd=(np.asarray(sd).tolist() if sd is not None else None))
            log(f"  P-B {src} test={c} {name}: RMSE {r['RMSE_pct']:.3f} % R² {r['R2']:.4f}")
    return recs, preds, sel_log


def run_PC(frames, src, qrefs, log):
    sel = C.soh_sel(src); recs, preds, sel_log = [], {}, []
    rng = np.random.RandomState(SEED)
    for c in sorted(frames):
        f = frames[c]; n = len(f); idx = rng.permutation(n); nte = int(round(C.SOH_FIXED["PC_test_frac"] * n))
        te_i, tr_i = np.sort(idx[:nte]), np.sort(idx[nte:])
        if ONLY is not None and c not in ONLY:
            continue
        tr, te = f.iloc[tr_i], f.iloc[te_i]
        feats, _ = select_on(tr, c, sel["min_rho"], sel["top_k"])
        cols = cols_of(feats, sel["cyc_axis"])
        sel_log.append(dict(source=src, protocol="P-C", cell=c, selected=feats,
                            sel_rows_cyc=tr.cyc.tolist(), train_rows_cyc=tr.cyc.tolist(), test_rows_cyc=te.cyc.tolist()))
        for name in models_of(src):
            mu, sd, _m = M.fit_predict(name, tr[cols].to_numpy(), tr.soh.to_numpy(), te[cols].to_numpy(),
                                       sub=sel["sub"], model_kwargs=mk(name, sel))
            r = dict(source=src, protocol="P-C", cell=c, model=name, cyc_axis=sel["cyc_axis"],
                     n_train=int(len(tr)), feats=",".join(feats), reused=0, **cfg_cols(sel),
                     **score(te.soh.to_numpy(), mu, qrefs[c], sd))
            r["pass_T_SOH"] = judge(r); recs.append(r)
            preds[f"{src}|P-C|{c}|8020|{name}"] = dict(cyc=te.cyc.tolist(), y=te.soh.tolist(),
                                                       yhat=np.asarray(mu).tolist(),
                                                       sd=(np.asarray(sd).tolist() if sd is not None else None))
            log(f"  P-C {src}/{c} {name}: RMSE {r['RMSE_pct']:.3f} % R² {r['R2']:.4f}")
    return recs, preds, sel_log


# ── P-B+（A3 稀疏參考更新協定）────────────────────────────────────────────────
def resid_gp(cyc_ref, res_ref, cyc_q):
    """A3 殘差 GPR（核與選項登記於 config ARCH_SOH_V910.A3）。"""
    k = CK(1.0, (1e-4, 1e2)) * RBF(50.0, (5.0, 2000.0)) + WhiteKernel(1e-5, (1e-8, 1e-1))
    g = GaussianProcessRegressor(kernel=k, normalize_y=False, n_restarts_optimizer=2, random_state=0)
    g.fit(np.asarray(cyc_ref, float)[:, None], np.asarray(res_ref, float))
    return g.predict(np.asarray(cyc_q, float)[:, None])


def run_PBplus(frames, src, qrefs, Mref, log):
    """P-B+：跨顆 LOOCV 預訓練（同 run_PB）＋ 測試顆每 Mref 列一個參考點之因果殘差 GPR 校正。
    第 j 區塊只用索引 ≤ 區塊起點之參考點；評估只算非參考列。"""
    sel = C.soh_sel(src); dkey = sel["delta_key"]
    recs, preds, sel_log = [], {}, []
    cl = sorted(frames)
    if len(cl) < 2:
        return recs, preds, sel_log
    corr = S.correlation_table(frames, cl)
    for c in cl:
        if ONLY is not None and c not in ONLY:
            continue
        tr_cells = [x for x in cl if x != c]
        feats, _mn, _mean = S.select_features(corr, tr_cells, sel["min_rho"], sel["top_k"])
        cols = cols_of(feats, sel["cyc_axis"])
        Xtr = pd.concat([frames[x] for x in tr_cells], ignore_index=True)
        Xte = frames[c].reset_index(drop=True)
        n = len(Xte); idx = np.arange(n); is_ref = (idx % Mref == 0)
        cy = Xte.cyc.to_numpy(float); y = Xte.soh.to_numpy(float); ev = ~is_ref
        sel_log.append(dict(source=src, protocol="P-B+", M=Mref, test_cell=c, train_cells=tr_cells,
                            selected=feats, n_ref=int(is_ref.sum()), n_eval=int(ev.sum()),
                            ref_rows_cyc=cy[is_ref].tolist()))
        for name in models_of(src):
            cached = _BASE_PRED.get((src, c, name))
            if cached is not None and len(cached) == n:
                mu0 = np.asarray(cached, float)          # 沿用同一折同一模型之 P-B 預訓練預測（確定性相等）
            else:
                mu0, _sd, _m = M.fit_predict(name, Xtr[cols].to_numpy(), Xtr.soh.to_numpy(),
                                             Xte[cols].to_numpy(), sub=sel["sub"], model_kwargs=mk(name, sel))
                _BASE_PRED[(src, c, name)] = np.asarray(mu0, float)
            mu0 = np.asarray(mu0, float); pred = np.full(n, np.nan)
            for b0 in range(0, n, Mref):
                blk = idx[b0:min(b0 + Mref, n)]
                past = np.where(is_ref & (idx <= b0))[0]
                res = y[past] - mu0[past]
                if len(past) >= 2:
                    pred[blk] = mu0[blk] + resid_gp(cy[past], res, cy[blk])
                elif len(past) == 1:
                    pred[blk] = mu0[blk] + float(res.mean())
                else:
                    pred[blk] = mu0[blk]
            r = dict(source=src, protocol="P-B+", cell=c, model=name, cyc_axis=sel["cyc_axis"],
                     delta_V=dkey, M=Mref, n_train=int(len(Xtr)), n_train_cells=len(tr_cells),
                     n_ref=int(is_ref.sum()), n_eval=int(ev.sum()), feats=",".join(feats), reused=0, **cfg_cols(sel),
                     **score(y[ev], pred[ev], qrefs[c]))
            r["pass_T_SOH"] = judge(r); recs.append(r)
            preds[f"{src}|P-B+|{c}|M={Mref}|{name}"] = dict(cyc=cy[ev].tolist(), y=y[ev].tolist(),
                                                            yhat=pred[ev].tolist(), sd=None,
                                                            ref_cyc=cy[is_ref].tolist(),
                                                            base_yhat=mu0[ev].tolist())
            log(f"  P-B+ {src} M={Mref} test={c} {name}: RMSE {r['RMSE_pct']:.3f} % R² {r['R2']:.4f}")
    return recs, preds, sel_log


# ── ox P-B：沿用 SEARCH_SOH 檢查點 ────────────────────────────────────────────
def reuse_available(src):
    """搜尋階段是否留有可沿用之 P-B 檢查點（v9.11 之搜尋以 P-B+ 為單元，檔名不同 → 通常不存在）。"""
    sel = C.soh_sel(src); safe = sel["config_id"].replace("|", "__")
    return all(os.path.exists(os.path.join(SEARCH_CKPT, src, f"{safe}__P-B__{name}.json"))
               for name in models_of(src))


def reuse_PB(src, qrefs, log):
    """由 results_v911/search/_ckpt/<src>/<config>__P-B__<model>.json 還原 P-B 記錄（登記之計算成本例外）。"""
    sel = C.soh_sel(src); safe = sel["config_id"].replace("|", "__")
    recs, sel_log, used = [], [], []
    for name in models_of(src):
        p = os.path.join(SEARCH_CKPT, src, f"{safe}__P-B__{name}.json")
        d = json.load(open(p, encoding="utf-8"))
        for row in d["rows"]:
            c = row["cell"]; qref = qrefs[c]
            _BASE_PRED[(src, c, name)] = np.asarray(mu, float)
            r = dict(source=src, protocol="P-B", cell=c, model=name, cyc_axis=sel["cyc_axis"],
                     n_train=int(row["n_train"]), n_train_cells=None, feats=row["feats"], reused=1,
                     n_test=int(row["n_test"]), RMSE_pct=float(row["RMSE_pct"]), MAE_pct=float(row["MAE_pct"]),
                     R2=float(row["R2"]),
                     RMSE_Ah=float(row["RMSE_pct"]) / 100.0 * qref, MAE_Ah=float(row["MAE_pct"]) / 100.0 * qref,
                     MaxRE_pct=None, MAPE_pct=None, CI95_cov=None, CI95_halfwidth_soh=None,
                     reuse_src=os.path.relpath(p, os.path.dirname(SEARCH_CKPT)))
            r["pass_T_SOH"] = judge(r); recs.append(r)
        used.append(dict(source=src, protocol="P-B", model=name, file=os.path.relpath(p, os.path.dirname(SEARCH_CKPT)),
                         n_rows=len(d["rows"]), search_sec=float(d.get("sec", 0.0))))
        log(f"  P-B {src} {name}: 沿用搜尋檢查點（{len(d['rows'])} 顆，搜尋耗時 {d.get('sec', 0):.0f}s）")
    # 篩選集合（與搜尋相同規則：訓練顆）— 由 feats 欄回報
    for c in sorted({r["cell"] for r in recs}):
        fs = [r["feats"] for r in recs if r["cell"] == c][0]
        sel_log.append(dict(source=src, protocol="P-B", test_cell=c,
                            train_cells=[x for x in sorted({r['cell'] for r in recs}) if x != c],
                            selected=fs.split(","), reused=1))
    return recs, sel_log, used


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main(RES, log=print, search_ckpt=None):
    global SEARCH_CKPT
    SEARCH_CKPT = search_ckpt or os.path.join(RES, "search", "_ckpt")
    CK = os.path.join(RES, "_ckpt_soh"); os.makedirs(CK, exist_ok=True)
    meta = json.load(open(os.path.join(RES, "frames_meta_v911.json"), encoding="utf-8"))
    qrefs = {k: float(v) for k, v in meta["q_ref_Ah"].items()}
    all_recs, all_preds, all_sel, reuse_log = [], {}, [], []
    for src in C.DATA["sources_soh"]:
        frames0 = SD.frames_for(RES, src)
        dsel = C.soh_sel(src)
        dkey = dsel.get("delta_key")
        if dsel.get("source_file") is None:
            raise RuntimeError(f"SOH_SELECTED_V911[{src}] 尚未由 search_soh_v911 之結果登記（source_file 為 None），不得跑正式階段")
        frames = S.apply_delta(frames0, dkey)              # A1：依選定 δ 過濾（訓練與評估皆剔除）
        n_drop = sum(len(frames0[c]) - len(frames[c]) for c in frames0)
        gate = dsel.get("label_gate", "on")
        log(f"  SOH {src}: {len(frames)} 顆，{len(frames[sorted(frames)[0]])} 列/顆（首顆）；"
            f"A2 label_gate={gate}（循環軸 {'cyc_c' if gate == 'on' else 'cyc_A'}）；"
            f"A1 δ={dkey} 剔除 {n_drop} 列；主模型 {dsel.get('main_model')}"
            f"（週期核 {'on' if dsel['periodic'] else 'off'}／ARD {'on' if dsel['ard'] else 'off'}／sub {dsel['sub']}）；"
            f"組態 {dsel['config_id']}")
        jobs = [("P-B", run_PB), ("P-C", run_PC), ("P-A", run_PA)]
        jobs += [(f"P-B+|M={m}", (lambda m: (lambda fr, s, q, lg: run_PBplus(fr, s, q, m, lg)))(m)) for m in M_LIST]
        for proto, fn in jobs:
            ckp = os.path.join(CK, f"{src}__{proto.replace('|', '_').replace('=', '')}.json")
            if os.path.exists(ckp):
                d = json.load(open(ckp, encoding="utf-8"))
                for r in d["recs"]:                        # 檢查點寫於落欄之前 → 讀回時補上（只補中繼欄，不改任何數字）
                    if r.get("delta_V") is None:
                        r["delta_V"] = dkey
                    if r.get("label_gate") is None:
                        r["label_gate"] = gate
                all_recs += d["recs"]; all_preds.update(d["preds"]); all_sel += d["sel"]; reuse_log += d.get("reuse", [])
                log(f"  SOH {src} {proto}（checkpoint，{len(d['recs'])} 列）"); continue
            t0 = time.time()
            if src == "ox" and proto == "P-B" and reuse_available(src):
                recs, sl, used = reuse_PB(src, qrefs, log); preds = {}; reuse_log += used   # 登記之計算成本例外
            elif src == "ox" and proto == "P-B":
                log("  P-B ox：搜尋階段無對應之 P-B 檢查點（v9.11 搜尋單元為 P-B+），全部正式重算（reused=0）")
                recs, preds, sl = fn(frames, src, qrefs, log); used = []
            else:
                recs, preds, sl = fn(frames, src, qrefs, log); used = []
            json.dump(dict(recs=recs, preds=preds, sel=sl, reuse=used), open(ckp, "w", encoding="utf-8"),
                      ensure_ascii=False, default=lambda o: (o.item() if hasattr(o, "item") else str(o)))
            for r in recs:                                 # A1／A2：全協定共用同一 δ 與 label_gate（§3）
                if r.get("delta_V") is None:
                    r["delta_V"] = dkey
                if r.get("label_gate") is None:
                    r["label_gate"] = gate
            all_recs += recs; all_preds.update(preds); all_sel += sl
            log(f"  SOH {src} {proto} 完成（{len(recs)} 列，{time.time()-t0:.0f}s）")
    df = pd.DataFrame(all_recs)
    cols = ["source", "protocol", "cell", "s0", "model", "cyc_axis", "label_gate", "delta_V", "M",
            "window", "filter", "norm", "family", "top_k", "periodic", "ard", "sub", "config_id",
            "n_train", "n_train_cells",
            "n_test", "n_ref", "n_eval",
            "RMSE_pct", "MAE_pct", "R2", "RMSE_Ah", "MAE_Ah", "MaxRE_pct", "MAPE_pct",
            "CI95_cov", "CI95_halfwidth_soh", "pass_T_SOH", "reused", "reuse_src", "feats", "note"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].sort_values(["source", "protocol", "cell", "s0", "model"], na_position="first").reset_index(drop=True)
    df.to_csv(os.path.join(RES, "soh_records_v911.csv"), index=False)
    with open(os.path.join(RES, "soh_predictions_v911.json"), "w", encoding="utf-8") as f:
        json.dump(all_preds, f, ensure_ascii=False)
    with open(os.path.join(RES, "soh_selection_v911.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rule=C.RUL_STANDARD["no_future_info"], selections=all_sel), f, ensure_ascii=False)
    with open(os.path.join(RES, "soh_reuse_v911.json"), "w", encoding="utf-8") as f:
        json.dump(dict(policy=C.EXEC["reuse_policy"], reused_units=reuse_log), f, ensure_ascii=False, indent=1)
    log(f"  SOH 完成：{len(df)} 列（沿用單元 {len(reuse_log)}）")
    return df
