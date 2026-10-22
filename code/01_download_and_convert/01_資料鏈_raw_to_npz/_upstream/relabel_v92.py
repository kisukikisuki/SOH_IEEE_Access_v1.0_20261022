"""
S1-S3 收尾 v9.2（先驗登記 v6）：UCS-5c 兩側起始電壓帶 ＋ UCS-5d 參考放電序列（上包絡）辨識
＋ SOH 標籤改為相鄰參考放電之時間內插。可完全由 extract 產出的 .npz 快取重算。

v9.1 → v9.2 差異（資料層）：
  UCS-5c：起始電壓由「≥ 眾數 − 0.05 V」（單側）改為「眾數 − 0.05 ≤ V_start ≤ 眾數 + 0.15 V」（兩側）＋ 極少數（<4 段）起始電壓孤立群剔除。
          → 排除 NASA B0036 之 V_start 4.242／4.008 V 異常段（原使 Q_ref 虛高為 2.44 Ah）。
  UCS-5d：協定感知之參考放電辨識。Oxford 多數電芯之循環放電未充飽（CC 無 CV），容量較 RPT 低 12–34 %，
          與 RPT 呈雙序列。以「容量／前後各 5 段中位數」r_i 判定：r_i > 1.06 之段 ≥ 3 → 雙協定，僅 RPT 段可作
          標籤與 Q_ref；否則單協定，全部段皆參考但剔除孤立尖峰（r_i > 1.06）與殘缺段（r_i < 0.94）。
  標籤：  充電事件之 SOH ＝ 相鄰兩個參考放電容量之線性內插（依 segment 序）／ Q_ref；範圍外取最近者。
用法：HI_LO=3.60 HI_HI=4.00 CACHE=../cache_ox OUT=../dataset_v6 python3 relabel_v92.py
"""
import numpy as np, glob, os, json
import os as _os

HI_LO = float(_os.environ.get('HI_LO', 3.80)); HI_HI = float(_os.environ.get('HI_HI', 4.10))
MIN_SUPPORT = 3
OXFORD_CRATE_BAND = (0.40, 0.60)
VSTART_LO, VSTART_HI = 0.05, 0.15   # UCS-5c 兩側帶（−0.05／+0.15 V）
MIN_EVENTS = 10


def source_of(c):
    return 'oxford' if c.startswith('G') else ('nasa' if c.startswith('B0') else 'own')




NB_K, RPT_RATIO, PARTIAL_RATIO, MIN_RPT, DUAL_MIN_MED_RATIO = 5, 1.06, 0.94, 3, 1.08


def drop_vstart_anomaly_clusters(Dc, gap_V=0.015, min_cluster=4, depth=0):
    """UCS-5c 補充：起始電壓以最大間隙（> gap_V）切分後，少數群 < min_cluster 段者視為異常協定段剔除（遞迴）。"""
    n = Dc.shape[0]; log = []
    if n < 3 or depth > 4:
        return np.ones(n, bool), log
    o = np.argsort(Dc[:, 5]); vs = Dc[o, 5]; gaps = np.diff(vs)
    if gaps.size == 0 or gaps.max() <= gap_V:
        return np.ones(n, bool), log
    k = int(np.argmax(gaps)) + 1
    hiV = np.zeros(n, bool); hiV[o[k:]] = True
    nA, nB = int((~hiV).sum()), int(hiV.sum())
    if min(nA, nB) >= min_cluster:
        return np.ones(n, bool), log
    minor = hiV if nB < nA else ~hiV
    log.append(f"vstart_anomaly_dropped(n={int(minor.sum())},Vstart={np.round(Dc[minor,5],3).tolist()})")
    keep = ~minor
    sub, l2 = drop_vstart_anomaly_clusters(Dc[keep], gap_V, min_cluster, depth + 1)
    m = np.zeros(n, bool); m[np.flatnonzero(keep)[sub]] = True
    return m, log + l2


def local_ratio(Dc, k=NB_K):
    """每段容量相對『時間上前後各 k 段（不含自身）之中位數』之比值。"""
    o = np.argsort(Dc[:, 0]); cap = Dc[o, 1]; n = cap.size; r = np.ones(n)
    for i in range(n):
        nb = np.r_[cap[max(0, i - k):i], cap[i + 1:i + 1 + k]]
        if nb.size:
            r[i] = cap[i] / np.median(nb)
    out = np.ones(n); out[o] = r
    return out


def classify_reference(Dc):
    """
    UCS-5d：參考放電辨識（協定感知）。
      r_i = 容量／前後各 5 段中位數。
      雙協定（r_i > 1.06 之段 ≥ 3 且其中位比 ≥ 1.08）：該顆循環放電未充飽（容量低 10–34 %），僅 r_i > 1.06 之『參考性能測試 RPT』
        可作標籤與 Q_ref；循環放電段不得混用。
      單協定：全部段皆為參考，但孤立尖峰（r_i > 1.06）與局部殘缺段（r_i < 0.94）剔除。
    """
    r = local_ratio(Dc)
    cand = r > RPT_RATIO
    if cand.sum() >= MIN_RPT and np.median(r[cand]) >= DUAL_MIN_MED_RATIO:
        return cand, "dual_protocol", f"dual_protocol:RPT={int(cand.sum())}/{Dc.shape[0]},ratio_med={np.median(r[cand]):.3f}"
    ok = (r >= PARTIAL_RATIO) & (r <= RPT_RATIO)
    return ok, "single_protocol", f"single_protocol:spikes={int((r > RPT_RATIO).sum())},partials={int((r < PARTIAL_RATIO).sum())}"


def qref_and_label(d, source):
    D = d['dsegs']                      # [seg, cap, n, I_mean, T_mean, V_start, V_min]
    if D.shape[0] == 0:
        return None
    vs, vm = float(np.median(D[:, 5])), float(np.median(D[:, 6]))
    comp = (D[:, 5] >= vs - VSTART_LO) & (D[:, 5] <= vs + VSTART_HI) & (D[:, 6] <= vm + 0.05)   # UCS-5c 兩側
    n_partial = int((~comp).sum())
    Df = D[comp] if comp.sum() >= 3 else D
    flag = ""
    if source == 'oxford':                                                     # UCS-6 Oxford 鎖 C/2 帶
        qr = float(Df[:, 1].max())
        for _ in range(5):
            cr = Df[:, 3] / qr
            m = (cr >= OXFORD_CRATE_BAND[0]) & (cr <= OXFORD_CRATE_BAND[1])
            if m.sum() < 3:
                break
            q1 = float(Df[m, 1].max())
            if abs(q1 - qr) < 1e-9:
                break
            qr = q1
        cr = Df[:, 3] / qr
        m = (cr >= OXFORD_CRATE_BAND[0]) & (cr <= OXFORD_CRATE_BAND[1])
        if m.sum() < 3:
            rate = float(np.median(Df[:, 3])); m = np.abs(Df[:, 3] - rate) <= 0.10 * rate
            flag = "NO_C2_FALLBACK_TO_MODAL_RATE"
        Dc = Df[m]
    else:
        rate = float(np.median(Df[:, 3])); m = np.abs(Df[:, 3] - rate) <= 0.10 * rate
        Dc = Df[m] if m.sum() >= 3 else Df
    if Dc.shape[0] == 0:
        return None
    keep, log = drop_vstart_anomaly_clusters(Dc)                                # UCS-5c 補充
    Dc = Dc[keep]
    if Dc.shape[0] >= 3:
        ref, mode, msg = classify_reference(Dc)                                    # UCS-5d
    else:
        ref, mode, msg = np.ones(Dc.shape[0], bool), "single_protocol", "n<3"
    log.append(msg)
    Dr = Dc[ref]
    qr = float(Dr[:, 1].max())
    return dict(q_ref=qr, Dr=Dr[np.argsort(Dr[:, 0])], n_partial=n_partial, n_ref_seg=int(Dr.shape[0]),
                n_lower_series_excl=int(Dc.shape[0] - Dr.shape[0]), c_rate=float(np.median(Dr[:, 3])) / qr,
                flag=flag, ucs5d_mode=mode, ucs5d_log=" ; ".join(log))


def label_events(Dr, q_ref, seg_ids):
    """相鄰參考放電之線性內插（依 segment 序）；範圍外取最近者。"""
    x = Dr[:, 0]; y = Dr[:, 1] / q_ref
    return np.interp(np.asarray(seg_ids, float), x, y)


def build(cache=_os.environ.get('CACHE', 'cache'), out=_os.environ.get('OUT', 'dataset_v6'),
          group_mode=_os.environ.get('GROUP_MODE', 'cell')):
    """group_mode='cell'：逐顆溫度分組（Oxford／own）；'event'：逐事件分組（NASA，UCS-10 v4）。"""
    os.makedirs(out, exist_ok=True)
    audit, data = [], {}
    for f in sorted(glob.glob(os.path.join(cache, '*.npz'))):
        cell = os.path.basename(f)[:-4]; s = source_of(cell)
        d = np.load(f, allow_pickle=True); st = json.loads(str(d['st']))
        r = qref_and_label(d, s)
        Q, S, E, g = d['Q'], d['support'], d['emeta'], d['grid']
        hm = (g >= HI_LO) & (g <= HI_HI); n_ev_raw = int(Q.shape[0])
        row = dict(cell=cell, source=s, rows_in=st.get('rows_in', 0), rows_kept=st.get('rows_kept', 0),
                   ucs3_nan=st.get('ucs3_nan_rows', 0), ucs8_nonincQ=st.get('ucs8_nonincreasing_Q', 0),
                   s2_noninc_V=st.get('s2_nonincreasing_V', 0), ev_total=st.get('charge_events_total', 0),
                   ev_cc_short=st.get('s2_cc_too_short', 0), ev_coverage_fail=st.get('s2_coverage_insufficient', 0),
                   ev_pass_coverage=n_ev_raw, thr_band_rows=int(d['thr_band_rows']))
        if r is None or n_ev_raw == 0:
            row.update(status='EXCLUDED_no_capacity_or_events'); audit.append(row); continue
        sel = np.ones(n_ev_raw, bool)
        if s == 'oxford':
            sel &= E[:, 3] > 1.0                                  # UCS-16 僅 C/2 充電
        if E.shape[1] >= 6:                                        # NASA v9.1 之逐事件覆蓋欄
            sel &= (E[:, 4] <= HI_LO) & (E[:, 5] >= HI_HI)
        row['ev_c2_charge'] = int(sel.sum())
        with np.errstate(invalid='ignore'):
            med_sup = np.median(S[:, hm], axis=1)
        sel &= med_sup >= MIN_SUPPORT; row['ev_support_ok'] = int(sel.sum())
        sel &= np.isfinite(Q[:, hm]).all(axis=1); row['ev_grid_ok'] = int(sel.sum())
        Tev = E[:, 2]
        row['T_median'] = round(float(np.nanmedian(Tev)), 1) if np.isfinite(Tev).any() else None
        groups = {'main': (15.0, 35.0)} if group_mode == 'cell' else {'main': (15.0, 35.0), 'high_T': (35.0, 99.0), 'low_T': (-20.0, 15.0)}
        for gname, (lo, hi) in groups.items():
            gsel = sel & (Tev >= lo) & (Tev < hi) if group_mode == 'event' else sel.copy()
            if group_mode == 'cell':
                in_main = (Tev >= 15.0) & (Tev <= 35.0)
                tg = 'main' if (sel & in_main).sum() >= 10 else ('low_T' if row['T_median'] is not None and row['T_median'] < 15 else 'high_T' if row['T_median'] is not None and row['T_median'] > 35 else 'main')
                if tg == 'main':
                    gsel &= in_main
                gname = tg
            min_ev = MIN_EVENTS if group_mode == 'cell' else 25
            if gsel.sum() < min_ev:
                if group_mode == 'cell':
                    row.update(status='EXCLUDED_too_few_events', ev_final=int(gsel.sum())); audit.append(row)
                continue
            soh = label_events(r['Dr'], r['q_ref'], E[gsel, 0]); order = np.argsort(E[gsel, 0])
            key = f"{s}{'' if gname == 'main' else '-' + gname}__{cell}"
            data[key] = dict(source=s, cell_id=cell, grid=g, Q=Q[gsel][order], support=S[gsel][order], soh=soh[order],
                             seg=E[gsel, 0][order], I_max=E[gsel, 3][order], T=E[gsel, 2][order], q_ref=r['q_ref'],
                             temp_group='main', real_group=gname, cap_seg=r['Dr'][:, 0], cap_soh=r['Dr'][:, 1] / r['q_ref'])
            rr = dict(row); rr.update(status='OK', group=gname, ev_final=int(gsel.sum()), q_ref=round(r['q_ref'], 4),
                                      c_rate=round(r['c_rate'], 3), n_ref_seg=r['n_ref_seg'], n_partial_excl=r['n_partial'],
                                      n_lower_series_excl=r['n_lower_series_excl'], ucs5d_mode=r.get('ucs5d_mode',''), ucs5d_log=r.get('ucs5d_log',''), soh_min=round(float(soh.min()), 4),
                                      soh_max=round(float(soh.max()), 4), flag=r['flag'])
            audit.append(rr)
    for k, v in data.items():
        np.savez_compressed(os.path.join(out, f"{k}.npz"), **v)
    return audit, data


if __name__ == '__main__':
    import pandas as pd
    audit, data = build()
    A = pd.DataFrame(audit); A.to_csv('gate1_qc_audit_v92.csv', index=False)
    ok = A[A.status == 'OK']
    print(f"寫出 {len(data)} 顆／臂 | QC 稽核表 gate1_qc_audit_v92.csv: {len(A)} 列")
    for (s, gname), g in ok.groupby(['source', 'group']):
        print(f"{s}-{gname}: {len(g)} 顆 | 事件 {int(g.ev_final.sum()):,} | Q_ref {g.q_ref.min():.3f}–{g.q_ref.max():.3f} | "
              f"SOH 最低 {g.soh_min.min():.4f} | 下序列排除段數合計 {int(g.n_lower_series_excl.sum())} | 兩側起始電壓排除合計 {int(g.n_partial_excl.sum())}")
        print(g[['cell', 'ev_final', 'q_ref', 'c_rate', 'n_ref_seg', 'n_lower_series_excl', 'n_partial_excl', 'soh_min', 'soh_max']].to_string(index=False))
