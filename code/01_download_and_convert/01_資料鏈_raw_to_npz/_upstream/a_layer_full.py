#!/usr/bin/env python3
"""
S1-A 全量 A 層（先驗登記 v9 §a_layer A1–A7）—— 串流讀原始 CSV，重建 regime／segment（同 extract_*2.py 之 segc），
每段統計 Ah_int=∫|I|dt、Ah_col=Q 欄極差、V_start/V_end/V_min、I_mean、T_mean、cycle_native；
放電段建立 cyc_A、Ah_cum、EFC；B 層對照（dataset_v8 之 cap_seg／cap_soh／seg）以旗標寫回，不刪列。
可續跑：預算內未完成則存 .state.npz；下次呼叫接續。
用法：a_layer_full.py <csv> <outdir> <budget_s>
"""
import io, json, os, sys, time, glob
import numpy as np, pandas as pd

COLS = ["source","cell_id","cycle","step","regime","time_s","test_time_s","voltage_V","current_A","capacity_Ah","temperature_C"]
USE  = ["cycle","time_s","voltage_V","current_A","capacity_Ah","temperature_C"]
UIDX = [COLS.index(c) for c in USE]
V_MIN,V_MAX,I_ABS_MAX,T_MIN,T_MAX = 2.00,4.30,6.0,-20.0,80.0
BLOCK = 64*1024*1024
VSTART_LO, VSTART_HI = 0.05, 0.15
HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(HERE, "..", "dataset_v8")

def source_of(c): return 'oxford' if c.startswith('G') else ('nasa' if c.startswith('B0') else 'own')

def find_dataset(cell):
    fs = sorted(glob.glob(os.path.join(DS, f"*__{cell}.npz")))
    return fs

def main():
    csv, outdir, budget = sys.argv[1], sys.argv[2], float(sys.argv[3])
    cell = os.path.basename(csv)[:-4]; src = source_of(cell)
    os.makedirs(outdir, exist_ok=True)
    sf = os.path.join(outdir, cell + ".state.npz"); of = os.path.join(outdir, cell + "_A_layer_segments.csv")
    if os.path.exists(of): print(json.dumps({"cell":cell,"status":"done_already"})); return
    size = os.path.getsize(csv); QPROV = 3.0 if src=='oxford' else 2.0; THR = 0.02*QPROV
    if os.path.exists(sf):
        S = np.load(sf, allow_pickle=True); st = json.loads(str(S["st"])); off = int(S["off"]); segc = int(S["segc"]); lastr = str(S["lastr"])
        rows = json.loads(str(S["rows"])); cur = json.loads(str(S["cur"])); ct, ci = list(S["ct"]), list(S["ci"])
    else:
        st = {}; off = 0; segc = 0; lastr = ""; rows = []; cur = None; ct = []; ci = []
    t0 = time.time()

    def finalize():
        nonlocal cur, ct, ci
        if cur is None: return
        t = np.asarray(ct, float); a = np.asarray(ci, float)
        if t.size >= 2:
            dt = np.diff(t); bad = (dt <= 0) | (dt > 3600)
            if bad.any():
                med = float(np.median(dt[~bad])) if (~bad).any() else 1.0
                dt = np.where(bad, med, dt); cur["n_dt_fixed"] = int(bad.sum())
            cur["dur_s"] = float(dt.sum()); cur["Ah_int"] = float(np.sum((a[1:]+a[:-1])/2*dt)/3600.0)
        else:
            cur["dur_s"] = 0.0; cur["Ah_int"] = 0.0
        cur["Ah_col"] = cur.pop("qmax") - cur.pop("qmin")
        cur["I_mean"] = cur.pop("isum")/cur["n"]; tn = cur.pop("tn"); cur["T_mean"] = cur.pop("tsum")/tn if tn else float("nan")
        cyc = cur.pop("cyc"); cur["cycle_native"] = int(max(cyc, key=cyc.get)); cur["n_native_cycles_in_seg"] = len(cyc)
        rows.append(cur); cur = None; ct = []; ci = []

    fh = open(csv, "rb"); fh.seek(off)
    if off == 0: fh.readline()
    eof = False
    while time.time()-t0 < budget:
        start = fh.tell(); buf = fh.read(BLOCK)
        if not buf: eof = True; break
        if len(buf) == BLOCK:
            nl = buf.rfind(b"\n")
            if nl >= 0: buf = buf[:nl+1]
        fh.seek(start+len(buf))
        try:
            df = pd.read_csv(io.BytesIO(buf), header=None, names=COLS, usecols=UIDX, on_bad_lines="skip", engine="c")
        except Exception:
            off = fh.tell(); continue
        off = fh.tell(); st["rows_in"] = st.get("rows_in",0)+len(df)
        V = df["voltage_V"].to_numpy(float); I = df["current_A"].to_numpy(float); Q = df["capacity_Ah"].to_numpy(float)
        T = df["temperature_C"].to_numpy(float); tt = df["time_s"].to_numpy(float); cy = pd.to_numeric(df["cycle"], errors="coerce").fillna(-1).to_numpy(int)
        ok = np.isfinite(V)&np.isfinite(I)&np.isfinite(Q)
        ok &= (V>=V_MIN)&(V<=V_MAX)&(np.abs(I)<=I_ABS_MAX)&(np.nan_to_num(T,nan=25)>=T_MIN)&(np.nan_to_num(T,nan=25)<=T_MAX)
        V,I,Q,T,tt,cy = V[ok],I[ok],Q[ok],T[ok],tt[ok],cy[ok]
        if V.size == 0: continue
        st["rows_kept"] = st.get("rows_kept",0)+V.size
        a = np.abs(I); reg = np.where(a<THR,"rest",np.where(I>0,"charge","discharge"))
        chg = np.flatnonzero(np.concatenate([[True], reg[1:]!=reg[:-1]])); bnds = list(chg)+[reg.size]
        for bi in range(len(bnds)-1):
            s,e = bnds[bi],bnds[bi+1]; r = reg[s]
            newseg = not (bi==0 and r==lastr)
            if newseg:
                finalize(); segc += 1
                cur = dict(seg=segc, regime=r, n=0, qmin=float(np.min(Q[s:e])), qmax=float(np.max(Q[s:e])), isum=0.0, tsum=0.0, tn=0,
                           V_start=float(V[s]), V_end=float(V[e-1]), V_min=float(np.min(V[s:e])), V_max=float(np.max(V[s:e])), cyc={}, n_dt_fixed=0)
            cur["n"] += int(e-s); cur["qmin"] = min(cur["qmin"], float(np.min(Q[s:e]))); cur["qmax"] = max(cur["qmax"], float(np.max(Q[s:e])))
            cur["isum"] += float(np.sum(a[s:e])); tf = T[s:e]; tf = tf[np.isfinite(tf)]; cur["tsum"] += float(tf.sum()); cur["tn"] += int(tf.size)
            cur["V_end"] = float(V[e-1]); cur["V_min"] = min(cur["V_min"], float(np.min(V[s:e]))); cur["V_max"] = max(cur["V_max"], float(np.max(V[s:e])))
            u, c = np.unique(cy[s:e], return_counts=True)
            for uu, cc in zip(u, c): cur["cyc"][str(int(uu))] = cur["cyc"].get(str(int(uu)),0)+int(cc)
            ct.extend(tt[s:e].tolist()); ci.extend(a[s:e].tolist())
            lastr = r
    fh.close()
    if not eof:
        np.savez_compressed(sf, off=off, segc=segc, lastr=lastr, st=json.dumps(st), rows=json.dumps(rows), cur=json.dumps(cur), ct=np.array(ct), ci=np.array(ci))
        print(json.dumps({"cell":cell,"status":"partial","pct":round(100*off/size,2),"segs":len(rows)})); return
    finalize()
    A = pd.DataFrame(rows)
    A = A[["seg","regime","n","dur_s","n_dt_fixed","cycle_native","n_native_cycles_in_seg","Ah_int","Ah_col","V_start","V_end","V_min","V_max","I_mean","T_mean"]]
    # ---------------- 放電段 A 層軸 ----------------
    dis = A[A.regime=="discharge"].copy().reset_index(drop=True)
    dis["b_ucs5c_n50"] = dis.n >= 50                                               # UCS-5c n≥50（extract 之 dsegs 條件）
    capmax = dis.loc[dis.b_ucs5c_n50, "Ah_col"].max() if dis.b_ucs5c_n50.any() else dis.Ah_col.max()
    dis["a_flag_shallow"] = dis.Ah_int < 0.3*dis.Ah_int.max()
    dis["cyc_A"] = np.cumsum(~dis.a_flag_shallow)
    dis["Ah_cum"] = dis.Ah_int.cumsum(); dis["Ah_cum_shallow"] = (dis.Ah_int*dis.a_flag_shallow).cumsum()
    dis["b_ucs5c_pass"] = dis.b_ucs5c_n50 & (dis.Ah_col > 0.3*capmax)
    # ---------------- B 層對照（dataset_v8） ----------------
    fs = find_dataset(cell); summ = dict(cell=cell, source=src, size_GB=round(size/1e9,3), rows_in=st.get("rows_in",0), rows_kept=st.get("rows_kept",0),
        n_seg_total=int(len(A)), n_discharge_seg=int(len(dis)), n_charge_seg=int((A.regime=="charge").sum()), n_shallow=int(dis.a_flag_shallow.sum()),
        n_cycles_A=int(dis.cyc_A.max()) if len(dis) else 0, shallow_Ah_share=round(float(dis.Ah_cum_shallow.iloc[-1]/dis.Ah_cum.iloc[-1]),4) if len(dis) else None,
        n_ucs5c_pass=int(dis.b_ucs5c_pass.sum()), n_native_cycles_discharge=int(dis.cycle_native.nunique()),
        discharge_seg_one_native_cycle=int((dis.n_native_cycles_in_seg==1).sum()), native_cycle_reused=int(dis.cycle_native.duplicated().sum()),
        n_dt_fixed_rows=int(A.n_dt_fixed.sum()), Ah_total=round(float(dis.Ah_int.sum()),3), in_dataset_v8=bool(fs), arms=";".join(os.path.basename(f).split("__")[0] for f in fs))
    full = dis[dis.b_ucs5c_pass]
    if len(full):
        rel = ((full.Ah_int-full.Ah_col)/full.Ah_col).abs()
        summ.update(Ah_int_vs_col_relmax=round(float(rel.max()),5), Ah_int_vs_col_relmed=round(float(rel.median()),5))
    if fs:
        d = np.load(fs[0], allow_pickle=True); q_ref = float(d["q_ref"]); cap_seg = d["cap_seg"].astype(int); cap_soh = d["cap_soh"]; ev_seg = d["seg"].astype(int); soh_B = d["soh"]
        dis["EFC"] = dis.Ah_cum/q_ref
        dis["b_label_ref"] = dis.seg.isin(cap_seg)
        vs_med = float(np.median(dis.loc[dis.b_ucs5c_pass,"V_start"])) if dis.b_ucs5c_pass.any() else float(np.median(dis.V_start))
        dis["a_flag_vstart"] = ~((dis.V_start>=vs_med-VSTART_LO)&(dis.V_start<=vs_med+VSTART_HI))
        B = pd.DataFrame(dict(seg=cap_seg, cap_B=cap_soh*q_ref)); Mg = dis.merge(B, on="seg", how="inner")
        dAB = ((Mg.Ah_int-Mg.cap_B)/Mg.cap_B).abs()
        ev = pd.DataFrame(dict(seg=ev_seg)); ev["event_idx"] = np.arange(len(ev))
        idx = np.searchsorted(dis.seg.to_numpy(), ev.seg.to_numpy(), side="left")-1; idx = np.clip(idx, 0, len(dis)-1)
        ev["prev_dis_seg"] = dis.seg.to_numpy()[idx]; ev["cyc_A"] = dis.cyc_A.to_numpy()[idx]; ev["EFC"] = dis.EFC.to_numpy()[idx]; ev["Ah_cum"] = dis.Ah_cum.to_numpy()[idx]; ev["soh_B"] = soh_B
        eol = np.flatnonzero(ev.soh_B.to_numpy() <= 0.80)
        drift = (ev.cyc_A-ev.event_idx)-(ev.cyc_A.iloc[0]-ev.event_idx.iloc[0])
        ev.to_csv(os.path.join(outdir, f"{cell}_event_axis_map.csv"), index=False)
        summ.update(q_ref_B=round(q_ref,4), EFC_total=round(float(dis.EFC.iloc[-1]),2), n_ref_label_B=int(len(cap_seg)), n_matched_ref=int(len(Mg)),
            Ah_A_vs_capB_relmax=round(float(dAB.max()),6) if len(Mg) else None, n_vstart_flag=int((dis.a_flag_vstart&dis.b_ucs5c_pass).sum()),
            n_events_S9=int(len(ev)), event_axis_drift_max=int(np.abs(drift).max()), soh_min_B=round(float(soh_B.min()),4),
            eol_event_idx=int(eol[0]) if eol.size else None, eol_cyc_A=int(ev.cyc_A.iloc[eol[0]]) if eol.size else None, eol_EFC=round(float(ev.EFC.iloc[eol[0]]),2) if eol.size else None)
    else:
        q_ref = float(capmax) if np.isfinite(capmax) else float("nan"); dis["EFC"] = dis.Ah_cum/q_ref; dis["b_label_ref"] = False; dis["a_flag_vstart"] = False
        summ.update(q_ref_B=None, EFC_total=round(float(dis.EFC.iloc[-1]),2) if len(dis) else None, note="not_in_dataset_v8:EFC_uses_max_ucs5c_capacity")
    A.to_csv(of, index=False); dis.to_csv(os.path.join(outdir, f"{cell}_A_layer_discharge.csv"), index=False)
    json.dump(summ, open(os.path.join(outdir, f"{cell}_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    if os.path.exists(sf): os.remove(sf)
    print(json.dumps({"cell":cell,"status":"done","segs":len(A),"dis":len(dis),"cyc_A":summ["n_cycles_A"],"EFC":summ.get("EFC_total"),"relAB":summ.get("Ah_A_vs_capB_relmax"),"t":round(time.time()-t0,1)}))

main()
