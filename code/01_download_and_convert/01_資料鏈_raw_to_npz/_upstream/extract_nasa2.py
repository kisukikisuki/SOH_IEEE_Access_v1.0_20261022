#!/usr/bin/env python3
"""
S1-S3 串流擷取器（device 端，僅需 numpy + pandas）
單一 pass / 位元組偏移可續跑。每顆電芯輸出一個 .npz。
UCS v1: 2,3,4,5,5b,5c,7,8,9,14  |  輸出 grid 上之 Q(V) + 放電容量段
用法: extract.py <csv> <outdir> <budget_s> [dv_mV]
"""
import io, json, os, sys, time
import numpy as np, pandas as pd

COLS = ["source","cell_id","cycle","step","regime","time_s","test_time_s",
        "voltage_V","current_A","capacity_Ah","temperature_C"]
USE  = ["regime","time_s","voltage_V","current_A","capacity_Ah","temperature_C"]
UIDX = [COLS.index(c) for c in USE]
V_MIN,V_MAX,I_ABS_MAX,T_MIN,T_MAX = 2.00,4.30,6.0,-20.0,80.0
GRID_LO,GRID_HI = 3.70,4.25
HI_LO,HI_HI = 3.95,4.05
BLOCK = 96*1024*1024

def grid_of(dv): 
    n=int(round((GRID_HI-GRID_LO)/(dv/1000.)))+1
    return GRID_LO+(dv/1000.)*np.arange(n)

def resample(v,q,g):
    o=np.argsort(v); v,q=v[o],q[o]
    k=np.concatenate([[True],np.diff(v)>0]); v,q=v[k],q[k]
    if v.size<4: return np.full(g.shape,np.nan),np.zeros(g.size,int)
    out=np.interp(g,v,q,left=np.nan,right=np.nan)
    out[(g<v[0])|(g>v[-1])]=np.nan
    dv=g[1]-g[0]; e=np.concatenate([g-dv/2,[g[-1]+dv/2]])
    return out,np.histogram(v,bins=e)[0]

def monotonize(v,q,st):
    q=q-q[0]; keep=np.ones(v.size,bool); qm=-np.inf; vm=-np.inf
    for i in range(v.size):
        if q[i]<qm-1e-12: keep[i]=False; st["ucs8_nonincreasing_Q"]=st.get("ucs8_nonincreasing_Q",0)+1; continue
        if v[i]<vm-1e-12: keep[i]=False; st["s2_nonincreasing_V"]=st.get("s2_nonincreasing_V",0)+1; continue
        qm=max(qm,q[i]); vm=max(vm,v[i])
    return v[keep],q[keep]

def main():
    csv,outdir,budget = sys.argv[1],sys.argv[2],float(sys.argv[3])
    dv = float(sys.argv[4]) if len(sys.argv)>4 else 5.0
    cell=os.path.basename(csv)[:-4]
    os.makedirs(outdir,exist_ok=True)
    sf=os.path.join(outdir,cell+".state.npz"); of=os.path.join(outdir,cell+".npz")
    if os.path.exists(of): print(json.dumps({"cell":cell,"status":"done_already"})); return
    g=grid_of(dv); size=os.path.getsize(csv)

    if os.path.exists(sf):
        S=np.load(sf,allow_pickle=True); st=json.loads(str(S["st"]))
        off=int(S["off"]); segc=int(S["segc"]); lastr=str(S["lastr"])
        dsegs=S["dsegs"].tolist(); Qs=list(S["Qs"]); sups=list(S["sups"])
        emeta=S["emeta"].tolist()
        cv,ci,cq,ct=list(S["cv"]),list(S["ci"]),list(S["cq"]),list(S["ct"])
        dacc=S["dacc"].tolist(); imin_nz=float(S["imin_nz"]); imax_rest=float(S["imax_rest"])
        ihist=S["ihist"]
    else:
        st={}; off=0; segc=0; lastr=""; dsegs=[]; Qs=[]; sups=[]; emeta=[]
        cv=[];ci=[];cq=[];ct=[]; dacc=[]; imin_nz=np.inf; imax_rest=0.0; ihist=np.zeros(400,np.int64)

    QPROV=3.0 if "oxford" in csv or "G" in cell else 2.0   # 暫定門檻，收尾時證明分類不變
    THR=0.02*QPROV
    t0=time.time()

    def flush_charge():
        nonlocal cv,ci,cq,ct
        if len(cv)>=10:
            V=np.array(cv);I=np.array(ci);Q=np.array(cq);T=np.array(ct)
            imx=np.nanmax(I)
            m=I>0.9*imx if imx>0 else np.zeros(I.size,bool)   # UCS-9 CC 段
            if m.sum()>=10:
                v2,q2=monotonize(V[m],Q[m],st)
                if v2.size>=10 and v2[0]<=HI_LO and v2[-1]>=HI_HI:
                    r,s=resample(v2,q2,g); Qs.append(r); sups.append(s)
                    emeta.append([segc,float(np.nanmean(I[m])),float(np.nanmean(T)),float(imx),float(v2[0]),float(v2[-1])])
                else: st["s2_coverage_insufficient"]=st.get("s2_coverage_insufficient",0)+1
            else: st["s2_cc_too_short"]=st.get("s2_cc_too_short",0)+1
            st["charge_events_total"]=st.get("charge_events_total",0)+1
        cv=[];ci=[];cq=[];ct=[]

    def flush_discharge():
        nonlocal dacc
        if dacc and dacc[0]>=50:                              # UCS-5c n>=50
            n,qmn,qmx,isum,tsum,vst,vmn=dacc
            dsegs.append([segc,qmx-qmn,n,isum/n,tsum/n,vst,vmn])
        dacc=[]

    fh=open(csv,"rb"); fh.seek(off)
    if off==0: fh.readline()                                   # header
    eof=False
    while time.time()-t0 < budget:
        start=fh.tell()
        buf=fh.read(BLOCK)
        if not buf: eof=True; break
        if len(buf)==BLOCK:
            nl=buf.rfind(b"\n")
            if nl>=0: buf=buf[:nl+1]
        fh.seek(start+len(buf))
        try:
            df=pd.read_csv(io.BytesIO(buf),header=None,names=COLS,usecols=UIDX,
                           on_bad_lines="skip",engine="c")
        except Exception:
            off=fh.tell(); continue
        off=fh.tell()
        st["rows_in"]=st.get("rows_in",0)+len(df)
        V=df["voltage_V"].to_numpy(float); I=df["current_A"].to_numpy(float)
        Q=df["capacity_Ah"].to_numpy(float); T=df["temperature_C"].to_numpy(float)
        ok=np.isfinite(V)&np.isfinite(I)&np.isfinite(Q)
        st["ucs3_nan_rows"]=st.get("ucs3_nan_rows",0)+int((~ok).sum())
        ok&=(V>=V_MIN)&(V<=V_MAX)&(np.abs(I)<=I_ABS_MAX)&(np.nan_to_num(T,nan=25)>=T_MIN)&(np.nan_to_num(T,nan=25)<=T_MAX)
        st["ucs2_out"]=st.get("ucs2_out",0)+int(len(df)-ok.sum()-st.get("_x",0)*0)
        V,I,Q,T=V[ok],I[ok],Q[ok],T[ok]
        if V.size==0: continue
        st["rows_kept"]=st.get("rows_kept",0)+V.size
        a=np.abs(I); nz=a[a>=THR]
        if nz.size: imin_nz=min(imin_nz,float(nz.min()))
        rs=a[a<THR]
        if rs.size: imax_rest=max(imax_rest,float(rs.max()))
        ihist+=np.histogram(a,bins=400,range=(0.0,0.2))[0]     # |I| 0-0.2 A, 0.5 mA/bin
        reg=np.where(a<THR,"rest",np.where(I>0,"charge","discharge"))
        chg=np.flatnonzero(np.concatenate([[True],reg[1:]!=reg[:-1]]))
        bnds=list(chg)+[reg.size]
        for bi in range(len(bnds)-1):
            s,e=bnds[bi],bnds[bi+1]; r=reg[s]
            newseg = not (bi==0 and r==lastr)
            if newseg:
                if lastr=="charge": flush_charge()
                elif lastr=="discharge": flush_discharge()
                segc+=1
            if r=="charge":
                cv.extend(V[s:e]); ci.extend(I[s:e]); cq.extend(Q[s:e]); ct.extend(T[s:e])
            elif r=="discharge":
                q=Q[s:e]
                if not dacc: dacc=[0,float(np.min(q)),float(np.max(q)),0.0,0.0,float(V[s]),float(np.min(V[s:e]))]
                dacc[0]+=int(e-s); dacc[1]=min(dacc[1],float(np.min(q))); dacc[2]=max(dacc[2],float(np.max(q)))
                dacc[3]+=float(np.sum(np.abs(I[s:e]))); dacc[4]+=float(np.nansum(T[s:e]))
                dacc[6]=min(dacc[6],float(np.min(V[s:e])))
            lastr=r
    fh.close()

    if not eof:
        np.savez_compressed(sf,off=off,segc=segc,lastr=lastr,st=json.dumps(st),
            dsegs=np.array(dsegs,float) if dsegs else np.zeros((0,7)),
            Qs=np.array(Qs) if Qs else np.zeros((0,g.size)),
            sups=np.array(sups) if sups else np.zeros((0,g.size),int),
            emeta=np.array(emeta,float) if emeta else np.zeros((0,6)),
            cv=np.array(cv),ci=np.array(ci),cq=np.array(cq),ct=np.array(ct),
            dacc=np.array(dacc,float) if dacc else np.zeros(0),
            imin_nz=imin_nz,imax_rest=imax_rest,ihist=ihist)
        print(json.dumps({"cell":cell,"status":"partial","pct":round(100*off/size,2),
                          "events":len(Qs),"dsegs":len(dsegs)})); return

    if lastr=="charge": flush_charge()
    elif lastr=="discharge": flush_discharge()
    D=np.array(dsegs,float)
    if D.size==0: print(json.dumps({"cell":cell,"status":"no_discharge"})); return
    D=D[D[:,1]>0.3*D[:,1].max()]                              # UCS-5c 容量下限
    if D.size==0: print(json.dumps({"cell":cell,"status":"no_valid_discharge"})); return
    # UCS-5c 強化（資料驅動完整性）：僅「起於眾數起始電壓、止於眾數截止電壓」之段
    # 方可作為容量標籤 —— 部分放電段會冒充低 SOH，必須排除。
    vs_ref=float(np.median(D[:,5])); vm_ref=float(np.median(D[:,6]))
    comp=(D[:,5]>=vs_ref-0.05)&(D[:,6]<=vm_ref+0.05)
    st["ucs5c_partial_discharge_excluded"]=int((~comp).sum())
    Dfull=D[comp] if comp.sum()>=3 else D
    # UCS-6：參考放電倍率由資料決定（各源不同：Oxford C/2、own 0.65C、NASA 1.09C）
    rate_ref=float(np.median(Dfull[:,3]))
    keep=np.abs(Dfull[:,3]-rate_ref)<=0.10*rate_ref
    Dc=Dfull[keep] if keep.sum()>=3 else Dfull
    qr=float(Dc[:,1].max())                                    # 觀測到的最大容量（禁用標稱）
    E=np.array(emeta,float) if emeta else np.zeros((0,4))
    soh=np.full(E.shape[0],np.nan)
    if Dc.shape[0]:
        for i in range(E.shape[0]):
            soh[i]=Dc[np.argmin(np.abs(Dc[:,0]-E[i,0])),1]/qr
    # 門檻不變性檢查：暫定門檻 THR 與真實門檻 0.02*Q_ref 之間的樣本數
    thr_true=0.02*qr; lo,hi=sorted((THR,thr_true))
    edges=np.linspace(0.0,0.2,401)
    band=int(ihist[(edges[:-1]>=lo)&(edges[:-1]<hi)].sum())
    np.savez_compressed(of,cell_id=cell,grid=g,dv_mV=dv,thr_true=thr_true,
        rate_ref=rate_ref,c_rate_ref=rate_ref/qr,vs_ref=vs_ref,vm_ref=vm_ref,
        thr_band_rows=band,ihist=ihist,
        Q=np.array(Qs) if Qs else np.zeros((0,g.size)),
        support=np.array(sups) if sups else np.zeros((0,g.size),int),
        soh=soh,emeta=E,q_ref=qr,dsegs=D,dsegs_c2=Dc,
        imin_nonrest=imin_nz,imax_rest=imax_rest,thr_prov=THR,
        st=json.dumps(st))
    if os.path.exists(sf): os.remove(sf)
    print(json.dumps({"cell":cell,"status":"done","q_ref":round(qr,4),"n_c2":int(Dc.shape[0]),
        "n_dseg":int(D.shape[0]),"events":len(Qs),
        "soh_min":round(float(np.nanmin(soh)),4) if np.isfinite(soh).any() else None,
        "c_rate":round(rate_ref/qr,3),"n_partial_excl":int(st.get("ucs5c_partial_discharge_excluded",0)),
        "thr_prov":round(THR,5),"thr_true":round(thr_true,5),
        "rows_between_thresholds":band}))

main()
