#!/usr/bin/env python3
"""s1_rebuild_chain_dedup.py — 審稿第 1 項／步驟書 §2.2 步驟 1：去重後重建 Oxford G1/G3/G5 9 顆之資料鏈（extract→relabel→v8→A 層→v9）。
複製自：5版/02_程式/01_資料鏈_raw_to_npz/rebuild_chain.py（全部階段邏輯、呼叫之上游程式、參數逐字沿用）。
只改：
  (1) L10 UP 改為原 `01_資料鏈_raw_to_npz\_upstream` 絕對路徑（本檔位於 05_審稿補跑，上游程式仍呼叫原檔、不複製改寫）；
  (2) L16–20 csv_of：Oxford 來源改為 <raw-root>\oxford\<--ox-subdir>（預設 oxford_dedup；回歸檢查用 --ox-subdir oxford＝原檔）；
  (3) L25 預設 cells 改為 Oxford G1/G3/G5 9 顆（其餘 25 顆不重建，直接沿用 R1 npz，步驟書 §2.2-2）；
  (4) v9 階段改呼叫 NEW\s1_make_dataset_v9_subset.py（make_dataset_v9.py 只改 L40 之 40 顆斷言為「＝本次 dataset_v8 主臂顆數」）。
Q_ref 規則沿 relabel_v92.py L120 `qr = Dr[:,1].max()` 不改。
用法：s1_rebuild_chain_dedup.py <stage> --raw-root RAW --work WORK [--procs N] [--ox-subdir oxford_dedup]
事後聲明：資料層重建；事後（post hoc）、未調整、不在 50 項 Holm 檢定族。
"""
import os, sys, json, glob, time, shutil, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor
HERE=os.path.dirname(os.path.abspath(__file__))
UP=os.path.join(os.path.dirname(HERE),'01_資料鏈_raw_to_npz','_upstream'); PY=sys.executable          # 只改 (1)
ap=argparse.ArgumentParser(); ap.add_argument('stage'); ap.add_argument('--raw-root',required=True); ap.add_argument('--work',required=True)
ap.add_argument('--procs',type=int,default=8); ap.add_argument('--cells',default=None); ap.add_argument('--ox-subdir',default='oxford_dedup'); a=ap.parse_args()
RAW=a.raw_root; W=a.work; os.makedirs(W,exist_ok=True)
NASA=['B0005','B0006','B0007','B0018','B0029','B0030','B0031','B0032','B0033','B0034','B0036','B0038','B0039','B0040','B0042','B0043','B0044']
OWN=['cell2','cell3']
OX=[f'G{g}__Cell{c}' for g,cs in ((1,(9,15,20)),(3,(10,11,14)),(5,(5,6,16))) for c in cs]                    # 只改 (3)
def csv_of(cell):
    if cell in NASA: return os.path.join(RAW,'nasa','nasa',cell+'.csv')
    if cell in OWN: return os.path.join(RAW,'own','own',cell+'.csv')
    return os.path.join(RAW,'oxford',a.ox_subdir,cell+'.csv')                                            # 只改 (2)
def kind(cell): return 'nasa' if cell in NASA else 'own' if cell in OWN else 'ox'
EXTRACT={'nasa':'extract_nasa2.py','own':'extract_own.py','ox':'extract_ox2.py'}
HI={'nasa':('3.90','4.10'),'own':('3.80','4.10'),'ox':('3.60','4.00')}
GM={'nasa':'event','own':'cell','ox':'cell'}
cells=a.cells.split(',') if a.cells else OX                                                             # 只改 (3)
P94=os.path.join(W,'pv94'); PRE=os.path.join(P94,'01_pretrain_programs'); DS8=os.path.join(P94,'dataset_v8'); AL=os.path.join(P94,'results_v94','a_layer')
def size(c): return os.path.getsize(csv_of(c))
def run(cmd,log,env=None,cwd=None):
    t=time.time(); e=dict(os.environ); e.update(env or {})
    r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf8',errors='replace',env=e,cwd=cwd)
    open(log,'a',encoding='utf8').write(r.stdout+r.stderr)
    return r,time.time()-t
def pool(jobs,fn):
    t0=time.time(); res={}
    with ThreadPoolExecutor(a.procs) as ex:
        futs={c:ex.submit(fn,c) for c in jobs}
        for c,f in futs.items():
            res[c]=f.result(); print(f'  done {c} {res[c]:.0f}s (elapsed {time.time()-t0:.0f}s)',flush=True)
    return res
stage=a.stage; timings={}
if stage=='extract':
    for k in EXTRACT: os.makedirs(os.path.join(W,'cache_'+k),exist_ok=True)
    def job(c):
        k=kind(c); r,t=run([PY,os.path.join(UP,EXTRACT[k]),csv_of(c),os.path.join(W,'cache_'+k),'1e9','5.0'],os.path.join(W,'extract_'+k+'.log'))
        if r.returncode: print('FAIL',c,r.stderr[-500:])
        return t
    order=sorted(cells,key=lambda c:-size(c)); timings=pool(order,job)
elif stage=='relabel':
    sys.path.insert(0,UP)
    for k in ('nasa','own','ox'):
        if not glob.glob(os.path.join(W,'cache_'+k,'*.npz')): continue
        code=f"import os,sys,json; sys.path.insert(0,{UP!r}); import relabel_v92 as RB, pandas as pd; audit,data=RB.build(cache={os.path.join(W,'cache_'+k)!r},out={os.path.join(W,'v6_'+k)!r},group_mode={GM[k]!r}); pd.DataFrame(audit).to_csv({os.path.join(W,'relabel_audit_'+k+'.csv')!r},index=False); print({k!r},len(data))"
        t=time.time(); r,_=run([PY,'-c',code],os.path.join(W,'relabel_'+k+'.log'),env={'HI_LO':HI[k][0],'HI_HI':HI[k][1]}); print(k,r.stdout.strip(),r.stderr[-300:],f'{time.time()-t:.1f}s')
elif stage=='v8':
    v6=os.path.join(W,'v6_all'); shutil.rmtree(v6,ignore_errors=True); os.makedirs(v6)
    for k in ('nasa','own','ox'):
        for f in glob.glob(os.path.join(W,'v6_'+k,'*.npz')): shutil.copy(f,v6)
    os.makedirs(PRE,exist_ok=True); os.makedirs(os.path.join(PRE,'prereg'),exist_ok=True)
    import filecmp
    def cp(src,dst):
        if not (os.path.exists(dst) and filecmp.cmp(src,dst,shallow=False)): shutil.copy(src,dst)
    for fn in ('make_dataset_v8.py','a_layer_full.py','a_layer_fix_v9a.py'): cp(os.path.join(UP,fn),os.path.join(PRE,fn))
    cp(os.path.join(UP,'prereg','arms_v93.json'),os.path.join(PRE,'prereg','arms_v93.json'))
    shutil.rmtree(DS8,ignore_errors=True)
    r,t=run([PY,os.path.join(PRE,'make_dataset_v8.py'),v6,DS8],os.path.join(W,'v8.log')); print(r.stdout,r.stderr[-300:])
elif stage=='alayer':
    os.makedirs(AL,exist_ok=True)
    def job(c):
        r,t=run([PY,os.path.join(PRE,'a_layer_full.py'),csv_of(c),AL,'1e9'],os.path.join(W,'alayer.log'))
        if r.returncode: print('FAIL',c,r.stderr[-500:])
        return t
    order=sorted(cells,key=lambda c:-size(c)); timings=pool(order,job)
    r,t=run([PY,os.path.join(PRE,'a_layer_fix_v9a.py')],os.path.join(W,'alayer_fix.log'),cwd=PRE); print('fix',r.returncode,f'{t:.1f}s')
elif stage=='v9':
    out=os.path.join(W,'dataset_v9'); shutil.rmtree(out,ignore_errors=True)
    d9=os.path.join(HERE,'s1_make_dataset_v9_subset.py')                                                # 只改 (4)
    r,t=run([PY,d9,P94,out],os.path.join(W,'v9.log')); print(r.stdout[-1500:],r.stderr[-800:])
json.dump(timings,open(os.path.join(W,f'timings_{stage}.json'),'w'),indent=1)
