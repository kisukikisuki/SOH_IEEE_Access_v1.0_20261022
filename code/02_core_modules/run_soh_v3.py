"""run_soh_v3.py — SOH 第3版重跑驅動（只換資料夾路徑與平行排程；模型／協定／特徵邏輯一律沿用 1版 pv911_run_SOH，prereg v17 凍結）。

階段：ENV → PREREG17（沿用凍結檔，不重寫）→ DATA → SOH → (EXPORT／VERIFY 見 export_soh_v3.py／verify_soh_v3.py)
平行化：以「來源 × 測試電芯」為單位（LOO 家族：P-B＋P-B+ M=10/20 同一程序，保留 P-B→P-B+ 之預訓練共用；P-C；P-A），
每單元獨立程序（單執行緒 BLAS）；P-C 之隨機排列依電芯排序串流全抽（見 patch_programs.py），故與全序列版逐位相同。
用法：python run_soh_v3.py --dataset <npz資料夾> --results <輸出> [--workers 18] [--stages ENV,PREREG17,DATA,SOH]
      python run_soh_v3.py --dataset ... --results ... --serial-check own   （以未修補之全序列路徑重算該來源並與平行結果逐值比對）
"""
import os, sys, json, time, argparse, hashlib, platform, datetime, warnings, shutil
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = os.environ.get("V3_BLAS_THREADS", "1")   # 預設單執行緒；R0 變體測試可覆寫
HERE = os.path.dirname(os.path.abspath(__file__))
PRG = os.path.join(HERE, "pv911_run_SOH"); PRO = os.path.join(PRG, "02_programs")
sys.path.insert(0, PRO)
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np, pandas as pd

_ST = {}


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dump(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=lambda o: (o.item() if hasattr(o, "item") else str(o)))


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def setup(dataset):
    """載入 1版 程式；僅替換資料集資料夾位置，並停用 RUL 用之老師電芯容量表（SOH 篇不使用）。"""
    if _ST.get("ds") == dataset:
        return _ST
    import data_v911 as D
    D.DATASET_V9 = dataset
    D.load_own_capacity_cells = lambda: {}
    try:
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter("ignore", ConvergenceWarning)
    except Exception:
        pass
    import config_v911 as C, soh_v911 as S, stage_data_v911 as SD, stage_soh_v911 as SS
    _ST.update(ds=dataset, D=D, C=C, S=S, SD=SD, SS=SS, frames={})
    return _ST


def get_frames(res, src):
    st = _ST
    if src not in st["frames"]:
        SD, S, C = st["SD"], st["S"], st["C"]
        meta = json.load(open(os.path.join(res, "frames_meta_v911.json"), encoding="utf-8"))
        qrefs = {k: float(v) for k, v in meta["q_ref_Ah"].items()}
        frames0 = SD.frames_for(res, src)
        dkey = C.soh_sel(src).get("delta_key")
        st["frames"][src] = (S.apply_delta(frames0, dkey), qrefs)
    return _ST["frames"][src]


def run_task(spec):
    """worker：一個（類型, 來源, 測試電芯）單元。回傳 ckpt 路徑。"""
    kind, src, cell, res, ds = spec["kind"], spec["src"], spec["cell"], spec["res"], spec["ds"]
    if kind == "sunit":
        return search_task(spec)
    ckp = os.path.join(res, "_ckpt_task", f"{kind}__{src}__{cell}.json")
    if os.path.exists(ckp):
        return ckp
    t0 = time.time(); st = setup(ds); SS = st["SS"]
    frames, qrefs = get_frames(res, src)
    SS.ONLY = {cell}; lg = lambda m: None
    out = {}
    if kind == "loo":
        out["P-B"] = SS.run_PB(frames, src, qrefs, lg)
        for m in SS.M_LIST:
            out[f"P-B+|M={m}"] = SS.run_PBplus(frames, src, qrefs, m, lg)
    elif kind == "pc":
        out["P-C"] = SS.run_PC(frames, src, qrefs, lg)
    elif kind == "pa":
        out["P-A"] = SS.run_PA(frames, src, qrefs, lg)
    else:
        raise ValueError(kind)
    out = {k: dict(recs=v[0], preds=v[1], sel=v[2]) for k, v in out.items()}
    out["_sec"] = time.time() - t0; out["_pid"] = os.getpid()
    os.makedirs(os.path.dirname(ckp), exist_ok=True)
    tmp = ckp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=lambda o: (o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, ckp)
    return ckp


def search_task(spec):
    """worker：SEARCH_SOH 之一個（組態單元, 測試電芯）：P-B+（M=20，主模型 GPR-LSP）之逐顆列。"""
    res, ds, src, cell = spec["res"], spec["ds"], spec["src"], spec["cell"]
    ckp = os.path.join(res, "search", "_tasks", spec["tid"] + ".json")
    if os.path.exists(ckp):
        return ckp
    t0 = time.time(); setup(ds)
    import search_soh_v911 as SR
    SR.RES = os.path.join(res, "search"); SR.CKPT = os.path.join(SR.RES, "_ckpt"); SR.LOGP = os.path.join(SR.RES, "logs", "search_soh.log")
    SR.log = lambda m: None
    cand = spec["cand"]; sel = SR.resolve(src, cand)
    frames = SR.frames_of(src, cand, {})
    fd = _ST["S"].apply_delta(frames, sel["delta_key"])
    nrows = {c: len(f) for c, f in fd.items()}
    out = dict(skipped=False)
    if (not nrows) or min(nrows.values()) < 30:
        out = dict(skipped=True, reason=f"min rows {min(nrows.values()) if nrows else 0} < 30")
    else:
        SR.ONLY = {cell}
        out["rows"] = SR.run_PBplus(fd, src, sel, SR.M_MAIN)
    out["sec"] = time.time() - t0
    os.makedirs(os.path.dirname(ckp), exist_ok=True)
    tmp = ckp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=lambda o: (o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, ckp)
    return ckp


def stage_search(res, ds, workers, log, sources=("nasa", "own", "ox")):
    """SEARCH_SOH 重跑：原程式之選擇邏輯（階 1 一次一維、階 2 前 8 名組合、選擇規則）逐字沿用；
    僅將「單元評估」改為（組態單元×測試電芯）平行計算並寫成與原檢查點相同格式；不套用 90 分鐘預算跳過（deadline 設為無限）。"""
    st = setup(ds)
    import search_soh_v911 as SR
    SRES = os.path.join(res, "search"); os.makedirs(SRES, exist_ok=True)
    SR.RES = SRES; SR.CKPT = os.path.join(SRES, "_ckpt"); SR.LOGP = os.path.join(SRES, "logs", "search_soh.log")
    os.makedirs(SR.CKPT, exist_ok=True)
    notes_p = os.path.join(SRES, "_notes.json")
    notes = json.load(open(notes_p, encoding="utf-8")) if os.path.exists(notes_p) else {}
    notes["budget"] = dict(total_sec=None, note="v3 重跑：不套用 90 分鐘預算（deadline=∞）；全部候選皆計算")
    orig_eval = SR.eval_unit; pending = {}
    DIMS = SR.DIMS

    def fake_eval(src, cand, stage, nt):
        d = SR.cached(src, cand)
        if d is not None:
            return d["rows"], d.get("sec", 0.0), True
        uid = SR.uid(src, cand)
        if os.path.exists(os.path.join(SRES, "_skip", f"{src}__{uid}.json")):
            return None, 0.0, False
        pending[(src, uid)] = (src, cand, stage)
        return None, 0.0, False
    far = time.time() + 1e12
    t_all = time.time(); rnd_i = 0
    SR.log = lambda m: None
    while True:
        pending.clear(); SR.eval_unit = fake_eval
        for src in sources:
            SR.run_source(src, notes, far)
        if not pending:
            break
        rnd_i += 1
        specs = []
        for (src, uid), (src_, cand, stage) in sorted(pending.items()):
            cl = sorted(SR.cells_of(src))
            for c in cl:
                specs.append(dict(kind="sunit", src=src, cell=c, res=res, ds=ds, uid=uid, stage=stage,
                                  cand=json.loads(json.dumps(cand, default=str)), tid=f"s__{src}__{uid}__{c}"))
        specs.sort(key=lambda x: (0 if x["src"] == "ox" else 1, x["uid"], x["cell"]))
        log(f"SEARCH 第 {rnd_i} 輪：{len(pending)} 個組態單元、{len(specs)} 個（單元×電芯）任務")
        run_workers(specs, os.path.join(SRES, "_tasks"), res, workers, log, tag=f"SEARCH{rnd_i}")
        for (src, uid), (src_, cand, stage) in pending.items():
            cl = sorted(SR.cells_of(src)); rows = []; sec = 0.0; skipped = None
            for c in cl:
                d = json.load(open(os.path.join(SRES, "_tasks", f"s__{src}__{uid}__{c}.json"), encoding="utf-8"))
                sec += d["sec"]
                if d.get("skipped"):
                    skipped = d["reason"]; break
                rows += d["rows"]
            if skipped:
                os.makedirs(os.path.join(SRES, "_skip"), exist_ok=True)
                json.dump(dict(reason=skipped), open(os.path.join(SRES, "_skip", f"{src}__{uid}.json"), "w"))
                continue
            for r in rows:
                r["stage"] = stage; r["diff_dims"] = ",".join(SR.diff_dims(src, cand))
            os.makedirs(os.path.dirname(SR.ck_path(src, cand)), exist_ok=True)
            json.dump(dict(rows=rows, sec=sec, cand={k: cand[k] for k in DIMS}, config_id=SR.resolve(src, cand)["config_id"], stage=stage),
                      open(SR.ck_path(src, cand), "w", encoding="utf-8"), ensure_ascii=False,
                      default=lambda o: (o.item() if hasattr(o, "item") else str(o)))
    # 最後一輪：原 eval_unit（全部已有檢查點；跳過者重新偵測並記入 notes）
    SR.eval_unit = orig_eval
    per_src, stage1_all = {}, {}
    for src in sources:
        r_, s1 = SR.run_source(src, notes, far); per_src[src] = r_; stage1_all[src] = s1
    notes["total_sec_wall"] = round(time.time() - t_all, 1); notes["rounds"] = rnd_i
    json.dump(notes, open(notes_p, "w", encoding="utf-8"), ensure_ascii=False, indent=1,
              default=lambda o: (o.item() if hasattr(o, "item") else str(o)))
    SR.write_outputs(list(sources), per_src, stage1_all, notes)
    log(f"SEARCH 完成：{ {s: len(per_src[s]) for s in sources} } 個組態；牆鐘 {time.time()-t_all:.0f}s")


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True); self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{utc()}] {msg}"; print(line, flush=True); self.f.write(line + "\n"); self.f.flush()


def stage_env(res, ds, workers, log):
    import sklearn, scipy
    try:
        from threadpoolctl import threadpool_info
        blas = [(d.get("internal_api"), d.get("version")) for d in threadpool_info() if d.get("user_api") == "blas"]
    except Exception:
        blas = None
    env = dict(python=sys.version.split()[0], platform=platform.platform(), machine=platform.node(), n_cpu=os.cpu_count(),
               numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
               blas=blas, utc=utc(), cwd=HERE, dataset=ds, workers=workers, blas_threads_per_process=int(os.environ.get("V3_BLAS_THREADS", "1")),
               note="v3 重跑：Windows 本機（非 1版 之 Linux 2 核）；平行單位＝來源×測試電芯")
    dump(env, os.path.join(res, "env_v911.json")); log(f"ENV {env}")


def stage_prereg(res, prereg_dir, log):
    p = os.path.join(prereg_dir, "selection_preregistration_v17.json")
    s = open(os.path.join(prereg_dir, "selection_preregistration_v17.sha256"), encoding="utf-8").read().split()[0]
    h = sha256_file(p)
    assert h == s, (h, s)
    reg = json.load(open(p, encoding="utf-8"))["registered_utc"]
    log(f"PREREG17 沿用凍結登記檔（不重寫）sha256 {h} registered_utc {reg}")


def stage_data(res, ds, log):
    st = setup(ds); SD = st["SD"]
    SD.main(res, log)



def task_id(sp):
    return sp.get("tid") or f"{sp['kind']}__{sp['src']}__{sp['cell']}"


def worker_loop(tasks_path, ckdir):
    """worker 程序：依序以 mkdir 原子認領單元（無管線／共用記憶體），逐一執行；每程序快取各來源 frames。"""
    specs = json.load(open(tasks_path, encoding="utf-8")); lockdir = os.path.join(os.path.dirname(tasks_path), "_locks")
    os.makedirs(lockdir, exist_ok=True); n = 0
    for sp in specs:
        tid = task_id(sp)
        if os.path.exists(os.path.join(ckdir, tid + ".json")):
            continue
        try:
            os.mkdir(os.path.join(lockdir, tid))
        except FileExistsError:
            continue
        run_task(sp); n += 1
    print(f"worker {os.getpid()} done, {n} tasks", flush=True)


def run_workers(specs, ckdir, res, workers, log, tag="SOH", extra_args=None):
    import subprocess
    os.makedirs(ckdir, exist_ok=True)
    lockdir = os.path.join(res, "_locks"); shutil.rmtree(lockdir, ignore_errors=True)
    tp = os.path.join(res, f"_tasks_{tag}.json"); json.dump(specs, open(tp, "w", encoding="utf-8"))
    todo = [sp for sp in specs if not os.path.exists(os.path.join(ckdir, task_id(sp) + ".json"))]
    log(f"{tag} start：{len(specs)} 單元（待算 {len(todo)}），{workers} 個 worker 程序")
    t0 = time.time(); procs = []
    for i in range(min(workers, max(1, len(todo)))):
        fo = open(os.path.join(res, "logs", f"worker_{tag}_{i:02d}.txt"), "w", encoding="utf-8")
        procs.append((subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", tp, "--ckdir", ckdir,
                                        "--dataset", "x", "--results", res] + (extra_args or []), stdout=fo, stderr=subprocess.STDOUT, cwd=HERE), fo))
    last = -1
    while any(p.poll() is None for p, _ in procs):
        time.sleep(15)
        n = sum(os.path.exists(os.path.join(ckdir, task_id(sp) + ".json")) for sp in specs)
        if n != last and (n % 10 == 0 or n == len(specs)):
            log(f"  {tag} 單元 {n}/{len(specs)} 完成（{time.time()-t0:.0f}s）"); last = n
    bad = [p.returncode for p, _ in procs if p.returncode != 0]
    for _, fo in procs:
        fo.close()
    n = sum(os.path.exists(os.path.join(ckdir, task_id(sp) + ".json")) for sp in specs)
    assert not bad and n == len(specs), (bad, n, len(specs))
    log(f"{tag} 全部單元完成（{time.time()-t0:.0f}s）")


def stage_soh(res, ds, workers, log):
    import subprocess
    st = setup(ds); C, SS = st["C"], st["SS"]
    meta = json.load(open(os.path.join(res, "frames_meta_v911.json"), encoding="utf-8"))
    srcs = C.DATA["sources_soh"]
    cells = {s: sorted(meta["sources"][s]["cells"]) for s in srcs}
    specs = []
    for s in srcs:
        for c in cells[s]:
            for k in ("loo", "pc", "pa"):
                specs.append(dict(kind=k, src=s, cell=c, res=res, ds=ds))
    pri = {"loo": 0, "pc": 1, "pa": 2}
    specs.sort(key=lambda x: (0 if x["src"] == "ox" else 1, pri[x["kind"]], x["src"], x["cell"]))   # 大單元優先（排程用，不影響結果）
    run_workers(specs, os.path.join(res, "_ckpt_task"), res, workers, log, tag="SOH")
    t0 = time.time()
    # ── 組裝（與 stage_soh_v911.main 尾段相同順序）──
    all_recs, all_preds, all_sel = [], {}, []
    M_LIST = SS.M_LIST
    timing = {}
    for src in srcs:
        dsel = C.soh_sel(src); dkey = dsel.get("delta_key"); gate = dsel.get("label_gate", "on")
        protos = ["P-B", "P-C", "P-A"] + [f"P-B+|M={m}" for m in M_LIST]
        src_of = {"P-B": "loo", "P-C": "pc", "P-A": "pa"}
        for proto in protos:
            kind = src_of.get(proto, "loo")
            recs, preds, sel = [], {}, []
            for c in cells[src]:
                d = json.load(open(os.path.join(res, "_ckpt_task", f"{kind}__{src}__{c}.json"), encoding="utf-8"))
                timing[f"{kind}__{src}__{c}"] = d["_sec"]
                x = d[proto]; recs += x["recs"]; preds.update(x["preds"]); sel += x["sel"]
            for r in recs:
                if r.get("delta_V") is None:
                    r["delta_V"] = dkey
                if r.get("label_gate") is None:
                    r["label_gate"] = gate
            all_recs += recs; all_preds.update(preds); all_sel += sel
            log(f"  SOH {src} {proto}：{len(recs)} 列")
    df = pd.DataFrame(all_recs)
    cols = ["source", "protocol", "cell", "s0", "model", "cyc_axis", "label_gate", "delta_V", "M",
            "window", "filter", "norm", "family", "top_k", "periodic", "ard", "sub", "config_id",
            "n_train", "n_train_cells", "n_test", "n_ref", "n_eval",
            "RMSE_pct", "MAE_pct", "R2", "RMSE_Ah", "MAE_Ah", "MaxRE_pct", "MAPE_pct",
            "CI95_cov", "CI95_halfwidth_soh", "pass_T_SOH", "reused", "reuse_src", "feats", "note"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].sort_values(["source", "protocol", "cell", "s0", "model"], na_position="first").reset_index(drop=True)
    df.to_csv(os.path.join(res, "soh_records_v911.csv"), index=False)
    with open(os.path.join(res, "soh_predictions_v911.json"), "w", encoding="utf-8") as f:
        json.dump(all_preds, f, ensure_ascii=False)
    with open(os.path.join(res, "soh_selection_v911.json"), "w", encoding="utf-8") as f:
        json.dump(dict(rule=C.RUL_STANDARD["no_future_info"], selections=all_sel), f, ensure_ascii=False)
    with open(os.path.join(res, "soh_reuse_v911.json"), "w", encoding="utf-8") as f:
        json.dump(dict(policy=C.EXEC["reuse_policy"], reused_units=[]), f, ensure_ascii=False, indent=1)
    dump(dict(wall_sec=time.time() - t0, workers=workers, n_units=len(specs), unit_sec=timing, cpu_sec_sum=sum(timing.values())),
         os.path.join(res, "soh_timing_v3.json"))
    log(f"SOH 完成：{len(df)} 列；牆鐘 {time.time()-t0:.0f}s；單元 CPU 秒合計 {sum(timing.values()):.0f}")
    return df


def serial_check(res, ds, src, log):
    """以未修補之全序列路徑（ONLY=None、單一程序）重算某來源，與平行結果（soh_records／predictions）逐值比對。"""
    st = setup(ds); SS = st["SS"]; SS.ONLY = None
    frames, qrefs = get_frames(res, src); lg = lambda m: None
    t0 = time.time(); recs, preds = [], {}
    for fn in (SS.run_PB, SS.run_PC, SS.run_PA):
        r, p, s = fn(frames, src, qrefs, lg); recs += r; preds.update(p)
    for m in SS.M_LIST:
        r, p, s = SS.run_PBplus(frames, src, qrefs, m, lg); recs += r; preds.update(p)
    df = pd.DataFrame(recs)
    old = pd.read_csv(os.path.join(res, "soh_records_v911.csv")); old = old[old.source == src]
    key = ["protocol", "cell", "s0", "model", "M"]
    for k in key:
        if k not in df.columns:
            df[k] = np.nan
    m = df.merge(old, on=key, suffixes=("_n", "_o"), how="outer", indicator=True)
    assert (m._merge == "both").all(), m._merge.value_counts()
    d = {c: float((m[f"{c}_n"].astype(float) - m[f"{c}_o"].astype(float)).abs().max()) for c in ("RMSE_pct", "MAE_pct", "R2")}
    P = json.load(open(os.path.join(res, "soh_predictions_v911.json"), encoding="utf-8"))
    dp = 0.0
    for k, v in preds.items():
        dp = max(dp, float(np.max(np.abs(np.asarray(v["yhat"], float) - np.asarray(P[k]["yhat"], float)))))
    out = dict(src=src, n_rows=len(m), max_abs_diff=d, max_abs_diff_pred=dp, serial_sec=time.time() - t0)
    log(f"SERIAL_CHECK {out}")
    dump(out, os.path.join(res, f"serial_check_{src}.json"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--results", required=True)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--stages", default="ENV,PREREG17,DATA,SOH")
    ap.add_argument("--serial-check", default=None)
    ap.add_argument("--worker", default=None); ap.add_argument("--ckdir", default=None)
    ap.add_argument("--search-sources", default="nasa,own,ox")
    ap.add_argument("--prereg-dir", default=os.path.join(PRG, "01_prereg", "prereg"))
    a = ap.parse_args()
    if a.worker:
        worker_loop(a.worker, a.ckdir); return
    res = os.path.abspath(a.results); os.makedirs(res, exist_ok=True)
    log = Log(os.path.join(res, "logs", "run_soh_v3.log"))
    if a.serial_check:
        serial_check(res, a.dataset, a.serial_check, log); return
    st = a.stages.split(",")
    log(f"RUN_START stages={st} dataset={a.dataset} workers={a.workers}")
    if "ENV" in st:
        stage_env(res, a.dataset, a.workers, log)
    if "PREREG17" in st:
        stage_prereg(res, a.prereg_dir, log)
    if "DATA" in st:
        t0 = time.time(); stage_data(res, a.dataset, log); log(f"STAGE_DONE DATA {time.time()-t0:.0f}s")
    if "SOH" in st:
        stage_soh(res, a.dataset, a.workers, log); log("STAGE_DONE SOH")
    if "SEARCH" in st:
        stage_search(res, a.dataset, a.workers, log, tuple(a.search_sources.split(","))); log("STAGE_DONE SEARCH")
    if "EXPORT" in st:
        import export_soh_v3 as EX
        t0 = time.time(); EX.main(res, a.prereg_dir, log); log(f"STAGE_DONE EXPORT {time.time()-t0:.0f}s")
    if "VERIFY" in st:
        setup(a.dataset)   # 驗證 9 需由 npz 重建 frames 之列數
        import verify_soh_v3 as VF
        t0 = time.time(); nf = VF.main(res, a.prereg_dir, log); log(f"STAGE_DONE VERIFY n_fail={nf} {time.time()-t0:.0f}s")
    log("ALL_DONE")


if __name__ == "__main__":
    main()
