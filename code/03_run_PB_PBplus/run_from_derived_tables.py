"""run_from_derived_tables.py — 由逐列衍生表重跑 P-B 與 P-B+（M = 10、20）；v34 第 III-A 節 2)、3)，第 III-C 節，表 5、表 6、表 8。

新撰（公開程式包之最小包裝）；邏輯引用（行號為原檔行號；原檔逐位元複製於 ../02_core_modules/）：
  ROOT\\5版\\02_程式\\02_SOH重跑\\run_soh_v3.py
    L39–53  setup()        —— 以 import 直接呼叫 run_soh_v3.setup()（setup() 將 data_v911.DATASET_V9 設為 --data 資料夾路徑；
                              本 runner 之資料載入不經 data_v911，不讀任何 npz）
    L56–65  get_frames()   —— 複製改寫：原 SD.frames_for()（由 npz 於記憶體內重建特徵表）換成 load_derived()；
                              其後同原程式套 S.apply_delta(frames, C.soh_sel(src)["delta_key"])（閘門 A1）。
                              qrefs（Q_ref, Ah）原由 frames_meta_v911.json 之 q_ref_Ah 讀入，本程式改由 cells_summary.csv
                              之 q_ref_Ah 欄（或 --qref 指定之 frames_meta_v911.json）讀入。
    L68–97  run_task()     —— 複製，只保留 kind="loo" 分支（P-B，再 P-B+ 之 M ∈ SS.M_LIST＝{10, 20}）；P-A／P-C／search 分支刪除。
    L238–255 task_id()／worker_loop() —— 複製，未改。
    L258–281 run_workers() —— 複製；子程序改為呼叫本檔（參數 --data／--qref／--out）。
    L284–341 stage_soh()   —— 複製組裝段；protos 只取 "P-B" 與 "P-B+|M=…"；records 欄位順序（L322–330）與排序不變；
                              輸出 soh_records_v911.csv（只含 P-B、P-B+ 列）、soh_predictions_v911.json（鍵格式同原程式）、
                              soh_selection_v911.json、soh_timing.json。
  ROOT\\5版\\02_程式\\02_SOH重跑\\pv911_run_SOH\\02_programs\\stage_data_v911.py
    L111–129 load_frames() —— 複製改寫為 load_derived()：讀 <data>/<src>__<cell>.csv，先丟棄 ann_* 欄（公開衍生表之右側標註欄），
                              其餘規則不變（label_gate＝on：drop_flag==0 列、cyc＝cyc_c；off：cyc＝cyc_c＝cyc_A；dropna；REL 正規化）。
                              資料載入共改寫兩處：(1) 丟棄 ann_* 欄；(2) pd.read_csv 加 float_precision="round_trip"。原 load_frames 用 pandas 預設浮點解析，
                              讀回值與記憶體內 build_frames 有末位差（本機實測最大 2.3e-13），足以使 SVR／MLP 之誤差改變達 0.01–1.3 pp；
                              以 round_trip 解析則 34 顆特徵表與 build_frames 逐值相等（差 0）。
                              （作者內部工作紀錄，未收錄：pandas 預設解析之特徵末位差最大約 2.3e-13。）
不呼叫：stage_soh_v911.run_PA／run_PC、search_soh_v911（範圍外，未複製）。
模型、特徵篩選、殘差 GPR、種子、組態全部由 ../02_core_modules/ 之原程式（config_v911.SOH_SELECTED_V911，計畫第 17 版鎖定組態）決定，本檔不覆寫任何組態。

用法：
  python run_from_derived_tables.py --data <PUB>/data/derived_per_row --out <輸出夾> [--qref <PUB>/data/cells_summary.csv] [--workers 18] [--sources nasa,own,ox]
"""
import os, sys, json, time, argparse, shutil, platform, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.abspath(os.path.join(HERE, "..", "02_core_modules"))
sys.path.insert(0, CORE)
import run_soh_v3 as RS          # 匯入時即設 BLAS 執行緒（V3_BLAS_THREADS，預設 1）並把 02_programs 加入 sys.path（原檔 L10–14）
import numpy as np, pandas as pd

ALL_SOURCES = ("nasa", "own", "ox")


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── 資料載入（複製改寫自 stage_data_v911.load_frames L111–129）──────────────────────
def load_derived(data_dir, src, st):
    S, C = st["S"], st["C"]
    sel = C.soh_sel(src)
    out = {}
    for fn in sorted(os.listdir(data_dir)):
        if not fn.startswith(src + "__") or not fn.endswith(".csv"):
            continue
        c = fn[len(src) + 2:-4]
        f = pd.read_csv(os.path.join(data_dir, fn), float_precision="round_trip")   # 見檔頭：逐位元還原 %.17g 落地值
        f = f[[k for k in f.columns if not str(k).startswith("ann_")]]   # 改寫處 (1)：丟棄公開衍生表之 ann_* 標註欄（改寫處 (2) 為上一行之 round_trip 解析）
        gate = str(f["label_gate"].iloc[0])
        if gate == "on":
            f = f[f.drop_flag == 0].copy(); f["cyc"] = f["cyc_c"]
        else:
            f = f.copy(); f["cyc"] = f["cyc_A"]; f["cyc_c"] = f["cyc_A"]
        f = f.dropna().reset_index(drop=True)
        out[c] = f
    if not out:
        raise FileNotFoundError(f"{data_dir} 內找不到 {src}__*.csv")
    if sel["rel_nref"]:
        out = {c: S.rel_normalize(f, nref=sel["rel_nref"]) for c, f in out.items()}
    return out


def load_qrefs(qref_path):
    """Q_ref（Ah）：cells_summary.csv（欄 cell、q_ref_Ah；可另含 source）或 frames_meta_v911.json（鍵 q_ref_Ah）。"""
    if qref_path.lower().endswith(".json"):
        meta = json.load(open(qref_path, encoding="utf-8"))
        return {k: float(v) for k, v in meta["q_ref_Ah"].items()}
    t = pd.read_csv(qref_path, float_precision="round_trip")
    if "cell" not in t.columns or "q_ref_Ah" not in t.columns:
        raise KeyError(f"{qref_path} 需有 cell 與 q_ref_Ah 欄；實際欄位 {list(t.columns)}")
    q = {}
    for r in t.itertuples():
        c = str(r.cell)
        s = str(getattr(r, "source", "")) if "source" in t.columns else ""
        if s and c.startswith(s + "__"):
            c = c[len(s) + 2:]
        q[c] = float(r.q_ref_Ah)
    return q


_FR = {}


def get_frames(data_dir, qref_path, src):
    """複製改寫自 run_soh_v3.get_frames L56–65（frames 來源改為衍生表）。"""
    st = RS.setup(data_dir)
    if src not in _FR:
        S, C = st["S"], st["C"]
        qrefs = load_qrefs(qref_path)
        frames0 = load_derived(data_dir, src, st)
        dkey = C.soh_sel(src).get("delta_key")
        _FR[src] = (S.apply_delta(frames0, dkey), qrefs)
    return _FR[src]


def run_task(spec):
    """複製自 run_soh_v3.run_task L68–97，只保留 kind="loo"。"""
    kind, src, cell, res = spec["kind"], spec["src"], spec["cell"], spec["res"]
    assert kind == "loo", kind
    ckp = os.path.join(res, "_ckpt_task", f"{kind}__{src}__{cell}.json")
    if os.path.exists(ckp):
        return ckp
    t0 = time.time(); st = RS.setup(spec["data"]); SS = st["SS"]
    frames, qrefs = get_frames(spec["data"], spec["qref"], src)
    SS.ONLY = {cell}; lg = lambda m: None
    out = {}
    out["P-B"] = SS.run_PB(frames, src, qrefs, lg)
    for m in SS.M_LIST:
        out[f"P-B+|M={m}"] = SS.run_PBplus(frames, src, qrefs, m, lg)
    out = {k: dict(recs=v[0], preds=v[1], sel=v[2]) for k, v in out.items()}
    out["_sec"] = time.time() - t0; out["_pid"] = os.getpid()
    os.makedirs(os.path.dirname(ckp), exist_ok=True)
    tmp = ckp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=lambda o: (o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, ckp)
    return ckp


def task_id(sp):                                   # 複製自 run_soh_v3 L238–239
    return sp.get("tid") or f"{sp['kind']}__{sp['src']}__{sp['cell']}"


def worker_loop(tasks_path, ckdir):                # 複製自 run_soh_v3 L242–255
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


def run_workers(specs, ckdir, res, workers, log, tag="SOH"):   # 複製自 run_soh_v3 L258–281（子程序改呼叫本檔）
    import subprocess
    os.makedirs(ckdir, exist_ok=True); os.makedirs(os.path.join(res, "logs"), exist_ok=True)
    lockdir = os.path.join(res, "_locks"); shutil.rmtree(lockdir, ignore_errors=True)
    tp = os.path.join(res, f"_tasks_{tag}.json"); json.dump(specs, open(tp, "w", encoding="utf-8"))
    todo = [sp for sp in specs if not os.path.exists(os.path.join(ckdir, task_id(sp) + ".json"))]
    log(f"{tag} start：{len(specs)} 單元（待算 {len(todo)}），{workers} 個 worker 程序")
    t0 = time.time(); procs = []
    for i in range(min(workers, max(1, len(todo)))):
        fo = open(os.path.join(res, "logs", f"worker_{tag}_{i:02d}.txt"), "w", encoding="utf-8")
        procs.append((subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", tp, "--ckdir", ckdir,
                                        "--data", specs[0]["data"], "--out", res],
                                       stdout=fo, stderr=subprocess.STDOUT, cwd=HERE), fo))
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


def stage_soh(data_dir, qref_path, res, workers, log, sources):
    """複製自 run_soh_v3.stage_soh L284–341；只排 kind="loo" 單元、只組裝 P-B 與 P-B+。"""
    st = RS.setup(data_dir); C, SS = st["C"], st["SS"]
    srcs = [s for s in C.DATA["sources_soh"] if s in sources]
    cells = {}
    for s in srcs:
        cells[s] = sorted(fn[len(s) + 2:-4] for fn in os.listdir(data_dir) if fn.startswith(s + "__") and fn.endswith(".csv"))
    specs = [dict(kind="loo", src=s, cell=c, res=res, data=data_dir, qref=qref_path) for s in srcs for c in cells[s]]
    specs.sort(key=lambda x: (0 if x["src"] == "ox" else 1, x["src"], x["cell"]))   # 大單元優先（排程用，不影響結果）
    run_workers(specs, os.path.join(res, "_ckpt_task"), res, workers, log, tag="SOH")
    t0 = time.time()
    all_recs, all_preds, all_sel = [], {}, []
    M_LIST = SS.M_LIST
    timing = {}
    for src in srcs:
        dsel = C.soh_sel(src); dkey = dsel.get("delta_key"); gate = dsel.get("label_gate", "on")
        protos = ["P-B"] + [f"P-B+|M={m}" for m in M_LIST]
        for proto in protos:
            recs, preds, sel = [], {}, []
            for c in cells[src]:
                d = json.load(open(os.path.join(res, "_ckpt_task", f"loo__{src}__{c}.json"), encoding="utf-8"))
                timing[f"loo__{src}__{c}"] = d["_sec"]
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
    RS.dump(dict(wall_sec_assemble=time.time() - t0, workers=workers, n_units=len(specs), unit_sec=timing,
                 cpu_sec_sum=sum(timing.values())), os.path.join(res, "soh_timing.json"))
    log(f"SOH 完成：{len(df)} 列；單元 CPU 秒合計 {sum(timing.values()):.0f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "..", "data", "derived_per_row"))
    ap.add_argument("--qref", default=None, help="cells_summary.csv（預設 <data>/../cells_summary.csv）或 frames_meta_v911.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--sources", default=",".join(ALL_SOURCES))
    ap.add_argument("--worker", default=None); ap.add_argument("--ckdir", default=None)
    a = ap.parse_args()
    if a.worker:
        worker_loop(a.worker, a.ckdir); return
    data_dir = os.path.abspath(a.data)
    qref = os.path.abspath(a.qref) if a.qref else os.path.abspath(os.path.join(data_dir, "..", "cells_summary.csv"))
    res = os.path.abspath(a.out); os.makedirs(os.path.join(res, "logs"), exist_ok=True)
    log = RS.Log(os.path.join(res, "logs", "run_from_derived_tables.log"))
    t0 = time.time(); start = utc()
    log(f"RUN_START data={data_dir} qref={qref} workers={a.workers} sources={a.sources}")
    import sklearn, scipy
    stage_soh(data_dir, qref, res, a.workers, log, tuple(a.sources.split(",")))
    RS.dump(dict(start_utc=start, end_utc=utc(), wall_sec=time.time() - t0, argv=sys.argv, data=data_dir, qref=qref,
                 workers=a.workers, blas_threads_per_process=int(os.environ.get("V3_BLAS_THREADS", "1")),
                 python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__,
                 pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__),
            os.path.join(res, "run_info.json"))
    log(f"ALL_DONE 牆鐘 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
