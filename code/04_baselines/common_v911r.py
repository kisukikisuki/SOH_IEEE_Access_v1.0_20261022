"""v9.11r 審稿回應後處理共用設定。
原則：只讀 0921/pv911_run/results_v911 既有結果，不重新訓練任何模型、不改架構。"""
import os, json, hashlib, math
import numpy as np
BASE = os.environ.get("PV911_RESULTS",
    os.path.expanduser("~/mnt/電池資料/0921/pv911_run/results_v911"))
OUT = os.environ.get("PV911R_OUT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_v911r"))
os.makedirs(OUT, exist_ok=True)
SEED = 20260830

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def load_json(name):
    with open(os.path.join(BASE, name), encoding="utf-8") as f: return json.load(f)

def dump(name, obj):
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f: json.dump(obj, f, ensure_ascii=False, indent=1, default=float)
    return p

def metrics(y, yh):
    y = np.asarray(y, float); yh = np.asarray(yh, float)
    e = yh - y
    rmse = 100 * math.sqrt(np.mean(e ** 2)); mae = 100 * np.mean(np.abs(e))
    ss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(e ** 2) / ss if ss > 0 else float("nan")
    return dict(RMSE_pct=rmse, MAE_pct=mae, R2=r2, n=len(y))

def passes(m):
    return int(m["RMSE_pct"] < 1 and m["MAE_pct"] < 1 and m["R2"] > 0.99)

def boot_ci(x, B=10000, seed=SEED):
    x = np.asarray(x, float)
    if len(x) < 3: return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
