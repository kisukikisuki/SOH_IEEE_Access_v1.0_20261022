"""
ic_v911.py — 架構一前段：IC 曲線（dQ/dV）、濾波（Gaussian／SG）、部分 IC 區間特徵、離群清洗、相關係數篩選。
全部依 A 類期刊之定義實作：
  IC = dQ/dV（CC 充電段；22 Eq.3、23 Eq.3、21 Eq.1、24 Eq.6）
  Gaussian 濾波（22：μ=17、σ=5；25：window 15、σ=3）→ 本版 σ 以 mV 登記；SG 濾波（23；21：window 51）作對照
  特徵：FV_k＝區間內每 30 mV 之 IC 值（22 Table 2：3.8–4.1 V 共 11 點）；HI1 峰值、HI2 峰下面積（21）；
        截距、斜率（23：部分 IC 線性擬合）；t_pk＝IC-T 峰時間（24：CC 下 t = Q/I）；V_pk 峰值電壓（05 HF13）
  離群：滑動視窗 Robust Z-Score（rolling median／MAD，0.6745 校正）＋邊界中位清洗＋線性內插取代（05 Eq.3–6）
  相關：Pearson（22 Eq.6）與 Spearman（25 Eq.11、05 Eq.7、21 Eq.4）
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import pearsonr, spearmanr

DV = 0.005  # 網格 5 mV


def fill_Q(q):
    """Q(V) 補值：前導 NaN → 0（充電尚未開始，累積容量為零）；中段 NaN → 線性內插；尾端 NaN 保留。"""
    q = np.asarray(q, float).copy(); ok = np.isfinite(q)
    if ok.sum() < 8:
        return np.full(q.shape, np.nan)
    first, last = np.argmax(ok), len(q) - 1 - np.argmax(ok[::-1])
    q[:first] = 0.0
    idx = np.arange(len(q)); mid = np.arange(first, last + 1); okm = ok[first:last + 1]
    q[first:last + 1] = np.interp(mid, mid[okm], q[first:last + 1][okm])
    return q


def ic_raw(q_filled, grid):
    """IC（Ah/V）＝ dQ/dV，中央差分（尾端 NaN 保留）。"""
    ok = np.isfinite(q_filled)
    ic = np.full(q_filled.shape, np.nan)
    if ok.sum() < 8:
        return ic
    last = len(q_filled) - 1 - np.argmax(ok[::-1])
    ic[:last + 1] = np.gradient(q_filled[:last + 1], grid[:last + 1])
    return ic


def gs_filter(ic, sigma_mV):
    """Gaussian 濾波；σ 以 mV 登記（網格 5 mV）。尾端 NaN 以邊緣值延伸後濾波再還原 NaN。"""
    ok = np.isfinite(ic)
    if ok.sum() < 8:
        return ic
    x = ic.copy(); last = len(ic) - 1 - np.argmax(ok[::-1]); x[last + 1:] = x[last]
    y = gaussian_filter1d(x, sigma=sigma_mV / (DV * 1000.0), mode="nearest")
    y[~ok] = np.nan
    return y


def sg_filter(ic, window_pts=11, order=3):
    ok = np.isfinite(ic)
    if ok.sum() < window_pts + 2:
        return ic
    x = ic.copy(); last = len(ic) - 1 - np.argmax(ok[::-1]); x[last + 1:] = x[last]
    y = savgol_filter(x, window_length=window_pts, polyorder=order, mode="nearest")
    y[~ok] = np.nan
    return y


def fv_points(lo, hi, step=0.03):
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    return np.round(lo + step * np.arange(n), 3)


def features(ic_f, q_f, grid, window, I_cc):
    """回傳 dict：FV_<V>（30 mV 格點 IC 值）、peak_h、V_pk、area、intercept、slope、t_pk_h。"""
    lo, hi = window; m = (grid >= lo - 1e-9) & (grid <= hi + 1e-9)
    g = grid[m]; ic = ic_f[m]; q = q_f[m]
    out = {}
    for v in fv_points(lo, hi):
        j = int(np.argmin(np.abs(grid - v)))
        out[f"FV_{v:.2f}"] = float(ic_f[j]) if np.isfinite(ic_f[j]) else np.nan
    ok = np.isfinite(ic)
    if ok.sum() >= 5:
        k = int(np.nanargmax(ic)); out["peak_h"] = float(ic[k]); out["V_pk"] = float(g[k])
        out["area"] = float(np.nansum(0.5 * (ic[ok][1:] + ic[ok][:-1]) * np.diff(g[ok])))           # ∫IC dV（Ah）
        A = np.vstack([np.ones(ok.sum()), g[ok]]).T; coef, *_ = np.linalg.lstsq(A, ic[ok], rcond=None)
        out["intercept"] = float(coef[0]); out["slope"] = float(coef[1])
        out["t_pk_h"] = float(q[k] / I_cc) if (np.isfinite(q[k]) and I_cc > 0) else np.nan          # IC-T：CC 下 t = Q/I（h）
    else:
        for k in ["peak_h", "V_pk", "area", "intercept", "slope", "t_pk_h"]:
            out[k] = np.nan
    return out


def robust_z_clean(x, window=15, z_thr=3.5, edge_ratio=0.10, eps=1e-6):
    """05 之滑動視窗 Robust Z-Score 離群偵測（Eq.3–5）＋邊界中位清洗（Eq.6）；離群以線性內插取代。回傳 (cleaned, n_outlier)."""
    x = np.asarray(x, float).copy(); n = len(x); flag = np.zeros(n, bool)
    if n < window + 2:
        return x, 0
    h = window // 2
    for t in range(h, n - h):
        w = x[t - h:t + h + 1]; w = w[np.isfinite(w)]
        if len(w) < 5:
            continue
        med = np.median(w); mad = np.median(np.abs(w - med))
        if mad <= eps:
            continue
        z = 0.6745 * (x[t] - med) / mad
        if np.isfinite(z) and abs(z) > z_thr:
            flag[t] = True
    # 邊界：頭尾 h 點以相鄰穩定段中位為基準
    ref_head = np.nanmedian(x[h:2 * h + 1]); ref_tail = np.nanmedian(x[n - 2 * h - 1:n - h])
    for t in range(0, h):
        if np.isfinite(x[t]) and abs(x[t] - ref_head) > max(abs(ref_head * edge_ratio), eps):
            flag[t] = True
    for t in range(n - h, n):
        if np.isfinite(x[t]) and abs(x[t] - ref_tail) > max(abs(ref_tail * edge_ratio), eps):
            flag[t] = True
    flag |= ~np.isfinite(x)
    if flag.any() and (~flag).sum() >= 2:
        idx = np.arange(n); x[flag] = np.interp(idx[flag], idx[~flag], x[~flag])
    return x, int(flag.sum())


def correlations(F, y):
    """每個特徵欄與 y 之 Pearson／Spearman；回傳 dict name → (r, rho)."""
    out = {}
    for c in F.columns:
        a = F[c].to_numpy(float); ok = np.isfinite(a) & np.isfinite(y)
        if ok.sum() < 5 or np.nanstd(a[ok]) == 0:
            out[c] = (np.nan, np.nan); continue
        out[c] = (float(pearsonr(a[ok], y[ok])[0]), float(spearmanr(a[ok], y[ok])[0]))
    return out


def dq_points(lo, hi, step=0.05):
    """F_dQ 之電壓格點（視窗下緣之後每 step V 一點；不含下緣本身）。"""
    n = int(np.floor((hi - lo) / step + 1e-9))
    return np.round(lo + step * np.arange(1, n + 1), 3)


def dq_features(q_filled, grid, window, step=0.05):
    """F_dQ（v9.11 新增）：部分充電容量增量 ΔQ(V) = Q(V) − Q(V_lo)，視窗內每 step V 一點（50 mV）。
    Q 為 fill_Q 後之累積充電容量（Ah）；V_lo 為視窗下緣最近格點。與 F_IC 同樣進 robust_z_clean。"""
    lo, hi = window
    j0 = int(np.argmin(np.abs(grid - lo)))
    q0 = q_filled[j0]
    out = {}
    for v in dq_points(lo, hi, step):
        j = int(np.argmin(np.abs(grid - v)))
        out[f"dQ_{v:.2f}"] = float(q_filled[j] - q0) if (np.isfinite(q_filled[j]) and np.isfinite(q0)) else np.nan
    return out


def seg_bounds(q_row, grid):
    """A1（v9.11）：該循環充電段 Q(V) 之第一個／最後一個「有限格點」電壓（未補值之原始 Q(V) 列）。
    回傳 (seg_first_V, seg_last_V)；全列無有限值回 (nan, nan)。"""
    ok = np.isfinite(np.asarray(q_row, float))
    if not ok.any():
        return float("nan"), float("nan")
    first = int(np.argmax(ok)); last = int(len(ok) - 1 - np.argmax(ok[::-1]))
    return float(grid[first]), float(grid[last])


def seg_valid(seg_first_V, seg_last_V, window, delta_V):
    """A1 充電段有效性閘：起始格點 ≤ V_lo+δ 且終止格點 ≥ V_hi−δ。δ=inf → 恆為有效（v9.9 行為）。
    seg_first_V／seg_last_V 為 nan（該循環無任何有限格點）→ 無效。"""
    lo, hi = window
    if not np.isfinite(delta_V):
        return True
    if not (np.isfinite(seg_first_V) and np.isfinite(seg_last_V)):
        return False
    return bool((seg_first_V <= lo + delta_V + 1e-9) and (seg_last_V >= hi - delta_V - 1e-9))


def window_coverage(Q, grid, window):
    """視窗覆蓋率（census 之 win_cov_bins 同義）：視窗內格點為有限值之比例（逐循環 Q(V) 原始表）。"""
    lo, hi = window
    m = (grid >= lo - 1e-9) & (grid <= hi + 1e-9)
    return float(np.isfinite(np.asarray(Q, float)[:, m]).mean())
