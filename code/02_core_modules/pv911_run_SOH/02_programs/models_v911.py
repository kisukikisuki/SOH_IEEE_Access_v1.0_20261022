"""
models_v911.py — 五模型（全部出自 A 類期刊）與軌跡外推器。
  GPR-SE  ：零均值＋SE 核（ARD 長度尺度 P）＋白噪聲，NLML 最佳化（22 Eq.8–17、23 Eq.6–19）
  GPR-LSP ：線性均值＋SE＋週期核（21 Eq.17–18；sklearn 以 DotProduct 核實現線性均值之貝氏等價）
  Ridge   ：多變數線性回歸＋Ridge 正則（20 Eq.5–6）
  SVR     ：05 之 RUL 基線；超參數沿 v9.5 凍結值（C=10、ε=0.01、γ=scale）
  MLP     ：25 之 ANN（2–6–5–1 → 隱藏層 (6,5)）；BP 訓練
外推器（架構二 B1／B2）：
  E21 GPR-LSP（x＝循環數）；E24 灰色模型 GM(1,1)＋多項式項（24 Eq.9–16，結構 N∈{0,1,2} 由訓練段末 25 % 留出誤差選、末點對齊）；E23 二次多項式（23 §3.2）
  AR-GPR（22 §3.2）：輸入 [SOH(k−2), SOH(k−1), SOH(k), RUL_pred(k−1)] → RUL(k)
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel, DotProduct, ExpSineSquared
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

SEED = 20260830
MODEL_NAMES = ["GPR-SE", "GPR-LSP", "Ridge", "SVR", "MLP"]
MODEL_SOURCE = {"GPR-SE": "22 Li 2019 JPS／23 Li 2020 JPS", "GPR-LSP": "21 Tang 2024 Sci Rep（線性均值＋SE＋週期核）",
                "Ridge": "20 Sanz-Gorrachategui 2021 TIM（MLR＋Ridge）", "SVR": "05 Zhou 2026 JES（RUL 基線）", "MLP": "25 Zhang 2019 JES（ANN 2–6–5–1）／20（MLP）"}


def gpr_se(d, restarts=3, seed=SEED):
    k = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1e-3, (1e-8, 1e1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=restarts, random_state=seed)


def gpr_lsp(d, restarts=3, seed=SEED, period_bounds=(0.05, 20.0)):
    k = (C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
         + C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
         + C(0.1, (1e-5, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=1.0, length_scale_bounds=(1e-2, 1e3), periodicity_bounds=period_bounds)
         + WhiteKernel(1e-3, (1e-8, 1e1)))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=restarts, random_state=seed)


# ── v9.11 新增：GPR-LSP 可調核（PLAN §3.1；登記於 config SEARCH_SPACE.soh.gpr_lsp）────
# 初值／界一律沿 v9.x 之 gpr_lsp（C=1、DotProduct σ0=1、RBF l=1、period 1（界 0.05–20）、White 1e-3，
# n_restarts=3、normalize_y=True）；本函式只加兩個開關，不改既有 gpr_lsp 之行為：
#   periodic=False → 去掉 ExpSineSquared 項（21 之週期核動機為抑制 nasa 之容量再生；ox／own 可能不需要）
#   ard=True       → SE（RBF）之 length_scale 改為長度 d 之向量（每維一個長度尺度，22／23 之 P＝{l²_i} 形式）
def gpr_lsp_v911(d, periodic=True, ard=False, restarts=3, seed=SEED, period_bounds=(0.05, 20.0)):
    ls = (np.ones(d) if ard else 1.0)
    k = (C(1.0, (1e-3, 1e3)) * DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3))
         + C(1.0, (1e-3, 1e3)) * RBF(length_scale=ls, length_scale_bounds=(1e-2, 1e3)))
    if periodic:
        k = k + C(0.1, (1e-5, 1e2)) * ExpSineSquared(length_scale=1.0, periodicity=1.0,
                                                     length_scale_bounds=(1e-2, 1e3), periodicity_bounds=period_bounds)
    k = k + WhiteKernel(1e-3, (1e-8, 1e1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=restarts, random_state=seed)


def make_model(name, d, seed=SEED, **kw):
    """kw（v9.11 新增，只對 GPR-LSP 生效）：periodic、ard、restarts、period_bounds；未給時行為與 v9.10 完全相同。"""
    if name == "GPR-SE":
        return Pipeline([("sc", StandardScaler()), ("m", gpr_se(d, seed=seed))])
    if name == "GPR-LSP":
        if kw:
            return Pipeline([("sc", StandardScaler()), ("m", gpr_lsp_v911(d, seed=seed, **kw))])
        return Pipeline([("sc", StandardScaler()), ("m", gpr_lsp(d, seed=seed))])
    if name == "Ridge":
        return Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=np.logspace(-4, 3, 29)))])
    if name == "SVR":
        return Pipeline([("sc", StandardScaler()), ("m", SVR(C=10.0, epsilon=0.01, gamma="scale"))])
    if name == "MLP":
        from sklearn.compose import TransformedTargetRegressor
        inner = Pipeline([("sc", StandardScaler()), ("m", MLPRegressor(hidden_layer_sizes=(6, 5), activation="tanh", solver="lbfgs", max_iter=5000, tol=1e-8, random_state=seed))])
        return TransformedTargetRegressor(regressor=inner, transformer=StandardScaler())   # 目標標準化（25 之 BP 訓練需輸出尺度一致）
    raise ValueError(name)


def fit_predict(name, Xtr, ytr, Xte, seed=SEED, sub=None, model_kwargs=None):
    """訓練並預測；GPR 回傳 (mean, std)，其他 (mean, None)。sub＝GPR 訓練子抽樣上限（登記；None＝不抽）。
    model_kwargs（v9.11 新增）：轉交 make_model；只對 GPR-LSP 生效，None／{} 時行為同 v9.10。"""
    Xtr = np.asarray(Xtr, float); ytr = np.asarray(ytr, float); Xte = np.asarray(Xte, float)
    if name.startswith("GPR") and sub is not None and len(ytr) > sub:
        rng = np.random.RandomState(seed); idx = np.sort(rng.choice(len(ytr), sub, replace=False)); Xtr, ytr = Xtr[idx], ytr[idx]
    m = make_model(name, Xtr.shape[1], seed, **(model_kwargs or {}))
    m.fit(Xtr, ytr)
    if name.startswith("GPR"):
        mu, sd = m.named_steps["m"].predict(m.named_steps["sc"].transform(Xte), return_std=True)
        return mu, sd, m
    return m.predict(Xte), None, m


# ── 外推器 ────────────────────────────────────────────────────────────────────
def extrap_gpr_lsp(t_tr, y_tr, t_te, seed=SEED):
    """E21：x＝循環數（尺度化）→ y；回傳 (mean, std)。"""
    X = np.asarray(t_tr, float)[:, None]; m = Pipeline([("sc", StandardScaler()), ("m", gpr_lsp(1, seed=seed))]); m.fit(X, y_tr)
    mu, sd = m.named_steps["m"].predict(m.named_steps["sc"].transform(np.asarray(t_te, float)[:, None]), return_std=True)
    return mu, sd


def _rlowess(y, span=15):
    """24 §4.2：rlowess 平滑（step 15）；以穩健局部線性（tricube 權重＋一次 bisquare 重加權）實作。"""
    y = np.asarray(y, float); n = len(y); out = y.copy()
    if n < span:
        return out
    h = span // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1); x = np.arange(lo, hi, dtype=float); yy = y[lo:hi]
        d = np.abs(x - i) / max(np.abs(x - i).max(), 1e-9); w = (1 - d ** 3) ** 3
        for _ in range(2):
            A = np.vstack([np.ones_like(x), x - i]).T * np.sqrt(w)[:, None]; b = yy * np.sqrt(w)
            coef, *_ = np.linalg.lstsq(A, b, rcond=None); r = yy - (coef[0] + coef[1] * (x - i))
            s = np.median(np.abs(r)) * 6 + 1e-12; w = w * np.clip(1 - (r / s) ** 2, 0, None) ** 2
        out[i] = coef[0]
    return out


def _grey_core(x, n_ahead, N):
    """單一結構 N 之 GM(1,N)：回傳長度 k+n_ahead 之還原序列 x̂（前 k 為樣本內擬合）。"""
    k = len(x); yacc = np.cumsum(x)
    rows = []; rhs = []
    for i in range(2, k + 1):
        z = 0.5 * (yacc[i - 1] + yacc[i - 2]); row = [z] + [(i ** (j + 1) - (i - 1) ** (j + 1)) / (j + 1) for j in range(N + 1)]
        rows.append(row); rhs.append(x[i - 1])
    A = np.asarray(rows); b = np.asarray(rhs)
    theta, *_ = np.linalg.lstsq(A, b, rcond=None); a = theta[0]; bs = theta[1:]
    T = k + n_ahead; yh = np.zeros(T); yh[0] = x[0]
    f = lambda tt, yy: a * yy + sum(bs[j] * tt ** j for j in range(N + 1))
    for t in range(1, T):
        k1 = f(t, yh[t - 1]); k2 = f(t + 0.5, yh[t - 1] + 0.5 * k1); k3 = f(t + 0.5, yh[t - 1] + 0.5 * k2); k4 = f(t + 1, yh[t - 1] + k3)
        yh[t] = yh[t - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return np.concatenate([[yh[0]], np.diff(yh)])


def grey_fit_predict(y_tr, n_ahead, N_candidates=(0, 1, 2)):
    """E24：GM(1,1)＋多項式項 dy/dt = a·y + Σ b_j t^j（24 Eq.9）；離散化 Eq.10、最小平方 Eq.13、還原 Eq.14–16。
    結構 N（24 之「自適應結構辨識」）以訓練段末 25 % 之留出誤差選定（前 75 % 擬合），再以全段重估；
    預測序列以訓練段末點對齊（消除 GM 擬合值與末觀測值之水平差）。回傳 (未來 n_ahead 步預測, N)。"""
    x = np.asarray(y_tr, float); k = len(x)
    if k < 8:
        return np.full(n_ahead, np.nan), None
    k_fit = max(int(round(0.75 * k)), 6); best = None
    for N in N_candidates:
        try:
            xh = _grey_core(x[:k_fit], k - k_fit, N)
        except np.linalg.LinAlgError:
            continue
        err = np.mean(np.abs(xh[k_fit:] - x[k_fit:]))
        if np.isfinite(err) and (best is None or err < best[0]):
            best = (err, N)
    if best is None:
        return np.full(n_ahead, np.nan), None
    N = best[1]
    try:
        xh = _grey_core(x, n_ahead, N)
    except np.linalg.LinAlgError:
        return np.full(n_ahead, np.nan), None
    pred = xh[k:] + (x[-1] - xh[k - 1])
    return pred, N


def _grey_fit_predict_legacy(y_tr, n_ahead, N_candidates=(0, 1, 2)):
    x = np.asarray(y_tr, float); k = len(x)
    if k < 6:
        return np.full(n_ahead, np.nan), None
    yacc = np.cumsum(x)
    best = None
    for N in N_candidates:
        rows = []; rhs = []
        for i in range(2, k + 1):                       # i = 2..k（Eq.10）
            z = 0.5 * (yacc[i - 1] + yacc[i - 2]); row = [z] + [(i ** (j + 1) - (i - 1) ** (j + 1)) / (j + 1) for j in range(N + 1)]
            rows.append(row); rhs.append(x[i - 1])
        A = np.asarray(rows); b = np.asarray(rhs)
        try:
            theta, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue
        a = theta[0]; bs = theta[1:]
        # 連續解：y(t) = e^{a(t-1)}·(x(1) − p(1)) + p(t)，p 為多項式特解 → 以數值積分（梯形）求解 dy/dt = a y + Σ b_j t^j，t=1..k+n_ahead
        T = k + n_ahead; yh = np.zeros(T); yh[0] = x[0]
        for t in range(1, T):
            f = lambda tt, yy: a * yy + sum(bs[j] * tt ** j for j in range(N + 1))
            tt = t; k1 = f(tt, yh[t - 1]); k2 = f(tt + 0.5, yh[t - 1] + 0.5 * k1); k3 = f(tt + 0.5, yh[t - 1] + 0.5 * k2); k4 = f(tt + 1, yh[t - 1] + k3)
            yh[t] = yh[t - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        xh = np.concatenate([[yh[0]], np.diff(yh)])          # 還原 Eq.16
        mape = np.mean(np.abs((xh[:k] - x) / np.where(np.abs(x) > 1e-9, x, 1e-9)))
        if not np.isfinite(mape):
            continue
        if best is None or mape < best[0]:
            best = (mape, N, xh)
    if best is None:
        return np.full(n_ahead, np.nan), None
    return best[2][k:], best[1]


def extrap_grey(t_tr, y_tr, t_te, span=15):
    """E24 完整流程：rlowess 平滑 → 灰色模型 → 外推到 t_te（要求 t 為連續整數循環）。"""
    ys = _rlowess(y_tr, span); t_tr = np.asarray(t_tr); t_te = np.asarray(t_te)
    n_ahead = int(t_te.max() - t_tr.max()); pred, N = grey_fit_predict(ys, max(n_ahead, 1))
    tt = t_tr.max() + np.arange(1, len(pred) + 1)
    return np.interp(t_te, tt, pred, left=pred[0] if len(pred) else np.nan, right=pred[-1] if len(pred) else np.nan), N


def extrap_poly(t_tr, y_tr, t_te, deg=2):
    """E23：多項式回歸（Eq.20）於循環數，最小平方；deg 登記為 2。"""
    t_tr = np.asarray(t_tr, float); t0 = t_tr.mean(); s = max(t_tr.std(), 1.0)
    coef = np.polyfit((t_tr - t0) / s, y_tr, deg)
    return np.polyval(coef, (np.asarray(t_te, float) - t0) / s)


# ── v9.11 新增：E24 可調參數版（PLAN_v911 §1.3／§3.2；登記於 config_v911.SEARCH_SPACE.rul.e24）──
# 文件命名（v9.11 更正）：E24 之模型為「自適應結構灰色模型（adaptive structure grey forecasting model，
# 24 §3.1、§5 原文名）」；「GM(1,1)＋多項式項 N」為 v9.x 之實作形式，非原文用語（LIT §D-1-2：譯本全篇
# 未出現 GM(1,1) 字樣）。原文未明述之三項（多項式階數 N、背景值構造、尾端對齊）在 v9.x 實作中之處理
# 一律揭露：N 由訓練段留出誤差選（原文未明述）、背景值以係數 α 參數化（原文未明述）、
# 預測序列以訓練段末點對齊（原文未明述）。
E24_NAME = "自適應結構灰色模型（24 §3.1；v9.x 實作：GM(1,1)＋多項式項 N）"
E24_UNSPECIFIED = ("24 原文（譯本）未明述者，v9.x 實作之處理全部揭露：(i) 多項式項階數 N —— 原文只稱"
                   "「自適應結構」，v9.x 以訓練段末 25 % 留出誤差自 N_cands 選定；(ii) 背景值構造 —— "
                   "原文未載，v9.x 以 z(k)=α·x1(k)+(1−α)·x1(k−1) 參數化，α=0.5 為標準緊鄰均值；"
                   "(iii) 尾端對齊 —— 原文未載，v9.x 以訓練段末點水平對齊（pred += x[-1] − x̂[k−1]）。")
GREY_BACKGROUND_FORMULA = ("背景值（background value）z(k) = α·x1(k) + (1−α)·x1(k−1)，x1＝一次累加（1-AGO）序列；"
                           "α＝0.5 即標準緊鄰均值生成（與 v9.10 之 _grey_core 逐字等價）；"
                           "v9.11 搜尋維度 α ∈ {0.5（標準）, 0.4, 0.6}。24 原文未明述背景值構造（LIT §D-1-2），"
                           "故以此等價參數化加入並登記。")
GREY_NFIT_NOTE = ("灰色擬合視窗 n_fit（v9.11 新增）：只取訓練段（1..u）之末 n_fit 點做灰色擬合，"
                  "結構 N 之留出選擇亦只在該視窗內進行（前 75 % 擬合、末 25 % 留出）；n_fit='all' 即沿 v9.10 用全訓練段。"
                  "動機：24 §3.1 明述灰色模型適合小樣本建模，用全部 1..u 點擬合違反其設計（LIT §D-1-2）。"
                  "n_fit 只回看訓練段，不含任何未來資訊。")
E24_DEFAULTS = dict(span=15, n_fit=None, N_cands=(0, 1, 2), alpha=0.5)


def _grey_core_v911(x, n_ahead, N, alpha=0.5):
    """單一結構 N 之灰色核（背景值係數 α；α=0.5 與 _grey_core 逐字等價）。
    回傳長度 k+n_ahead 之還原序列 x̂（前 k 為樣本內擬合）。"""
    x = np.asarray(x, float); k = len(x); yacc = np.cumsum(x)
    rows = []; rhs = []
    for i in range(2, k + 1):
        z = alpha * yacc[i - 1] + (1.0 - alpha) * yacc[i - 2]        # GREY_BACKGROUND_FORMULA
        row = [z] + [(i ** (j + 1) - (i - 1) ** (j + 1)) / (j + 1) for j in range(N + 1)]
        rows.append(row); rhs.append(x[i - 1])
    A = np.asarray(rows); b = np.asarray(rhs)
    theta, *_ = np.linalg.lstsq(A, b, rcond=None); a = theta[0]; bs = theta[1:]
    T = k + n_ahead; yh = np.zeros(T); yh[0] = x[0]
    f = lambda tt, yy: a * yy + sum(bs[j] * tt ** j for j in range(N + 1))
    for t in range(1, T):
        k1 = f(t, yh[t - 1]); k2 = f(t + 0.5, yh[t - 1] + 0.5 * k1); k3 = f(t + 0.5, yh[t - 1] + 0.5 * k2); k4 = f(t + 1, yh[t - 1] + k3)
        yh[t] = yh[t - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return np.concatenate([[yh[0]], np.diff(yh)])


def grey_fit_predict_v911(y_tr, n_ahead, N_candidates=(0, 1, 2), n_fit=None, alpha=0.5):
    """E24 之灰色擬合／外推（v9.11 可調版）。n_fit=None／'all'／>=len → 用全訓練段（＝v9.10 grey_fit_predict）。
    結構 N 以（視窗內）末 25 % 留出誤差選定，再以（視窗內）全段重估；末點對齊。回傳 (未來 n_ahead 步預測, N)。"""
    x_all = np.asarray(y_tr, float)
    if n_fit is not None and not (isinstance(n_fit, str) and str(n_fit) == "all"):
        nf = int(n_fit)
        x = x_all[-nf:] if nf < len(x_all) else x_all
    else:
        x = x_all
    k = len(x)
    if k < 8:
        return np.full(n_ahead, np.nan), None
    k_fit = max(int(round(0.75 * k)), 6); best = None
    for N in N_candidates:
        try:
            xh = _grey_core_v911(x[:k_fit], k - k_fit, N, alpha)
        except np.linalg.LinAlgError:
            continue
        err = np.mean(np.abs(xh[k_fit:] - x[k_fit:]))
        if np.isfinite(err) and (best is None or err < best[0]):
            best = (err, N)
    if best is None:
        return np.full(n_ahead, np.nan), None
    N = best[1]
    try:
        xh = _grey_core_v911(x, n_ahead, N, alpha)
    except np.linalg.LinAlgError:
        return np.full(n_ahead, np.nan), None
    return xh[k:] + (x[-1] - xh[k - 1]), N


def extrap_grey_v911(t_tr, y_tr, t_te, span=15, n_fit=None, N_cands=(0, 1, 2), alpha=0.5):
    """E24 完整流程（v9.11 可調版）：rlowess(span) 平滑（只作用訓練段）→ 灰色擬合（視窗 n_fit、背景值 α、
    結構候選 N_cands）→ 外推到 t_te。span=15、n_fit=None、N_cands=(0,1,2)、alpha=0.5 時與 v9.10
    extrap_grey 逐值等價。回傳 (mean, N)。"""
    ys = _rlowess(np.asarray(y_tr, float), int(span))
    t_tr = np.asarray(t_tr); t_te = np.asarray(t_te)
    n_ahead = int(t_te.max() - t_tr.max())
    pred, N = grey_fit_predict_v911(ys, max(n_ahead, 1), N_candidates=tuple(N_cands), n_fit=n_fit, alpha=float(alpha))
    tt = t_tr.max() + np.arange(1, len(pred) + 1)
    return np.interp(t_te, tt, pred, left=pred[0] if len(pred) else np.nan,
                     right=pred[-1] if len(pred) else np.nan), N
