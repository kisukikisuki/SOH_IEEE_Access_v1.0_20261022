"""s6_cluster_bootstrap.py — 審稿第 7 項／步驟書 §7 列 7-a（cluster bootstrap＋MixedLM；後處理，不重訓）
複製自：無（新撰）；邏輯引用 ROOT\\5版\\02_程式\\03_審查後處理\\addendum_C2.py
  L199–205 units()：d＝PBplus_RMSE_pct−ZOH_RMSE_pct；cluster＝非 Oxford 逐顆、Oxford 每組（group＝cell.split('__')[0]）
  L107–108 subsets()：Measured-label subset (25)＝source!='ox' 或 G7–G10；All cells (34)
  Own 批次：post_review\\c2_own_campaign_split.csv 之 campaign 欄（cell2/cell3 vs GA1-GA7；addendum L117）
只改：(1) Own 由「逐顆」改為「批次 cluster」（步驟書 §7 7-a 指定）；(2) 以 cluster bootstrap／MixedLM 取代 Wilcoxon。
輸入：--a04 <A04_reference_only_baselines_percell.csv>  --camp <c2_own_campaign_split.csv>
輸出（--out 目錄）：c7_cluster_bootstrap.csv、c7_mixed_effects.json、c7_S14_table_data.csv、_run_s6_cluster_bootstrap.json
事後聲明：事後（post hoc）、未調整、不在 50 項 Holm 檢定族。

統計式（寫定於執行前）：
  θ̂ ＝ 子集內逐顆 d_i 之平均（每顆等權）。cluster g 之 S_g＝Σ_{i∈g} d_i、n_g＝顆數。
  bootstrap：對 G 個 cluster 有放回抽 G 個（B=10000；每「子集×M」各以 numpy.random.default_rng(20260830) 新建），
    θ*_b ＝ Σ_k S_{g_bk} / Σ_k n_{g_bk}。percentile CI＝θ* 之 2.5／97.5 百分位（numpy.quantile, method='linear'）。
  BCa（Efron 1987）：z0＝Φ⁻¹( (#{θ*<θ̂} + 0.5·#{θ*=θ̂}) / B )；
    acceleration a 以 cluster 為單位之 jackknife：θ_(−g)＝去掉 cluster g 全部顆後之逐顆平均，θ_(·)＝mean_g θ_(−g)，
    a＝Σ_g (θ_(·)−θ_(−g))³ / { 6·[Σ_g (θ_(·)−θ_(−g))²]^{3/2} }；
    α_j＝Φ( z0 + (z0+z_j)/(1−a(z0+z_j)) )，z_j＝Φ⁻¹(0.025)、Φ⁻¹(0.975)；CI＝θ* 之 α_1、α_2 百分位。
  MixedLM：statsmodels（_vendor，--no-deps 安裝）smf.mixedlm('d ~ 1', groups=cluster).fit(reml=True)；
    報截距、SE、Wald 95% CI（fit.conf_int）、cluster 變異（cov_re）、殘差變異（scale）、converged、warnings。
"""
import os, sys, json, argparse, hashlib, warnings
from datetime import datetime, timezone
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, '_vendor')
sys.path.insert(0, VENDOR)
import numpy as np, pandas as pd
from scipy import stats

SEED = 20260830; B = 10000; MS = (10, 20)
MEAS_GROUPS = ('G7', 'G8', 'G9', 'G10'); INTERP_GROUPS = ('G1', 'G3', 'G5')
FEW_CLUSTER_MAX = 4   # n_clusters ≤ 4 標「cluster 過少，CI 不可靠」（揭露旗標，非檢定）


def sha(fp):
    h = hashlib.sha256()
    with open(fp, 'rb') as f:
        for ch in iter(lambda: f.read(1 << 20), b''):
            h.update(ch)
    return h.hexdigest()


def load(a04_fp, camp_fp):
    a = pd.read_csv(a04_fp)
    a['group'] = [c.split('__')[0] if s == 'ox' else '' for s, c in zip(a.source, a.cell)]
    camp = pd.read_csv(camp_fp)
    cmap = camp.drop_duplicates('cell').set_index('cell').campaign.to_dict()
    def cl(r):
        if r.source == 'nasa': return 'NASA_' + r.cell
        if r.source == 'own': return 'OWN_' + cmap[r.cell]
        return 'OX_' + r.group
    a['cluster'] = a.apply(cl, axis=1)
    a['d'] = a.PBplus_RMSE_pct - a.ZOH_RMSE_pct
    return a, camp


def subsets(am):
    return {'Measured-label subset (25)': am[(am.source != 'ox') | am.group.isin(MEAS_GROUPS)],
            'All cells (34)': am,
            'NASA': am[am.source == 'nasa'],
            'Own': am[am.source == 'own'],
            'Oxford (21 cells)': am[am.source == 'ox'],
            'Oxford G1/G3/G5 (interpolated, 9)': am[(am.source == 'ox') & am.group.isin(INTERP_GROUPS)]}


def cluster_boot(df):
    g = df.groupby('cluster')['d'].agg(['sum', 'size'])
    S = g['sum'].to_numpy(float); n = g['size'].to_numpy(float); G = len(S)
    theta = float(df.d.mean())
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, G, size=(B, G))
    ts = S[idx].sum(1) / n[idx].sum(1)
    pct = np.quantile(ts, [0.025, 0.975])
    jk = np.array([(S.sum() - S[k]) / (n.sum() - n[k]) for k in range(G)]) if G > 1 else np.array([np.nan])
    dj = jk.mean() - jk
    den = 6.0 * (np.sum(dj ** 2)) ** 1.5
    a = float(np.sum(dj ** 3) / den) if den > 0 else 0.0
    prop = (np.sum(ts < theta) + 0.5 * np.sum(ts == theta)) / B
    out = dict(theta=theta, G=G, boot_mean=float(ts.mean()), boot_se=float(ts.std(ddof=1)),
               pct_lo=float(pct[0]), pct_hi=float(pct[1]), z0=np.nan, accel=a, bca_lo=np.nan, bca_hi=np.nan,
               bca_alpha_lo=np.nan, bca_alpha_hi=np.nan, n_unique_boot_values=int(len(np.unique(np.round(ts, 12)))),
               jackknife={k: float(v) for k, v in zip(g.index, jk)}, cluster_sizes={k: int(v) for k, v in zip(g.index, n)},
               cluster_means={k: float(s / m) for k, s, m in zip(g.index, S, n)})
    if 0 < prop < 1:
        z0 = float(stats.norm.ppf(prop)); al = []
        for zj in (stats.norm.ppf(0.025), stats.norm.ppf(0.975)):
            al.append(float(stats.norm.cdf(z0 + (z0 + zj) / (1 - a * (z0 + zj)))))
        q = np.quantile(ts, al)
        out.update(z0=z0, bca_lo=float(q[0]), bca_hi=float(q[1]), bca_alpha_lo=al[0], bca_alpha_hi=al[1])
    return out


def mixed(df):
    try:
        import statsmodels, statsmodels.formula.api as smf
    except Exception as e:
        return dict(status='未能執行', error=repr(e))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            md = smf.mixedlm('d ~ 1', data=df[['d', 'cluster']].reset_index(drop=True), groups='cluster')
            fit = md.fit(reml=True)
            ci = fit.conf_int().loc['Intercept'].to_numpy(float)
            res = dict(status='ok', statsmodels_version=statsmodels.__version__, method='REML',
                       intercept=float(fit.fe_params['Intercept']), intercept_se=float(fit.bse_fe['Intercept']),
                       ci95_wald=[float(ci[0]), float(ci[1])], p_wald_z=float(fit.pvalues['Intercept']),
                       var_cluster=float(fit.cov_re.iloc[0, 0]), var_resid=float(fit.scale),
                       converged=bool(fit.converged), llf_reml=float(fit.llf), n_obs=int(fit.nobs), n_groups=int(len(fit.model.group_labels)))
        except Exception as e:
            res = dict(status='fit_failed', error=repr(e))
    res['warnings'] = sorted({f'{x.category.__name__}: {x.message}' for x in w})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a04', required=True); ap.add_argument('--camp', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='formal')
    A = ap.parse_args()
    t0 = datetime.now(timezone.utc); os.makedirs(A.out, exist_ok=True)
    a, camp = load(A.a04, A.camp)
    checks = {}
    # 回歸／一致性檢查：Own 列之 PBplus、ZOH 與 c2_own_campaign_split.csv 逐值
    mx = 0.0
    for _, r in camp.iterrows():
        x = a[(a.source == 'own') & (a.cell == r.cell) & (a.M == r.M)].iloc[0]
        mx = max(mx, abs(x.PBplus_RMSE_pct - r.PBplus), abs(x.ZOH_RMSE_pct - r.ZOH))
    checks['own_vs_campaign_csv_max_abs_diff'] = float(mx)
    checks['own_vs_campaign_csv_pass_1e-9'] = bool(mx <= 1e-9)
    for M in MS:
        am = a[a.M == M]
        checks[f'M{M}_n_cells'] = int(len(am))
        checks[f'M{M}_n_clusters_all'] = int(am.cluster.nunique())
        checks[f'M{M}_n_clusters_measured'] = int(subsets(am)['Measured-label subset (25)'].cluster.nunique())
    rows, mixed_out, boot_detail = [], {}, {}
    for M in MS:
        for sn, df in subsets(a[a.M == M]).items():
            bt = cluster_boot(df); me = mixed(df)
            key = f'M{M}|{sn}'
            boot_detail[key] = {k: bt[k] for k in ('jackknife', 'cluster_sizes', 'cluster_means')}
            mixed_out[key] = me
            rows.append(dict(M=M, subset=sn, n_cells=int(len(df)), n_clusters=bt['G'], clusters='|'.join(sorted(df.cluster.unique())),
                             mean_diff_PBplus_minus_ZOH=bt['theta'], n_cells_PBplus_lower=int((df.d < 0).sum()),
                             boot_B=B, boot_seed=SEED, boot_mean=bt['boot_mean'], boot_se=bt['boot_se'],
                             ci95_percentile_lo=bt['pct_lo'], ci95_percentile_hi=bt['pct_hi'],
                             bca_z0=bt['z0'], bca_accel=bt['accel'], bca_alpha_lo=bt['bca_alpha_lo'], bca_alpha_hi=bt['bca_alpha_hi'],
                             ci95_bca_lo=bt['bca_lo'], ci95_bca_hi=bt['bca_hi'], n_unique_boot_values=bt['n_unique_boot_values'],
                             mixedlm_status=me.get('status'), mixedlm_intercept=me.get('intercept'), mixedlm_se=me.get('intercept_se'),
                             mixedlm_ci95_lo=(me.get('ci95_wald') or [None, None])[0], mixedlm_ci95_hi=(me.get('ci95_wald') or [None, None])[1],
                             mixedlm_var_cluster=me.get('var_cluster'), mixedlm_var_resid=me.get('var_resid'), mixedlm_converged=me.get('converged'),
                             mixedlm_n_warnings=len(me.get('warnings', [])), mixedlm_warnings='; '.join(me.get('warnings', [])),
                             mixedlm_identifiable=bool(len(df) > bt['G']),   # 每 cluster 僅 1 顆時 cluster 變異與殘差不可分（NASA）
                             few_clusters_unreliable=bool(bt['G'] <= FEW_CLUSTER_MAX),
                             post_hoc='事後（post hoc）、未調整、不在 50 項 Holm 檢定族'))
    tab = pd.DataFrame(rows)
    def _note(r):   # S14 註記欄（主代理指示）：依輸出旗標與 statsmodels 警告文字組成
        n = []
        if not r.mixedlm_identifiable: n.append('MixedLM 不可識別（每 cluster 1 顆）')
        if r.mixedlm_converged is False or r.mixedlm_converged == False: n.append('MixedLM 未收斂')
        if 'boundary' in str(r.mixedlm_warnings): n.append('MixedLM 邊界解（cluster 變異≈0）')
        if r.few_clusters_unreliable: n.append(f'cluster ≤{FEW_CLUSTER_MAX}，CI 不可靠')
        return '；'.join(n)
    tab['S14_note'] = tab.apply(_note, axis=1)
    tab.to_csv(os.path.join(A.out, 'c7_cluster_bootstrap.csv'), index=False, encoding='utf-8')
    s14 = tab[['subset', 'M', 'n_cells', 'n_clusters', 'mean_diff_PBplus_minus_ZOH', 'ci95_percentile_lo', 'ci95_percentile_hi',
               'ci95_bca_lo', 'ci95_bca_hi', 'mixedlm_intercept', 'mixedlm_ci95_lo', 'mixedlm_ci95_hi', 'mixedlm_var_cluster',
               'mixedlm_var_resid', 'mixedlm_identifiable', 'mixedlm_warnings', 'few_clusters_unreliable', 'S14_note']]
    s14.to_csv(os.path.join(A.out, 'c7_S14_table_data.csv'), index=False, encoding='utf-8')
    json.dump(dict(tag=A.tag, model='d_i ~ 1 + (1|cluster), REML', post_hoc='事後（post hoc）、未調整、不在 50 項 Holm 檢定族',
                   d_definition='PBplus_RMSE_pct − ZOH_RMSE_pct（百分點；負值＝P-B+ 較低）', fits=mixed_out, bootstrap_detail=boot_detail,
                   checks=checks), open(os.path.join(A.out, 'c7_mixed_effects.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
    t1 = datetime.now(timezone.utc)
    json.dump(dict(tag=A.tag, argv=sys.argv, start_utc=t0.isoformat(), end_utc=t1.isoformat(), wall_sec=(t1 - t0).total_seconds(),
                   inputs={A.a04: sha(A.a04), A.camp: sha(A.camp)}, program={os.path.abspath(__file__): sha(os.path.abspath(__file__))},
                   numpy=np.__version__, pandas=pd.__version__, checks=checks),
              open(os.path.join(A.out, '_run_s6_cluster_bootstrap.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
    print(json.dumps(checks, ensure_ascii=False))
    print(tab[['M', 'subset', 'n_cells', 'n_clusters', 'mean_diff_PBplus_minus_ZOH', 'ci95_percentile_lo', 'ci95_percentile_hi', 'ci95_bca_lo', 'ci95_bca_hi',
               'mixedlm_intercept', 'mixedlm_ci95_lo', 'mixedlm_ci95_hi', 'mixedlm_var_cluster', 'mixedlm_var_resid', 'mixedlm_converged', 'mixedlm_n_warnings']].to_string())


if __name__ == '__main__':
    main()
