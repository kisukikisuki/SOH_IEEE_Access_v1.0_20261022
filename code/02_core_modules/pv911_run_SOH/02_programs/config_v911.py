"""
config_v911.py — v9.11（雙架構層級修訂）全部登記參數（先驗登記 v16 之內容；登記後不改）。

內容涵蓋 PLAN_v911 §1–§6：
  §1 資料與納入規則（沿 v9.7／v9.8）
  §2 可搜尋之細部（搜尋空間；搜尋結果導向＝R26 揭露）
  §3 選擇規則
  §4 執行（階段、環境、checkpoint）
  §5 判準修正（RUL 主比較集合、四指標定義、排除未來資訊、循環軸）
  §6 SELECT（Fable 依 §3 自 search 結果定案之主設定；filter 欄以 "GS"/"SG" ＋ sigma_mV 正確登記）
"""

SEED = 20260830
VERSION = "v9.11"
PREREG = "v17"

# ── §1 資料 ────────────────────────────────────────────────────────────────────
DATA = dict(
    dataset="pv95_run/dataset_v9（40 顆 npz；Q(V) 5 mV 網格＝充電段累積容量）",
    own_capacity_dir="pv98_run/data_own_capacity（老師電芯 NCR18650GA 1–7 逐循環容量表；唯讀引用）",
    grids={"nasa": [3.70, 4.25], "own": [3.70, 4.20], "ox": [3.50, 4.25]},
    sources_soh=["nasa", "own", "ox"],
    sources_rul=["nasa", "own", "ox", "GA"],
    journal_nasa_cells=["B0005", "B0006", "B0007", "B0018"],
    rated_Ah={"nasa": 2.0, "own": 3.35},
    eol_defs={"EOL_A": "80 % Q_ref（初始容量；主定義）",
              "EOL_B": "70 % 額定（nasa 1.4 Ah；21／05）",
              "EOL_C": "80 % 額定（nasa 1.6 Ah；24）"},
    min_cycles_for_start=100, min_soh_range=0.05, min_window_cov=0.90,
    inclusion="同 v13 census：架構一 27 顆（nasa 4、ox 21、own 2）；架構二有 EOL 23 顆＋右設限 10 顆＋GA 6 顆（GA4 86 循環不起算）",
    own_gap_rule="own_gap_rule_v98（gaps_v911.apply_rule；dsoh_thr=0.010、med_win=5、med_thr=0.010）：斷點過渡循環拉掉、cyc_c 連續重編；主結果於 cyc_c，cyc_A 並存（§5.4）",
    gap_params=dict(dsoh_thr=0.010, med_win=5, med_thr=0.010),
)

# ── §2 搜尋空間（R26：搜尋結果導向，全部 leaderboard 揭露）──────────────────────
# ── 濾波／正規化之登記字典（v9.11；供每來源前處理搜尋以字串鍵登記）────────────
# SG 之 (視窗點數, 階數)：SG51＝21 之凍結值（視窗 51 點、階數 3；21 明述視窗取奇數以避免邊界效應、
#   階數未明述 → 沿 v9.x 之 sg_order=3 並揭露）；SG11＝v9.x 既有對照值。
# GS 之 σ 以 mV 登記（網格 5 mV）：GS5＝22（σ=5）／25（σ=3，放電段）之量級；GS10／GS15／GS20 為 v15 網格值。
FILTERS = {
    "GS5":  dict(filter="GS", sigma_mV=5.0,  sg_window_pts=None, source="22 Li 2019 JPS（μ=17、σ=5）／25（σ=3）之量級"),
    "GS10": dict(filter="GS", sigma_mV=10.0, sg_window_pts=None, source="v15 網格"),
    "GS15": dict(filter="GS", sigma_mV=15.0, sg_window_pts=None, source="v15 網格（nasa v16 選定）"),
    "GS20": dict(filter="GS", sigma_mV=20.0, sg_window_pts=None, source="v15 網格（own v16 選定）"),
    "SG51": dict(filter="SG", sigma_mV=None, sg_window_pts=51, source="21 Tang 2024 Sci Rep（視窗長度 51，奇數）"),
    "SG11": dict(filter="SG", sigma_mV=None, sg_window_pts=11, source="v9.x 既有 SG 對照值"),
}
# 正規化：REL_n＝每顆每特徵除以該顆「前 n 個循環之中位數」（只用訓練段初始循環）
NORMS = {"none": None, "REL3": 3, "REL5": 5, "REL": 3}   # "REL"＝v15 舊鍵，等同 REL3（相容保留）

SEARCH_SPACE = dict(
    soh=dict(
        delta_V=[0.0, 0.05, 0.10, float("inf")],
        delta_keys=["d0", "d50", "d100", "inf"],
        label_gate=["off", "on"],
        label_gate_by_source={"own": ["on"], "nasa": ["off", "on"], "ox": ["off", "on"]},
        label_gate_note=("A2 標籤一致性閘改為每來源可搜尋維度（Fable 判讀後之追加，2026-09-20；仍在登記前，"
                         "屬 R26 結果導向搜尋，全部揭露）：own 固定 on（沿 v9.8，不搜）；nasa／ox 兩者都跑。"
                         "理由：nasa 之 SOH 階梯為靜置後容量再生（物理現象，非量測協定異常），"
                         "拉掉並重編 cyc_c 反而扭曲了作為特徵之循環軸（B0006 v9.9 P-B 1.103 % → A2 後 base 1.240 %；"
                         "Fable 探針無 A2、原 cyc 軸之 P-B+ GPR-LSP 為 0.949 %）。"
                         "on＝拉掉過渡循環且循環軸用 cyc_c；off＝不剔除且循環軸用 cyc_A（原物理循環）；"
                         "兩者 frames 皆保留 drop_flag 欄，另加 label_gate 欄供 verify 用。"),
        M=[10, 20], M_main=20,
        M_cancelled=dict(M=50, reason=("v9.11 取消 M=50（登記）：M=50 時 ox 多顆之非參考列區塊跨越過大、"
                                       "參考點過稀，v9.10 已顯示其逐顆 RMSE 一律劣於 M=20，且正式階段時間預算"
                                       "優先給每來源前處理之五模型全跑。M ∈ {10, 20} 皆正式跑，主 M=20。")),
        models=["GPR-SE", "GPR-LSP", "Ridge", "SVR", "MLP"],
        base_config="沿 v15 SOH_SELECTED（視窗／濾波／特徵族／正規化／min_rho／top_k／循環軸不重搜；PLAN §2.1 A4）",
        protocol="P-B+（稀疏參考更新；LOOCV 預訓練＋因果殘差 GPR 校正）；主 M=20",
        stages="單階：每來源 × δ(4) × M=20 × 五模型 × LOOCV 全部顆",
        reduction=("計算成本例外（沿 v15 縮減先例）：ox 之 GPR-SE 單一 P-B 單元於 v15 量得 3618 s，"
                   "本版 ox 只跑 Ridge／SVR／GPR-LSP／MLP，GPR-SE 略過並於 search/_notes.json 揭露。"),

        # ── v9.11 每來源前處理網格（PLAN_v911 §3.1；主模型固定 GPR-LSP、協定 P-B+ M=20）──
        # base＝v16 選定值（PLAN §3.1 表之 *）；alts＝該維其餘候選。搜尋為「一次一維」＋階 2 全組合前 8 名。
        v911=dict(
            main_model="GPR-LSP", protocol="P-B+", M_main=20,
            dims=["window", "filter_key", "family", "norm", "label_gate", "delta_key", "periodic", "ard", "sub", "top_k"],
            grid_by_source={
                "nasa": dict(window=[[3.80, 4.10], [3.75, 4.15]],
                             filter_key=["GS15", "GS5", "SG51", "SG11"],
                             family=["F_both", "F_IC"], norm=["REL3", "REL5", "none"],
                             label_gate=["off", "on"], delta_key=["inf", "d50"],
                             periodic=[True, False], ard=[False, True], sub=[2000], top_k=[12]),
                "own": dict(window=[[3.70, 4.20], [3.80, 4.10], [3.75, 4.15]],
                            filter_key=["GS20", "GS15", "SG51"],
                            family=["F_both", "F_IC"], norm=["none", "REL3"],
                            label_gate=["on"], delta_key=["d0", "d50"],
                            periodic=[True, False], ard=[False, True], sub=[2000], top_k=[12]),
                "ox": dict(window=[[3.50, 4.10], [3.60, 4.00], [3.55, 4.05]],
                           filter_key=["GS10", "GS5", "SG51"],
                           family=["F_IC", "F_both"], norm=["none", "REL3"],
                           label_gate=["on", "off"], delta_key=["d0", "d50"],
                           periodic=[True, False], ard=[False, True], sub=[1000, 2000], top_k=[12]),
            },
            base_note=("每維之第一個值＝v16 選定值（基準）：nasa window 3.80–4.10／GS15／F_both／REL3／A2 off／δ=inf；"
                       "own 3.70–4.20／GS20／F_both／none／A2 on／δ=0；ox 3.50–4.10／GS10／F_IC／none／A2 on／δ=0。"
                       "核組態基準＝週期核 on、ARD off（21 之原文核型式；21 未述 ARD）；"
                       "sub 基準＝nasa／own 2000（全量 ≤ 2000）、ox 1000（v15 之值）。"),
            stages=("階 1「一次一維」：以基準組態為中心，每維各換一個候選值各跑一單元"
                    "（nasa 11、own 9、ox 11 個變體＋基準 1）；"
                    "階 2：取階 1 各維之最佳值，做「基準值 × 最佳值」之全組合，依階 1 之逐顆 RMSE 平均"
                    "加成排序取前 8 名加跑（已於階 1 跑過者不重跑）。"),
            win_cov_rule=("新增視窗（nasa 3.75–4.15 等）沿 v15 規則做覆蓋率檢查："
                          "逐顆 win_cov＝該顆逐循環 Q(V) 於視窗內格點之有限值比例（ic_v911.window_coverage）；"
                          "任一顆 < 0.90（DATA.min_window_cov）→ 該視窗候選標記跳過並於 _notes.json 揭露。"),
            gpr_lsp=dict(impl="models_v911.gpr_lsp_v911（新增；不改既有 gpr_lsp）",
                         init=dict(const=1.0, const_bounds=[1e-3, 1e3],
                                   dot_sigma0=1.0, dot_sigma0_bounds=[1e-3, 1e3],
                                   rbf_length_scale=1.0, rbf_length_scale_bounds=[1e-2, 1e3],
                                   periodic_const=0.1, periodic_const_bounds=[1e-5, 1e2],
                                   periodic_length_scale=1.0, periodicity=1.0, periodicity_bounds=[0.05, 20.0],
                                   white=1e-3, white_bounds=[1e-8, 1e1],
                                   n_restarts_optimizer=3, normalize_y=True),
                         periodic="on＝含 ExpSineSquared 項（21 原文）；off＝去掉該項",
                         ard="on＝RBF 之 length_scale 為長度 d 之向量（22／23 之 P＝{l²_i} 形式）；off＝純量（21 原文）",
                         sub="GPR 訓練子抽樣上限（以 SEED 固定抽樣）",
                         source="21 Tang 2024 Sci Rep 式 17–18（線性均值＋SE＋週期核）；初值／ARD／噪聲原文未明述（LIT §C-1）"),
            budget=dict(total_sec=5400, note="總搜尋 ≤ 90 min；超出先縮減階 2（再縮減階 1 之高成本單元），全部於 _notes.json 揭露"),
        ),
    ),
    soh_v15_grid=dict(
        windows={"nasa": [[3.80, 4.10], [3.75, 4.15], [3.80, 4.20], [3.70, 4.20]],
                 "own": [[3.80, 4.10], [3.75, 4.15], [3.80, 4.15], [3.70, 4.20]],
                 "ox": [[3.60, 4.00], [3.55, 4.05], [3.60, 4.10], [3.50, 4.10]]},
        filters=[["GS", 10.0], ["GS", 15.0], ["GS", 20.0], ["SG", None]],
        families=["F_IC", "F_dQ", "F_both"], norms=["none", "REL"],
        sel_min_rho=0.85, top_k=[6, 12], cyc_axes=["cyc", "efc", "none"],
        note="v15（prereg v15）之 SOH 組態網格；v9.11 不重搜，選定值見 SOH_SELECTED。",
    ),
    rul=dict(
        starts_fixed=[60, 80, 100], starts_ratio=[0.30, 0.50], starts_late=[0.70, 0.80],
        extraps=["E21", "E24", "E23", "SVR", "DEXP", "LIN20", "LIN30", "LIN50", "POW", "MED", "SIM"],
        smooths=["none", "rlowess15"],
        knee=["off", "on"], knee_applicable=["LIN20", "LIN30", "LIN50", "POW"],
        k=[20, 10], k_main=20,
        update_protocol=("滑動視窗（PLAN §2.2 B1）：更新點 u = s, s+k, …（有 EOL：u ≤ EOL_true−1；右設限：u ≤ 資料末−1）；"
                         "每更新點只用 1..u（前處理只作用於訓練段）擬合外推器，預測 (u, u+k] 之容量與 EOL 交點"
                         "（cap_fut 單調化後 first_cross）。固定／比例／晚期起算點皆用同一滑動協定（一致並揭露）。"),
        stages="搜尋：每來源 × 主比較集合案例 × 11 外推器 × 2 前處理 × knee × k=20（knee 只對 LIN_n／POW 展開）",
        sim=("SIM＝相似度軌跡庫：庫為同來源其餘「有 EOL_true（依該來源主門檻）」之顆之 SOH(t)（rlowess15＋單調化 "
             "np.minimum.accumulate）；α ∈ [0.4, 2.5] 對數等距 31 點、偏移 b＝觀測後半段（max(10, n/2) 點）殘差平均、"
             "貼合度 r＝RMSE、權重 exp(−(r/σ)²)（σ＝r 中位數，下限 1e-4）、預測＝逐 t 加權平均（忽略 NaN）、"
             "庫曲線耗盡處以末 20 點線性延伸。own 與 GA 同型 NCR18650GA → 聯合庫（揭露）；"
             "nasa 因主門檻 EOL_B＝1.4 Ah 固定且各顆 qref 不同，庫與匹配、交點皆在 Ah 軸，其餘來源在 SOH 比值軸。"
             "測試顆本身只用 1..u（無未來資訊）；庫顆使用其全程資料（跨顆先驗，登記揭露）。"),
        hyb=("HYB-X（雙時距外推；搜尋期依結果新增，R26 揭露、登記前）：容量指標取自局部外推器 X、"
             "EOL 交點與 RUL 側指標取自 SIM 之同案例同前處理結果；X ∈ ['POW', 'LIN20', 'LIN50', 'DEXP', 'E24']；"
             "knee 沿 X；完全由既有案例列組合，不重跑擬合（rul_v911.hyb_rows／hyb_update_rows）。"),
        med="MED＝{E21, E24, LIN20, POW, DEXP} 之逐點中位數（同一更新點只算一次成員外推，供 MED 與各成員共用）",
        # ── v9.11 新增：E24（主外推器）之可調參數網格（PLAN_v911 §1.3／§3.2；LIT §D-1）──
        e24=dict(
            model_name="自適應結構灰色模型（adaptive structure grey forecasting model；24 §3.1、§5 原文名）",
            implementation_name="v9.x 實作：GM(1,1)＋多項式項 N",
            naming_note=("v9.11 文件命名更正（登記）：24 之譯本全篇未出現「GM(1,1)」或「GM(1,1,N)」字樣"
                         "（LIT_EXTRACT_v911 §D-1-2），故正式名稱一律寫「自適應結構灰色模型"
                         "（v9.x 實作：GM(1,1)＋多項式項 N）」，不得單寫 GM(1,1)+N 而不加實作註記。"),
            unspecified_in_source=("24 原文未明述、由 v9.x 實作補上並全部揭露者：(i) 多項式項階數 N（原文只稱"
                                   "「自適應結構」）—— v9.x 由訓練段末 25 % 留出誤差自 N 候選中選；"
                                   "(ii) 背景值（background value）構造 —— 原文未載，v9.x 以 "
                                   "z(k)=α·x1(k)+(1−α)·x1(k−1) 參數化，α=0.5 即標準緊鄰均值生成；"
                                   "(iii) 尾端對齊 —— 原文未載，v9.x 以訓練段末點水平對齊 "
                                   "(pred += x[−1] − x̂[k−1])。原文明述者僅：rlowess step=15（§4.2）、"
                                   "初值 ŷ(1)=x(1)（式 14 前）、最小平方估參（式 12–13）。"),
            span=[5, 15, 25],
            span_note="rlowess 平滑步長；只作用於訓練段（1..u），不含任何未來資訊。15 為 24 §4.2 之凍結值。",
            n_fit=[20, 30, 50, "all"],
            n_fit_note=("灰色擬合視窗（v9.11 新增）：只取訓練段末 n_fit 點做灰色擬合，結構 N 之留出選擇亦"
                        "只在該視窗內（前 75 % 擬合、末 25 % 留出）；'all' 即沿 v9.10 用全訓練段。"
                        "動機：24 §3.1 明述灰色模型適合小樣本建模，用全部 1..u 點擬合違反其設計（LIT §D-1-2）。"),
            N_cands=["0,1,2", "0,1"],
            alpha=[0.5, 0.4, 0.6],
            alpha_formula="背景值 z(k) = α·x1(k) + (1−α)·x1(k−1)，x1＝一次累加（1-AGO）序列；α=0.5 與 v9.10 逐字等價。",
            knee=["off", "on"], k=[20],
            cross=["self", "SIM"],
            cross_note=("交點來源：self＝E24 自身之 EOL 交點；SIM＝HYB-E24（容量取 E24、EOL 交點取同案例 SIM），"
                        "由 rul_v911.hyb_rows 組合既有案例列產生，不重跑任何擬合。"),
            two_stage=("完整網格 3×4×2×3×2 = 144 個參數組（再乘 2 前處理 × 2 交點來源 = 576）超出 90 min 預算，"
                       "故採兩階段：階 1 一次一維（基準＝span 15／n_fit all／N{0,1,2}／α 0.5／knee off／self，"
                       "共 10 個參數組）；階 2 由階 1 取各維前 2 名值全組合、以名次和最小取前 8 個未跑過之參數組。"
                       "全部候選與 leaderboard 於 search_rul_e24_grid／summary／stage1 CSV 與 selected json 揭露（R26）。"),
            starts=("搜尋同時跑固定起算點 {60,80,100}（主比較集合，入選擇規則）與比例起算點 {30 %, 50 %}"
                    "（只供 PLAN §1.4(b) 之 05 協定 RE_05 報告，不入選擇規則）。"),
            sim_recompute=("SIM 於 E24 參數搜尋中以同一協定重算（未沿用 v9.10 search checkpoint），因需要 r30／r50 之"
                           "SIM 列（v9.10 搜尋未跑比例起算點）；另抽查 2 個共同案例與 v9.10 數值等值"
                           "（確定性計算，結果寫入 search/_notes.json 之 rul.e24_sim_crosscheck）。"),
            outputs=["search_rul_e24_grid_v911.csv", "search_rul_e24_summary_v911.csv",
                     "search_rul_e24_stage1_v911.csv", "search_rul_e24_updates_v911.csv",
                     "search_rul_e24_selected_v911.json"],
        ),
        reduction_gpr=("計算成本例外（登記）：B2 滑動視窗每更新點重訓 E21（GPR-LSP），成本 O(n³)；"
                       "ox 長顆 n≈900 單次逾 120 s。故 B2 之 E21（含 MED 之 E21 成員）訓練點上限 "
                       "gpr_sub_b2=100（以 SEED 固定抽樣）；B1 仍用 gpr_sub=1000。"),
        b1="HI＝該來源 SOH 主設定於訓練段 1..s 內篩選之 top_k 特徵；C-GPR＝GPR-LSP；外推器同 B2；"
           "B1 沿 v9.9 之固定起算點單次外推、report-only（登記為計算成本例外，PLAN §2.2 B5）",
        report_only="B1（HI 二階段）、B3-25、B3-20、B4-22 沿 v9.9，report-only、不判準（PLAN §2.2 B5）",
    ),
    disclosure="R26：§2 全部候選與 leaderboard（search_soh_v911.csv／search_rul_B2_v911.csv 及其 summary／selected json）皆隨結果揭露；搜尋為結果導向，不宣稱事前唯一設定。",
)

# ── §3 選擇規則 ────────────────────────────────────────────────────────────────
SELECTION_RULE = dict(
    soh=("v9.11（PLAN_v911 §3.1）：每來源獨立；主模型固定 GPR-LSP、協定 P-B+（M=20）；"
         "於 SEARCH_SPACE.soh.v911.grid_by_source 之每來源前處理網格（視窗／濾波／特徵族／正規化／"
         "label_gate／δ／週期核／ARD／子抽樣／top_k），以兩階段搜尋（階 1 一次一維、階 2 前 8 名全組合）；"
         "選定規則：逐顆「達標顆數（RMSE < 1 %、MAE < 1 %、R² > 0.99）」最多 → 逐顆 RMSE 平均最小 → "
         "較簡（＝維持 v16 選定值之組態優先，其次維度變更數少者）。P-A／P-B／P-C／P-B+ 正式階段"
         "共用同一選定前處理；M ∈ {10, 20} 皆跑、主 M=20（M=50 取消，見 SEARCH_SPACE.soh.M_cancelled）。"),
    soh_v910=("v9.10（PLAN §3）：每來源獨立；於 δ ∈ {0, 0.05, 0.10, inf} × M=20 之網格，以 P-B+ 五模型中"
         "最佳模型之「逐顆 RMSE 平均」最小者為主設定（δ, 主模型）；同分取較簡（δ 較大＝較少剔除）。"
         "P-A／P-B／P-C 用同一 δ。其餘組態（視窗／濾波／特徵族／正規化／top_k／循環軸）沿 v15 SOH_SELECTED 不重搜。"
         "追加維度 label_gate（A2 開關）同網格一併搜尋（own 固定 on）；同分取較簡＝label_gate=off、δ 較大。"),
    soh_v15="（v15 舊規則，保留供對照）每來源以 P-B LOOCV 五模型中最佳模型之逐顆 RMSE 平均最小者為主設定；同分取較簡。",
    rul=("v9.11（PLAN §3）：每來源獨立；於 (外推器 × 前處理 × knee × k=20) 網格，以主比較集合"
         "（固定起算點 60/80/100 × 主門檻 × 有 EOL_true）之「四項全過案例數」最多者為主組合；"
         "同分依 RE 平均最小、再依 MAPE_Q 平均最小（仍同分依 (extrap, smooth, knee) 字典序）。"
         "k=10 為敏感度另列；起算點集合不選、全部報。"),
    rul_e24=("v9.11 主外推器 E24 之參數選擇（PLAN §3.2）：每來源獨立；於 E24 之 "
             "(rlowess span × 灰色擬合視窗 n_fit × N 候選 × 背景值 α × knee × 前處理 × 交點來源 self／SIM)、k=20 網格，"
             "以主比較集合（固定起算點 60/80/100 × 主門檻 × 有 EOL_true）之「四項全過案例數」最多者為選定；"
             "同分依 RE 平均最小、再依 MAPE_Q 平均最小（仍同分依 (e24_cfg, smooth, cross) 字典序）——"
             "與 SELECTION_RULE.rul 同一規則（v16 承接）。兩階段見 SEARCH_SPACE.rul.e24.two_stage。"
             "比例起算點 r30／r50 之 RE_05、條件式 MRE 只並列報告，不進入選擇。"),
    rul_v99="（v9.9 舊規則，保留供對照）每來源以主比較集合 MAPE_Q 平均最小之（外推器, 前處理）為主外推器。",
    judgement="T-SOH／T-RUL 於每來源 × 每協定／每起算點逐一判達／未達；平均值與最差顆並列；不得只報達標之子集。",
)

# ── §4 執行 ────────────────────────────────────────────────────────────────────
EXEC = dict(
    env="雲端 Linux 2 核（sklearn 1.8.0）",
    stages=["ENV", "PREREG17", "DATA", "SOH", "RUL", "EXPORT", "VERIFY"],
    gpr_subsample=1000, seed=SEED,
    checkpoint="逐顆／逐折／逐單元 checkpoint（results_v911/_ckpt_*），可續跑；_done_<STAGE>.json 標記階段完成",
    reuse_policy=("計算成本例外（登記於 v16，先於任何正式結果檔；沿 v15 之同類先例）："
                  "ox P-B（五模型 × 21 折，GPR-SE 單一單元 3618 s）之數值直接沿用 SEARCH_SOH 階段之檢查點 "
                  "results_v911/search/_ckpt/ox/*__P-B__*.json；其組態、協定、模型、種子、特徵篩選規則（僅用訓練顆）"
                  "與正式階段完全相同，為確定性計算，數值必然相等。VERIFY 以 Ridge 重算 2 顆做等值抽查。"
                  "沿用單元於 soh_records_v911.csv 之 reused 欄標 1，並列於 frozen_config／summary／verify 報告。"
                  "沿用單元不含 CI95 涵蓋率與逐點預測（搜尋檢查點未保存），該兩欄留空並揭露；其餘全部重算。"),
    soh_models_by_source={"nasa": ["GPR-SE", "GPR-LSP", "Ridge", "SVR", "MLP"],
                          "own": ["GPR-SE", "GPR-LSP", "Ridge", "SVR", "MLP"],
                          "ox": ["Ridge", "SVR", "GPR-LSP", "MLP"]},
    soh_stage_reduction=("計算成本例外（登記於 v16，沿 v15／搜尋階段之同一縮減）：ox 之 GPR-SE 於全部協定"
                         "（P-A／P-B／P-C／P-B+ M=10/20/50）一律略過（單一 P-B 單元 3618 s）；"
                         "ox 之 P-B 五模型沿用 SEARCH_SOH 檢查點（reuse_policy），其 GPR-SE 列即來自搜尋檢查點；"
                         "P-B+ 之主 M=20 全部來源一律正式重算，不沿用。"),
    rul_stage_reduction=("計算成本例外（登記於 v16，先於任何正式結果檔）：正式階段 B2 之"
                         "(a) 來源 ox 之 E21（GPR-LSP）與 MED（成員含 E21）略過（沿搜尋階段之同一縮減，"
                         "單顆逾 25 min）；(b) E21／MED 只跑 rlowess15 前處理（沿搜尋階段）；"
                         "(c) 敏感度 k=10 只跑該來源之選定外推器（RUL_SELECTED_V910 之 extrap，HYB-X 展開為 X）"
                         "與 SIM（HYB 之 EOL 來源），兩種前處理與 knee 維度照跑；主 k=20 不縮減，"
                         "全部外推器 × 前處理 × knee × 全部起算點照跑。全部縮減逐項寫入 "
                         "results_v911/rul_reduction_v911.json 並於執行紀錄揭露。"),
    rul_e24_note=("v9.11 RUL 正式階段 B2（PLAN §3.2、§4）：主外推器固定 E24，其 (rlowess span, 灰色擬合視窗 n_fit, "
                  "N 候選, 背景值 α, knee, 前處理, 交點來源) 一律取該來源之 RUL_SELECTED_V911[src]['e24'] 選定值"
                  "（stage_rul_v911.plan_calls 將 E24 獨立成一個呼叫並帶入 e24_params）；"
                  "其餘外推器（E21／E23／SVR／DEXP／LIN20／LIN30／LIN50／POW／MED／SIM 與 HYB-X）"
                  "一律以基線參數全跑並全報。ox 仍依既有計算成本例外略過 E21／MED（登記）。"
                  "選定參數若尚未填入（RUL_SELECTED_V911 為空），E24 即沿 v9.10 凍結參數"
                  "（span 15／n_fit all／N{0,1,2}／α 0.5），數值與 v9.10 逐值相同。"),
    rul_note=("v9.11 RUL：B2 採滑動視窗更新協定（SEARCH_SPACE.rul.update_protocol），主 k=20、k=10 為敏感度；"
              "搜尋只跑主比較集合（固定起算點）之 k=20，正式階段 stage_rul 全部起算點 × k ∈ {20,10} 全報；"
              "E21 之 gpr_sub_b2=100 為登記之計算成本例外；搜尋期若對 E21／MED 再限前處理，於 search/_notes.json 揭露。"),
    b1_starts_note=("計算成本例外（登記於 v16，沿 v15 同類先例）：B1（HI 二階段）之起算點僅跑主比較集合之固定起算點 {60,80,100}；"
                    "比例／晚期起算點之敏感度僅由 B2 提供。理由：B1 每顆需 top_k 條 HI × 外推器 × 前處理之外推，"
                    "晚期起算點（s≈0.8·EOL）單顆單起算點即逾 10 min，27 顆逾 7 h，超出本次 wall budget。"),
)

# ── §5 判準（修正後標準）───────────────────────────────────────────────────────
TARGETS = dict(
    T_SOH=dict(RMSE_pct=1.0, MAE_pct=1.0, R2=0.99,
               rule="逐顆判：RMSE_SOH < 1 %、MAE_SOH < 1 %、R² > 0.99；來源 × 協定 × 模型之整體達標＝該組合全部顆皆達"),
    T_RUL=dict(MAPE_Q_pct=1.0, RMSE_Q_frac_Qref=0.010, MAE_Q_frac_Qref=0.0075, RE_pct_mean=5.0,
               rule=("主比較集合逐案例判：MAPE_Q ≤ 1 %、RMSE_Q ≤ 0.010×Q_ref、MAE_Q ≤ 0.0075×Q_ref；"
                     "RE_RUL＝AE/(EOL_true−s)×100 之案例平均 ≤ 5 %。2 Ah 尺度＝0.020／0.015 Ah；3.10 Ah 尺度＝0.0310／0.0233 Ah。")),
)
RUL_STANDARD = dict(
    update_protocol=("v9.11（PLAN §2.2 B1）滑動視窗更新：更新點 u = s, s+k, s+2k, …（有 EOL：u ≤ EOL_true−1；"
                     "右設限：u ≤ 資料末−1）；每更新點只用 1..u（前處理只作用於訓練段）擬合外推器，"
                     "預測 (u, u+k] 之容量並於單調化後求 EOL 交點。k ∈ {20, 10}，主 k=20。"),
    main_case_set=("案例＝(顆, 起算點 s, 外推器, 前處理, knee, k)，s ∈ {60, 80, 100}（固定起算點；第一更新點 u=s），"
                   "且該顆有 EOL_true 且 s < EOL_true；起算點已過 EOL 標「不適用」；右設限顆只報容量誤差、RUL 欄不填 0。"),
    primary_threshold={"nasa": "EOL_B", "own": "EOL_A", "ox": "EOL_A", "GA": "EOL_A"},
    capacity_window="s+1 … EOL_true（有 EOL）或 s+1 … 資料末（右設限），事先指定",
    metrics=("MAPE_Q／RMSE_Q／MAE_Q／NRMSE：s+1 … EOL_true（右設限 → 資料末）全區間之「k 步前瞻」預測 vs 實測；"
             "MAPE_Q＝mean|Q̂−Q|/Q×100；RMSE_Q、MAE_Q 以 Ah 4 位小數；NRMSE＝RMSE_Q/Q_ref×100；"
             "RE_pct＝更新點平均之 |EOL_pred(u)−EOL_true|/(EOL_true−s)×100（分母固定起算點，依 RUL_Q1 文件），"
             "只計 EOL_true−u ≥ k 之更新點；同列 RE_u_pct（分母 EOL_true−u）、RE_first_pct（第一更新點"
             "＝v9.9 型單次外推之對照）、AE_mean、signed_err_mean、n_updates、n_no_cross；"
             "無交點更新點該點 RE 視為 NaN 且 n_no_cross+1；案例若全部更新點皆無交點則 RE 留空、no_cross=1（計入失敗、不刪）；"
             "逐更新點紀錄（案例 id、u、EOL_pred、RE(u)）另存 rul_B2_updates_v911.csv"),
    sensitivity=("比例起算點 {30 %, 50 %} 與晚期 {70 %, 80 %} 另列為敏感度，不入主表（與固定起算點用同一滑動協定，"
                 "一致並揭露）；k=10 亦為敏感度另列。"),
    no_future_info=("架構二每個更新點：平滑、擬合、膝點偵測與外推器結構選擇（E24 之 N）只用 1..u；"
                    "SIM 之庫顆為其他顆之全程軌跡（跨顆先驗，登記揭露），測試顆本身不用未來資訊。"
                    "架構二每個起算點：B1 特徵篩選、正規化、外推器結構選擇（E24 之 N）只用 1..s 訓練段；"
                    "架構一：P-B 折內只用訓練顆篩選；P-A 篩選只用前 30 循環訓練段；P-C 篩選只用訓練 80 %；"
                    "REL 參考＝該顆前 3 循環（屬訓練段）。"),
    cycle_axis="own cell2／cell3 同時保留 cyc_A（物理循環）與 cyc_c；主結果於 cyc_c（v9.8 E1），缺號性質待查證並於文件揭露。",
    # ── v9.11 新增（PLAN §1.4）：三種 Q1 RUL 報法並列，皆程式產生、皆登記；(ii)(iii) 不取代判準 ──
    q1_reporting=dict(
        judged=("(i) 判準：RE_pct＝更新點平均 |EOL_pred(u)−EOL_true|/(EOL_true−s)×100（分母固定起算點）；"
                "門檻 RE ≤ 5 %（TARGETS.T_RUL）。達標與否一律依此判。"),
        ae=("(ii-a) AE（絕對圈數誤差，24 式 27 之定義＝|EOL_pre−EOL_true|）：records 同列 "
            "AE_mean（更新點平均）與 AE_first（第一更新點 u=s 之一次性外推）。24 之 NASA 結果為 AE 0–4 圈。"),
        re05=("(ii-b) 05 協定之 RE：RE_05_pct＝|EOL_pred(u=s)−EOL_true|/RUL_true×100，RUL_true＝EOL_true−s，"
              "為「同顆內、起算點＝該顆總循環 30 %／50 %、一次性外推（不以後續實測更新）」之定義"
              "（LIT §D-3：05 為七篇中唯一報 RE ≤ 5 % 者，NASA 平均 0.53 %）。"
              "records 之 protocol_05 欄標記 start_kind=='ratio'（r30／r50）之案例；於 sensitivity 表列出。"
              "同一欄於固定起算點案例亦計算（即第一更新點之一次性外推 RE），供對照。"),
        cond_mre=("(iii) 條件式 MRE（援引 20）：主比較集合中 RUL_true > 150 圈之案例之 RE 平均"
                  "（20 之 MES 整體 MRE 15.2 %、RUL > 150 循環時 9.79 %）。"
                  "動機：RUL_true < 30 圈時 RE ≤ 5 % 等於要求 AE ≤ 1 圈，超過循環解析度（PLAN §1.4）。"),
        rule="三者皆逐案例入 records／updates、皆於 summary 與 tables 並列；判準只用 (i)，不得以 (ii)(iii) 取代。",
        no_r2="RUL 不報 R²（LIT §F-1：七篇零篇對 RUL 報 R²）。",
    ),
)
REPORT_ONLY_LAWS = {
    "B3-25": "同顆隨機 80/20 特徵→RUL（25 Table 5 格式）：同顆切分、測試點與訓練點同分布，降為只報、不入主結果、不判準（§5.3）。",
    "B3-20": "混顆逐循環 60/20/20 Ridge／MLP32（20 Table II 格式）：同顆隨機切分，降為只報、不判準（§5.3）。",
    "B4-22": "AR-GPR（22）之輸入含前一步 RUL 且起點為訓練段末真值，實質為倒數計時；誤差僅反映 ŜOH 擾動，降為只報、不判準（§5.3）。",
}

# ── 架構一固定參數（非搜尋維度）────────────────────────────────────────────────
SOH_FIXED = dict(
    fv_step_V=0.03, dq_step_V=0.05, sg_window_pts=11, sg_order=3,
    # v9.11：sg_window_pts 改為「依組態」（由 FILTERS[filter_key] 提供；上列 11 為未指定 filter_key 時之舊預設）
    sg_window_pts_by_filter_key={k: v["sg_window_pts"] for k, v in FILTERS.items() if v["sg_window_pts"]},
    sg_window_note="SG 視窗點數依選定 filter_key（SG51＝21 之凍結值 51；SG11＝v9.x 舊值）；階數固定 3（21 未明述，沿 v9.x 並揭露）",
    norms=NORMS, filters={k: {kk: vv for kk, vv in v.items()} for k, v in FILTERS.items()},
    clean_window=15, clean_z=3.5, clean_edge_ratio=0.10,
    sel_min_rho=0.85, PA_ntrain=30, PA_starts=[1, 40], PC_test_frac=0.20,
    PB_gpr_subsample=1000, rel_nref=3,
    models=["GPR-SE", "GPR-LSP", "Ridge", "SVR", "MLP"],
)

# ── §6 SELECT（登記後不改）─────────────────────────────────────────────────────
# filter 以 "GS"/"SG" ＋ sigma_mV 正確登記（修正 search_soh_selected_v911.json 之 bound-method 瑕疵）
SOH_SELECTED = {
    "nasa": dict(config_id="nasa|3.80-4.10|GS15|F_both|REL|k12|cyc",
                 window=[3.80, 4.10], filter="GS", sigma_mV=15.0, family="F_both", norm="REL",
                 min_rho=0.85, top_k=12, cyc_axis="cyc", main_model="GPR-LSP",
                 search_PB_RMSE_pct_mean=0.8418345840818193),
    "own": dict(config_id="own|3.70-4.20|GS20|F_both|none|k12|none",
                window=[3.70, 4.20], filter="GS", sigma_mV=20.0, family="F_both", norm="none",
                min_rho=0.85, top_k=12, cyc_axis="none", main_model="GPR-LSP",
                search_PB_RMSE_pct_mean=1.3013374036368983),
    "ox": dict(config_id="ox|3.50-4.10|GS10|F_IC|none|k12|cyc",
               window=[3.50, 4.10], filter="GS", sigma_mV=10.0, family="F_IC", norm="none",
               min_rho=0.85, top_k=12, cyc_axis="cyc", main_model="SVR",
               search_PB_RMSE_pct_mean=1.3441369040512972,
               note="階 2 僅此一組完成（單一 P-B 單元逾 900 s 觸發縮減）；階 1 Ridge 第 1 名；R26 揭露"),
}
# ── v9.11 架構層級修訂（PLAN §2.1；登記於 prereg v16）────────────────────────
ARCH_SOH_V910 = dict(
    A1=dict(name="充電段有效性閘",
            where="「充電段 Q(V)」之後、「IC」之前（data／stage_data 層）",
            rule=("每循環之充電段 Q(V) 第一個有限格點電壓 ≤ V_lo+δ 且最後一個有限格點 ≥ V_hi−δ 才有效"
                  "（V_lo／V_hi＝該來源 v15 選定 IC 視窗，見 SOH_SELECTED[src]['window']）；"
                  "無效循環不進訓練亦不進評估"),
            delta_V=[0.0, 0.05, 0.10, float("inf")],
            frame_cols=["seg_first_V", "seg_last_V", "seg_valid_d0", "seg_valid_d50", "seg_valid_d100"],
            census_cols=["n_seg_invalid_d0", "n_seg_invalid_d50", "n_seg_invalid_d100"],
            impl="ic_v911.seg_bounds／ic_v911.seg_valid；soh_v911.build_cell_frame_v911 落欄；soh_v911.apply_delta 過濾"),
    A2=dict(name="標籤一致性閘（清洗方塊擴充）",
            rule=("v9.8 own_gap_rule_v98（gaps_v911.apply_rule；dsoh_thr=0.010、med_win=5、med_thr=0.010）"
                  "由 own 擴及 nasa／ox：|ΔSOH| > 0.010 且前後 5 循環中位差 > 0.010 之過渡循環拉掉，"
                  "索引缺口重編為連續 cyc_c；frames 全來源皆有 cyc_A／cyc_c／drop_flag"),
            main_axis="on → cyc_c；off → cyc_A（原物理循環）",
            census_cols=["n_label_dropped", "dropped_cycA"],
            searchable="label_gate ∈ {off, on}（own 固定 on；nasa／ox 搜尋）；見 SEARCH_SPACE.soh.label_gate_note",
            impl="stage_data_v911.build_frames／search_soh_v911.frames_of 對全來源套用，依 label_gate 決定是否剔除與循環軸"),
    A3=dict(name="P-B+ 稀疏參考更新協定",
            rule=("跨顆 LOOCV 預訓練（同 P-B，特徵篩選只用訓練顆）；測試顆之列以索引 i%M==0 為參考點（RPT）；"
                  "以每 M 列為一區塊迭代，第 j 區塊只用索引 ≤ 區塊起點之參考點擬合殘差 GPR 並把殘差預測加回；"
                  "評估只算非參考列（不用未來資訊）"),
            gpr_kernel="ConstantKernel(1.0,(1e-4,1e2))*RBF(50,(5,2000))+WhiteKernel(1e-5,(1e-8,1e-1))",
            gpr_opts=dict(normalize_y=False, n_restarts_optimizer=2, random_state=0),
            fallback="參考點 < 2 時以參考殘差平均加回",
            M=[10, 20, 50], M_main=20,
            main_judgement="T-SOH 之跨顆主判準改以 P-B+；P-B 零校正仍全報為消融",
            impl="stage_soh_v911.run_PBplus／search_soh_v911.run_PBplus"),
    A4="模型與特徵組態不變（五模型；視窗／濾波／特徵族／REL／top_k／循環軸沿 v15 §6）",
)

# ── §6 v9.11 SOH 選定（由 search_soh_v911 依 §3 規則產生；數值一律不手打，見 source_file）──
# 規則（§3）：每來源於 δ ∈ {0, 0.05, 0.10, inf} × M=20 之 P-B+ 網格，取五模型中最佳模型之
#   「逐顆 RMSE 平均」最小者；同分取較簡（δ 較大＝較少剔除）。
# 本處只登記「選定之類別值」（δ 與主模型）；全部指標數值請逕查 source_file 之 leaderboard。
_SEL_SRC = "results_v911/search/search_soh_selected_v911.json（search_soh_v911.py，2026-09-20）"
SOH_SELECTED_V910 = {
    # nasa：d100 與 inf 之 P-B+ 逐顆 RMSE 平均相同（該來源 δ=0.10 無任何無效循環）→ 同分取較簡 δ=inf
    # nasa：label_gate=off 全面優於 on（B0006 1.276 → 0.949 %）；off 下 d100 與 inf 同分（無 d100 無效循環）
    #        → 同分取較簡 δ=inf。A2 規則偵測到之 20 個過渡循環於 off 下不剔除（判為靜置後容量再生）
    "nasa": dict(label_gate="off", delta_key="inf", delta_V=None, main_model="GPR-LSP", M_main=20,
                 protocol="P-B+", source_file=_SEL_SRC),
    "own": dict(label_gate="on", delta_key="d0", delta_V=0.0, main_model="GPR-LSP", M_main=20,
                protocol="P-B+", source_file=_SEL_SRC),
    "ox": dict(label_gate="on", delta_key="d0", delta_V=0.0, main_model="SVR", M_main=20,
               protocol="P-B+", source_file=_SEL_SRC,
               note="ox 之 GPR-SE 依登記之計算成本例外略過（SEARCH_SPACE.soh.reduction）；"
                    "主模型自 Ridge／SVR／GPR-LSP／MLP 中選出"),
}

# ── §6 v9.11 SOH 選定（每來源前處理；由 search_soh_v911.py 依 SELECTION_RULE.soh 產生後填入）──
# 下列各欄之初值＝v16 選定值（＝搜尋之基準組態）；搜尋完成後由 fill_soh_selected_v911.py 依
# results_v911/search/search_soh_selected_v911.json 覆寫（數值一律不手打，source_file 記錄來源檔與產生時間）。
_SEL911_SRC = 'results_v911/search/search_soh_selected_v911.json（search_soh_v911.py，2026-09-20T14:18:51Z）'
SOH_SELECTED_V911 = {
    'nasa': dict(window=[3.80, 4.10], filter_key='GS15', family='F_both', norm='REL3',
                  label_gate='off', delta_key='inf', periodic=True, ard=False, sub=2000, top_k=12,
                  min_rho=0.85, cyc_axis='cyc', main_model='GPR-LSP', M_main=20, protocol='P-B+',
                  search_stage='stage1_base', search_config_id='nasa|3.80-4.10|GS15|F_both|REL3|k12|cyc|A2off|inf|Pon|ARDoff|sub2000', search_changed_vs_v16=[],
                  search_n_pass=3, search_n_cells=4, search_RMSE_pct_mean=0.7133258420563545, search_RMSE_pct_max=0.9490967198469881,
                  search_MAE_pct_mean=0.4854811492131583, search_R2_min=0.9889643692578413,
                  search_PB_base_RMSE_pct_mean=0.8486643106390358, search_PB_base_n_pass=2, source_file=_SEL911_SRC),
    'own': dict(window=[3.70, 4.20], filter_key='GS20', family='F_both', norm='none',
                 label_gate='on', delta_key='d0', periodic=True, ard=False, sub=2000, top_k=12,
                 min_rho=0.85, cyc_axis='none', main_model='GPR-LSP', M_main=20, protocol='P-B+',
                 search_stage='stage1_base', search_config_id='own|3.70-4.20|GS20|F_both|none|k12|none|A2on|d0|Pon|ARDoff|sub2000', search_changed_vs_v16=[],
                 search_n_pass=1, search_n_cells=2, search_RMSE_pct_mean=0.575723209937539, search_RMSE_pct_max=0.8467558839565745,
                 search_MAE_pct_mean=0.37824224698461084, search_R2_min=0.9771469243339178,
                 search_PB_base_RMSE_pct_mean=0.6614573805751371, search_PB_base_n_pass=1, source_file=_SEL911_SRC),
    'ox': dict(window=[3.50, 4.10], filter_key='GS10', family='F_IC', norm='none',
                label_gate='on', delta_key='d0', periodic=False, ard=False, sub=1000, top_k=12,
                min_rho=0.85, cyc_axis='cyc', main_model='GPR-LSP', M_main=20, protocol='P-B+',
                search_stage='stage1', search_config_id='ox|3.50-4.10|GS10|F_IC|none|k12|cyc|A2on|d0|Poff|ARDoff|sub1000', search_changed_vs_v16=['periodic'],
                search_n_pass=2, search_n_cells=21, search_RMSE_pct_mean=0.9329919697774952, search_RMSE_pct_max=1.9524760371207692,
                search_MAE_pct_mean=0.5580474901841519, search_R2_min=0.8902395778631581,
                search_PB_base_RMSE_pct_mean=1.1781036786282686, search_PB_base_n_pass=3, source_file=_SEL911_SRC),
}

# v9.9 之選定（固定起算點單次外推），保留供 v9.11 對照
RUL_SELECTED_V99 = {
    "nasa": dict(extrap="E21", smooth="rlowess15", primary_threshold="EOL_B", search_MAPE_Q_mean=3.057),
    "own": dict(extrap="POW", smooth="rlowess15", primary_threshold="EOL_A", search_MAPE_Q_mean=2.3361),
    "GA": dict(extrap="E24", smooth="rlowess15", primary_threshold="EOL_A", search_MAPE_Q_mean=3.4809),
    "ox": dict(extrap="LIN20", smooth="rlowess15", primary_threshold="EOL_A", search_MAPE_Q_mean=4.5254),
}

# v9.11 B2 滑動視窗協定之固定參數（非搜尋維度）
RUL_PROTOCOL_V910 = dict(
    update_protocol="sliding_window",
    k_list=(20, 10), k_main=20,
    sim_axis={"nasa": "ah", "own": "soh", "GA": "soh", "ox": "soh"},
    gpr_sub_b2=100,
    knee_formula=("兩段連續線性最小平方（Bacon-Watts 之 γ→0 極限，等價形式）："
                  "y(t) = a0 + a1·(t−c) + a2·(t−c)·1{t>c}；於候選斷點 c 之網格（訓練段第 10 點至倒數第 20 點，"
                  "最多 60 個等距候選）以 OLS 最小化 SSE 選 c；膝前斜率 a1、膝後斜率 a1+a2。"
                  "判定膝點成立：a1+a2 < 0 且 a1+a2 < 2×a1 且膝後點數 ≥ 20。成立時 LIN_n／POW 只用膝後段擬合。"),
    source_file="results_v911/search/search_rul_selected_v911.json（SEARCH_RUL 階段寫出）",
)

# v9.11 選定組合（由 search_rul_v911.py 依 SELECTION_RULE.rul 選定後填入；search_* 欄為主比較集合統計）
RUL_SELECTED_V910 = {
    "nasa": dict(extrap='E24', smooth='none', knee='n/a', k=20,
                 primary_threshold='EOL_B',
                 search_n_cases=8, search_n_pass_all4=1,
                 search_MAPE_Q_mean=2.0231, search_RMSE_Q_mean=0.0411,
                 search_MAE_Q_mean=0.0301, search_RE_pct_mean=36.133,
                 search_RE_pct_max=61.979, search_RE_first_pct_mean=55.228,
                 search_n_cases_no_cross=3, search_n_no_cross_updates=1),
    "own": dict(extrap='HYB-E24', smooth='rlowess15', knee='n/a', k=20,
                 primary_threshold='EOL_A',
                 search_n_cases=6, search_n_pass_all4=0,
                 search_MAPE_Q_mean=0.4879, search_RMSE_Q_mean=0.0194,
                 search_MAE_Q_mean=0.0129, search_RE_pct_mean=12.333,
                 search_RE_pct_max=15.205, search_RE_first_pct_mean=22.105,
                 search_n_cases_no_cross=0, search_n_no_cross_updates=0),
    "GA": dict(extrap='E21', smooth='rlowess15', knee='n/a', k=20,
                 primary_threshold='EOL_A',
                 search_n_cases=15, search_n_pass_all4=0,
                 search_MAPE_Q_mean=1.17, search_RMSE_Q_mean=0.0445,
                 search_MAE_Q_mean=0.0316, search_RE_pct_mean=16.413,
                 search_RE_pct_max=27.778, search_RE_first_pct_mean=27.652,
                 search_n_cases_no_cross=0, search_n_no_cross_updates=6),
    "ox": dict(extrap='E24', smooth='rlowess15', knee='n/a', k=20,
                 primary_threshold='EOL_A',
                 search_n_cases=36, search_n_pass_all4=6,
                 search_MAPE_Q_mean=1.4115, search_RMSE_Q_mean=0.1606,
                 search_MAE_Q_mean=0.0379, search_RE_pct_mean=19.841,
                 search_RE_pct_max=46.75, search_RE_first_pct_mean=53.544,
                 search_n_cases_no_cross=0, search_n_no_cross_updates=131),
}
RUL_SELECTED_V910["_protocol"] = RUL_PROTOCOL_V910

# ── §6 v9.11 RUL 選定（主外推器固定 E24；E24 參數由 search_rul_v911.py --e24-grid 依
#    SELECTION_RULE.rul_e24 選定後填入。e24 欄＝(span, n_fit, N_cands, alpha)；cross＝交點來源）──
RUL_PROTOCOL_V911 = dict(RUL_PROTOCOL_V910)
RUL_PROTOCOL_V911.update(
    main_extrap="E24",
    main_extrap_note=("v9.11 主外推器固定 E24（自適應結構灰色模型；PLAN §2 架構二）；其餘外推器"
                      "（E21／E23／SVR／DEXP／LIN20／LIN30／LIN50／POW／MED／SIM／HYB-X）全部照跑為基線。"),
    e24_defaults=dict(span=15, n_fit="all", N_cands="0,1,2", alpha=0.5),
    e24_default_note="＝v9.10 凍結值；rul_v911.extrapolate 之 e24_params=None 即走 v9.10 原路徑（逐值相同）。",
    source_file_e24="results_v911/search/search_rul_e24_selected_v911.json（SEARCH_RUL 階段寫出）",
)
RUL_SELECTED_V911 = {
    # 由 search_rul_v911.py --e24-grid 依 SELECTION_RULE.rul_e24 選定後程式填入（零手打數字）；
    # 來源檔 results_v911/search/search_rul_e24_selected_v911.json，完整 leaderboard 見同目錄 CSV。
    'nasa': dict(
        extrap='E24', smooth='none', knee='off', cross='self', k=20,
        primary_threshold='EOL_B',
        e24=dict(span=15.0, n_fit=20, N_cands='0,1,2', alpha=0.6,
                 knee='off', cross='self', cfg_id='sp15_nf20_N012_a0.6_knoff'),
        search_n_cases=8,
        search_n_pass_all4=2,
        search_MAPE_Q_mean=1.2954,
        search_RMSE_Q_mean=0.0224,
        search_MAE_Q_mean=0.019,
        search_RE_pct_mean=17.3124,
        search_RE_pct_max=47.3684,
        search_RE_first_pct_mean=20.39,
        search_AE_mean_mean=7.7667,
        search_AE_first_mean=10.1429,
        search_RE_05_r30_pct_mean=150.9933,
        search_RE_05_r50_pct_mean=64.891,
        search_MRE_cond_gt150_pct=None,
        search_n_cases_gt150=0,
        search_n_cases_no_cross=3,
        search_n_no_cross_updates=2,
        search_n_pass_MAPE=3,
        search_n_pass_RMSE=3,
        search_n_pass_MAE=3,
        search_n_pass_RE=2,
    ),
    'own': dict(
        extrap='HYB-E24', smooth='rlowess15', knee='off', cross='SIM', k=20,
        primary_threshold='EOL_A',
        e24=dict(span=15.0, n_fit='all', N_cands='0,1', alpha=0.6,
                 knee='off', cross='SIM', cfg_id='sp15_nfall_N01_a0.6_knoff'),
        search_n_cases=6,
        search_n_pass_all4=0,
        search_MAPE_Q_mean=0.4719,
        search_RMSE_Q_mean=0.019,
        search_MAE_Q_mean=0.0125,
        search_RE_pct_mean=12.3333,
        search_RE_pct_max=15.2047,
        search_RE_first_pct_mean=22.1053,
        search_AE_mean_mean=37.2319,
        search_AE_first_mean=65.0,
        search_RE_05_r30_pct_mean=18.0312,
        search_RE_05_r50_pct_mean=11.3537,
        search_MRE_cond_gt150_pct=12.3333,
        search_n_cases_gt150=6,
        search_n_cases_no_cross=0,
        search_n_no_cross_updates=0,
        search_n_pass_MAPE=6,
        search_n_pass_RMSE=6,
        search_n_pass_MAE=6,
        search_n_pass_RE=0,
    ),
    'GA': dict(
        extrap='E24', smooth='rlowess15', knee='off', cross='self', k=20,
        primary_threshold='EOL_A',
        e24=dict(span=15.0, n_fit=20, N_cands='0,1', alpha=0.6,
                 knee='off', cross='self', cfg_id='sp15_nf20_N01_a0.6_knoff'),
        search_n_cases=15,
        search_n_pass_all4=0,
        search_MAPE_Q_mean=3.7201,
        search_RMSE_Q_mean=0.221,
        search_MAE_Q_mean=0.0967,
        search_RE_pct_mean=19.3322,
        search_RE_pct_max=35.2,
        search_RE_first_pct_mean=26.0314,
        search_AE_mean_mean=22.8178,
        search_AE_first_mean=38.4286,
        search_RE_05_r30_pct_mean=20.8183,
        search_RE_05_r50_pct_mean=49.3954,
        search_MRE_cond_gt150_pct=30.0,
        search_n_cases_gt150=1,
        search_n_cases_no_cross=0,
        search_n_no_cross_updates=26,
        search_n_pass_MAPE=6,
        search_n_pass_RMSE=3,
        search_n_pass_MAE=4,
        search_n_pass_RE=0,
    ),
    'ox': dict(
        extrap='E24', smooth='none', knee='on', cross='self', k=20,
        primary_threshold='EOL_A',
        e24=dict(span=15.0, n_fit='all', N_cands='0,1,2', alpha=0.4,
                 knee='on', cross='self', cfg_id='sp15_nfall_N012_a0.4_knon'),
        search_n_cases=36,
        search_n_pass_all4=6,
        search_MAPE_Q_mean=1.0089,
        search_RMSE_Q_mean=0.1032,
        search_MAE_Q_mean=0.0264,
        search_RE_pct_mean=17.0609,
        search_RE_pct_max=44.466,
        search_RE_first_pct_mean=53.3231,
        search_AE_mean_mean=101.3075,
        search_AE_first_mean=321.88,
        search_RE_05_r30_pct_mean=41.764,
        search_RE_05_r50_pct_mean=43.8297,
        search_MRE_cond_gt150_pct=17.0609,
        search_n_cases_gt150=36,
        search_n_cases_no_cross=0,
        search_n_no_cross_updates=178,
        search_n_pass_MAPE=27,
        search_n_pass_RMSE=18,
        search_n_pass_MAE=27,
        search_n_pass_RE=6,
    ),
}
RUL_SELECTED_V911["_protocol"] = RUL_PROTOCOL_V911
# 相容別名（export_v911／tables_v911 仍以 RUL_SELECTED 取用；v9.11 起指向 RUL_SELECTED_V911，
# 未填時回退 RUL_SELECTED_V910）
RUL_SELECTED = {k: v for k, v in (RUL_SELECTED_V911 if len(RUL_SELECTED_V911) > 1 else RUL_SELECTED_V910).items()
                if not k.startswith("_")}
B1_SETTINGS = dict(cgpr_model="GPR-LSP",
                   hi="該來源 SOH 主設定之 top_k 特徵，於每個起算點以訓練段 1..s 之 Spearman 相關重新篩選（§5.3）",
                   extraps="同 B2 全報；主外推器同 B2",
                   starts="固定起算點 {60, 80, 100}（見 EXEC.b1_starts_note）")

SEARCH_PRIOR_FINDINGS = (
    "搜尋期已知結論（進正式文件之判準前提，R26 揭露）：T-SOH 於 P-B 三來源皆未達（最差顆 > 1 %）；"
    "P-C nasa／own 於搜尋期可達（需以訓練段篩選重跑確認）；ox P-C 因低變化顆 R² 未過；"
    "T-RUL 四來源 main-set 皆未達（MAPE_Q 2.3–4.5 %、RE 27–54 %）。正式跑如實照登，不得只報達標子集。")

CFG911 = dict(
    version=VERSION, prereg=PREREG, seed=SEED, plan="PLAN_v911.md（Fable 5.1，2026-09-20）",
    data=DATA, search_space=SEARCH_SPACE, selection_rule=SELECTION_RULE, exec=EXEC,
    targets=TARGETS, rul_standard=RUL_STANDARD, report_only_laws=REPORT_ONLY_LAWS,
    soh_fixed=SOH_FIXED, soh_selected=SOH_SELECTED, soh_selected_v910=SOH_SELECTED_V910,
    soh_selected_v911=SOH_SELECTED_V911, soh_filters=FILTERS, soh_norms=NORMS,
    arch_soh_v911=ARCH_SOH_V910, rul_selected=RUL_SELECTED_V910, rul_selected_v99=RUL_SELECTED_V99,
    rul_selected_v911=RUL_SELECTED_V911,
    rul_protocol_v911=RUL_PROTOCOL_V911,
    b1_settings=B1_SETTINGS, search_prior_findings=SEARCH_PRIOR_FINDINGS,
)


def resolve_soh(sel, src=None):
    """（v9.11）把一個「選定／候選」組態 dict 展開為完整欄位：filter／sigma_mV／sg_window_pts／rel_nref／config_id。
    sel 需含 window、filter_key、family、norm、label_gate、delta_key、periodic、ard、sub、top_k（其餘沿基準）。"""
    s = dict(sel); f = FILTERS[s["filter_key"]]
    s["filter"] = f["filter"]; s["sigma_mV"] = f["sigma_mV"]
    s["sg_window_pts"] = f["sg_window_pts"] or SOH_FIXED["sg_window_pts"]
    s["sg_order"] = SOH_FIXED["sg_order"]
    s["rel_nref"] = NORMS[s["norm"]]
    s.setdefault("min_rho", SOH_FIXED["sel_min_rho"])
    s.setdefault("cyc_axis", "cyc")
    w = s["window"]
    s["config_id"] = (f"{src or s.get('source','?')}|{w[0]:.2f}-{w[1]:.2f}|{s['filter_key']}|{s['family']}|"
                      f"{s['norm']}|k{s['top_k']}|{s['cyc_axis']}|A2{s['label_gate']}|{s['delta_key']}|"
                      f"P{'on' if s['periodic'] else 'off'}|ARD{'on' if s['ard'] else 'off'}|sub{s['sub']}")
    return s


def soh_sel(src):
    """（v9.11）該來源之 SOH 選定組態（展開後）。正式階段（stage_data／stage_soh）一律由此取值。"""
    return resolve_soh(SOH_SELECTED_V911[src], src)


def soh_cfg_of(sel):
    """（v9.11）把展開後之組態轉成 build_cell_frame_v911 所需之 cfg dict。"""
    c = dict(SOH_FIXED)
    c.update(filter=sel["filter"], sigma_mV=(sel["sigma_mV"] if sel["sigma_mV"] is not None else float("nan")),
             sg_window_pts=sel["sg_window_pts"], sg_order=sel["sg_order"])
    return c


def gpr_kwargs(sel):
    """（v9.11）主模型 GPR-LSP 之核組態 kwargs（只對 GPR-LSP 生效；其餘四模型用基線設定）。"""
    return dict(periodic=bool(sel["periodic"]), ard=bool(sel["ard"]))


def soh_cfg(src):
    """把 §6 選定之組態轉成 build_cell_frame_v911 所需之 cfg dict。"""
    s = SOH_SELECTED[src]
    c = dict(SOH_FIXED)
    c.update(filter=s["filter"], sigma_mV=(s["sigma_mV"] if s["sigma_mV"] is not None else float("nan")))
    return c


def ah_thresholds(qref):
    """§5.2：依 Q_ref 尺度換算之 Ah 門檻。"""
    return dict(RMSE_Ah=TARGETS["T_RUL"]["RMSE_Q_frac_Qref"] * qref,
                MAE_Ah=TARGETS["T_RUL"]["MAE_Q_frac_Qref"] * qref)
