# 資料字典 — `data/derived_per_row/` 與 `data/cells_summary.csv`

> 新撰（公開程式包）。欄位意義一律由程式碼確認，引用處標「檔:行」（行號為作者封存樹 `5版/02_程式/` 之原檔；核心模組之複本見 `code/02_core_modules/`，轉檔程式之複本見 `code/01_download_and_convert/`）。式號與章節指稿件 v34。
> 產生程式：`code/07_build_derived_tables/build_derived_tables.py`；執行紀錄與全部驗證結果：同夾 `build_log.json`、`build_stdout.txt`。

## 1. 檔案

- `derived_per_row/<src>__<cell>.csv`：34 檔，檔名同作者封存 `5版/03_實驗數據/results_R3_dedup/frames/<src>__<cell>.csv`（`src`＝`nasa`／`own`／`ox`）。
  - 左側欄位＝封存 frames 之**全部原欄位與原字串**（不重新格式化；封存檔由 `stage_data_v911.py` L66 以 `float_format="%.17g"` 寫出）。回讀後與封存檔逐值相等（`build_log.json` → `checks.all_orig_columns_equal_archived = true`）。換行符為 LF（封存檔為 CRLF）。
  - 右側附加 9 個 `ann_` 標註欄（§4）。
- `cells_summary.csv`：每顆一列（§5）。

**一列＝一個循環索引列**（v34 第 II-A 節）：同一循環（同一 `cyc_A`）之多個充電事件合併為一列，Q(V) 逐網格點取事件平均、SOH 標籤／EFC／溫度／I_max 取平均（`data_v911.py` L78–87 `per_cycle_table`）。列為通過資料建置溫度閘門（15–35 °C）之事件所成之循環，位於閘門 A1、A2 **之前**。34 檔合計 16 567 列（NASA 629、自有 2847、Oxford 13 091），即 v34 表 2「16 567 個循環索引列」；`n_ev` 欄合計為閘門前充電事件數（NASA 632、自有 2847、Oxford 14 174，v34 第 II-A 節）。

**特徵為候選全集，未經篩選與標準化**：衍生表只含封存 frames 原有之全部特徵候選欄（已經滾動穩健 z 分數清洗，見 §3），**不做** Spearman 篩選 (6)、標準化與 REL (5)。Spearman 篩選與標準化只用每折訓練部分；REL 依各電芯（含測試電芯）自身前三個保留列，屬稿件第 II-D 節所述之離線步驟（`soh_v911.py` L236–245；鎖定組態只有 NASA 使用 REL）。篩選之實作定義（`soh_v911.py` L79–88 `select_features`；P-B 與 P-B+ 之呼叫見 `stage_soh_v911.py` L133、L201）：各候選特徵於每顆訓練電芯內與 SOH 之 Spearman 相關，取各訓練電芯 |ρ| 之最小值 ≥ 0.85 者，以平均 |ρ| 排序取前 12；若通過者不足 12，則退回以全部候選之平均 |ρ| 排序取前 12。鎖定組態 34 折皆未觸發退回（各折通過 |ρ| 門檻之候選數：NASA 21、自有 18–19、Oxford 12–13）。標準化在模型管線內。因此篩選與標準化不能預先套在整表上。NASA 之 REL3 於載入時（`stage_data_v911.py` L127–128）才套用，**衍生表中之 NASA 特徵值為 REL 前之值**。

## 2. 每來源之欄集合

| 來源 | IC 視窗（v34 第 II-C 節） | 特徵族（v34 第 II-D 節） | FV 欄 | dQ 欄 |
|---|---|---|---|---|
| nasa | 3.80–4.10 V | F_both | FV_3.80 … FV_4.10（每 30 mV，11 欄） | dQ_3.85 … dQ_4.10（每 50 mV，6 欄） |
| own | 3.70–4.20 V | F_both | FV_3.70 … FV_4.18（17 欄） | dQ_3.75 … dQ_4.20（10 欄） |
| ox | 3.50–4.10 V | F_IC | FV_3.50 … FV_4.10（21 欄） | 無 |

視窗與特徵族取自 `config_v911.py` L410–432（`SOH_SELECTED_V911`），封存 `frames_meta_v911.json` 之 `soh_resolved` 與之相同。

## 3. 原欄位（封存 frames）

| 欄 | 單位 | 定義（程式出處） | v34 |
|---|---|---|---|
| `cyc` | 循環 | 該列之循環計數＝`cyc_A`（`data_v911.py` L80 `np.unique(cyc)`，`cyc` 來自 npz 之 `cyc_A`，L41）。**全表欄**中與 `cyc_A` 相同；建模時才依 A2 改成 `cyc_c` 或 `cyc_A`（`stage_data_v911.py` L122、L124）。 | II-A |
| `efc` | 等效完整循環 | 該循環各事件 EFC 之平均（`data_v911.py` L86）；事件 EFC＝其前一放電段之累計放電 Ah ÷ Q_ref（`a_layer_full.py` L124、L131–132；`a_layer_fix_v9a.py`） | — |
| `soh` | 比值（×100 為 %） | 該循環各事件 SOH 標籤之平均（`data_v911.py` L86）；事件標籤依式 (1)：相鄰參考放電容量依事件序線性內插 ÷ Q_ref（`relabel_v92.py` L126–129；Q_ref＝參考放電最大容量，L120） | II-B 式 (1) |
| `cap_Ah` | Ah | `soh × Q_ref`（`soh_v911.py` L227） | — |
| `T` | °C | 該循環各事件之充電段平均溫度之平均（事件值：`extract_*2.py` L76 `nanmean(T)`；循環平均：`data_v911.py` L86）。稿件表 2 之電芯溫度＝以 `n_ev` 加權之事件平均 Σ(T·n_ev)/Σn_ev（與封存 `census_v911.csv` 之 `T_mean` 相同，34 顆最大差 7.1 × 10⁻¹⁵）；若對 `T` 欄逐列平均，Oxford 上限為 26.5 °C 而非表 2 之 26.4 °C | II-A（溫度閘門以事件值判定，`relabel_v92.py` L160–170）；表 2 |
| `I_max` | A | 該循環各事件充電段最大電流之平均（事件值：`extract_*2.py` L70、L76；循環平均：`data_v911.py` L86） | — |
| `n_ev` | 個 | 該循環合併之充電事件數（`data_v911.py` L83、L87） | II-A |
| `seg_first_V`、`seg_last_V` | V | 該循環（事件平均後）Q(V) 第一個／最後一個有限網格點電壓（`ic_v911.py` L149–156；`soh_v911.py` L213） | II-C（閘門 A1） |
| `seg_valid_d0`、`seg_valid_d50`、`seg_valid_d100` | 0／1 | A1 有效性：`seg_first_V ≤ V_lo + δ` 且 `seg_last_V ≥ V_hi − δ`，δ＝0、0.05、0.10 V（`ic_v911.py` L159–168；`soh_v911.py` L193、L230–232） | II-C（閘門 A1） |
| `FV_<V>` | Ah/V | 濾波後 IC 於最接近 `<V>` 之網格點之值（`ic_v911.py` L73–75）；IC＝中央差分 (3)（L31–39，`np.gradient`，5 mV 網格），濾波＝高斯核 (4)（L42–50；σ：NASA 15、自有 20、Oxford 10 mV），Q(V) 前導缺值補 0、中段線性內插（L19–28） | II-D 式 (3)(4) |
| `peak_h` | Ah/V | 視窗內濾波後 IC 之最大值（`ic_v911.py` L78） | II-D |
| `V_pk` | V | `peak_h` 所在網格電壓（`ic_v911.py` L78） | II-D |
| `area` | Ah | 視窗內有限點之 IC 梯形積分（`ic_v911.py` L79） | II-D |
| `intercept`、`slope` | Ah/V；Ah/V² | 視窗內 IC 對電壓之最小平方直線之截距與斜率（`ic_v911.py` L80–81） | II-D |
| `t_pk_h` | h | 峰值網格點之累積電量 ÷ 充電電流（`ic_v911.py` L82；電流為該循環 `I_max`，`soh_v911.py` L219） | II-D（峰時間） |
| `dQ_<V>` | Ah | 部分充入電量增量 Q(`<V>`) − Q(V_lo)（`ic_v911.py` L136–146；每 50 mV） | II-D（F_both） |
| `cyc_A` | 循環 | 原循環計數（`gaps_v911.py` L43） | II-C、III-A |
| `drop_flag` | 0／1 | 閘門 A2 判定之過渡循環（式 (2)，τ₁＝τ₂＝0.010、前後各 5 循環中位數；`gaps_v911.py` L15–33、L44；參數 `config_v911.py` L32）。**對所有來源皆計算**；是否生效由 `label_gate` 決定 | II-C 式 (2) |
| `cyc_c` | 循環 | A2 剔除過渡循環後之連續重編索引（1..N）；過渡循環為空值（`gaps_v911.py` L46、L53–54） | III-A |
| `label_gate` | on／off | 該來源鎖定之 A2 開關（NASA off、自有 on、Oxford on；`stage_data_v911.py` L31） | II-C |

全部特徵欄（FV、peak_h、V_pk、area、intercept、slope、t_pk_h、dQ）於整條電芯序列上以視窗 15 之置中滾動穩健 z 分數（|z| > 3.5）及首尾各 7 點邊界規則（偏離 > 10%）清洗並以線性內插取代（`ic_v911.py` L89–116；`soh_v911.py` L223–225；參數 `config_v911.py` L332）。

## 4. 附加標註欄（`ann_`）

| 欄 | 值 | 定義（程式出處） | v34 |
|---|---|---|---|
| `ann_measured_label` | 0／1 | 該列 SOH 標籤是否含實測值：該循環（同 `cyc_A`）之任一充電事件為「實測事件」即為 1，否則（純內插）為 0。實測事件＝其後、下一個可用事件之前接有參考放電（`seg_i < cap_seg < seg_{i+1}`）。逐事件規則：`03_審查後處理/addendum_C2.py` L19–22（＝`05_審稿補跑/s5_common.py` L288–292）；逐事件→逐列：`s5_common.py` L295–307、`s7_meas_flag.py` L14–23 | II-B（實測參考比例） |
| `ann_n_meas_ev` | 個 | 該循環之實測事件數（`s7_meas_flag.py` L21–22 `n_meas_ev`） | II-B |
| `ann_A1_pass` | 0／1 | 依該來源鎖定 δ 之 A1 結果：NASA δ＝∞ 恆為 1；自有、Oxford δ＝0 取 `seg_valid_d0`（`soh_v911.py` L196–201） | II-C |
| `ann_A2_drop` | 0／1 | `drop_flag = 1` 且該來源 A2 開啟（`label_gate = on`）；NASA 恆為 0 | II-C |
| `ann_in_model` | 0／1 | 是否為建模／評估列（保留列）：A2 開啟時取 `drop_flag = 0` 之列，A2 關閉時取全部列，去除含空值之列（`stage_data_v911.py` L120–126），再依 A1 過濾（`soh_v911.py` L196–201）；呼叫順序同 `run_soh_v3.py` L56–65 | III-A（保留列） |
| `ann_eval_idx` | 整數或空 | 建模列內依原順序之 0 起序號 i；非建模列為空（`stage_soh_v911.py` L205 `idx = arange(n)`） | III-A |
| `ann_cyc_axis` | 循環 | P-B+ 殘差 GPR 之循環軸值 cᵢ：A2 開啟時＝`cyc_c`，關閉時＝`cyc_A`（取該欄原字串；A2 剔除列為空）。A1 移除列處保留空缺 | III-A 式 (8)(9) |
| `ann_ref_M10`、`ann_ref_M20` | 0／1 | 建模列中 i mod M = 0 者為參考列（`stage_soh_v911.py` L205）；參考列不計入評估集 | III-A 式 (7)、III-B |

**逐列旗標與逐顆實測參考比例之關係**：v34 之「實測參考比例」以**閘門前之充電事件**計數（第 II-B 節），＝`Σ ann_n_meas_ev ÷ Σ n_ev`（對該顆全部列加總）；本包 `cells_summary.csv` 之 `measured_ref_fraction` 即此值。`ann_measured_label` 則是**逐循環列**旗標（同循環任一事件實測即為 1），其列比例與事件比例不同：例如 B0005 有 167 個事件（對應 168 次參考放電）、166 個實測事件，事件比例 0.9940，而 166 列皆為實測列（列比例 1.000）。參考列（`ann_ref_M*`）依保留列位置選取，不限於實測列（v34 第 III-A 節）。

## 5. `cells_summary.csv`

| 欄 | 定義 |
|---|---|
| `source`、`cell` | 來源與電芯 |
| `group` | Oxford 老化組（`G1`…`G10`）；其餘空白 |
| `own_campaign` | 自有測試批次（`cell2/cell3`、`GA1-GA7`），取自 `post_review/c2_own_campaign_split.csv` |
| `cluster` | cluster bootstrap 之 cluster：NASA＝`NASA_<cell>`、自有＝`OWN_<campaign>`、Oxford＝`OX_<組>`（`05_審稿補跑/s6_cluster_bootstrap.py` L44–54） |
| `label_stratum` | `measured`（實測標籤子集 25 顆：NASA、自有、Oxford G7–G10）或 `interpolated`（內插子集 9 顆：Oxford G1、G3、G5）（`s5_common.py` L240；v34 第 II-B 節） |
| `q_ref_Ah` | Q_ref（Ah），取自封存 `results_R3_dedup/frames_meta_v911.json` 之 `q_ref_Ah` |
| `n_rows` | 循環索引列數（＝表 2 之每顆循環數） |
| `n_events_pre_gate` | 閘門 A1／A2 前之充電事件數（`Σ n_ev`） |
| `n_A1_removed` | A2 保留列中 A1 不通過之列數（與 `stage_data_v911.py` L36–37 之計數方式相同） |
| `n_A2_removed` | A2 移除之列數 |
| `n_dropna_removed` | 因空值移除之列數（本次全部為 0） |
| `n_in_model` | 建模／評估列（保留列）數 |
| `n_measured_events`、`measured_ref_fraction` | 實測事件數與實測參考比例（事件計，見 §4） |
| `n_rows_measured` | `ann_measured_label = 1` 之列數 |
| `n_ref_M10`、`n_ref_M20`、`n_eval_M10`、`n_eval_M20` | P-B+ 之參考列數與非參考評估列數 |

## 6. 驗證摘要（數值由 `build_log.json` 之 `checks` 抄錄；另註明出處者除外）

- 34 顆；列數 16 567（NASA 629、自有 2847、Oxford 13 091）；每顆列數範圍 NASA 131–166、自有 185–581、Oxford 227–1306；閘門前事件 NASA 632、自有 2847、Oxford 14 174。
- A1 移除：cell2 78、Oxford 合計 25；A2 移除：自有 48（GA1–GA7 42、cell2＋cell3 6）、Oxford 69、NASA 0。與封存 census_v911.csv 逐顆相同（依鎖定設定換算：NASA 因 δ = ∞ 與 A2 關閉，期望值為 0；自有與 Oxford 直接取 census 之 n_seg_invalid_d0／n_label_dropped）。census 中 NASA 之 n_seg_invalid_d0 合計 198（B0005 79、B0006 87、B0007 27、B0018 5；取自封存 census_v911.csv），即稿件第 II-C 節『若 δ = 0 會拒絕 629 中 198』之出處。
- 參考列數（平均 [範圍]）：NASA M=10 16.25 [14–17]、M=20 8.5 [7–9]；自有 30.78 [19–58]、15.67 [10–29]；Oxford 62.29 [23–129]、31.38 [12–65]；Oxford G1/G3/G5 M=10 88–129、M=20 44–65。逐顆 `n_ref`／`n_eval` 與封存 `soh_records_v911.csv` 之 P-B+ 列（298 列、全部模型）全部相同；參考列與評估列之 `ann_cyc_axis` 與封存 `soh_predictions_v911.json` 之 P-B+（GPR-LSP，68 鍵）`ref_cyc`／`cyc` 逐值相同。
- 建模列與以核心模組 `stage_data_v911.load_frames` ＋ `soh_v911.apply_delta` 載入封存 frames 之結果逐顆相同（34／34）。
- 實測參考比例（事件計）：NASA 0.7784–0.9940、自有 0.7983–1.0000（cell2 0.7983、cell3 0.9707、GA1–GA7 0.9946–1.0000）、Oxford G1/G3/G5 0.0206–0.0298、G7–G10 0.7753–0.8980；與封存 `post_review_R3/c2_measured_fraction_per_cell.csv` 之事件數與實測事件數逐顆相同，比例：以 `float_precision="round_trip"` 解析兩檔則 34 顆逐值相同（差 0）；以 pandas 預設浮點解析讀取已發布 `cells_summary.csv` 與封存檔，最大差 2.2 × 10⁻¹⁶（解析末位誤差）。逐列旗標與封存 `results_R3_post/figs/frames_meas_flag_R3.csv` 16 567 列全部相同。

## 7. 來源與授權

- Oxford 衍生列（`ox__*.csv`）源自 Oxford 路徑相依劣化資料（T. Raj，牛津大學；Part 1 DOI 10.5287/bodleian:v0ervBv6p、Part 2 DOI 10.5287/bodleian:2zvyknyRg、Part 3 DOI 10.5287/bodleian:j1a2eD7ow），該資料依 Open Data Commons Open Database License 取得（v34「資料可用性」原文）。衍生列之再散布條件：待作者填寫。
- NASA 衍生列（`nasa__*.csv`）源自 NASA Ames 預後卓越中心之 Battery Data Set（B. Saha 與 K. Goebel，2007）。其授權條款：待作者填寫。
- 自有衍生列（`own__*.csv`）：源自作者實驗室資料；v34 載明原始循環資料不公開。本包只含其逐列衍生值，是否公開之最終決定：待作者填寫。
- 本包之授權：待作者填寫。
- 不含 v34 排除之電芯（Oxford 第 2、4、6 組與 G3 Cell15）。
