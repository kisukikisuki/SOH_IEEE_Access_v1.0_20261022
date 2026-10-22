# 參考感知跨電芯 SOH 評估——分析程式與衍生資料（繁體中文說明）

公開包版本 1.0（對應稿件 v34，2026-10-22）。本文與 `README.md` 內容對應；指令區塊與程式樹以 `README.md` 為準。

> 〈部分增量容量特徵能否勝過保持最近一次實測參考容量？18650 鋰離子電芯健康狀態估測之參考感知跨電芯評估〉
> 作者：MING-JIA CAI、HONG-JIE DAI、TSUNG-HSUN WU；預定投稿 *IEEE Access*。
> 作者順序待作者確認：中文版 v34 為 Cai、Dai、Wu，英文版 v34-EN 為 Dai、Cai、Wu。

論文 DOI：待作者填寫　Zenodo DOI：待作者填寫　ORCID：待作者填寫　授權：待作者填寫（見第 6 節）

## 本包內容

讀者可由 `data/` 之 34 顆逐列衍生表重跑 P-B／P-B+、三種僅用參考容量之基線與 cluster bootstrap，並與封存結果逐值比對。產生衍生表之前處理程式一併收錄供檢視；能否由原始資料重跑見第 4 節。

| 分析（稿件章節） | 資料夾 | 本包可否重跑 |
|---|---|---|
| 前處理：NASA／Oxford 原始 CSV→npz→逐列特徵、SOH 標籤 (1)、閘門 A1/A2、IC (3)(4)、特徵族、清洗（第 II 節） | `code/01_download_and_convert/`、`code/02_core_modules/`、`code/07_build_derived_tables/` | 程式供檢視。官方下載檔→其所讀 CSV 之轉換程式缺；npz 不散布；公開設定下之 raw→npz→特徵路徑作者未執行驗證（第 4 節） |
| Spearman 篩選 (6) 與標準化於每折內執行；REL (5) 逐電芯套用、僅 NASA（第 II-D 節） | `code/02_core_modules/` | 可，隨 P-B／P-B+ 執行 |
| P-B（留一電芯）與 P-B+（稀疏參考更新、殘差 GPR (7)–(9)，M = 10、20），五種迴歸器（Oxford 無 GPR-SE）（第 III-A、III-C 節，表 4） | `code/02_core_modules/`、`code/03_run_PB_PBplus/` | 可 |
| 相同非參考評估列上之 ZOH、LIN2、LINALL（第 III-B 節） | `code/04_baselines/` | 可 |
| cluster bootstrap（percentile 與 BCa，B = 10 000，種子 20260830）與輔助之 REML 隨機截距模型（第 III-E 節、補充表 S14） | `code/05_cluster_bootstrap/` | 可 |
| 與封存結果比對、重算稿件數字 | `code/06_verify/` | 可 |
| 34 顆電芯逐列衍生表 | `data/` | — |

**不含**（不在本次公開範圍）：可執行之 P-A、P-C 流程；組態搜尋與留一電芯重新選擇（表 9、圖 8）；50 項 Holm 檢定族及其他 Wilcoxon／Mann–Whitney／Hodges–Lehmann 分析；表 7 之拆解對照 (a)–(c)；NASA 去洩漏變體（表 10）；留一組與跨批次（補充表 S12）；M = 5／30／50／100 之交叉週期與只在第 0 列一次參考（補充表 S7）；參考雜訊（補充表 S8）；人工稀疏化與實測參考列（補充表 S16）；圖表與稿件排版程式。

仍含本包未使用內容之原樣複製檔：`stage_soh_v911.py`（P-A／P-C 函式）、`soh_v911.py`（P-A／P-C 輔助）、`run_soh_v3.py`（search、P-A、P-C 與 ENV／PREREG17／DATA 階段；PREREG17 所需檔案未收錄，請勿執行；DATA 階段僅供對照，公開設定下未驗證）、`config_v911.py`（搜尋空間與另一項 RUL 研究之設定）。runner 會 import 這些檔案，但不呼叫上述未使用部分。

## 1. 快速重現（由衍生表起跑）

封存執行之環境：Python 3.11.17、numpy 2.4.4、pandas 3.0.2、scikit-learn 1.8.0、scipy 1.17.1、statsmodels 0.15.0、patsy 1.0.3（`requirements.txt`），每程序 1 個 BLAS 執行緒（runner 以 `V3_BLAS_THREADS` 設定，預設 1）。Windows 中文路徑另設 `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1`。

statsmodels：封存執行時 statsmodels 0.15.0 與 patsy 1.0.3 以 `pip install --no-deps` 安裝，公式介面經 patsy 執行，未安裝 formulaic。欲與之相同，請照 `README.md` 第 1 節之兩行安裝指令。若無法 import statsmodels，第 3 步仍以回傳碼 0 結束，但混合效應模型欄寫「未能執行」，第 4 步會判為不符。

五步指令（runner → 基線 → bootstrap → 與封存比對 → 重算稿件數字）見 `README.md` 第 1 節。

原基線程式 `soh_post_v911r.py` 於模組頂層會一併計算配對 Wilcoxon 檢定、平均 RMSE 之 bootstrap 區間與 CI 涵蓋率摘要，皆屬本包範圍外之分析；`run_baselines.py` 於執行後刪除這兩個附帶檔（`A04_A09_B05_soh_post_summary.json`、`A12_ci95_coverage.json`），並不保存原程式之標準輸出（含範圍外之檢定與涵蓋率列印），只將回傳碼、標準錯誤與刪除之檔名記入 `log_run_baselines.txt`。第 3–5 步只使用 `A04_reference_only_baselines_percell.csv`。

於原工作站（Windows 10、20 邏輯核心、`--workers 18`、每程序 1 個 BLAS 執行緒、statsmodels／patsy 如上）2026-10-22 三次獨立執行（其中一次由獨立審查者執行）之結果：

- 第 1 步產生 447 列（P-B 149、P-B+ M = 10 149、P-B+ M = 20 149）。runner 牆鐘（`run_info.json` 之 `wall_sec`）：1923 s、1938 s、1908 s。最後一次使用本包 1.0 版檔案；前兩次使用發布前版本之新撰檔（其後曾修改，含基線包裝之附帶輸出處理）；三次之比較結果皆與封存逐值相同。
- 第 4 步：所有比對值與封存結果完全相同（容差 1e-9，最大差 0）；第 5 步：重算之 145 項稿件數字全部相符。`reproduce_v34_numbers.py` 以 Python format 捨入（半數取偶），此點只影響一個值：NASA M = 10 之 n_ref 平均 16.25，表 8 印為 16.2。

**跨平台限制（稿件第 III-F 節）**：在另一平台上，特徵與 Ridge、SVR 誤差可重現，GPR 與 MLP 誤差並非位元相同；`compare_to_archive.py` 會列出所有差值。

讀表注意：runner 以 `float_precision="round_trip"` 解析 CSV；若用 pandas 預設解析，特徵末位會差（最大約 2e-13），試跑時使 MLP 之 RMSE 至多改變約 1.3 pp、SVR 至多約 0.04 pp。

## 2. 解讀限制（依稿件原文）

- P-B+ 與基線之比較及標籤分層，皆為事後、探索性分析（第 I 節）。
- cluster bootstrap 與混合效應模型另未經多重比較調整（第 III-E 節）；稿件中經 Holm 調整之檢定不在本包內。
- cluster 數不多於 4 之子集，其 bootstrap 區間不可靠；NASA 每 cluster 僅 1 顆，混合效應模型不可識別（第 III-E 節、補充表 S14）。
- 整體流程為離線、非因果：閘門 A2 使用階梯之後的標籤、特徵清洗為置中視窗、REL 以各電芯（含被留出之測試電芯）自身前三個保留列之中位數相除、Q_ref 為整段紀錄之最大值（第 II-B、II-C、II-D、III-A 節）；只有 P-B+ 之殘差校正步驟為因果。
- 鎖定組態係在相同電芯上選出（第 III-E 節），對原始 27 顆電芯為樂觀；P-B+ 於參考列使用測試電芯真實且無雜訊之實驗室容量。
- 全部結果僅以單一種子取得（20260830；殘差 GPR 重啟種子 0）；GPR 超參數、對數邊際概似與收斂狀態未存檔，最佳化收斂警告被抑制（第 III-C、III-F 節）。

## 3. 衍生資料（`data/`）

- `data/derived_per_row/<來源>__<電芯>.csv`：每顆一檔（NASA 4、自有 NCR18650GA 9、Oxford 21；合計 16 567 個循環索引列）。左側為鎖定組態執行實際使用之逐列表原欄位原值；右側 `ann_` 欄為本次加註：實測／內插標籤、A1 通過、A2 剔除、是否進入建模、評估列序號 *i*、殘差 GPR 循環軸 *c_i*、M = 10／20 之參考列旗標（*i* mod M = 0）。
- `data/cells_summary.csv`：每顆一列，含 bootstrap 用 cluster（Oxford 老化組、自有測試批次、NASA 電芯）、標籤分層（實測 25／內插 9）、Q_ref、列數、A1/A2 移除數、實測參考比例、M = 10／20 之參考列數與評估列數。
- `data/DATA_DICTIONARY.md`：每欄定義、單位與程式出處。

特徵為候選全集，檔內未做篩選、標準化或 REL。每折內，程式只以訓練列標準化、只以訓練電芯篩選。篩選之實作（`soh_v911.select_features`）：對每顆訓練電芯分別計算 Spearman ρ；保留「各訓練電芯 |ρ| 之最小值 ≥ 0.85」之特徵；依各訓練電芯 |ρ| 之平均排序取前 12。通過者不足 12 個時，程式會退回由全部候選排序取前 12；鎖定組態執行之 34 折皆未觸發此退回分支。各折通過門檻之候選數：NASA 21、自有 18–19、Oxford 12–13；Oxford 21 折中有 20 折恰好 12 個通過，即全部採用。REL（僅 NASA）依第 II-D 節逐電芯套用，含測試電芯。runner 建模前會丟棄所有 `ann_` 欄。

## 4. 由原始資料重建（僅 NASA 與 Oxford）

見 `code/01_download_and_convert/download_instructions.md` 與 `README_convert.md`；`check_raw_files.py` 可不連網檢查本機原始檔之檔名、表頭、列數，以及 9 顆去重後 Oxford 檔之 SHA-256。npz→逐列特徵為原驅動程式之 DATA 階段（`README_convert.md` 第 2C 節）；`code/07_build_derived_tables/` 記錄封存 frames 如何轉為 `data/`。

已知缺口（照實說明）：

- 將 NASA、Oxford 官方下載檔轉成 `extract_nasa2.py`／`extract_ox2.py` 所讀之 11 欄 CSV 的程式不在作者封存中，使用者需自行轉換，再以 `check_raw_files.py` 所列列數（Oxford 9 顆另有 SHA-256）核對。
- 依稿件所述，Oxford Research Archive 之原始 Part 1／Part 2 檔未取得，未逐檔比對。
- 依稿件第 III-F 節，由原始檔重建時原始 27 顆電芯之標籤差異至多 1.4 × 10⁻¹⁴，但 Q(V) 網格略有改變、原因未定；在其他機器重建可能非位元相同。
- 只用 NASA 與 Oxford 原始資料之路徑，作者未在公開設定下執行驗證（`README_convert.md` 第 5 節）。
- `code/01_download_and_convert/` 中部分原樣複製之腳本含作者機器之硬編碼路徑，行號列於 `README_convert.md`。其他位置另有兩個複製檔含此類路徑：`code/02_core_modules/pv911_run_SOH/02_programs/data_v911.py` L11（註解）與 `code/04_baselines/common_v911r.py` L5–6（預設結果路徑，`run_baselines.py` 以環境變數 `PV911_RESULTS` 覆蓋）。原檔皆未修改。
- 自有 NCR18650GA 原始時間序列不公開（稿件「資料可用性」），其轉檔程式亦不收錄；自有 9 顆只能由衍生表重跑。

## 5. 程式結構與來源

結構見 `README.md` 第 5 節。由作者封存複製之檔案與原檔逐位元相同；`MANIFEST.csv` 列出除其本身與 `SHA256SUMS.txt` 以外每個檔案之封存原路徑（新撰者為 `NEW`）、SHA-256、角色與對應稿件章節；`SHA256SUMS.txt` 涵蓋全包。原樣複製檔保留作者原始註解與資料夾名稱，其中提及內部分析計畫版本、另一項 RUL 研究與稿件修訂流程，這些註解不是本文之宣稱。新撰檔案檔頭皆註明所呼叫或複製之原程式與行號。固定設定（分析計畫第 17 版鎖定組態（表 4）、全域種子 20260830、殘差 GPR 重啟種子 0、bootstrap B = 10 000、種子 20260830）取自原程式，不開放為選項。

## 6. 資料來源與授權

- NASA 電池資料集（B. Saha 與 K. Goebel，NASA Ames 預後卓越中心，2007；B0005、B0006、B0007、B0018）：https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ ——授權：待作者填寫。
- Oxford 路徑相依劣化資料（T. Raj，牛津大學；Part 1 為 2020 年，Part 2 與 Part 3 為 2021 年），Oxford Research Archive，Open Data Commons Open Database License：Part 1 https://doi.org/10.5287/bodleian:v0ervBv6p、Part 2 https://doi.org/10.5287/bodleian:2zvyknyRg、Part 3 https://doi.org/10.5287/bodleian:j1a2eD7ow。`data/` 中 Oxford 各列衍生自此 ODbL 資料。
- 自有 NCR18650GA（cell2、cell3、GA1–GA7）：原始資料不公開，僅含逐列衍生表。
- 本包程式與衍生表之授權：待作者填寫。

## 7. 引用

待作者於接受後填寫（論文 DOI 與 Zenodo DOI）。
