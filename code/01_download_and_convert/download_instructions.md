# 原始資料下載與放置說明（NASA／Oxford）

> 新撰（公開程式包）。本檔所列網址只取自稿件 v34「資料可用性」段；本包**不附任何原始資料**，亦不提供自動下載。
> 作者自有 NCR18650GA 電芯（cell2、cell3、GA1–GA7）之原始時間序列不公開（v34「資料可用性」），本包不含其轉檔程式。

## 1. 資料來源（照錄 v34「資料可用性」）

| 資料 | v34 所載出處 | 本研究使用之電芯 |
|---|---|---|
| NASA 電池資料集（「Battery Data Set」，B. Saha 與 K. Goebel，NASA Ames 預後卓越中心，2007） | NASA Ames 預後卓越中心資料集儲存庫：https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ | B0005、B0006、B0007、B0018 |
| Oxford 路徑相依劣化資料 Part 1（T. Raj，牛津大學，2020） | https://doi.org/10.5287/bodleian:v0ervBv6p | 第 1、3 組（第 1、3 組之檔案始於 Part 1 並延續至 Part 2） |
| Oxford 路徑相依劣化資料 Part 2（T. Raj，牛津大學，2021） | https://doi.org/10.5287/bodleian:2zvyknyRg | 第 1、3 組之延續段；第 5 組 |
| Oxford 路徑相依劣化資料 Part 3（T. Raj，牛津大學，2021） | https://doi.org/10.5287/bodleian:j1a2eD7ow | 第 7、8、9、10 組 |

Oxford 資料依 Open Data Commons Open Database License 取得（v34 原文）。NASA 資料之授權條款：待作者填寫。

v34 第 II-A 節原文：「第 1、3 組各電芯之 Part 1／Part 2 接合列，以充放電機累計測試時間（test_time_s）之重置定位，每顆皆得唯一接合列（接合段之處理見第 II-B 節）；**Oxford Research Archive 之原始 Part 1／Part 2 檔未取得，未逐檔比對。**」

使用之 21 顆 Oxford 電芯（v34 表 2；檔名寫法依轉檔程式 `rebuild_chain.py` L16）：
G1__Cell9、G1__Cell15、G1__Cell20、G3__Cell10、G3__Cell11、G3__Cell14、G5__Cell5、G5__Cell6、G5__Cell16、G7__Cell4、G7__Cell19、G7__Cell27、G8__Cell5、G8__Cell18、G8__Cell20、G9__Cell1、G9__Cell8、G9__Cell9、G10__Cell2、G10__Cell22、G10__Cell25。
（v34 第 II-A 節：第 2、4 組、日曆老化對照組第 6 組及截斷檔 G3 Cell15 不納入。）

## 2. 轉檔程式預期之輸入格式（由程式碼讀出）

轉檔鏈（見 `README_convert.md`）之第一步（`_upstream/extract_nasa2.py`、`extract_ox2.py`、`a_layer_full.py`、`05_審稿補跑/s1_find_join.py`）讀取的是**每顆電芯一個、已整理為 11 欄之 CSV**，而非 NASA／Oxford 官方發布之原始檔格式：

- 表頭（逐字；`extract_ox2.py` L11–12 `COLS`）：
  `source,cell_id,cycle,step,regime,time_s,test_time_s,voltage_V,current_A,capacity_Ah,temperature_C`
- 程式實際使用之欄：
  - `extract_*2.py` L13（`USE`）：`regime, time_s, voltage_V, current_A, capacity_Ah, temperature_C`；其中充電／放電／靜置之區分由電流重新判定（L121：|I| < 0.02·QPROV 為靜置，I > 0 為充電，I < 0 為放電），故 **充電電流須為正、放電電流須為負**。
  - `a_layer_full.py` L13（`USE`）：`cycle, time_s, voltage_V, current_A, capacity_Ah, temperature_C`。
  - `s1_find_join.py` L15（`USE`）：另讀 `test_time_s`（只記錄各段首末值，接合列之定位方式見 `README_convert.md` §2 B）；`cycle` 欄於作者之 Oxford CSV 中為「檔段索引」，非物理循環（作者內部紀錄，未收錄）。
  - 列過濾（`extract_*2.py` L15）：2.00 ≤ V ≤ 4.30 V、|I| ≤ 6.0 A、−20 ≤ T ≤ 80 °C。
  - 單位依欄名：V、A、Ah、°C、s。
- 目錄與檔名（`rebuild_chain.py` L17–20；`s1_rebuild_chain_dedup.py` L23–26）：
  ```
  <RAW>/nasa/nasa/B0005.csv  …（NASA；作者鏈另含 13 顆非期刊 NASA 電芯，見 §3）
  <RAW>/oxford/oxford/G1__Cell9.csv  …（Oxford 21 顆，原檔）
  <RAW>/oxford/oxford_dedup/G1__Cell9.csv …（G1/G3/G5 9 顆去重後，由 s1_dedup_csv.py 產生）
  ```
  注意：`extract_*2.py` L62 以「路徑字串含 `oxford`」或「電芯名含 `G`」判定暫定電流門檻；`<RAW>` 路徑本身不可含 `oxford` 字樣，否則 NASA 電芯之門檻會被誤判（作者內部之 `run_data.bat` 亦註記同一限制；該檔未收錄）。

**重要缺口（照實記錄）**：將官方下載檔轉為本程式所讀 11 欄 CSV 之程式不在作者封存內；來源檔版本與轉換方式：待作者填寫。

因此本包無法提供「官方檔 → 11 欄 CSV」之步驟；使用者需自行轉成上述格式，並以 `check_raw_files.py` 對照作者封存之列數（及 9 顆 Oxford 之 SHA-256）判斷是否與作者所用檔相同。

## 3. 作者鏈所需之檔案清單與可核對之封存值

`check_raw_files.py` 內含下列預期值（不連網）：

- 每檔資料列數：取自作者封存 `build_logs/relabel_audit_nasa.csv`、`relabel_audit_ox.csv` 之 `rows_in`（extract 以 pandas 解析之列數）。
- SHA-256：只有 Oxford G1/G3/G5 9 顆之原始 CSV 與去重後 CSV 有封存值（`oxford_dedup/reference_archived/dedup_ranges_<cell>.json` 之 `sha256_in`／`sha256_out`，由 `s1_dedup_csv.py` 計算）。其餘檔之雜湊未封存，不檢查。
- NASA 非期刊電芯 B0029–B0044（13 顆）：作者之完整鏈（`rebuild_chain.py`）一併擷取，供 `make_dataset_v9.py` L40 之 40 顆斷言與普查排除理由；v34 之分析不使用。

用法：
```
python check_raw_files.py --raw-root <RAW>                      # 檢查存在與表頭
python check_raw_files.py --raw-root <RAW> --count-rows --sha256 --dedup   # 另核列數、9 顆 SHA-256、去重後檔
```
結束碼 0 表示 v34 所需檔案皆存在且所選檢查無不符。
