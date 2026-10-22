# 01_download_and_convert — NASA／Oxford 原始 CSV → dataset_v9 npz（含 Oxford 接合假影去重）

> 新撰說明（公開程式包）。本資料夾之 `.py`／`.json` 程式檔皆為作者封存程式之**逐位元複本**（保留原檔名與原相對子夾 `01_資料鏈_raw_to_npz/`、`05_審稿補跑/`，使原檔內之相對路徑仍成立）；來源路徑與 SHA-256 見公開包 MANIFEST。新撰檔為 `check_raw_files.py`、本說明與 `download_instructions.md`。
> 原始資料不附；下載與放置見 `download_instructions.md`。官方 NASA／Oxford 檔轉成 11 欄 CSV 之程式不在作者封存內，此步驟待作者填寫（見 `download_instructions.md` §2）。作者自有電芯之轉檔程式（`extract_own.py`、`convert_ga_raw.py`、`ga_make_dataset_v2.py` 等）不收錄。

## 1. 對應 v34 章節

| 步驟 | 程式 | v34 章節 |
|---|---|---|
| 擷取：逐充電事件 CC 段（I > 0.9·I_max）Q(V) 內插至 5 mV 網格；放電段容量 | `_upstream/extract_nasa2.py`、`extract_ox2.py` | II-C（CC 部分、5 mV 網格、擷取視窗 NASA 3.90–4.10 V／Oxford 3.60–4.00 V 由 relabel 之 HI_LO／HI_HI 判定） |
| 標籤：參考放電辨識（兩側起始電壓帶、C/2 帶、孤立尖峰 > 1.06／殘缺 < 0.94、雙序列判定 ≥ 3 段且中位比 ≥ 1.08）、Q_ref＝參考放電最大容量、相鄰參考放電線性內插、溫度閘門 15–35 °C | `_upstream/relabel_v92.py`（L30、L65–78、L81–123、L126–129、L160–170；逐顆模式為 15 ≤ T ≤ 35 °C、NASA 逐事件模式為 15 ≤ T < 35 °C） | II-A（溫度閘門）、II-B 式 (1) |
| 分臂／更名（標籤不變） | `_upstream/make_dataset_v8.py`＋`_upstream/prereg/arms_v93.json` | — |
| A 層：段表、循環計數 cyc_A（非淺放電之累計）、EFC | `_upstream/a_layer_full.py`（L107–108、L124、L130–132）、`a_layer_fix_v9a.py` | II-A（循環計數） |
| 組裝 dataset_v9 npz（加 cyc_A、EFC、F_cond 欄） | `_upstream/make_dataset_v9.py`（作者完整鏈）；`05_審稿補跑/s1_make_dataset_v9_subset.py`（去重 9 顆；只改 L40 之顆數斷言） | II-A |
| 驅動（五階段 extract → relabel → v8 → alayer → v9） | `01_資料鏈_raw_to_npz/rebuild_chain.py`；去重版 `05_審稿補跑/s1_rebuild_chain_dedup.py`（去重版五階段之逐步呼叫見 §2 B） | — |
| Oxford 接合列與逐列複本定位 | `05_審稿補跑/s1_find_join.py`、`s1_join_analyze.py` | II-A（接合列）、II-B（接合假影）；定位方式見 §2 B |
| Oxford 去重 CSV | `05_審稿補跑/s1_dedup_csv.py` | II-B |

npz → 逐列特徵表（frames）屬核心模組（`code/02_core_modules/`，`stage_data_v911.py`；由 `run_soh_v3.py` 之 DATA 階段執行），不在本資料夾。

## 2. 作者實際執行之順序（依封存紀錄）

依據：作者內部執行紀錄（未收錄）。以下指令依該紀錄整理；路徑一律以 `<RAW>`、`<W>` 等佔位。

**A. 全來源重建（2026-10-07，產生 `dataset_v9_npz`）**
```
python 01_資料鏈_raw_to_npz/rebuild_chain.py extract --raw-root <RAW> --work <W> --procs 12
python 01_資料鏈_raw_to_npz/rebuild_chain.py relabel --raw-root <RAW> --work <W>
python 01_資料鏈_raw_to_npz/rebuild_chain.py v8      --raw-root <RAW> --work <W>
python 01_資料鏈_raw_to_npz/rebuild_chain.py alayer  --raw-root <RAW> --work <W> --procs 13
python 01_資料鏈_raw_to_npz/rebuild_chain.py v9      --raw-root <RAW> --work <W>
```
- 輸入：`<RAW>/nasa/nasa/` 17 顆、`<RAW>/own/own/` cell2／cell3（不公開）、`<RAW>/oxford/oxford/` 21 顆（`rebuild_chain.py` L14–20）。輸出 `<W>/dataset_v9/`（40 顆）。
- 各階段輸入輸出（`rebuild_chain.py` L41–79）：extract → `<W>/cache_{nasa,own,ox}/<cell>.npz`；relabel → `<W>/v6_*/`、`relabel_audit_*.csv`（環境變數 HI_LO／HI_HI：NASA 3.90／4.10、own 3.80／4.10、Oxford 3.60／4.00；group_mode NASA＝event、其餘＝cell，L23–24）；v8 → `<W>/pv94/dataset_v8/`；alayer → `<W>/pv94/results_v94/a_layer/`；v9 → `<W>/dataset_v9/`。
- 其後作者另以未收錄之程式加入 GA1–GA7（自有）並以普查規則選出 34 顆（`assemble_and_compare.py`，未收錄）。

**B. Oxford G1/G3/G5 去重（2026-10-21，產生 `dataset_v9_npz_R3dedup`）**
```
python 05_審稿補跑/s1_find_join.py <RAW>/oxford/oxford/<cell>.csv <RES>/s1_join      # 9 顆各一次
python 05_審稿補跑/s1_join_analyze.py                                               # 產生 dup_blocks.csv、oxford_join_points.csv
python 05_審稿補跑/s1_dedup_csv.py <cell>                                            # 9 顆各一次 → <RAW>/oxford/oxford_dedup/<cell>.csv
python 05_審稿補跑/s1_rebuild_chain_dedup.py extract --raw-root <RAW> --work <W_dedup> --procs 9 --ox-subdir oxford_dedup
python 05_審稿補跑/s1_rebuild_chain_dedup.py relabel --raw-root <RAW> --work <W_dedup> --procs 9 --ox-subdir oxford_dedup
python 05_審稿補跑/s1_rebuild_chain_dedup.py v8      --raw-root <RAW> --work <W_dedup> --procs 9 --ox-subdir oxford_dedup
python 05_審稿補跑/s1_rebuild_chain_dedup.py alayer  --raw-root <RAW> --work <W_dedup> --procs 9 --ox-subdir oxford_dedup
python 05_審稿補跑/s1_rebuild_chain_dedup.py v9      --raw-root <RAW> --work <W_dedup> --procs 9 --ox-subdir oxford_dedup
```
- 作者執行上列五階段時之環境變數：`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`＝1，`PYTHONIOENCODING=utf-8`，`PYTHONUTF8=1`（Windows cp950 locale 下 `make_dataset_v8.py` 之 `json.load(open(...))` 未指定編碼，需此設定），`S1_N_CELLS=9`（`s1_make_dataset_v9_subset.py` 之顆數斷言）。五階段依序執行，每一階段結束後應檢查其回傳碼與該階段 log（`<W_dedup>/extract_ox.log`、`relabel_ox.log`、`v8.log`、`alayer.log`、`alayer_fix.log`、`v9.log`）再進行下一階段；`s1_rebuild_chain_dedup.py`（同 `rebuild_chain.py`）不檢查其子程序之回傳碼，子程序失敗時本身仍可能以 0 結束。
- 作者先以 `--ox-subdir oxford`（原檔）跑同一鏈作回歸，9 顆 npz 與 `dataset_v9_npz` 逐鍵相同（作者內部紀錄，未收錄；比對程式亦未收錄）。
- 去重後 9 顆 npz 放入 `dataset_v9_npz_R3dedup`，其餘 25 顆為 `dataset_v9_npz` 之逐位元複本；此複製步驟無對應程式檔（作者內部紀錄，未收錄）。
- **接合列之定位方式（依程式實作）**：
  1. `s1_find_join.py` 逐段（與 `extract_ox2.py` 相同之段切分）記錄原始列號、檔段索引（csv `cycle` 欄之首末值 `part_first`／`part_last`）、`test_time_s` 之首末值（`tt0`／`tt1`，L68–69）與段內 (V, I, Q, T) 之 SHA-1 指紋。
  2. `s1_join_analyze.py` 以逐段指紋找出逐列複本區段（`blocks_of`，L19–42）；其中複本來源落在檔首前 3 段內（`src_seg0 ≤ 3`）且起始列最早者為「檔首複本區段」，其首段之 `part_first` 記為 `part2_first_cycle`（L92–104）。
  3. `s1_dedup_csv.py` 以「接合檔段之前一段（`part_first < part2_first_cycle` 之最後一段）之 `row0`」至「檔首複本區段之 `row0`」為搜尋視窗（L29–33），於視窗內以 csv `cycle` 欄等於 `part2_first_cycle` 之第一列為接合列（L49–52；找不到則 L61 斷言失敗），再刪除接合列至檔首複本區段最後一列之整段（G1/G3 六顆；G5 三顆與 G1 Cell15 另刪逐列複本區段）。
  4. `test_time_s` 只於步驟 1 記錄（`tt0`／`tt1`）並寫入 `dup_blocks.csv`（`s1_join_analyze.py` L68）；程式未檢查 `test_time_s` 是否於接合列重置（`s1_dedup_csv.py` 檔頭 L4 之「test_time_s 於此重置」為註解，非檢查條件）。封存之 `dedup_ranges_<cell>.json` 所記 6 顆接合列原文（`join_line`）之 `test_time_s` 欄，5 顆為 0、G1 Cell9 為 10。
  稿件第 II-A 節寫「以 test_time_s 之重置定位」接合列，與上述程式做法（以 `cycle` 欄於指紋界定之視窗內定位）之關係：待作者確認。
- 逐顆刪除列數、接合列與輸入／輸出 SHA-256：`oxford_dedup/reference_archived/dedup_ranges_<cell>.json`（封存輸出之複本）。`s1_join_analyze.py` 需讀入之 98 列重複特徵列清單（由未收錄之 `addendum_C3R1.py` 產生）亦附於 `reference_archived/c3r1_duplicate_feature_rows.csv`。

**C. 逐列特徵表**：`run_soh_v3.py --dataset dataset_v9_npz_R3dedup` 之 DATA 階段（2026-10-21T01:38Z；作者內部執行紀錄，未收錄）→ `results_R3_dedup/frames/`（34 檔）。本包 `data/derived_per_row/` 即以此 34 檔為底附加標註（`code/07_build_derived_tables/`）。

## 3. 原檔內之硬編碼路徑與需調整之處（不改原檔；請在自己的副本中修改）

| 檔案 | 位置 | 內容 | 調整方式 |
|---|---|---|---|
| `01_資料鏈_raw_to_npz/rebuild_chain.py` | L17–20 `csv_of` | 原始 CSV 目錄配置 `<RAW>/nasa/nasa`、`<RAW>/own/own`、`<RAW>/oxford/oxford` | 以 `--raw-root` 指定；檔案依此配置放置 |
| 同上 | L25 | 預設電芯＝NASA 17＋own 2＋Oxford 21 | 無自有資料時以 `--cells` 指定 |
| `05_審稿補跑/s1_rebuild_chain_dedup.py` | L16 `UP` | 以本檔所在夾之上一層 `01_資料鏈_raw_to_npz/_upstream` 為上游程式位置 | 本包已保留此相對配置，無需修改 |
| 同上 | L23–26、L31 | Oxford 讀 `<RAW>/oxford/<--ox-subdir>`；預設電芯＝G1/G3/G5 9 顆 | 以 `--ox-subdir`、`--cells` 指定 |
| `05_審稿補跑/s1_find_join.py` | — | 無硬編碼（參數 `<csv> <outdir>`） | — |
| `05_審稿補跑/s1_join_analyze.py` | L12–14 | `ROOT`、`RES`、`SJ`、`DATA`（去重前 npz 夾）、`PR`（含 `c3r1_duplicate_feature_rows.csv` 之夾）為作者機器絕對路徑 | 改成本機路徑；`PR` 可指向 `oxford_dedup/reference_archived/` |
| `05_審稿補跑/s1_dedup_csv.py` | L17–19 | `ROOT`、`RES`、`SJ`、`RAW`（原 Oxford CSV 夾）、`OUT`（`oxford_dedup` 夾） | 改成本機路徑；`RES` 需含 `oxford_join_points.csv`，`SJ` 需含 `dup_blocks.csv` 與 `<cell>__segments.csv` |
| `_upstream/make_dataset_v9.py` | L40 | `assert len(files) == 40`（NASA 17＋own 2＋Oxford 21） | 無自有資料時無法通過；`05_審稿補跑/s1_make_dataset_v9_subset.py` 以環境變數 `S1_N_CELLS` 取代此斷言（作者只以 9 顆執行過） |
| `_upstream/a_layer_full.py`、`a_layer_fix_v9a.py` | L19；L8 | 以本檔位置之 `../dataset_v8`、`../results_v94/a_layer` 為輸入輸出 | 由驅動程式（v8 階段）複製到 `<W>/pv94/01_pretrain_programs/` 後執行，無需修改 |
| `_upstream/make_dataset_v8.py` | L5 | 讀本檔旁之 `prereg/arms_v93.json` | 同上（驅動程式一併複製） |
| `_upstream/extract_*2.py` | L62 | 路徑字串含 `oxford` 即採 Oxford 門檻 | `<RAW>` 與工作夾路徑不可含 `oxford` 字樣（Oxford 子夾本身除外） |

### 3.1 原檔內之硬編碼前篩（與稿件所述擷取視窗之關係）

| 檔案 | 位置 | 內容 |
|---|---|---|
| `_upstream/extract_nasa2.py` | L17、L74 | 擷取階段預篩：CC 段（I > 0.9·I_max）單調化後之電壓須涵蓋 3.95–4.05 V（`HI_LO, HI_HI = 3.95, 4.05`），否則不保留該事件 |
| `_upstream/extract_ox2.py` | L17、L74 | 同上，Oxford 預篩為 3.60–4.00 V |
| `_upstream/relabel_v92.py` | L152 | Oxford 只保留充電電流最大值 I_max > 1.0 A 之充電事件（`E[:, 3]`＝`extract_ox2.py` L76 記錄之 I_max；註解「僅 C/2 充電」） |
| 同上 | L153–154 | NASA 逐事件：CC 段電壓須涵蓋 relabel 之 HI_LO–HI_HI（3.90–4.10 V） |
| 同上 | L157–158 | 視窗（HI_LO–HI_HI）內各網格點支持樣本數之中位數須 ≥ 3（`MIN_SUPPORT = 3`，L18） |

實際生效之擷取視窗仍為稿件所述：NASA 3.90–4.10 V、Oxford 3.60–4.00 V（relabel 階段以環境變數 `HI_LO`／`HI_HI` 設定，`rebuild_chain.py` L23、`s1_rebuild_chain_dedup.py` L29）。NASA 之擷取預篩 3.95–4.05 V 被 3.90–4.10 V 之涵蓋條件包含，故不再額外排除事件；Oxford 預篩與生效視窗相同。

## 4. 由原始檔重建之已知差異（v34 第 III-F 節）

v34 第 III-F 節原文：「由原始檔重建全部 34 顆電芯，重現了原始 27 顆電芯的標籤，差異至多 1.4 × 10⁻¹⁴，而 Q(V) 網格略有改變，原因無法確定。主結果一律使用重建網格（Oxford 第 1、3、5 組為去重後之重建網格，第 II-B 節）；差異只在擷取階段之 Q 欄，三種隔離重跑（舊版 NumPy、關閉 AVX 分派、穩定排序）均未重現封存網格，最可能原因為電壓並列值之排序，未經驗證。」
因此他人以本鏈重建時，標籤欄可預期一致，但 Q(V) 網格及由其導出之 IC 特徵可能與 `data/derived_per_row/` 之封存值不完全相同；本包之衍生表為作者封存值（Oxford G1/G3/G5 已去重之版本）。

## 5. 執行環境（作者）

Windows 10；Python 3.11.17、numpy 2.4.4、pandas 3.0.2、scikit-learn 1.8.0、scipy 1.17.1（作者內部紀錄，未收錄）。公開使用者僅以 NASA／Oxford 執行本鏈之路徑（不含自有電芯）作者未執行驗證。
