#!/usr/bin/env python3
"""check_raw_files.py — 檢查使用者本機放置之 NASA／Oxford 逐顆 11 欄 CSV 是否齊全（不連網、不下載）。

新撰（公開程式包）。只讀檔、不改檔。
預期檔名與目錄配置取自轉檔程式：01_資料鏈_raw_to_npz/rebuild_chain.py L13–20（NASA、OX 清單與 csv_of）、
05_審稿補跑/s1_rebuild_chain_dedup.py L23–26（oxford_dedup 子夾）。
預期欄位取自 _upstream/extract_nasa2.py、extract_ox2.py L11–12（COLS，11 欄）。
預期資料列數取自作者封存之擷取／標籤稽核表（5版/01_電池資料/build_logs/relabel_audit_nasa.csv、relabel_audit_ox.csv 之 rows_in 欄；
該值為 extract 以 pandas 解析之列數，若檔內無格式錯誤列，等於檔案行數減 1）；
Oxford G1/G3/G5 9 顆原始 CSV 與去重後 CSV 之資料列數與 SHA-256 取自 s1_dedup_csv.py 之輸出
（oxford_dedup/reference_archived/dedup_ranges_<cell>.json 之 rows_in／rows_out／sha256_in／sha256_out）。
其餘 CSV 之 SHA-256 未封存，故不檢查。

用法：python check_raw_files.py --raw-root <RAW> [--count-rows] [--sha256] [--dedup]
  <RAW> 之配置：<RAW>/nasa/nasa/<B00xx>.csv、<RAW>/oxford/oxford/<Gg>__Cell<c>.csv、（--dedup）<RAW>/oxford/oxford_dedup/<Gg>__Cell<c>.csv
  --count-rows：逐檔計算行數（Oxford 單檔數 GB，需時數分鐘）；--sha256：對有封存雜湊之 9 顆計算 SHA-256。
結束碼：0＝v34 所需檔案皆存在（且所選檢查無不符）；1＝有缺檔或不符。
"""
import os, sys, json, argparse, hashlib

COLS = ["source", "cell_id", "cycle", "step", "regime", "time_s", "test_time_s",
        "voltage_V", "current_A", "capacity_Ah", "temperature_C"]          # extract_ox2.py L11–12

# v34 使用之電芯（表 2）
NASA_V34 = ["B0005", "B0006", "B0007", "B0018"]
OX_V34 = [f"G{g}__Cell{c}" for g, cs in ((1, (9, 15, 20)), (3, (10, 11, 14)), (5, (5, 6, 16)), (7, (4, 19, 27)),
                                         (8, (5, 18, 20)), (9, (1, 8, 9)), (10, (2, 22, 25))) for c in cs]   # rebuild_chain.py L16
# 作者鏈另擷取之 NASA 非期刊電芯（rebuild_chain.py L14；make_dataset_v9.py 斷言 40 顆時需要；不進 v34 分析）
NASA_EXTRA = ["B0029", "B0030", "B0031", "B0032", "B0033", "B0034", "B0036", "B0038", "B0039", "B0040", "B0042", "B0043", "B0044"]

ROWS_IN = {  # build_logs/relabel_audit_nasa.csv、relabel_audit_ox.csv 之 rows_in
    "B0005": 591458, "B0006": 591458, "B0007": 591458, "B0018": 314676, "B0029": 110487, "B0030": 110487, "B0031": 110487,
    "B0032": 110487, "B0033": 406812, "B0034": 406812, "B0036": 406812, "B0038": 138423, "B0039": 138423, "B0040": 138423,
    "B0042": 277136, "B0043": 277136, "B0044": 277136,
    "G10__Cell2": 45424341, "G10__Cell22": 49163154, "G10__Cell25": 49160986, "G1__Cell15": 77529569, "G1__Cell20": 81093897,
    "G1__Cell9": 83604339, "G3__Cell10": 80821355, "G3__Cell11": 80812881, "G3__Cell14": 74506235, "G5__Cell16": 23950769,
    "G5__Cell5": 25343585, "G5__Cell6": 27485084, "G7__Cell19": 31344109, "G7__Cell27": 30171857, "G7__Cell4": 31337649,
    "G8__Cell18": 31344194, "G8__Cell20": 31337273, "G8__Cell5": 31340098, "G9__Cell1": 31335368, "G9__Cell8": 31327953,
    "G9__Cell9": 31324573}


def load_dedup_ref():
    here = os.path.dirname(os.path.abspath(__file__)); d = os.path.join(here, "oxford_dedup", "reference_archived")
    out = {}
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.startswith("dedup_ranges_") and fn.endswith(".json"):
                j = json.load(open(os.path.join(d, fn), encoding="utf-8"))
                out[j["cell"]] = dict(rows_in=j["rows_in"], rows_out=j["rows_out"], sha256_in=j["sha256_in"], sha256_out=j["sha256_out"])
    return out


def count_rows(p):
    n = 0
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            n += b.count(b"\n")
    return n - 1   # 扣表頭（末行若無換行符則少計 1，於報告中標示）


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--raw-root", required=True)
    ap.add_argument("--count-rows", action="store_true"); ap.add_argument("--sha256", action="store_true"); ap.add_argument("--dedup", action="store_true")
    a = ap.parse_args(); DD = load_dedup_ref(); rows = []; bad = 0
    plan = [("nasa", c, os.path.join(a.raw_root, "nasa", "nasa", c + ".csv"), True) for c in NASA_V34] + \
           [("nasa", c, os.path.join(a.raw_root, "nasa", "nasa", c + ".csv"), False) for c in NASA_EXTRA] + \
           [("ox", c, os.path.join(a.raw_root, "oxford", "oxford", c + ".csv"), True) for c in OX_V34]
    if a.dedup:
        plan += [("ox_dedup", c, os.path.join(a.raw_root, "oxford", "oxford_dedup", c + ".csv"), True) for c in sorted(DD)]
    for kind, c, p, needed in plan:
        r = dict(kind=kind, cell=c, path=p, used_in_v34=needed, exists=os.path.isfile(p))
        if r["exists"]:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                hdr = f.readline().strip().split(",")
            r["header_ok"] = (hdr == COLS)
            exp_rows = DD[c]["rows_out"] if kind == "ox_dedup" else ROWS_IN.get(c)
            exp_sha = (DD[c]["sha256_out"] if kind == "ox_dedup" else DD[c]["sha256_in"]) if c in DD else None
            r["expected_rows"] = exp_rows; r["expected_sha256"] = exp_sha
            if a.count_rows:
                r["rows"] = count_rows(p); r["rows_ok"] = (r["rows"] == exp_rows) if exp_rows is not None else None
            if a.sha256 and exp_sha:
                r["sha256"] = sha256(p); r["sha256_ok"] = (r["sha256"] == exp_sha)
            if not r["header_ok"] or r.get("rows_ok") is False or r.get("sha256_ok") is False:
                bad += needed
        else:
            bad += needed
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    miss = [r["cell"] for r in rows if r["used_in_v34"] and not r["exists"]]
    print(json.dumps(dict(n_checked=len(rows), missing_used_in_v34=miss,
                          missing_extra_nasa=[r["cell"] for r in rows if not r["used_in_v34"] and not r["exists"]],
                          n_problems_used_in_v34=bad), ensure_ascii=False))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
