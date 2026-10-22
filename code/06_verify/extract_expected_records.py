"""extract_expected_records.py — 由封存 soh_records_v911.csv 篩出 P-B 與 P-B+ 列，作為 expected/ 比對基準（數值不改）。

新撰（公開程式包）。篩選方法：逐行讀入原檔（原檔每列一行、無內嵌換行），保留表頭行與 protocol 欄（以 csv 模組解析）
∈ {"P-B", "P-B+"} 之資料行，原位元組逐行照抄（含原行尾），不重新格式化任何數值。
同時寫出 <out>.provenance.json：原檔 SHA-256、原檔列數、保留列數、各 (source, protocol, M) 列數、篩選條件。
用法：python extract_expected_records.py --src <results_R3_dedup/soh_records_v911.csv> --origin-label <ROOT 相對路徑> --out <expected/soh_records_v911_PB_PBplus.csv>
"""
import csv, io, json, hashlib, argparse, collections

KEEP = ("P-B", "P-B+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--origin-label", default=None)
    a = ap.parse_args()
    raw = open(a.src, "rb").read()
    lines = raw.splitlines(keepends=True)
    header = next(csv.reader([lines[0].decode("utf-8")]))
    ip = header.index("protocol"); isrc = header.index("source"); im = header.index("M")
    kept = [lines[0]]; cnt = collections.Counter()
    for ln in lines[1:]:
        row = next(csv.reader([ln.decode("utf-8")]))
        if row[ip] in KEEP:
            kept.append(ln); cnt[f"{row[isrc]}|{row[ip]}|M={row[im]}"] += 1
    out = b"".join(kept)
    open(a.out, "wb").write(out)
    prov = dict(origin=a.origin_label or a.src, origin_sha256=hashlib.sha256(raw).hexdigest(), origin_n_rows=len(lines) - 1,
                filter="protocol ∈ {'P-B', 'P-B+'}（原行位元組照抄，含表頭）", kept_n_rows=len(kept) - 1,
                kept_sha256=hashlib.sha256(out).hexdigest(), counts=dict(sorted(cnt.items())))
    json.dump(prov, open(a.out + ".provenance.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(prov, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
