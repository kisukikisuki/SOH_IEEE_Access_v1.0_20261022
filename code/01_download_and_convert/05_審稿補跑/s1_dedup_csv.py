"""s1_dedup_csv.py — 審稿第 1 項／步驟書 §2.2 步驟 1：依接合列與重複區段刪除 per-cell csv 中之重複區段，輸出 oxford_dedup（原檔不動）。
複製自：新撰。
刪除規則（作者裁定 2026-10-21，見 results_R3_dedup/_notes.json "author_ruling"）：
  G1/G3 六顆：刪除「接合列（csv cycle 欄＝接合檔段之第一列；test_time_s 於此重置）」至「檔首複本區段最後一列」之整段
             （Part 2 開頭之特性測試＋檔首區段逐列複本，視為接合假影）；
  其餘逐列重複區段（G5 三顆之早期複本、G1 Cell15 檔尾複本）：刪除該區段 [row0, row1]。
  98 列中 21 列（G5 EoL 特徵相同列）經 s1_join_analyze 判定非原始資料重複 → 不刪。
只改：—（新撰）。原始列逐位元複製（不重新格式化）；只跳過刪除區間之資料列。
輸入：results_R3_dedup/oxford_join_points.csv、s1_join/dup_blocks.csv、s1_join/<cell>__segments.csv；RAW/oxford/oxford/<cell>.csv（唯讀）
輸出：RAW/oxford/oxford_dedup/<cell>.csv；results_R3_dedup/s1_join/dedup_ranges_<cell>.json
事後聲明：資料層修正；事後（post hoc）、未調整、不在 50 項 Holm 檢定族。
用法：python s1_dedup_csv.py <cell>
"""
import os, sys, json, time, hashlib
import pandas as pd

ROOT = r"D:\AGA分析\3.Meeting\0804-投稿\1006\電池資料\1007\SOH_跨電芯健康狀態估測_IEEE_Access_v3.0"
RES = os.path.join(ROOT, r"5版\03_實驗數據\results_R3_dedup"); SJ = os.path.join(RES, "s1_join")
RAW = r"D:\AGA分析\3.Meeting\0804-投稿\1006\電池資料\oxford\oxford"; OUT = r"D:\AGA分析\3.Meeting\0804-投稿\1006\電池資料\oxford\oxford_dedup"


def main():
    cell = sys.argv[1]; t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    J = pd.read_csv(os.path.join(RES, "oxford_join_points.csv")).set_index("cell")
    B = pd.read_csv(os.path.join(SJ, "dup_blocks.csv")); B = B[B.cell == cell]
    S = pd.read_csv(os.path.join(SJ, f"{cell}__segments.csv")).set_index("seg")
    ranges = []; join_part = None; win = None
    if pd.notna(J.loc[cell, "join_row"]):
        head = B[B.src_seg0 <= 3].sort_values("row0").iloc[0]
        join_part = int(J.loc[cell, "part2_first_cycle"])
        # 接合列一定落在「接合檔段之前一段」之 row0 與重複區段起點之間；只在此視窗內解析 cycle 欄
        segs_before = S[(S.index < int(head.seg0)) & (S.part_first < join_part)]
        win = (int(segs_before.row0.iloc[-1]), int(head.row0))
        ranges.append(dict(kind="join_to_end_of_head_copy", row0=None, row1=int(head.row1), block=int(head.block)))
        for _, b in B[B.block != head.block].iterrows():
            ranges.append(dict(kind="row_duplicate_block", row0=int(b.row0), row1=int(b.row1), block=int(b.block)))
    else:
        for _, b in B.iterrows():
            ranges.append(dict(kind="row_duplicate_block", row0=int(b.row0), row1=int(b.row1), block=int(b.block)))
    src = os.path.join(RAW, cell + ".csv"); dst = os.path.join(OUT, cell + ".csv"); tmp = dst + ".tmp"
    hin, hout = hashlib.sha256(), hashlib.sha256()
    n = -1; n_del = 0; join_row = None; join_line = None
    with open(src, "rb") as fi, open(tmp, "wb", buffering=1 << 24) as fo:
        for line in fi:
            hin.update(line)
            if n < 0:                                   # 表頭
                fo.write(line); hout.update(line); n = 0; continue
            i = n; n += 1
            if win is not None and join_row is None and win[0] <= i <= win[1]:
                if int(line.split(b",", 3)[2]) == join_part:
                    join_row = i; join_line = line.decode("utf-8", "replace").strip()
                    ranges[0]["row0"] = i
            drop = False
            for r in ranges:
                if r["row0"] is not None and r["row0"] <= i <= r["row1"]:
                    drop = True; break
            if drop:
                n_del += 1; continue
            fo.write(line); hout.update(line)
    if win is not None:
        assert join_row is not None, f"{cell}: 未在視窗內找到 cycle=={join_part} 之列"
    os.replace(tmp, dst)
    out = dict(cell=cell, rows_in=n, rows_deleted=n_del, rows_out=n - n_del, join_part=join_part, join_row=join_row, join_line=join_line,
               ranges=ranges, sha256_in=hin.hexdigest(), sha256_out=hout.hexdigest(), sec=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(SJ, f"dedup_ranges_{cell}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
