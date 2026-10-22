"""s1_join_analyze.py — 審稿第 1 項／步驟書 §2.1 步驟 0（第 2 段）：由 s1_find_join.py 之逐段指紋找出重複區段、定位接合列、
對照 post_review/c3r1_duplicate_feature_rows.csv 之 98 列複製列與 RPT（cap_seg）位置。唯讀，不跑模型。
複製自：新撰；98 列之來源規則引用 03_審查後處理/addendum_C3R1.py L11–25（FV 向量相同之列）；事件→循環對應引用 data_v911.per_cycle_table（同 cyc_A 之事件聚合）。
只改：—（新撰分析程式）
輸入：results_R3_dedup/s1_join/<cell>__segments.csv；DATA_R1/ox__<cell>.npz（seg、cyc_A、cap_seg、cap_soh、q_ref）；PR/c3r1_duplicate_feature_rows.csv
輸出：results_R3_dedup/oxford_join_points.csv（步驟書 §2.1 欄位＋附加欄）、s1_join/dup_blocks.csv、s1_join/dup_rows_98_mapping.csv、s1_join/rpt_in_blocks.csv、s1_join/join_analysis.json
事後聲明：資料層稽核；事後（post hoc）、未調整、不在 50 項 Holm 檢定族。
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = r"D:\AGA分析\3.Meeting\0804-投稿\1006\電池資料\1007\SOH_跨電芯健康狀態估測_IEEE_Access_v3.0"
RES = os.path.join(ROOT, r"5版\03_實驗數據\results_R3_dedup"); SJ = os.path.join(RES, "s1_join")
DATA = os.path.join(ROOT, r"5版\01_電池資料\dataset_v9_npz"); PR = os.path.join(ROOT, r"5版\03_實驗數據\post_review")
CELLS = ["G1__Cell9", "G1__Cell15", "G1__Cell20", "G3__Cell10", "G3__Cell11", "G3__Cell14", "G5__Cell5", "G5__Cell6", "G5__Cell16"]
MIN_N_START = 10          # 區段起點之段至少 10 列（避免短 rest 段偶然同值）；區段內之後續段不設下限（逐段指紋必須相等）


def blocks_of(S):
    h = S.hash.to_numpy(); n = S.n.to_numpy(); N = len(S)
    first = {}
    src = np.full(N, -1)
    for j in range(N):
        k = (h[j], n[j])
        if k in first:
            src[j] = first[k]
        else:
            first[k] = j
    out = []; j = 0; used = np.zeros(N, bool)
    while j < N:
        if src[j] < 0 or n[j] < MIN_N_START or used[j]:
            j += 1; continue
        o = j - src[j]
        a = j
        while a - 1 >= 0 and a - 1 - o >= 0 and not used[a - 1] and h[a - 1] == h[a - 1 - o] and n[a - 1] == n[a - 1 - o] and a - 1 - o < a - 1:
            a -= 1
        b = j
        while b + 1 < N and h[b + 1] == h[b + 1 - o] and n[b + 1] == n[b + 1 - o]:
            b += 1
        used[a:b + 1] = True
        out.append((a, b, o)); j = b + 1
    return out


def main():
    dup98 = pd.read_csv(os.path.join(PR, "c3r1_duplicate_feature_rows.csv"))
    B_rows, J_rows, M_rows, R_rows = [], [], [], []
    summ = {}
    for cell in CELLS:
        S = pd.read_csv(os.path.join(SJ, f"{cell}__segments.csv")).sort_values("seg").reset_index(drop=True)
        assert (S.seg.to_numpy() == np.arange(1, len(S) + 1)).all(), cell
        z = np.load(os.path.join(DATA, f"ox__{cell}.npz"), allow_pickle=True)
        ev_seg = z["seg"].astype(int); ev_cyc = z["cyc_A"].astype(float); cap_seg = z["cap_seg"].astype(int)
        q = float(z["q_ref"]); cap = z["cap_soh"].astype(float) * q
        blks = blocks_of(S)
        seg2blk = {}
        for bi, (a, b, o) in enumerate(blks):
            sa, sb = S.iloc[a], S.iloc[b]; ra, rb = S.iloc[a - o], S.iloc[b - o]
            segs = S.seg.iloc[a:b + 1].to_numpy()
            for s in segs:
                seg2blk[int(s)] = (bi, o)
            rpt_in = [int(s) for s in cap_seg if a + 1 <= s <= b + 1]
            B_rows.append(dict(cell=cell, block=bi, seg0=int(sa.seg), seg1=int(sb.seg), src_seg0=int(ra.seg), src_seg1=int(rb.seg), offset_seg=int(o),
                               n_seg=int(b - a + 1), row0=int(sa.row0), row1=int(sb.row1), src_row0=int(ra.row0), src_row1=int(rb.row1),
                               n_rows_kept=int(S.n.iloc[a:b + 1].sum()), raw_rows_span=int(sb.row1 - sa.row0 + 1),
                               part_first=int(sa.part_first), part_last=int(sb.part_last), src_part_first=int(ra.part_first), src_part_last=int(rb.part_last),
                               n_charge=int((S.regime.iloc[a:b + 1] == "charge").sum()), n_discharge=int((S.regime.iloc[a:b + 1] == "discharge").sum()),
                               tt0=float(sa.tt0), src_tt0=float(ra.tt0), n_rpt_in_block=len(rpt_in),
                               rpt_caps_in_block=";".join(f"{cap[list(cap_seg).index(s)]:.5f}" for s in rpt_in),
                               ev_cycA_in_block=";".join(f"{c:g}" for c in sorted(set(ev_cyc[(ev_seg >= a + 1) & (ev_seg <= b + 1)])))))
        Bc = [r for r in B_rows if r["cell"] == cell]
        # RPT 逐一：是否落在重複區段（複本側）
        jm = int(np.argmax(cap))
        for k, s in enumerate(cap_seg):
            bo = seg2blk.get(int(s))
            R_rows.append(dict(cell=cell, rpt_idx_1based=k + 1, seg=int(s), cap_Ah=float(cap[k]), is_qref=bool(k == jm),
                               in_dup_block=bo is not None, block=(bo[0] if bo else None), src_seg=(int(s) - bo[1] if bo else None),
                               src_cap_Ah=(float(cap[list(cap_seg).index(int(s) - bo[1])]) if bo and (int(s) - bo[1]) in set(cap_seg) else None)))
        # 98 列對照
        dd = dup98[dup98.cell == f"ox__{cell}"]
        for _, r in dd.iterrows():
            es = ev_seg[ev_cyc == r.cycle]; esrc = ev_seg[ev_cyc == r.copies_cycle]
            inb = [int(s) for s in es if int(s) in seg2blk]
            ok_src = [int(s) for s in inb if (int(s) - seg2blk[int(s)][1]) in set(int(x) for x in esrc)]
            bl = sorted({seg2blk[s][0] for s in inb})
            parts = sorted({int(S.part_first.iloc[s - 1]) for s in es}) if len(es) else []
            sparts = sorted({int(S.part_first.iloc[s - seg2blk[s][1] - 1]) for s in inb})
            M_rows.append(dict(cell=f"ox__{cell}", cycle=int(r.cycle), copies_cycle=int(r.copies_cycle), early_copy=bool(r.early_copy),
                               n_events=int(len(es)), n_events_in_dup_block=len(inb), n_events_source_matches_copies_cycle=len(ok_src),
                               blocks=";".join(map(str, bl)), csv_part_of_row=";".join(map(str, parts)), csv_part_of_source=";".join(map(str, sparts)),
                               matched=bool(len(es) > 0 and len(ok_src) == len(es))))
        # 接合列：複本來源自檔首（src_seg0 落在前 3 段內）之重複區段
        head = [r for r in Bc if r["src_seg0"] <= 3]
        if head:
            h0 = min(head, key=lambda r: r["row0"])
            j = h0["seg0"]
            prev = S.iloc[j - 2] if j >= 2 else None
            before = ev_cyc[ev_seg < j]; after = ev_cyc[ev_seg > h0["seg1"]]
            J_rows.append(dict(cell=cell, join_row=int(h0["row0"]), join_seg=int(j), join_cyc=(float(before.max()) if before.size else None),
                               part1_last_cycle=(int(prev.part_last) if prev is not None else None), part2_first_cycle=int(h0["part_first"]),
                               duplicate_source_file=f"csv part {h0['src_part_first']}–{h0['src_part_last']}（同檔檔首區段之複本）",
                               duplicate_rows_matched=int(sum(m["matched"] for m in M_rows if m["cell"] == f"ox__{cell}")),
                               dup_block_row0=int(h0["row0"]), dup_block_row1=int(h0["row1"]), dup_block_raw_rows=int(h0["raw_rows_span"]),
                               first_cycA_after_block=(float(after.min()) if after.size else None), n_head_blocks=len(head)))
        else:
            J_rows.append(dict(cell=cell, join_row=None, join_seg=None, join_cyc=None, part1_last_cycle=None, part2_first_cycle=None,
                               duplicate_source_file="無檔首複本區段", duplicate_rows_matched=int(sum(m["matched"] for m in M_rows if m["cell"] == f"ox__{cell}")),
                               dup_block_row0=None, dup_block_row1=None, dup_block_raw_rows=None, first_cycA_after_block=None, n_head_blocks=0))
        summ[cell] = dict(n_seg=int(len(S)), n_blocks=len(Bc), blocks_rows=[r["raw_rows_span"] for r in Bc],
                          qref_rpt_idx_1based=jm + 1, qref_cap=float(cap[jm]), qref_rpt_in_dup_block=bool(seg2blk.get(int(cap_seg[jm])) is not None))
    pd.DataFrame(B_rows).to_csv(os.path.join(SJ, "dup_blocks.csv"), index=False)
    pd.DataFrame(M_rows).to_csv(os.path.join(SJ, "dup_rows_98_mapping.csv"), index=False)
    pd.DataFrame(R_rows).to_csv(os.path.join(SJ, "rpt_in_blocks.csv"), index=False)
    pd.DataFrame(J_rows).to_csv(os.path.join(RES, "oxford_join_points.csv"), index=False)
    M = pd.DataFrame(M_rows)
    summ["_98"] = dict(n=int(len(M)), n_matched=int(M.matched.sum()), n_early=int(M.early_copy.sum()), n_early_matched=int(M[M.early_copy].matched.sum()))
    json.dump(summ, open(os.path.join(SJ, "join_analysis.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
