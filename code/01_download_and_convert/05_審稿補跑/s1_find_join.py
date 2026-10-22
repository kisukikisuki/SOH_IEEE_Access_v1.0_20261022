"""s1_find_join.py — 審稿第 1 項／步驟書 §2.1 步驟 0：定位 Oxford 已拼接 csv 之接合列與重複區段（唯讀掃描，不跑模型）。
複製自：新撰；段切分規則逐字引用 01_資料鏈_raw_to_npz/_upstream/extract_ox2.py L11–17、L62–63、L100–139
        （列過濾 V/I/T 範圍、|I|<0.02·QPROV 為 rest、段計數 segc 跨區塊延續規則），故本程式之 segc 與 extract 之 emeta[:,0]／npz seg 一一對應。
只改：不擷取 Q(V)；改為逐段記錄原始資料列號、檔段索引（csv 之 cycle 欄）、test_time_s、與段內 (V, I, Q, T) 之 SHA-1 指紋。
輸入：<csv>（Oxford 已拼接之 per-cell csv，原檔唯讀）
輸出：<outdir>/<cell>__segments.csv（seg, regime, row0, row1, n, part_first, part_last, tt0, tt1, ts0, V0, Q0, hash）
事後聲明：資料層稽核；事後（post hoc）、未調整、不在 50 項 Holm 檢定族。
用法：python s1_find_join.py <csv> <outdir>
"""
import os, sys, io, json, time, hashlib
import numpy as np, pandas as pd

COLS = ["source", "cell_id", "cycle", "step", "regime", "time_s", "test_time_s",
        "voltage_V", "current_A", "capacity_Ah", "temperature_C"]
USE = ["cycle", "time_s", "test_time_s", "voltage_V", "current_A", "capacity_Ah", "temperature_C"]
UIDX = [COLS.index(c) for c in USE]
V_MIN, V_MAX, I_ABS_MAX, T_MIN, T_MAX = 2.00, 4.30, 6.0, -20.0, 80.0   # = extract_ox2.py L15
BLOCK = 96 * 1024 * 1024                                                # = extract_ox2.py L18


def main():
    csv, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    cell = os.path.basename(csv)[:-4]
    QPROV = 3.0 if "oxford" in csv or "G" in cell else 2.0               # = extract_ox2.py L62
    THR = 0.02 * QPROV                                                   # = extract_ox2.py L63
    t0 = time.time()
    segs = []; cur = None; lastr = ""; segc = 0; raw_base = 0; st = dict(rows_in=0, rows_kept=0)

    def close():
        nonlocal cur
        if cur is not None:
            cur["hash"] = cur.pop("h").hexdigest(); segs.append(cur); cur = None

    fh = open(csv, "rb"); fh.readline()
    while True:
        start = fh.tell(); buf = fh.read(BLOCK)
        if not buf:
            break
        if len(buf) == BLOCK:
            nl = buf.rfind(b"\n")
            if nl >= 0:
                buf = buf[:nl + 1]
        fh.seek(start + len(buf))
        df = pd.read_csv(io.BytesIO(buf), header=None, names=COLS, usecols=UIDX, on_bad_lines="skip", engine="c")
        nraw = len(df); st["rows_in"] += nraw
        V = df["voltage_V"].to_numpy(float); I = df["current_A"].to_numpy(float)
        Q = df["capacity_Ah"].to_numpy(float); T = df["temperature_C"].to_numpy(float)
        tt = df["test_time_s"].to_numpy(float); ts = df["time_s"].to_numpy(float)
        part = pd.to_numeric(df["cycle"], errors="coerce").fillna(-1).to_numpy(np.int64)
        ok = np.isfinite(V) & np.isfinite(I) & np.isfinite(Q)
        ok &= (V >= V_MIN) & (V <= V_MAX) & (np.abs(I) <= I_ABS_MAX) & (np.nan_to_num(T, nan=25) >= T_MIN) & (np.nan_to_num(T, nan=25) <= T_MAX)
        ridx = np.flatnonzero(ok) + raw_base
        V, I, Q, T, tt, ts, part = V[ok], I[ok], Q[ok], T[ok], tt[ok], ts[ok], part[ok]
        raw_base += nraw
        if V.size == 0:
            continue
        st["rows_kept"] += int(V.size)
        a = np.abs(I); reg = np.where(a < THR, "rest", np.where(I > 0, "charge", "discharge"))
        chg = np.flatnonzero(np.concatenate([[True], reg[1:] != reg[:-1]])); bnds = list(chg) + [reg.size]
        blob = np.column_stack([V, I, Q, np.nan_to_num(T, nan=-999.0)])
        for bi in range(len(bnds) - 1):
            s, e = bnds[bi], bnds[bi + 1]; r = reg[s]
            newseg = not (bi == 0 and r == lastr)                     # = extract_ox2.py L126
            if newseg:
                close(); segc += 1
                cur = dict(seg=segc, regime=r, row0=int(ridx[s]), row1=int(ridx[e - 1]), n=0, part_first=int(part[s]), part_last=int(part[e - 1]),
                           tt0=float(tt[s]), tt1=float(tt[e - 1]), ts0=float(ts[s]), V0=float(V[s]), Q0=float(Q[s]), h=hashlib.sha1())
            cur["n"] += int(e - s); cur["row1"] = int(ridx[e - 1]); cur["part_last"] = int(part[e - 1]); cur["tt1"] = float(tt[e - 1])
            cur["h"].update(np.ascontiguousarray(blob[s:e]).tobytes())
            lastr = r
    fh.close(); close()
    S = pd.DataFrame(segs)
    S.to_csv(os.path.join(outdir, f"{cell}__segments.csv"), index=False, float_format="%.10g")
    print(json.dumps(dict(cell=cell, rows_in=st["rows_in"], rows_kept=st["rows_kept"], n_seg=int(len(S)),
                          n_charge=int((S.regime == "charge").sum()), sec=round(time.time() - t0, 1))))


if __name__ == "__main__":
    main()
