"""
dataset_v9（v9.5 三來源臂）＝ dataset_v8（40 顆，標籤不變）＋ A 層事件軸（EFC／cyc_A）＋ S1-A A8 工況欄位 F_cond。
每顆一個 npz：{source, sub_arm, cell_id, grid, Q, support, soh, seg, I_max, T, q_ref, efc, cyc_A, event_idx,
               F_cond (n×5: T_mean, C_chg, C_dis_prev, proto_CCCV, cal_flag), fcond_names, I_max_over_qref}
A8 規則（先驗登記 v10 ②，全由 A 層段表推得，不回原始檔、不動任何標籤）：
  T_mean     ＝ 該充電事件段之 T_mean
  C_chg      ＝ 該充電事件段 I_mean ／ Q_ref
  C_dis_prev ＝ 該事件之前一個「有效放電段」（regime=discharge 且列數 n > 10）之 |I_mean| ／ Q_ref
               —— 揭露：v9.4 event_axis_map 之 prev_dis_seg 於 NASA 全部指向 1 列、0 s 之偽放電段（I_mean 為單點值），故此處不用該欄而改依上述規則重取
  proto_CCCV ＝ 子臂 metadata：ox-cc（G1/G3/G5）=0；ox-cccv（G7–G10）=1；nasa（1.5 A CC-CV）=1；own（2 A，段表 V_end 4.2 V 且 I_mean < 0.9·I_max，觀測為 CC-CV）=1
               另算資料驅動旗標 proto_CCCV_data = (V_end ≥ 4.19 V) & (I_mean/I_max < 0.9) 作 A8 一致性核對（只報告）
  cal_flag   ＝ 子臂 metadata：G1/G3/G7–G10 = 1（含日曆老化）；G5、nasa、own = 0
用法：python make_dataset_v9.py <pv94_run_dir> <dataset_v9_dir>   （另輸出 <dataset_v9_dir>/../results_v95/dataset_v9_census.csv 與 fcond_a8_check_v95.csv）
"""
import os, sys, glob, json, numpy as np, pandas as pd

SUB_ARM_OF_GROUP = {"G1": "ox-cc", "G3": "ox-cc", "G5": "ox-cc", "G7": "ox-cccv", "G8": "ox-cccv", "G9": "ox-cccv", "G10": "ox-cccv"}
SUB_ARM_OF_CELL = {**{c: "nasa-2A" for c in ["B0005", "B0006", "B0007", "B0018", "B0036", "B0042", "B0043", "B0044"]},
                   "B0033": "nasa-4A-40", "B0034": "nasa-4A-40", **{c: "nasa-4A-54" for c in ["B0029", "B0030", "B0031", "B0032"]},
                   **{c: "nasa-1A-47" for c in ["B0038", "B0039", "B0040"]}, "cell2": "own", "cell3": "own"}
SOURCE_OF_SUB = {"ox-cc": "ox", "ox-cccv": "ox", "nasa-2A": "nasa", "nasa-4A-40": "nasa", "nasa-4A-54": "nasa", "nasa-1A-47": "nasa", "own": "own"}
PROTO_CCCV = {"ox-cc": 0, "ox-cccv": 1, "nasa-2A": 1, "nasa-4A-40": 1, "nasa-4A-54": 1, "nasa-1A-47": 1, "own": 1}
CAL_FLAG_OF_GROUP = {"G1": 1, "G3": 1, "G5": 0, "G7": 1, "G8": 1, "G9": 1, "G10": 1}
# A8 通關之臂定義倍率（v8 臂標籤）：充電 CC 電流／放電電流（A）
NOMINAL = {"ox-cc": (1.5, 1.5), "ox-cccv": (1.5, 1.5), "nasa-2A": (1.5, 2.0), "nasa-4A-40": (1.5, 4.0), "nasa-4A-54": (1.5, 4.0), "nasa-1A-47": (1.5, 1.0), "own": (2.0, 2.0)}
FCOND = ["T_mean", "C_chg", "C_dis_prev", "proto_CCCV", "cal_flag"]


def sub_arm(cell):
    return SUB_ARM_OF_GROUP[cell.split("__")[0]] if "__" in cell else SUB_ARM_OF_CELL[cell]


def main(p94, out):
    os.makedirs(out, exist_ok=True); res = os.path.join(os.path.dirname(os.path.abspath(out)), "results_v95"); os.makedirs(res, exist_ok=True)
    al = os.path.join(p94, "results_v94", "a_layer"); files = {}
    for f in sorted(glob.glob(os.path.join(p94, "dataset_v8", "*.npz"))):
        arm, cell = os.path.basename(f)[:-4].split("__", 1)
        if arm in ("ox-cc", "ox-cccv", "own") or arm.startswith("nasa"):       # 主臂檔（次要子臂為同顆複本，不重複納入）
            files.setdefault(cell, f)
    assert len(files) == 40, len(files)
    census, a8 = [], []
    for cell, f in sorted(files.items()):
        d = {k: np.load(f, allow_pickle=True)[k] for k in np.load(f, allow_pickle=True).files}
        sa = sub_arm(cell); src = SOURCE_OF_SUB[sa]; qref = float(d["q_ref"])
        seg = pd.read_csv(os.path.join(al, f"{cell}_A_layer_segments.csv")).set_index("seg")
        ev = pd.read_csv(os.path.join(al, f"{cell}_event_axis_map.csv")).set_index("seg")
        s = np.asarray(d["seg"]).astype(int)
        assert all(x in ev.index for x in s), cell
        efc = ev.loc[s, "EFC"].to_numpy(float); cyc = ev.loc[s, "cyc_A"].to_numpy(float); evi = ev.loc[s, "event_idx"].to_numpy(float)
        ch = seg.loc[s]; assert (ch.regime == "charge").all(), cell
        dis = seg[(seg.regime == "discharge") & (seg.n > 10)]; dis_idx = dis.index.to_numpy()
        prev = np.full(len(s), np.nan); prev_seg = np.full(len(s), -1)
        for i, sg in enumerate(s):
            k = np.searchsorted(dis_idx, sg) - 1
            if k >= 0:
                prev[i] = abs(float(dis.loc[dis_idx[k], "I_mean"])); prev_seg[i] = int(dis_idx[k])
        # 無前一有效放電者（首事件）→ 以該顆中位數代入並旗標（不進標籤）
        n_noprev = int(np.isnan(prev).sum()); prev = np.where(np.isnan(prev), np.nanmedian(prev), prev)
        T_mean = ch.T_mean.to_numpy(float); C_chg = ch.I_mean.to_numpy(float) / qref; C_dis = prev / qref
        cal = CAL_FLAG_OF_GROUP[cell.split("__")[0]] if "__" in cell else 0
        F = np.column_stack([T_mean, C_chg, C_dis, np.full(len(s), PROTO_CCCV[sa], float), np.full(len(s), cal, float)])
        imax = np.asarray(d["I_max"], float)
        proto_data = ((ch.V_end.to_numpy(float) >= 4.19) & (ch.I_mean.to_numpy(float) / np.maximum(imax, 1e-9) < 0.9)).astype(int)
        np.savez_compressed(os.path.join(out, f"{src}__{cell}.npz"), **d, sub_arm=sa, src_arm=src, efc=efc, cyc_A=cyc, event_idx=evi,
                            F_cond=F, fcond_names=np.array(FCOND), I_max_over_qref=imax / qref, prev_dis_seg_v95=prev_seg, proto_CCCV_data=proto_data)
        census.append(dict(source=src, sub_arm=sa, cell=cell, n_events=len(s), q_ref=qref, soh_min=float(np.min(d["soh"])), soh_max=float(np.max(d["soh"])),
                           efc_min=float(efc.min()), efc_max=float(efc.max()), T_mean_med=float(np.median(T_mean)), T_mean_p5=float(np.percentile(T_mean, 5)), T_mean_p95=float(np.percentile(T_mean, 95)),
                           C_chg_med=float(np.median(C_chg)), C_dis_prev_med=float(np.median(C_dis)), proto_CCCV=PROTO_CCCV[sa], cal_flag=cal, n_no_prev_dis=n_noprev,
                           proto_CCCV_data_frac=float(proto_data.mean())))
        nom_c, nom_d = NOMINAL[sa]
        a8.append(dict(source=src, sub_arm=sa, cell=cell, C_chg_med=float(np.median(C_chg)), Imax_over_qref_med=float(np.median(imax / qref)), nominal_C_chg=nom_c / qref,
                       C_chg_Imean_ok=bool(abs(np.median(C_chg) - nom_c / qref) <= 0.10 * nom_c / qref), C_chg_Imax_ok=bool(abs(np.median(imax / qref) - nom_c / qref) <= 0.10 * nom_c / qref),
                       C_dis_prev_med=float(np.median(C_dis)), nominal_C_dis=nom_d / qref, C_dis_ok=bool(abs(np.median(C_dis) - nom_d / qref) <= 0.10 * nom_d / qref),
                       proto_CCCV_meta=PROTO_CCCV[sa], proto_CCCV_data_frac=float(proto_data.mean()), proto_ok=bool(round(float(proto_data.mean())) == PROTO_CCCV[sa])))
    pd.DataFrame(census).to_csv(os.path.join(res, "dataset_v9_census.csv"), index=False)
    A = pd.DataFrame(a8); A.to_csv(os.path.join(res, "fcond_a8_check_v95.csv"), index=False)
    summ = dict(n_cells=len(census), n_events=int(sum(c["n_events"] for c in census)),
                by_source={s: dict(n_cells=int((A.source == s).sum()), n_events=int(sum(c["n_events"] for c in census if c["source"] == s))) for s in ("ox", "nasa", "own")},
                a8_C_chg_Imax_pass=int(A.C_chg_Imax_ok.sum()), a8_C_chg_Imean_pass=int(A.C_chg_Imean_ok.sum()), a8_C_dis_pass=int(A.C_dis_ok.sum()), a8_proto_pass=int(A.proto_ok.sum()),
                a8_note="C_chg 通關以 CC 平台電流 I_max/Q_ref 核對（F_cond 之 C_chg 仍為設計定義 I_mean/Q_ref，CC-CV 事件之 I_mean 因 CV 尾而低於名目值，兩者皆列）；C_dis_prev 依 n>10 有效放電段重取（見檔頭揭露）")
    json.dump(summ, open(os.path.join(res, "dataset_v9_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1)); print("DATASET_V9_DONE")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
