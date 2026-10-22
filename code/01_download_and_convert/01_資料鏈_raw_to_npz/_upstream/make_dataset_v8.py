"""由 v9.2 之 dataset_v6（pv92_run/results_v92/dataset_v6_labels_v92.tar.gz 解壓）依 prereg/arms_v93.json 產生 dataset_v8（僅重新命名／分臂，標籤不變）。
用法：python3 make_dataset_v8.py <dataset_v6_dir> <dataset_v8_dir>"""
import json, os, sys, glob, shutil
src, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
cfg = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "prereg", "arms_v93.json")))["arms"]
n = 0
for arm, c in cfg.items():
    for f in glob.glob(os.path.join(src, "*.npz")):
        old_arm, cell = os.path.basename(f)[:-4].split("__", 1)
        ok = (old_arm == "oxford" and cell.split("__")[0] in c.get("groups", [])) or (cell in c.get("cells", []))
        if ok and cell != "G3__Cell15":
            shutil.copy(f, os.path.join(dst, f"{arm}__{cell}.npz")); n += 1
print("dataset_v8 files:", n)
