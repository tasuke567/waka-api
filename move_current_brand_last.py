# move_current_brand_last.py
import sys, arff, pathlib

def move_class_last(src_path, dst_path, class_attr="Current_brand"):
    with open(src_path, encoding="utf-8") as f:
        data = arff.load(f)

    names = [a[0] for a in data["attributes"]]
    if names[-1] == class_attr:
        print(f"{src_path}  ✔ already last")
        return

    idx = names.index(class_attr)
    # ย้าย header
    data["attributes"].append(data["attributes"].pop(idx))
    # ย้ายค่าจริง ๆ
    for row in data["data"]:
        row.append(row.pop(idx))

    with open(dst_path, "w", encoding="utf-8") as f:
        arff.dump(data, f)
    print(f"{src_path}  →  {dst_path}")

if __name__ == "__main__":
    for fname in sys.argv[1:]:
        p = pathlib.Path(fname)
        move_class_last(p, p)          # เขียนทับไฟล์เดิม
