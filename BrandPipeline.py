#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BrandPipeline  (stand-alone)
----------------------------------
1. CSV  →  ARFF (ย้าย Current_brand เป็นคอลัมน์ท้าย)
2. Train FilteredClassifier(StringToNominal ▸ J48)
3. 10-fold CV สรุปผล
4. Demo : ทำนายแถวแรก (Current_brand = ?)
"""

from __future__ import annotations
import os, subprocess, sys, tempfile
from pathlib import Path
import pandas as pd
import arff            # liac-arff

# ───────────── CONFIG ─────────────────────────────────
CSV_IN      = Path("data_full_english.csv")      # ← ไฟล์ CSV ต้นทาง
ARFF_OUT    = Path("sample.arff")
MODEL_OUT   = Path("brand.model")
CLASS_ATTR  = "Current_brand"

WEKA_JAR    = Path("model/weka.jar")
MTJ_JAR     = Path("model/mtj-1.0.4.jar")        # math-lib ของ J48
JAVA        = "java"
RAM_GB      = 2
# ──────────────────────────────────────────────────────


# ---------- helper to call Weka -----------------------
def weka(*args: str) -> str:
    cp = os.pathsep.join(map(str, [WEKA_JAR, MTJ_JAR, "."]))
    cmd = [JAVA, f"-Xmx{RAM_GB}G", "-cp", cp, *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode:
        sys.stderr.write(proc.stderr)
        raise RuntimeError("Weka error ↑")
    return proc.stdout


# ---------- 1) CSV → ARFF  ----------------------------
print("➡️  converting CSV → ARFF …")
df = pd.read_csv(CSV_IN, encoding="utf-8")

# ย้าย Current_brand ไปท้าย
cols = [c for c in df.columns if c != CLASS_ATTR] + [CLASS_ATTR]
df   = df[cols]

# pandas → ARFF (ระบุ nominal / numeric)
atts = []
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        atts.append([col, "NUMERIC"])
    else:
        vals = sorted(df[col].dropna().unique().tolist()) or ["?"]
        atts.append([col, vals])

with ARFF_OUT.open("w", encoding="utf-8") as f:
    arff.dump({"relation": "smartphone",
               "attributes": atts,
               "data": df.values.tolist()}, f)
print(f"✅  ARFF saved  → {ARFF_OUT}")

# ---------- 2) train model ----------------------------
print("\n⏳  training FilteredClassifier(J48) …")
weka(
    "weka.classifiers.meta.FilteredClassifier",
    "-F", "weka.filters.unsupervised.attribute.StringToNominal -R first-last",
    "-W", "weka.classifiers.trees.J48",
    "-t", str(ARFF_OUT),
    "-d", str(MODEL_OUT),
    "-c", "last", # class attribute = Current_brand
    "-x", "10",         # 10-fold CV
    "-o"                # summary only
)
print(f"✅  model saved  → {MODEL_OUT}")

def build_one_row_arff(row: pd.Series, header_src: Path) -> Path:
    """สร้าง temp ARFF 1 บรรทัด พร้อม header ตรงกับไฟล์เทรน"""
    with header_src.open(encoding="utf-8") as f:
        hdr = arff.load(f)["attributes"]

    values = row.tolist()
    values[-1] = "?"          # class unknown

    fd, name = tempfile.mkstemp(suffix=".arff")
    os.close(fd)
    p = Path(name)
    arff.dump({"relation": "sample", "attributes": hdr, "data": [values]},
              p.open("w", encoding="utf-8"))
    return p

# ----------------------------------------

# ---------- 3-4) demo predict first row ---------------
row0 = df.iloc[0].copy()
row0[CLASS_ATTR] = "?"        # ให้โมเดลเดา

# สร้าง temp ARFF 1 แถว (header เดิมครบ)
tmp_arff  = build_one_row_arff(row0, ARFF_OUT)

print("\n🔍  predicting …")
out = weka(
    "weka.classifiers.meta.FilteredClassifier",
    "-l", str(MODEL_OUT),
    "-T", str(tmp_arff),
    "-p", "0", "-distribution",          # พิมพ์ prediction + distribution
    "-c", "last"
)

# ------------- ดึงค่าทำนาย -----------------
# หาเฉพาะบรรทัดที่ขึ้นต้นด้วยตัวเลข (inst#)
pred_lines = [ln for ln in out.splitlines()
              if ln and ln.lstrip()[0].isdigit()]

if not pred_lines:
    raise RuntimeError("Weka didn’t return predictions!\n" + out)

cols  = pred_lines[0].split()
label = cols[2].split(":")[1]           # Apple
dist  = cols[-1].strip("()")            # '0.83,0.05,0.12,…'

print(f"🔮  Brand = {label}")
print(f"📊  Probabilities = [{dist}]")

# tmp_arff.unlink()      # uncomment ถ้าต้องการลบไฟล์ temp
