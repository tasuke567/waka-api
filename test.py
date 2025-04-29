def move_class_to_last(lines, class_attr="Current_brand"):
    header = []
    data = []
    is_data = False
    class_line = None

    for line in lines:
        if line.strip().lower().startswith("@data"):
            is_data = True
            header.append(line)
            continue

        if not is_data:
            if class_attr in line:
                class_line = line
            else:
                header.append(line)
        else:
            data.append(line)

    if not class_line:
        raise Exception(f"Attribute {class_attr} not found")

    # ย้าย class attribute ไปไว้ท้ายสุด
    header = [l for l in header if class_attr not in l]
    header.append(class_line)

    # ย้ายค่าคอลัมน์ class ไปท้ายสุดของแต่ละบรรทัด
    new_data = []
    for row in data:
        parts = [x.strip() for x in row.split(",")]
        class_val = parts.pop(header.index(class_line))  # ตำแหน่งเดิมของ class
        parts.append(class_val)
        new_data.append(",".join(parts))

    return header + new_data

# ใช้:
with open("sample.arff", encoding="utf8") as f:
    lines = f.readlines()

fixed = move_class_to_last(lines)

with open("sample_fixed.arff", "w", encoding="utf8") as f:
    f.write("\n".join(fixed))
