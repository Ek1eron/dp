"""
Генератор синтетичного датасету для прикладу.
Запусти один раз: python examples/_make_dataset.py
Створює train.csv, test.csv та dataset.zip у тій самій папці.
"""
import csv
import os
import random
import zipfile

random.seed(42)
HERE = os.path.dirname(os.path.abspath(__file__))


def gen_sample():
    f1 = random.gauss(0, 1)
    f2 = random.gauss(0, 1)
    f3 = random.gauss(0, 1)
    f4 = random.gauss(0, 1)
    # Простий лінійний паттерн з шумом → MLP має досягати ~92-96% точності
    score = f1 + f2 - f3 - f4 + random.gauss(0, 0.5)
    label = 1 if score > 0 else 0
    return f1, f2, f3, f4, label


def write_csv(path, n):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["f1", "f2", "f3", "f4", "label"])
        for _ in range(n):
            f1, f2, f3, f4, label = gen_sample()
            w.writerow([f"{f1:.6f}", f"{f2:.6f}", f"{f3:.6f}", f"{f4:.6f}", label])


train_csv = os.path.join(HERE, "train.csv")
test_csv  = os.path.join(HERE, "test.csv")
zip_path  = os.path.join(HERE, "dataset.zip")

write_csv(train_csv, 500)
write_csv(test_csv,  100)

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.write(train_csv, arcname="train.csv")
    zf.write(test_csv,  arcname="test.csv")

# Print summary
size_kb = os.path.getsize(zip_path) / 1024
print(f"Created:")
print(f"  {train_csv}  (500 rows)")
print(f"  {test_csv}   (100 rows)")
print(f"  {zip_path}   ({size_kb:.1f} KB)")
print()
print("Структура zip:")
with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        print(f"  {info.filename}  ({info.file_size} bytes)")
