"""
Приклад тренування MLP на завантаженому датасеті.

Очікує:
  DATA_DIR/train.csv  — навчальна вибірка (500 рядків × 4 фічі + label)
  DATA_DIR/test.csv   — тестова вибірка   (100 рядків × 4 фічі + label)

Зберігає у OUTPUT_DIR:
  model.pt        — ваги моделі
  metrics.txt     — фінальні метрики
  loss_curve.csv  — крива втрат
"""
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim


# ── Шляхи ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.environ.get("DATA_DIR",   "/workspace/data")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device:     {device}")
if device == "cuda":
    print(f"GPU:        {torch.cuda.get_device_name(0)}")
print(f"DATA_DIR:   {DATA_DIR}")
print(f"OUTPUT_DIR: {OUTPUT_DIR}")
print()


# ── Завантаження даних ────────────────────────────────────────────────────────
def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    X = torch.tensor(
        [[float(r["f1"]), float(r["f2"]), float(r["f3"]), float(r["f4"])] for r in rows],
        dtype=torch.float32,
    )
    y = torch.tensor([int(r["label"]) for r in rows], dtype=torch.long)
    return X.to(device), y.to(device)


X_train, y_train = load_csv(os.path.join(DATA_DIR, "train.csv"))
X_test,  y_test  = load_csv(os.path.join(DATA_DIR, "test.csv"))
print(f"Train: {tuple(X_train.shape)} | Test: {tuple(X_test.shape)}")
print(f"Class balance — train: {(y_train == 1).sum().item()}/{len(y_train)} positive\n")


# ── Модель ────────────────────────────────────────────────────────────────────
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 2),
).to(device)

opt     = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


# ── Тренування ────────────────────────────────────────────────────────────────
EPOCHS = 50
losses = []

for epoch in range(EPOCHS):
    model.train()
    opt.zero_grad()
    logits = model(X_train)
    loss   = loss_fn(logits, y_train)
    loss.backward()
    opt.step()
    losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(dim=1)
            acc   = (preds == y_test).float().mean().item()
        print(f"Epoch {epoch+1:3d} | loss = {loss.item():.4f} | test_acc = {acc:.4f}")


# ── Фінальна оцінка ───────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    train_preds = model(X_train).argmax(dim=1)
    test_preds  = model(X_test).argmax(dim=1)
    train_acc   = (train_preds == y_train).float().mean().item()
    test_acc    = (test_preds  == y_test).float().mean().item()

print(f"\n✓ Done!")
print(f"  Train accuracy: {train_acc:.4f}")
print(f"  Test accuracy:  {test_acc:.4f}")


# ── Збереження артефактів ─────────────────────────────────────────────────────
torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))

with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
    f.write(f"train_accuracy: {train_acc:.4f}\n")
    f.write(f"test_accuracy:  {test_acc:.4f}\n")
    f.write(f"final_loss:     {losses[-1]:.6f}\n")
    f.write(f"epochs:         {EPOCHS}\n")
    f.write(f"num_train:      {len(X_train)}\n")
    f.write(f"num_test:       {len(X_test)}\n")
    f.write(f"device:         {device}\n")

with open(os.path.join(OUTPUT_DIR, "loss_curve.csv"), "w") as f:
    f.write("epoch,loss\n")
    for i, l in enumerate(losses):
        f.write(f"{i+1},{l:.6f}\n")

print(f"\nArtifacts saved to {OUTPUT_DIR}:")
for name in sorted(os.listdir(OUTPUT_DIR)):
    p    = os.path.join(OUTPUT_DIR, name)
    size = os.path.getsize(p)
    print(f"  {name}  ({size} bytes)")
