"""
03 — Save Artifacts
───────────────────
Демонструє як зберігати **результати роботи** (модель, метрики, логи)
у `OUTPUT_DIR` — після завершення задачі вони з'являються у вкладці
**Artifacts** з кнопкою скачування.

▶ Як запустити:
    Mode:    Paste Code
    Runtime: pytorch-cu121
    CPUs:    1
    Memory:  2g
    Time:    ~10s

Що перевіряє:
  • Зчитування `os.environ['OUTPUT_DIR']`
  • Worker сканує цю папку після завершення і реєструє файли
  • Master віддає файли через `GET /jobs/{id}/artifacts/{filename}`
  • Path traversal захист (спробуй у URL змінити шлях — отримаєш 400)

Після завершення — клік на задачу у таблиці → таб **Artifacts** → Download.
"""
import json
import os

import torch
import torch.nn as nn

# OUTPUT_DIR задається worker-ом і вказує на /workspace/{job_id}_output/
output_dir = os.environ.get("OUTPUT_DIR", "/workspace/output")
os.makedirs(output_dir, exist_ok=True)

print(f"OUTPUT_DIR = {output_dir}")
print()

# 1) Тренуємо просту модель і зберігаємо ваги
model = nn.Sequential(
    nn.Linear(20, 16), nn.ReLU(),
    nn.Linear(16, 1),
).cuda()

opt     = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

x = torch.randn(200, 20, device="cuda")
y = torch.randn(200, 1, device="cuda")

losses = []
for epoch in range(30):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()
    losses.append(loss.item())

print(f"Final loss: {losses[-1]:.4f}")

# 2) Зберігаємо ваги моделі
model_path = os.path.join(output_dir, "model.pt")
torch.save(model.state_dict(), model_path)
print(f"✓ Saved model.pt        ({os.path.getsize(model_path):,} bytes)")

# 3) Зберігаємо метрики у JSON
metrics = {
    "final_loss":  round(losses[-1], 6),
    "best_loss":   round(min(losses), 6),
    "epochs":      len(losses),
    "params":      sum(p.numel() for p in model.parameters()),
    "device":      str(next(model.parameters()).device),
}
metrics_path = os.path.join(output_dir, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Saved metrics.json    ({os.path.getsize(metrics_path):,} bytes)")

# 4) Зберігаємо криву втрат як CSV
loss_path = os.path.join(output_dir, "loss_curve.csv")
with open(loss_path, "w") as f:
    f.write("epoch,loss\n")
    for i, l in enumerate(losses, 1):
        f.write(f"{i},{l:.6f}\n")
print(f"✓ Saved loss_curve.csv  ({os.path.getsize(loss_path):,} bytes)")

# 5) ASCII-графік як текстовий файл
plot_path = os.path.join(output_dir, "loss_plot.txt")
with open(plot_path, "w") as f:
    max_loss = max(losses)
    f.write("Loss curve (ASCII):\n")
    f.write("─" * 50 + "\n")
    for i, l in enumerate(losses, 1):
        bar_len = int((l / max_loss) * 40)
        bar     = "█" * bar_len
        f.write(f"Epoch {i:2d} | {l:.4f} | {bar}\n")
print(f"✓ Saved loss_plot.txt   ({os.path.getsize(plot_path):,} bytes)")

print()
print(f"All 4 artifacts saved → {output_dir}")
print("Перевір вкладку 'Artifacts' у dashboard — там кнопки Download.")
