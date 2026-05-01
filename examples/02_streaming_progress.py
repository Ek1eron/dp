"""
02 — Streaming Progress
───────────────────────
Симулює довге тренування з прогресом — щоб побачити **live стрімінг логів**
та **ETA таймер** у dashboard у дії.

▶ Як запустити:
    Mode:    Paste Code
    Runtime: pytorch-cu121
    CPUs:    1
    Memory:  1g
    Time:    ~30s

Що перевіряє:
  • SSE стрімінг (рядки з'являються у dashboard кожну секунду)
  • ETA таймер `Xs elapsed` оновлюється у реальному часі
  • Browser нотифікація на завершенні (якщо дозволено)
  • Ось чому фронтенд показує живі дані, а не очікує закінчення

Поки задача працює — закрий вкладку браузера. Через ~30 секунд отримаєш
системне сповіщення що задача завершилась.
"""
import sys
import time

import torch

EPOCHS = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
print(f"Simulating training for {EPOCHS} epochs")
print("─" * 60)
sys.stdout.flush()

# Простий MLP — щоб реально щось рахувалось на GPU
model = torch.nn.Sequential(
    torch.nn.Linear(256, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10),
).to(device)

opt     = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

x = torch.randn(512, 256, device=device)
y = torch.randint(0, 10, (512,), device=device)

start = time.time()
for epoch in range(1, EPOCHS + 1):
    opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()

    elapsed = time.time() - start
    pct     = epoch / EPOCHS * 100
    bar     = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))

    print(f"Epoch {epoch:2d}/{EPOCHS} [{bar}] {pct:5.1f}%  loss={loss.item():.4f}  t={elapsed:.1f}s")
    sys.stdout.flush()

    # Кожна епоха — секунда (щоб побачити стрімінг)
    time.sleep(1)

print("─" * 60)
print(f"✓ Training finished in {time.time() - start:.1f}s")
