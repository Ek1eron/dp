"""
05 — Intentional Failure (CUDA OOM)
────────────────────────────────────
**Спеціально падає** з помилкою CUDA out of memory — щоб перевірити
що система коректно показує `failed` статус і відображає stderr.

▶ Як запустити:
    Mode:    Paste Code
    Runtime: pytorch-cu121
    CPUs:    1
    Memory:  2g
    Time:    ~5s (швидко падає)

Що перевіряє:
  • Worker правильно ловить exit code != 0
  • Status у Redis встановлюється як `failed`
  • stderr з'являється у вкладці Errors з червоним підсвічуванням
  • Browser нотифікація з іконкою ✗ і першим рядком помилки
  • Кнопка ↻ Repeat дозволяє швидко перезапустити після виправлення

Очікувана помилка:
    torch.cuda.OutOfMemoryError: CUDA out of memory.
    Tried to allocate XX.XX GiB...

Це **навмисна** помилка для демонстрації error handling.
"""
import torch

print("Available GPU memory:")
print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"  Free:  {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
print()
print("Attempting to allocate a tensor of shape (50000, 50000, 50)…")
print("This is ~500 GB — guaranteed to OOM on any GPU.")
print()

# 50000 * 50000 * 50 * 4 bytes = 500 GB → не вміститься на жодній GPU
huge = torch.zeros(50000, 50000, 50, device="cuda", dtype=torch.float32)

# Цей рядок НЕ буде виконано — попередній має впасти з OOM
print(f"Allocated tensor of shape {huge.shape}")
print("If you see this, something is very wrong (or you have 500GB of VRAM).")
