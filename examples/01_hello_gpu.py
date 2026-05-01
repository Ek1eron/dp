"""
01 — Hello GPU
──────────────
Найпростіший тест: перевірити що CUDA доступна і виконати маленький tensor product.

▶ Як запустити (на dashboard):
    Mode:    Paste Code (або Upload .py)
    Runtime: pytorch-cu121
    CPUs:    1
    Memory:  1g
    Time:    ~10s (плюс перший раз — завантаження образу)

Що перевіряє:
  • Базове підключення до GPU через NVIDIA Container Toolkit
  • PyTorch у контейнері бачить CUDA
  • CUDA-операція повертає коректний результат
"""
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available:  {torch.cuda.is_available()}")
print(f"CUDA version:    {torch.version.cuda}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA не доступна — перевір NVIDIA Container Toolkit на сервері")

print(f"Device count:    {torch.cuda.device_count()}")
print(f"Device name:     {torch.cuda.get_device_name(0)}")
print(f"Compute cap:     {torch.cuda.get_device_capability(0)}")
print()
print("─" * 50)
print("Running a small tensor product on GPU…")

x = torch.randn(2000, 2000, device="cuda")
y = x @ x.T

print(f"  Input shape:  {tuple(x.shape)}")
print(f"  Output shape: {tuple(y.shape)}")
print(f"  Output sum:   {y.sum().item():.4f}")
print(f"  Output mean:  {y.mean().item():.4f}")
print()
print("✓ GPU is working correctly!")
