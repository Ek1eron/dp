"""
04 — Custom requirements.txt
────────────────────────────
Демонструє як використати пакет якого **немає** у базовому PyTorch образі.
Тут — `tabulate` (форматування таблиць) і `rich` (кольоровий вивід).

▶ Як запустити:
    Mode:           Paste Code
    Runtime:        pytorch-cu121
    CPUs:           1
    Memory:         2g
    requirements:   tabulate
                    rich
    Time:           ~30-60s (pip install + run)

Що перевіряє:
  • Worker записує requirements у файл
  • Контейнер запускається з `--network bridge` (інакше: `--network none`)
  • Виконується `pip install -r requirements.txt && python script.py`
  • Логи показують і pip install і вивід коду
  • Безпечний пакет встановлюється з PyPI

⚠ Якщо запустиш без requirements — впаде з ModuleNotFoundError.
"""
import torch
from rich.console import Console
from rich.table   import Table
from tabulate     import tabulate

console = Console()
console.print("[bold cyan]Custom Requirements Demo[/bold cyan]")
console.print(f"PyTorch: {torch.__version__}")
console.print(f"CUDA:    {torch.cuda.is_available()}\n")

# ── Створюємо синтетичні результати "експериментів" ─────────────────────────
results = []
for lr in [1e-2, 1e-3, 1e-4]:
    for batch_size in [16, 32, 64]:
        # імітуємо метрики
        x = torch.randn(batch_size, 100, device="cuda")
        loss = (x ** 2).mean().item() * lr  # фейкова "метрика"
        acc  = 0.5 + (1 - lr) * 0.4 + (batch_size / 1000)
        results.append({
            "lr":         lr,
            "batch_size": batch_size,
            "loss":       round(loss, 4),
            "accuracy":   round(min(acc, 0.99), 4),
        })

# ── tabulate (зворотньо-сумісний plain text) ────────────────────────────────
print("Plain table (tabulate):\n")
print(tabulate(results, headers="keys", tablefmt="github"))
print()

# ── rich (кольорова таблиця у stdout) ───────────────────────────────────────
table = Table(title="Hyperparameter Sweep Results", show_lines=True)
table.add_column("LR",         style="cyan",   justify="right")
table.add_column("Batch",      style="magenta",justify="right")
table.add_column("Loss",       style="yellow", justify="right")
table.add_column("Accuracy",   style="green",  justify="right")

for r in results:
    table.add_row(f"{r['lr']:.0e}", str(r["batch_size"]), f"{r['loss']:.4f}", f"{r['accuracy']:.4f}")

console.print(table)
console.print("\n[bold green]✓ All packages from requirements.txt loaded successfully![/bold green]")
