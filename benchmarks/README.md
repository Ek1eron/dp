# Benchmarks — GPU Job Scheduler

Набір скриптів для вимірювання характеристик системи і підготовки графіків для дипломної роботи.

## Вимоги

```bash
# З кореневої директорії проєкту
source venv/bin/activate
pip install matplotlib numpy requests

# Сервіс має бути запущений
docker compose up -d
```

## Скрипти

| Скрипт | Що вимірює | Час виконання | Пріоритет |
|---|---|---|---|
| `security_audit.py` | 23 атаки / захисти | ~2 хв | 🥇 Обов'язково |
| `lifecycle.py` | Фази задачі (черга → GPU → завершення) | ~15 хв (10 задач) | 🥇 |
| `cold_warm.py` | Cold vs warm start контейнера | ~10 хв | 🥈 |
| `concurrency.py` | Паралельні запуски, race condition | ~30 хв | 🥈 |
| `runtime_compare.py` | PyTorch 2.2 vs 2.1 vs TensorFlow | ~20 хв | 🥉 |

## Запуск

```bash
cd ~/projects/gpu-job-scheduler

# 1. Аудит безпеки (ПЕРШОЧЕРГОВИЙ)
ADMIN_API_KEY=your_key python benchmarks/security_audit.py

# 2. Розклад часу задачі (10 запусків)
python benchmarks/lifecycle.py

# або більше запусків:
N=20 python benchmarks/lifecycle.py

# 3. Cold/warm start (1 cold + 5 warm)
python benchmarks/cold_warm.py

# 4. Throughput і race condition (рівні: 1, 5, 10, 20 паралельних)
python benchmarks/concurrency.py

# Кастомні рівні:
LEVELS=1,5,10 python benchmarks/concurrency.py

# 5. Порівняння runtime'ів (3 запуски кожного)
python benchmarks/runtime_compare.py

RUNS=5 python benchmarks/runtime_compare.py
```

## Результати

Після кожного запуску в `benchmarks/results/` з'являються:
- `*.csv` — сирі дані (можна відкрити в Excel)
- `plots/*.png` — готові графіки для вставки в диплом (150 dpi)

```
benchmarks/results/
├── security_audit.csv
├── lifecycle.csv
├── cold_warm.csv
├── concurrency.csv
├── runtime_compare.csv
└── plots/
    ├── security_audit.png
    ├── lifecycle.png
    ├── cold_warm.png
    ├── concurrency.png
    └── runtime_compare.png
```

## Опис графіків для диплому

### `security_audit.png`
**Розділ: Безпека.** Два графіки:
- ліворуч — кількість PASS/FAIL по категоріях атак
- праворуч — час відповіді API на кожну перевірку (підтверджує що блокування відбувається швидко)

### `lifecycle.png`
**Розділ: Архітектура.** Стекований бар-чарт:
- синя частина = час очікування в черзі
- червона частина = час виконання на GPU

### `cold_warm.png`
**Розділ: Продуктивність.** Cold/warm start ratio:
- перший запуск контейнера (холодний)
- наступні запуски (прогрітий Docker layer cache)

### `concurrency.png`
**Розділ: Паралельне виконання.** Throughput vs навантаження:
- зелений бар = GPU ізоляція OK (race condition не виявлено)
- червоний бар = race condition (якщо виявлено — баг у системі)

### `runtime_compare.png`
**Розділ: Runtime профілі.** Порівняння трьох Docker-образів:
- PyTorch 2.2.2 (CUDA 12.1) vs PyTorch 2.1.2 (CUDA 11.8) vs TensorFlow 2.15

## Змінні середовища

| Змінна | Значення за замовчуванням | Опис |
|---|---|---|
| `BASE_URL` | `http://localhost:8001` | URL сервісу |
| `ADMIN_API_KEY` | (порожньо) | Для тестів адмін-захисту |
| `N` | `10` | Кількість задач у lifecycle.py |
| `WARM_RUNS` | `5` | Warm runs у cold_warm.py |
| `LEVELS` | `1,5,10,20` | Рівні паралелізму у concurrency.py |
| `RUNS` | `3` | Запусків на runtime у runtime_compare.py |
