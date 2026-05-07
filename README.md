# GPU Job Scheduler

Сервіс віддаленого виконання обчислювальних завдань на GPU в ізольованих контейнерах.

Система автоматизує процес приймання, черги та запуску студентського Python-коду на відеокартах NVIDIA на кафедральному Linux-сервері. Основний акцент — перехід від ручного налаштування середовищ до моделі **GPU-as-a-Service**: студент надсилає код і отримує результат, не турбуючись про версії CUDA, cuDNN чи залежності.

---

## Проблема яку вирішує система

На кафедральному сервері кілька студентів одночасно хочуть запускати GPU-задачі. Без системи виникають конфлікти:

- Студент А встановлює PyTorch з CUDA 11.8, студент Б — TensorFlow з CUDA 12.1, вони ламають середовище один одного
- Немає черги — задачі запускаються хаотично і конкурують за GPU
- Немає ізоляції — один студент може випадково зайняти всю пам'ять GPU

**Рішення:** кожна задача запускається в окремому Docker-контейнері з вибраним runtime-профілем. Scheduler розподіляє задачі по вільних GPU. Студент отримує stdout/stderr результат через API або Dashboard.

---

## Архітектура

```
Студент (браузер або curl)
        │
        ▼
Master API Server (FastAPI)  ← POST /submit-job, /admin/*
        │                       Dashboard + Admin Panel
        ▼
Redis Job Queue              ← задача ставиться в чергу
        │                      + API ключі, логи стрімінгу
        ▼
Worker (Scheduler)           ← вибирає вільну GPU
        │                      pip install -r requirements.txt
        ▼                      git clone репозиторіїв
Docker Container             ← --gpus device=N --network none
        │                      OUTPUT_DIR / DATA_DIR env vars
        ▼
CUDA / PyTorch / TensorFlow
        │
        ▼
GPU (NVIDIA)
```

Кожен компонент — окремий Docker-контейнер. Спілкування між master і worker відбувається через Redis.

---

## Можливості

### Для студентів
- **Вставити код** прямо в браузерний редактор (Monaco з підсвіткою синтаксису)
- **Завантажити `.py` файл** замість вставки коду
- **Підключити публічний GitHub/GitLab репозиторій** — worker його склонує і запустить
- **Завантажити датасет `.zip`** разом із кодом — розпаковується у `DATA_DIR`
- **Додаткові пакети через `requirements.txt`** — `pip install` перед запуском коду
- **Артефакти результату** — все, що код запише у `OUTPUT_DIR`, можна скачати після завершення
- **Live-стрімінг логів** через Server-Sent Events
- **Шаблони коду** — 5 готових прикладів (PyTorch, TensorFlow, MNIST)
- **Ім'я та опис задачі** для зручної ідентифікації
- **Repeat кнопка** — клонує параметри попередньої задачі у форму
- **Фільтр "only my jobs"** у таблиці задач
- **Оцінка часу очікування** на основі позиції в черзі і середнього часу

### Для адміністратора (викладача)
- **Адмін-панель** на `/admin` — статистика, управління, force-kill
- **API ключі** для автентифікації студентів
- **Per-student статистика** — кількість задач, GPU-години, найчастіші помилки
- **Runtime usage** — які профілі найпопулярніші
- **Force-kill** будь-якої задачі

### Безпека
- AST аналіз коду перед запуском (блокує `exec`, `eval`, `__import__`, `compile`)
- Path traversal захист при розпакуванні zip (zip-slip)
- Валідація `requirements.txt` — блокує `-e`, `git+`, `--index-url`, тощо
- Docker-level: `--cap-drop ALL`, `--security-opt no-new-privileges`, `--pids-limit 200`
- `--network none` за замовчуванням (`bridge` лише коли є requirements)
- Per-student queue limit (HTTP 429)
- Path traversal захист при скачуванні артефактів

---

## Технологічний стек

| Компонент | Технологія |
|---|---|
| API сервер | Python, FastAPI |
| Real-time стрімінг | Server-Sent Events (sse-starlette) |
| Code editor | Monaco Editor |
| Черга задач | Redis |
| Виконання задач | Docker, NVIDIA Container Toolkit |
| GPU фреймворки | PyTorch, TensorFlow, CUDA |
| Оркестрація | Docker Compose |
| Середовище | Linux / WSL2 |

---

## Структура проєкту

```
gpu-job-scheduler/
│
├── master/
│   ├── Dockerfile
│   ├── requirements.txt        # fastapi, sse-starlette, python-multipart, ...
│   └── app/
│       ├── master.py           # FastAPI сервер + admin API
│       └── templates/
│           ├── dashboard.html  # веб-інтерфейс для студентів
│           └── admin.html      # адмін-панель
│
├── worker/
│   ├── Dockerfile              # містить git, docker-cli
│   ├── requirements.txt
│   └── app/
│       └── worker.py           # scheduler + виконавець + стрімінг логів
│
├── benchmarks/                 # скрипти вимірювання продуктивності
│   ├── security_audit.py       # 23 атаки / захисти → PNG
│   ├── lifecycle.py            # фази задачі (черга → GPU → завершення) → PNG
│   ├── cold_warm.py            # cold vs warm container startup → PNG
│   ├── concurrency.py          # throughput + race condition detection → PNG
│   ├── runtime_compare.py      # PyTorch cu121/cu118 vs TensorFlow → PNG
│   ├── sse_vs_polling.py       # SSE latency vs polling → PNG
│   ├── diagram.py              # архітектурні діаграми → PNG
│   └── results/                # CSV + plots/*.png (генерується при запуску)
│
├── examples/
│   ├── train.py                # приклад тренування MLP
│   ├── dataset.zip             # синтетичний датасет (CSV)
│   └── _make_dataset.py        # генератор датасету
│
├── tests/
│   └── test_api.py             # pytest інтеграційні тести (50+ тестів)
│
├── logs/                       # логи worker-а (ротація, .gitignore)
├── docker-compose.yml
├── .env.example                # шаблон змінних середовища
├── README.md
└── DEPLOY.md                   # інструкція розгортання на сервері
```

---

## Швидкий старт (локально / WSL2)

**1. Клонувати репозиторій**
```bash
git clone https://github.com/Ek1eron/dp
cd dp
```

**2. Створити `.env` файл**
```bash
cp .env.example .env
```

**3. Згенерувати ADMIN_API_KEY** (для доступу до адмін-панелі)
```bash
echo "ADMIN_API_KEY=$(openssl rand -hex 24)" >> .env
```
(або просто залишити порожнім — тоді адмін-панель буде вимкнена)

**4. Запустити систему**
```bash
docker compose up --build -d
```

**5. Відкрити Dashboard**

[http://localhost:8001](http://localhost:8001) — для студентів
[http://localhost:8001/admin](http://localhost:8001/admin) — адмін-панель (треба ADMIN_API_KEY)

**6. Перевірити що GPU знайдена**
```bash
docker compose logs worker
```

> Для розгортання на кафедральному сервері — див. [DEPLOY.md](DEPLOY.md)

---

## Runtime профілі

Система використовує попередньо визначені Docker-образи щоб уникнути конфліктів версій CUDA і бібліотек. Студент обирає профіль — система забезпечує ізольоване середовище.

| Профіль | Образ | Використання |
|---|---|---|
| `pytorch-cu121` | `pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime` | PyTorch + CUDA 12.1 |
| `pytorch-cu118` | `pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime` | PyTorch + CUDA 11.8 |
| `tensorflow` | `tensorflow/tensorflow:2.15.0-gpu` | TensorFlow з GPU |

Якщо потрібен пакет якого нема в образі — використовуй `requirements.txt` (див. нижче).

---

## CPUs та Memory — як обирати

| Поле | Призначення |
|---|---|
| `cpus` | М'який ліміт на CPU. `1.0` = одне ядро, `2.5` = 2.5 ядра. Допустимо: `0.1 - 8.0` |
| `memory` | **Жорсткий** ліміт RAM. Якщо перевищити — OOM killer вб'є процес. Формат: `512m`, `2g` |

**Рекомендації для GPU задач** (основна робота на GPU, тож CPU не вузьке місце):

| Тип задачі | CPUs | Memory |
|---|---|---|
| Простий inference, `torch.cuda.is_available()` | `1` | `1g` |
| Тренування малих моделей (MNIST, наш приклад) | `1-2` | `2g` |
| Середні моделі (CIFAR, ResNet) | `2` | `4g` |
| Великі моделі / великі датасети | `4` | `8g` |
| Transformers, NLP з великими батчами | `4-8` | `8-16g` |

**Підказки:**
- Exit code `137` (`Killed`) у stderr — мало RAM, збільшити memory
- GPU navantazena малою при повільному тренуванні — мало CPU (DataLoader не встигає)
- VRAM (пам'ять GPU) — окреме питання, не контролюється цими полями

---

## API

### Public API

| Endpoint | Опис |
|---|---|
| `POST /submit-job` | Відправити задачу (JSON) |
| `POST /submit-job-form` | Відправити з multipart (файл, репо, датасет) |
| `GET /jobs` | Список задач (`?status=running`) |
| `GET /jobs/{id}` | Деталі задачі |
| `GET /jobs/{id}/logs/stream` | SSE-стрім live логів |
| `GET /jobs/{id}/artifacts` | Список артефактів |
| `GET /jobs/{id}/artifacts/{file}` | Скачати артефакт |
| `POST /jobs/{id}/cancel` | Скасувати задачу |
| `DELETE /jobs/cleanup` | Видалити завершені *(потребує X-Admin-Key)* |
| `GET /gpus`, `/cluster-status` | Стан GPU і кластеру |
| `GET /runtimes` | Список runtime профілів |
| `GET /health` | Health check |

### Admin API (потребує `X-Admin-Key` header)

| Endpoint | Опис |
|---|---|
| `GET /admin/check` | Перевірка ключа |
| `GET /admin/stats` | Per-student + runtime + daily статистика |
| `POST /admin/keys` | Створити API ключ для студента |
| `GET /admin/keys` | Список ключів |
| `DELETE /admin/keys/{key}` | Відкликати ключ |
| `POST /admin/jobs/{id}/kill` | Force-kill будь-якої задачі |

### Приклад відправки задачі

```bash
curl -X POST http://localhost:8001/submit-job \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import torch\nprint(torch.cuda.is_available())",
    "runtime": "pytorch-cu121",
    "cpus": 1.0,
    "memory": "2g",
    "name": "GPU check",
    "student_id": "ivan_petrenko"
  }'
```

З API ключем:
```bash
curl -X POST http://localhost:8001/submit-job \
  -H "X-API-Key: key_..." \
  -H "Content-Type: application/json" \
  -d '{"code": "...", "runtime": "pytorch-cu121"}'
```
Якщо ключ валідний — `student_id` береться з ключа (захист від підробки).

---

## Адмін-панель

Доступна на `/admin` за наявності `ADMIN_API_KEY` у `.env`.

**Можливості:**
- Огляд кластера: загальна кількість студентів, задач, середній час виконання
- Per-student таблиця: задач, completed/failed/cancelled, total GPU time
- Runtime usage: які профілі найпопулярніші (горизонтальні бари)
- Управління API ключами: створити, скопіювати, відкликати
- Список усіх задач з force-kill кнопкою

**Створити ключ:**
1. Зайти на `/admin`, ввести ADMIN_API_KEY
2. У секції "API keys" вписати Student ID + Display name → Generate Key
3. Ключ автоматично копіюється в буфер
4. Передати студенту — він вставить його у поле "API key" на dashboard

**Чому це корисно:**
- Студенти не можуть видавати себе один за одного у статистиці кафедри
- Можна вимагати ключ для всіх задач: `REQUIRE_STUDENT_KEY=true` в `.env`
- Викладач бачить хто скільки навантажує сервер

---

## requirements.txt — додаткові пакети

Якщо студенту потрібен пакет якого нема у runtime образі (наприклад `transformers`, `wandb`, `opencv-python`) — він вказує його в полі **"Extra packages (requirements.txt)"** на dashboard.

**Як це працює:**
1. Студент пише пакети у textarea (`one per line`)
2. Worker записує їх у файл та запускає у контейнері:
   ```bash
   pip install -r requirements.txt && python user_code.py
   ```
3. Контейнер з requirements отримує `--network bridge` (для доступу до PyPI), без — `--network none`

**Приклад requirements:**
```
transformers==4.40.0
datasets
accelerate
```

**Обмеження безпеки** (валідація на рівні API):
- ❌ `-e ./local` (editable install) — заблоковано
- ❌ `git+https://...` (git репо) — заблоковано
- ❌ `--index-url`, `--find-links` — заблоковано
- ❌ Розмір файлу > 10 КБ — заблоковано
- ✅ Звичайні назви пакетів (`numpy==1.24`) — дозволено

---

## Артефакти

Код студента може зберігати файли (модель, метрики, графіки) у `os.environ['OUTPUT_DIR']`. Після завершення задачі вони з'являються у вкладці **Artifacts** з кнопкою скачування.

```python
import os
output_dir = os.environ['OUTPUT_DIR']
os.makedirs(output_dir, exist_ok=True)

torch.save(model.state_dict(), f"{output_dir}/model.pt")
with open(f"{output_dir}/metrics.txt", 'w') as f:
    f.write(f"accuracy: {acc:.4f}\n")
```

---

## Завантаження датасету

Студент може прикріпити `.zip` архів (до 500 МБ) до задачі. Worker розпаковує його у тимчасову директорію, шлях передається у `os.environ['DATA_DIR']`:

```python
import os
data_dir = os.environ['DATA_DIR']
df = pd.read_csv(f"{data_dir}/train.csv")
```

Готовий приклад: [examples/](examples/) — датасет (бінарна класифікація, 600 семплів) + train.py.

---

## Життєвий цикл задачі

```
queued → running → completed
                 ↘ failed
                 ↘ cancelled
```

- `queued` — задача в черзі Redis, чекає вільну GPU
- `running` — контейнер запущено на конкретній GPU
- `completed` — код виконався з exit code 0
- `failed` — помилка або таймаут
- `cancelled` — скасовано користувачем або адміністратором

Задачі зберігаються в Redis 24 години, після чого видаляються автоматично.

---

## Ізоляція та безпека

Кожна задача запускається з обмеженнями:

```
--gpus device=N             доступ тільки до виділеної GPU
--cpus 1.0                  обмеження CPU
--memory 2g                 обмеження RAM
--network none              без мережі (default)
--network bridge            мережа лише коли є requirements.txt
--cap-drop ALL              без Linux capabilities
--security-opt no-new-privileges
--pids-limit 200            захист від fork bomb
--rm                        контейнер видаляється після завершення
```

На рівні API:
- AST аналіз коду — блокує `exec`, `eval`, `compile`, `__import__`
- Розмір коду ≤ 200 КБ
- Zip-slip перевірка при розпакуванні датасету
- Path traversal захист при скачуванні артефактів
- Per-student queue limit (`MAX_QUEUE_PER_STUDENT`, default `3`)

---

## Змінні середовища

| Змінна | За замовчуванням | Опис |
|---|---|---|
| `GPU_COUNT` | `1` | Кількість GPU на сервері |
| `JOB_TIMEOUT_SECONDS` | `300` | Максимальний час виконання задачі |
| `REDIS_HOST` | `redis` | Хост Redis |
| `REDIS_PORT` | `6379` | Порт Redis |
| `LOG_LEVEL` | `INFO` | Рівень логування |
| `MAX_QUEUE_PER_STUDENT` | `3` | Скільки активних задач може мати один студент |
| `MAX_DATASET_SIZE_MB` | `500` | Максимальний розмір .zip датасету |
| `ADMIN_API_KEY` | (порожній) | Ключ для доступу до адмін-панелі. Порожній = адмін вимкнений |
| `REQUIRE_STUDENT_KEY` | `false` | `true` = всі submit-job вимагають валідний `X-API-Key` |

---

## Тестування

```bash
# Базові тести (без admin):
pytest tests/test_api.py -v

# З admin тестами:
ADMIN_API_KEY=<key_from_env> pytest tests/test_api.py -v

# Лише швидкі (без виконання задач на GPU):
pytest tests/test_api.py::TestHealth tests/test_api.py::TestCodeSafety -v
```

---

## Бенчмарки та вимірювання

Скрипти у `benchmarks/` вимірюють характеристики системи і генерують PNG-графіки для дипломної роботи. Вимагають запущеного `docker compose up -d` і активованого venv.

```bash
source venv/bin/activate
pip install matplotlib numpy requests

# Аудит безпеки (23 атаки → таблиця PASS/FAIL):
ADMIN_API_KEY=<key> python benchmarks/security_audit.py

# Розклад часу задачі (черга + виконання):
python benchmarks/lifecycle.py

# Cold vs warm container startup:
python benchmarks/cold_warm.py

# Throughput при 1/5/10/20 паралельних задачах:
python benchmarks/concurrency.py

# PyTorch cu121 / cu118 vs TensorFlow:
python benchmarks/runtime_compare.py

# SSE streaming vs HTTP polling latency:
python benchmarks/sse_vs_polling.py

# Архітектурні діаграми (без GPU):
python benchmarks/diagram.py
```

Результати зберігаються у `benchmarks/results/` (CSV) та `benchmarks/results/plots/` (PNG).

---
