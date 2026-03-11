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
Master API Server (FastAPI)  ← POST /submit-job
        │
        ▼
Redis Job Queue              ← задача ставиться в чергу
        │
        ▼
Worker (Scheduler)           ← вибирає вільну GPU
        │
        ▼
Docker Container             ← --gpus device=N --network none
        │
        ▼
CUDA / PyTorch / TensorFlow
        │
        ▼
GPU (NVIDIA)
```

Кожен компонент — окремий Docker-контейнер. Спілкування між master і worker відбувається через Redis.

---

## Технологічний стек

| Компонент | Технологія |
|---|---|
| API сервер | Python, FastAPI |
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
│   ├── requirements.txt
│   └── app/
│       ├── master.py          # FastAPI сервер
│       └── templates/
│           └── dashboard.html # веб-інтерфейс
│
├── worker/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       └── worker.py          # scheduler + виконавець задач
│
├── logs/                      # логи контейнерів задач
├── docker-compose.yml
├── .env.example               # шаблон змінних середовища
├── Makefile
├── README.md
└── DEPLOY.md                  # інструкція розгортання на сервері
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

За замовчуванням `GPU_COUNT=1`. Якщо GPU більше — змінити значення у `.env`.

**3. Запустити систему**
```bash
docker compose up --build -d
```

**4. Відкрити Dashboard**

Перейти у браузері: [http://localhost:8001](http://localhost:8001)

**5. Перевірити що GPU знайдена**
```bash
docker compose logs worker
```

Очікуваний вивід:
```
Worker started. GPU_COUNT=1, JOB_TIMEOUT=300s
[GPU] Detected GPUs: 1
[GPU]   [0] NVIDIA GeForce RTX ...
[GPU] OK: GPU_COUNT=1 matches real GPU count.
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

---

## API

### Відправити задачу
```
POST /submit-job
```
```json
{
  "code": "import torch\nprint(torch.cuda.is_available())",
  "runtime": "pytorch-cu121",
  "cpus": 1.0,
  "memory": "2g"
}
```
Відповідь:
```json
{
  "job_id": "4a63eb1d-...",
  "status": "queued",
  "runtime": "pytorch-cu121"
}
```

### Статус задачі
```
GET /jobs/{job_id}
```

### Список задач
```
GET /jobs
GET /jobs?status=running
```

### Скасувати задачу
```
POST /jobs/{job_id}/cancel
```

### Стан GPU
```
GET /gpus
GET /cluster-status
```

### Список runtime профілів
```
GET /runtimes
```

### Здоров'я системи
```
GET /health
```

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
- `cancelled` — скасовано користувачем

Задачі зберігаються в Redis 24 години, після чого видаляються автоматично.

---

## Ізоляція та безпека

Кожна задача запускається з обмеженнями:

```
--gpus device=N    доступ тільки до виділеної GPU
--cpus 1.0         обмеження CPU
--memory 2g        обмеження RAM
--network none     без доступу до мережі
--rm               контейнер видаляється після завершення
```

---

## Змінні середовища

| Змінна | За замовчуванням | Опис |
|---|---|---|
| `GPU_COUNT` | `1` | Кількість GPU на сервері |
| `JOB_TIMEOUT_SECONDS` | `300` | Максимальний час виконання задачі (секунди) |
| `REDIS_HOST` | `redis` | Хост Redis |
| `REDIS_PORT` | `6379` | Порт Redis |

---
