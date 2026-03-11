# Інструкція розгортання на кафедральному сервері

Цей документ описує повний процес розгортання GPU Job Scheduler на Linux-сервері з одною або кількома відеокартами NVIDIA з нуля.

---

## Вимоги до сервера

- ОС: Ubuntu 20.04 / 22.04 (або інший Linux з apt)
- GPU: одна або кілька відеокарт NVIDIA
- RAM: мінімум 4 GB
- Диск: мінімум 20 GB вільного місця (для Docker образів)
- Доступ: права sudo

---

## Крок 1 — Встановити NVIDIA драйвер

Перевірити чи драйвер вже встановлений:
```bash
nvidia-smi
```

Якщо команда не знайдена — встановити драйвер:
```bash
sudo apt update
sudo ubuntu-drivers autoinstall
sudo reboot
```

Після перезавантаження перевірити:
```bash
nvidia-smi
```

Має вивести таблицю з GPU, версією драйвера і CUDA версією.

---

## Крок 2 — Встановити Docker

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

Додати поточного користувача до групи docker (щоб не писати sudo):
```bash
sudo usermod -aG docker $USER
newgrp docker
```

Перевірити:
```bash
docker --version
docker compose version
```

---

## Крок 3 — Встановити NVIDIA Container Toolkit

Цей компонент дозволяє Docker-контейнерам отримувати доступ до GPU.

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
```

Налаштувати інтеграцію з Docker:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Перевірити що GPU доступна в контейнері:
```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

Має вивести ту саму таблицю що і `nvidia-smi` на хості.

---

## Крок 4 — Клонувати репозиторій

```bash
git clone https://github.com/Ek1eron/dp
cd dp
```

---

## Крок 5 — Налаштувати `.env`

```bash
cp .env.example .env
nano .env
```

Змінити `GPU_COUNT` на реальну кількість GPU сервера:
```bash
# Дізнатись кількість GPU:
nvidia-smi --query-gpu=name --format=csv,noheader | wc -l
```

Приклад `.env` для сервера з 4 GPU:
```
GPU_COUNT=4
JOB_TIMEOUT_SECONDS=300
REDIS_HOST=redis
REDIS_PORT=6379
```

---

## Крок 6 — Завантажити Docker образи заздалегідь

Це займе час але зробить перший запуск задач швидшим:
```bash
docker pull pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
docker pull pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
docker pull tensorflow/tensorflow:2.15.0-gpu
```

---

## Крок 7 — Запустити систему

```bash
docker compose up --build -d
```

Перевірити що всі сервіси запущені:
```bash
docker compose ps
```

Має показати три контейнери зі статусом `Up`:
```
NAME                  SERVICE   STATUS
dp-master-1           master    Up
dp-redis-1            redis     Up
dp-worker-1           worker    Up
```

---

## Крок 8 — Перевірити роботу

**Перевірити логи worker-а:**
```bash
docker compose logs worker
```

Очікуваний вивід:
```
Worker started. GPU_COUNT=4, JOB_TIMEOUT=300s
[GPU] Detected GPUs: 4
[GPU]   [0] NVIDIA A100-SXM4-40GB
[GPU]   [1] NVIDIA A100-SXM4-40GB
[GPU]   [2] NVIDIA A100-SXM4-40GB
[GPU]   [3] NVIDIA A100-SXM4-40GB
[GPU] OK: GPU_COUNT=4 matches real GPU count.
[Recovery] Checking for stale GPU reservations...
```

**Перевірити API:**
```bash
curl http://localhost:8001/health
```

Відповідь:
```json
{"status": "ok", "redis": "ok", "gpu_count": 4}
```

**Відкрити Dashboard:**

У браузері: `http://<IP_сервера>:8001`

---

## Крок 9 — Тестова задача

```bash
curl -X POST http://localhost:8001/submit-job \
  -H "Content-Type: application/json" \
  -d '{
    "code": "import torch\nprint(torch.cuda.is_available())\nprint(torch.cuda.get_device_name(0))",
    "runtime": "pytorch-cu121"
  }'
```

Отримати результат (підставити job_id з відповіді):
```bash
curl http://localhost:8001/jobs/<job_id>
```

---

## Керування системою

**Зупинити:**
```bash
docker compose down
```

**Перезапустити:**
```bash
docker compose restart
```

**Подивитись логи в реальному часі:**
```bash
docker compose logs -f worker
```

**Очистити завершені задачі:**
```bash
curl -X DELETE http://localhost:8001/jobs/cleanup
```

**Оновити систему після змін у коді:**
```bash
git pull
docker compose down
docker compose up --build -d
```

---

## Відкриття доступу для студентів

За замовчуванням Dashboard доступний тільки локально. Щоб студенти могли підключатись з інших комп'ютерів у мережі кафедри — нічого додатково налаштовувати не потрібно, достатньо відкрити порт 8001 у firewall:

```bash
sudo ufw allow 8001/tcp
```

Студенти підключаються через браузер: `http://<IP_сервера>:8001`

---

## Усунення проблем

**GPU не знаходиться в контейнері:**
```bash
# Перевірити що toolkit встановлений правильно
nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**Worker не запускається:**
```bash
docker compose logs worker
# Переконатись що /var/run/docker.sock існує
ls -la /var/run/docker.sock
```

**Задача зависла:**
```bash
# Скасувати через API
curl -X POST http://localhost:8001/jobs/<job_id>/cancel

# Або примусово зупинити контейнер
docker kill job-<job_id>
```

**Перевірити стан Redis:**
```bash
docker compose exec redis redis-cli ping
# Відповідь: PONG
```