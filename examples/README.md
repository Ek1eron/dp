# Examples

Готові приклади для тестування системи. Кожен файл — самостійна Python-програма
з докстрингом який пояснює що саме перевіряє.

## Як використовувати

1. Відкрий [http://localhost:8001](http://localhost:8001)
2. У формі Submit Job обери **Upload .py** або **Paste Code**
3. Якщо Upload — виберіть файл з цієї папки. Якщо Paste — скопіюй вміст
4. Зміни Runtime / CPUs / Memory як описано в коментарі
5. (для `train.py`) — прикріпи `dataset.zip` у поле Dataset
6. Submit Job

## Перелік прикладів

| Файл | Що тестує | Runtime | CPU/Mem | Час | Особливе |
|---|---|---|---|---|---|
| [01_hello_gpu.py](01_hello_gpu.py) | Базова перевірка GPU | `pytorch-cu121` | 1 / 1g | ~10s | — |
| [02_streaming_progress.py](02_streaming_progress.py) | Live стрімінг + ETA таймер | `pytorch-cu121` | 1 / 1g | ~30s | Спробуй закрити вкладку → отримаєш нотифікацію |
| [03_save_artifacts.py](03_save_artifacts.py) | Завантаження артефактів | `pytorch-cu121` | 1 / 2g | ~10s | 4 файли у вкладці Artifacts |
| [04_with_requirements.py](04_with_requirements.py) | `pip install` додаткових пакетів | `pytorch-cu121` | 1 / 2g | ~30-60s | Розгорни «Extra packages», вкажи `tabulate\nrich` |
| [05_failing_oom.py](05_failing_oom.py) | Обробка помилок (CUDA OOM) | `pytorch-cu121` | 1 / 2g | ~5s | Навмисно падає — перевір `failed` статус і stderr |
| [06_tensorflow_demo.py](06_tensorflow_demo.py) | Альтернативний runtime | `tensorflow` | 1 / 2g | ~15s | Інший Docker образ, той самий механізм |
| [train.py](train.py) | Повний цикл з датасетом | `pytorch-cu121` | 1 / 2g | ~30s | Прикріпи `dataset.zip` до задачі |

---

## Сценарії демонстрації

### Швидкий тест (~3 хвилини на захисті)
1. **01_hello_gpu.py** — система працює, GPU видимий
2. **02_streaming_progress.py** — live логи прямо у браузері
3. **train.py** з `dataset.zip` — повний цикл з артефактами

### Повний тест усіх фіч (~10 хвилин)
1. **01_hello_gpu.py** — швидкий sanity check
2. **02_streaming_progress.py** — закрий вкладку → нотифікація
3. **03_save_artifacts.py** — скачай model.pt
4. **04_with_requirements.py** — кастомні пакети встановлюються
5. **05_failing_oom.py** — error handling
6. **06_tensorflow_demo.py** — альтернативний runtime
7. **train.py** з датасетом — full ML cycle
8. Натисни **↻ Repeat** на якійсь задачі — форма заповниться
9. Зайди на **/admin** → побач статистику по студенту

### Тест безпеки
Спробуй вставити такий код — система його **відхилить**:
```python
import os
os.system("rm -rf /")     # → 400 Bad Request (немає у нашому AST блокуванні,
                           #   але `--cap-drop ALL` зробить це безпечним всередині)
exec("print('hello')")    # → 400: Use of exec() is not allowed
eval("1+1")               # → 400: Use of eval() is not allowed
__import__("os")          # → 400: Use of __import__() is not allowed
```

Спробуй злий `requirements.txt`:
```
git+https://github.com/evil/repo
-i https://malicious.example.com
-e ./local
```
Усі ці рядки заблоковані валідатором `check_requirements_safety`.

---

## Файли датасету (для `train.py`)

- [dataset.zip](dataset.zip) — 600 семплів, 4 фічі, бінарна мітка (synthetic)
- [train.csv](train.csv) / [test.csv](test.csv) — розпакована версія для перегляду
- [_make_dataset.py](_make_dataset.py) — генератор (запускати повторно не треба)
