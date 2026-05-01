"""
06 — TensorFlow Demo
────────────────────
Той самий тест Hello GPU, але **через інший runtime** — TensorFlow.
Демонструє що система ізолює середовища: PyTorch і TensorFlow задачі
працюють паралельно без конфліктів CUDA.

▶ Як запустити:
    Mode:    Paste Code
    Runtime: tensorflow         ← важливо! не pytorch
    CPUs:    1
    Memory:  2g
    Time:    ~15s

Що перевіряє:
  • Виконання у `tensorflow/tensorflow:2.15.0-gpu` образі
  • TF бачить GPU через NVIDIA Container Toolkit (той самий механізм що й PyTorch)
  • Студент може чергувати PyTorch і TensorFlow задачі без переналаштувань
"""
import tensorflow as tf

print(f"TensorFlow:  {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print()

gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs detected: {len(gpus)}")
for i, g in enumerate(gpus):
    print(f"  [{i}] {g.name}")
print()

if not gpus:
    raise RuntimeError("TF не бачить GPU — перевір nvidia-container-toolkit")

# ── Простий обчислювальний тест ─────────────────────────────────────────────
print("─" * 50)
print("Running matrix operations on GPU…")

with tf.device("/GPU:0"):
    a = tf.random.normal((2000, 2000))
    b = tf.random.normal((2000, 2000))
    c = tf.matmul(a, b)
    result = tf.reduce_sum(c).numpy()

print(f"  Shape:        {c.shape}")
print(f"  Sum:          {result:.2f}")
print(f"  Mean:         {tf.reduce_mean(c).numpy():.4f}")
print()

# ── Маленька Keras-модель для перевірки повного флоу ────────────────────────
print("Training a tiny Keras model…")

x = tf.random.normal((512, 32))
y = tf.random.uniform((512,), maxval=4, dtype=tf.int32)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(4),
])
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(x, y, epochs=5, verbose=2, batch_size=64)

print()
print("✓ TensorFlow runtime working correctly!")
