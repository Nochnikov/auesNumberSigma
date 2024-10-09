import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Уменьшаем вывод логов TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Загружаем датасет MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

# Преобразуем изображения в вектор размером 28*28 = 784
x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28 * 28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28 * 28])

# One-hot кодирование меток
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Определение модели (двухслойная Dense сеть)
class DenseNet(tf.Module):
    def __init__(self, outputs, activate="relu"):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.fl_init = False

    def __call__(self, x, *args, **kwargs):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name="w")
            self.b = tf.zeros([self.outputs], name="b", dtype=tf.float32)

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)
            self.fl_init = True

        y = x @ self.w + self.b

        if self.activate == "relu":
            return tf.nn.relu(y)
        elif self.activate == "softmax":
            return tf.nn.softmax(y)

        return y

layer_1 = DenseNet(128)
layer_2 = DenseNet(10, activate="softmax")

def model_predict(x):
    y = layer_1(x)
    y = layer_2(y)
    return y

# Определяем функцию потерь и параметры для обучения
loss_fn = tf.losses.CategoricalCrossentropy()
BATCH_SIZE = 32
EPOCHS = 30

# Функция одного шага обучения
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model_predict(x)
        loss = loss_fn(y, predictions)
    gradients = tape.gradient(loss, [layer_1.w, layer_1.b, layer_2.w, layer_2.b])
    optimizer.apply_gradients(zip(gradients, [layer_1.w, layer_1.b, layer_2.w, layer_2.b]))
    return loss

optimizer = tf.optimizers.Adam()

# Сохраняем и загружаем модель
checkpoint_dir = './model_weights'
checkpoint = tf.train.Checkpoint(layer_1=layer_1, layer_2=layer_2)

if os.path.exists(checkpoint_dir):
    print("Загрузка сохраненных весов модели...")
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
else:
    print("Сохраненные веса не найдены, начнем тренировку...")

# Тренировочный цикл
if not os.path.exists(checkpoint_dir):  # Обучение выполняется только если модель не найдена
    for epoch in range(EPOCHS):
        for batch_idx in range(0, len(x_train), BATCH_SIZE):
            x_batch = x_train[batch_idx:batch_idx + BATCH_SIZE]
            y_batch = y_train[batch_idx:batch_idx + BATCH_SIZE]
            loss = train_step(x_batch, y_batch)
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

    # Сохранение весов модели после обучения
    print("Сохранение весов модели...")
    checkpoint.save(os.path.join(checkpoint_dir, 'model.ckpt'))

# Прогнозирование
y = model_predict(x_test)
y2 = tf.argmax(y, axis=-1).numpy()

# Оценка точности
y_test_labels = tf.argmax(y_test, axis=-1).numpy()
correct_predictions = tf.reduce_sum(tf.cast(y_test_labels == y2, tf.float32))
acc = correct_predictions / y_test.shape[0] * 100
# print(f'Accuracy: {acc.numpy()}%')
