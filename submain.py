from PIL import Image
import numpy as np
import tensorflow as tf
from main import model_predict

def preprocess_image(image_path):
    # Открываем изображение
    img = Image.open(image_path).convert('L')  # Преобразуем в черно-белое изображение
    img = img.resize((28, 28))  # Изменяем размер до 28x28 пикселей

    # Преобразуем изображение в numpy массив
    img_array = np.array(img)

    # Инвертируем цвета (если необходимо)
    img_array = 255 - img_array

    # Нормализуем значения пикселей
    img_array = img_array / 255.0

    # Плоское изображение (28*28 -> 784)
    img_array = img_array.reshape(-1, 28 * 28)

    return tf.convert_to_tensor(img_array, dtype=tf.float32)


# Пример использования:
image_path = r'C:\Users\Nurdaulet.DESKTOP-KDILCUN\Desktop\test1.png'  # Замените на путь к вашему изображению
processed_image = preprocess_image(image_path)

# Прогноз модели для пользовательского изображения
prediction = model_predict(processed_image)
predicted_label = tf.argmax(prediction, axis=-1).numpy()[0]
print(f'Predicted Label: {predicted_label}')

