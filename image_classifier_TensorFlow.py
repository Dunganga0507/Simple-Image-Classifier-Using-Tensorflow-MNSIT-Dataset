from random import randint
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# MNIST veri setini yükleyelim
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Verileri normalize edelim (0-255 arası, 0-1 arası yapalım)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Modeli oluşturalım
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Giriş katmanı, 28x28 resimleri düzleştirir
    layers.Dense(128, activation='relu'),  # Gizli katman, 128 nöronlu
    layers.Dense(10, activation='softmax')  # Çıktı katmanı, 10 sınıf için softmax
])

# Modeli derleyelim
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Modeli eğitelim
history = model.fit(train_images, train_labels, epochs=5)

# Test verisi ile modelin başarımını değerlendirelim
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test başarı oranı: {test_acc}')

# Eğitim sürecini görselleştirelim
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Eğitim Süreci')
plt.ylabel('Değer')
plt.xlabel('Epoch')
plt.legend(['Doğruluk', 'Kaybın'], loc='upper left')
plt.show()

r = randint(0, 10000)
plt.imshow(test_images[r].reshape(28,28), cmap='gray')
plt.title(f"Predicted Label: {test_labels[r]}")
plt.show()