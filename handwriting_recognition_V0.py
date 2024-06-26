import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

# Chargement des données MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisation des images
x_train, x_test = x_train / 255.0, x_test / 255.0

# Affichage de quelques exemples d'images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Définition du modèle
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Reshape des données pour correspondre à l'entrée du modèle
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Entraînement du modèle
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))



# Évaluation du modèle
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')




import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np

class DigitRecognizerApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model

        self.canvas = tk.Canvas(root, width=200, height=200, bg='white')
        self.canvas.pack()

        self.button_predict = tk.Button(root, text="Reconnaître", command=self.predict_digit)
        self.button_predict.pack()

        self.button_clear = tk.Button(root, text="Effacer", command=self.clear_canvas)
        self.button_clear.pack()

        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = Image.new("L", (200, 200), 255)
        self.draw_image = ImageDraw.Draw(self.image)

    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='black', width=5)
        self.draw_image.ellipse([x-5, y-5, x+5, y+5], fill='black')

    def predict_digit(self):
        img = self.image.resize((28, 28))
        img = ImageOps.invert(img)
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        prediction = self.model.predict(img)
        digit = np.argmax(prediction)

        print(f'Le chiffre prédit est: {digit}')

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw_image.rectangle([0, 0, 200, 200], fill='white')

root = tk.Tk()
app = DigitRecognizerApp(root, model)
root.mainloop()
