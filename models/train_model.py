import tensorflow as tf
from tensorflow.keras import layers, models

def create_crnn_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Reshape for RNN
    model.add(layers.Reshape((-1, 128)))

    # Recurrent layers
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))

    # Fully connected layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (128, 32, 1)  # Example input shape, adjust according to your data
num_classes = 80  # Example number of classes, adjust according to your data

model = create_crnn_model(input_shape, num_classes)
model.summary()

# Assuming X_train and y_train are your training data and labels
# X_train should be of shape (num_samples, height, width, channels)
# y_train should be one-hot encoded labels of shape (num_samples, num_classes)

# Example placeholders
X_train = ...  # Load your training images here
y_train = ...  # Load your training labels here

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('handwriting_model.h5')
