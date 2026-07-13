from tensorflow import keras
from tensorflow.keras import layers


def build_mnist_cnn() -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dropout(0.35),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_cnn",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
