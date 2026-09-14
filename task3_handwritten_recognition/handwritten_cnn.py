"""
Task 3: Handwritten Digit Recognition using CNN
------------------------------------------------
Run:
    python task3_handwritten_recognition/handwritten_cnn.py
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)


def build_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def save_training_plots(history):
    hist = history.history

    plt.figure(figsize=(7, 5))
    plt.plot(hist["accuracy"], label="Training Accuracy")
    plt.plot(hist["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "accuracy_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(hist["loss"], label="Training Loss")
    plt.plot(hist["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "loss_curve.png", dpi=160)
    plt.close()


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("MNIST Confusion Matrix")
    plt.xlabel("Predicted Digit")
    plt.ylabel("Actual Digit")
    plt.colorbar()
    ticks = np.arange(10)
    plt.xticks(ticks)
    plt.yticks(ticks)

    threshold = cm.max() / 2
    for i in range(10):
        for j in range(10):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.tight_layout()
    plt.savefig(OUTPUT / "confusion_matrix.png", dpi=160)
    plt.close()


def save_prediction_grid(x_test, y_test, predictions):
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(x_test), 12, replace=False)

    plt.figure(figsize=(10, 8))
    for plot_number, idx in enumerate(indices, start=1):
        ax = plt.subplot(3, 4, plot_number)
        ax.imshow(x_test[idx].squeeze(), cmap="gray")
        ax.set_title(
            f"True: {y_test[idx]} | Pred: {predictions[idx]}"
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT / "sample_predictions.png", dpi=160)
    plt.close()


def main():
    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True
        )
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.10,
        callbacks=callbacks,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, batch_size=256, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}\n")

    print("Classification report:")
    report = classification_report(y_test, predictions, digits=4)
    print(report)

    model.save(OUTPUT / "mnist_cnn.keras")
    save_training_plots(history)
    save_confusion_matrix(y_test, predictions)
    save_prediction_grid(x_test, y_test, predictions)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy)
    }
    with open(OUTPUT / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(OUTPUT / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Outputs saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
