import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


SEED = 42
EPOCHS = 5
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
SONUC_KLASORU = Path(__file__).resolve().parent / "sonuclar"


def convolution_incele():
    print("\n3. Convolution")
    tf.keras.utils.set_random_seed(SEED)
    conv = tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=(3, 3),
        activation="relu",
        kernel_initializer="he_normal",
    )
    dummy_input = tf.random.normal((1, 128, 128, 1))
    output = conv(dummy_input)
    weights, biases = conv.get_weights()

    print("Giriş boyutu:", tuple(dummy_input.shape))
    print("Çıkış boyutu:", tuple(output.shape))
    print("Weights boyutu:", weights.shape)
    print("Bias boyutu:", biases.shape)
    print("İlk filtrenin weight değerleri:")
    print(weights[:, :, 0, 0])

    fig, axes = plt.subplots(4, 4, figsize=(7.5, 5.2), layout="constrained")
    limit = np.abs(weights).max()
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(weights[:, :, 0, i], cmap="coolwarm",
                       vmin=-limit, vmax=limit)
        ax.set_title(f"Filtre {i + 1}")
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.85, label="Weight")
    fig.savefig(SONUC_KLASORU / "convolution_filtreleri.png", dpi=180)


def max_pooling_incele():
    print("\n4. Max Pooling")
    input_data = tf.constant([
        [1., 3., 2., 4.],
        [5., 6., 1., 2.],
        [7., 2., 8., 3.],
        [4., 1., 5., 9.],
    ])
    input_data = tf.reshape(input_data, (1, 4, 4, 1))
    pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
    output = pool(input_data)

    print("Giriş boyutu:", tuple(input_data.shape))
    print("Giriş matrisi:\n", input_data.numpy()[0, :, :, 0], sep="")
    print("Çıkış boyutu:", tuple(output.shape))
    print("Çıkış matrisi:\n", output.numpy()[0, :, :, 0], sep="")


def aktivasyonlari_incele():
    print("\n5. ReLU ve Softmax")
    x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
    relu_output = tf.keras.layers.ReLU()(x)
    print("ReLU girişi:", x.numpy())
    print("ReLU çıkışı:", relu_output.numpy())

    logits = tf.constant([[2.0, 1.0, 0.1]])
    probabilities = tf.keras.layers.Softmax()(logits)
    print("Softmax olasılıkları:", probabilities.numpy())
    print("Olasılıkların toplamı:", tf.reduce_sum(probabilities).numpy())
    print("Tahmin edilen sınıf indisi:", int(tf.argmax(probabilities[0])))


def model_olustur(ilk_filtre_sayisi):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(ilk_filtre_sayisi, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def modeli_egit(ilk_filtre_sayisi, x_train, y_train, x_test, y_test):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    model = model_olustur(ilk_filtre_sayisi)
    if ilk_filtre_sayisi == 8:
        model.summary()

    print(f"\nİlk Conv2D katmanı: {ilk_filtre_sayisi} filtre")
    baslangic = perf_counter()
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=2,
    )
    sure = perf_counter() - baslangic
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    sonuc = {
        "filtre_sayisi": ilk_filtre_sayisi,
        "egitim_accuracy": history.history["accuracy"][-1],
        "test_accuracy": test_accuracy,
        "test_loss": test_loss,
        "egitim_suresi_saniye": sure,
    }
    print(f"Test Accuracy: {test_accuracy:.2%}")
    print(f"Test Loss: {test_loss:.4f}")
    return history, sonuc


def egitim_grafigini_ciz(history):
    epochs = range(1, len(history.history["accuracy"]) + 1)
    fig = plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["accuracy"], label="Training")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.xticks(epochs)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["loss"], label="Training")
    plt.plot(epochs, history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.xticks(epochs)
    plt.legend()

    plt.tight_layout()
    fig.savefig(SONUC_KLASORU / "egitim_ve_validation.png", dpi=180)


def karsilastirmayi_kaydet(sonuclar):
    print("\n8. Filtre sayısı karşılaştırması")
    print(f"{'Filtre':>6} {'Eğitim accuracy':>17} {'Test accuracy':>15} "
          f"{'Test loss':>11} {'Eğitim süresi':>15}")
    for sonuc in sonuclar:
        print(
            f"{sonuc['filtre_sayisi']:>6} {sonuc['egitim_accuracy']:>17.2%} "
            f"{sonuc['test_accuracy']:>15.2%} {sonuc['test_loss']:>11.4f} "
            f"{sonuc['egitim_suresi_saniye']:>12.2f} sn"
        )

    yol = SONUC_KLASORU / "deney_sonuclari.csv"
    with yol.open("w", encoding="utf-8-sig", newline="") as dosya:
        yazici = csv.DictWriter(dosya, fieldnames=list(sonuclar[0]))
        yazici.writeheader()
        yazici.writerows(sonuclar)


def main():
    SONUC_KLASORU.mkdir(exist_ok=True)
    print("TensorFlow sürümü:", tf.__version__)
    convolution_incele()
    max_pooling_incele()
    aktivasyonlari_incele()

    print("\n6–7. Fashion-MNIST ve CNN modeli")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]
    print("Eğitim verisi:", x_train.shape)
    print("Test verisi:", x_test.shape)

    history_8, sonuc_8 = modeli_egit(8, x_train, y_train, x_test, y_test)
    egitim_grafigini_ciz(history_8)

    # 8 filtreli modelin eğitim sonucu ilk deney olarak kullanılır.
    _, sonuc_16 = modeli_egit(16, x_train, y_train, x_test, y_test)
    karsilastirmayi_kaydet([sonuc_8, sonuc_16])
    print(f"\nSonuç dosyaları: {SONUC_KLASORU}")
    plt.show()


if __name__ == "__main__":
    main()
