import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers



# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

# insira seu código aqui
def carregar_dados():
    #Carrega o dataset MNIST, normaliza as imagens e ajusta o formato
    #para (28, 28, 1), conforme exigido pelo projeto.

    # Carrega o conjunto de treinamento e teste
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normaliza os pixels para o intervalo [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Adiciona a dimensão do canal (grayscale)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    # Embaralha os dados de treinamento
    indices = tf.random.shuffle(tf.range(len(x_train)))
    x_train = tf.gather(x_train, indices)
    y_train = tf.gather(y_train, indices)

    # Define o tamanho do conjunto de validação (20%)
    validation_size = int(len(x_train) * 0.2)

    # Separa treino e validação
    x_val = x_train[:validation_size]
    y_val = y_train[:validation_size]

    x_train = x_train[validation_size:]
    y_train = y_train[validation_size:]

    return x_train, y_train, x_val, y_val, x_test, y_test

def criar_modelo():
    modelo = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),

        # Bloco convolucional 1
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloco convolucional 2
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Bloco convolucional 3
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classificador
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(10, activation="softmax")
    ])

    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return modelo


def treinar_modelo(modelo, x_train, y_train, x_val, y_val):
    #Treina o modelo utilizando early stopping e salva o modelo treinado.
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    historico = modelo.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=15,
        batch_size=64,
        callbacks=[early_stopping]
    )

    val_accuracy = max(historico.history["val_accuracy"])

    print(f"Acurácia final de validação: {val_accuracy:.4f}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    modelo.save(os.path.join(script_dir, "model.h5"))

    print("Modelo salvo como model.h5")


def main():
    x_train, y_train, x_val, y_val, x_test, y_test = carregar_dados()

    print(f"Treino: {x_train.shape}")
    print(f"Validação: {x_val.shape}")
    print(f"Teste: {x_test.shape}")

    modelo = criar_modelo()

    treinar_modelo(
        modelo,
        x_train,
        y_train,
        x_val,
        y_val
    )

if __name__ == "__main__":
    main()