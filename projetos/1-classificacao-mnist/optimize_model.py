import tensorflow as tf
import os
from tensorflow import keras

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

# insira seu código aqui


def otimizar_modelo():
    # Converte o modelo Keras (.h5) para TensorFlow Lite
    # aplicando Dynamic Range Quantization.

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Carrega o modelo treinado
    modelo = keras.models.load_model(os.path.join(script_dir, "model.h5"))

    # Cria o conversor TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(modelo)

    # Aplica otimização
    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    # Realiza a conversão
    modelo_tflite = converter.convert()

    # Salva o modelo otimizado
    with open(os.path.join(script_dir, "model.tflite"), "wb") as arquivo:
        arquivo.write(modelo_tflite)

    print("Modelo convertido e salvo como model.tflite")


def main():
    otimizar_modelo()
    

if __name__ == "__main__":
    main()