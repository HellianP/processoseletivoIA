# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 Hellian Sampaio Silva Peixinho

### 1️⃣ Resumo da Arquitetura do Modelo

Foi desenvolvida uma Rede Neural Convolucional (CNN) para classificação de dígitos manuscritos do conjunto MNIST.

A arquitetura possui três blocos convolucionais compostos por camadas Conv2D, BatchNormalization e MaxPooling2D. Após os blocos convolucionais, foi utilizada uma camada Flatten seguida de uma camada totalmente conectada (Dense) com 128 neurônios. Antes da camada de saída foi aplicada uma camada Dropout (0.5) para reduzir o risco de overfitting. A camada de saída possui 10 neurônios com função de ativação Softmax, correspondentes às classes de dígitos de 0 a 9.

O treinamento foi realizado utilizando EarlyStopping monitorando a perda no conjunto de validação (val_loss), com restauração automática dos melhores pesos obtidos durante o treinamento.

### 2️⃣ Bibliotecas Utilizadas

TensorFlow 2.19 e Keras
NumPy 2.1

### 3️⃣ Técnica de Otimização do Modelo

Foi utilizada a técnica Dynamic Range Quantization durante a conversão do modelo Keras (.h5) para TensorFlow Lite (.tflite).

A otimização foi aplicada utilizando:

converter.optimizations = [tf.lite.Optimize.DEFAULT]

Essa técnica reduz significativamente o tamanho do modelo sem necessidade de novo treinamento, mantendo desempenho adequado para execução em dispositivos Edge AI.

### 4️⃣ Resultados Obtidos

Informe a acurácia de validação obtida e o tamanho dos arquivos `model.h5` e `model.tflite`.
Melhor acurácia de validação foi de 98,71%
Tamanho do modelo treinado (model.h5): 1.406.904 bytes (aproximadamente 1,34 MB)
Tamanho do modelo otimizado (model.tflite): 125.368 bytes (aproximadamente 122,43 KB)
A redução de tamanho foi de aproximadamente 91,1%, mantendo desempenho adequado para inferência.

### 5️⃣ Comentários Adicionais (Opcional)

Durante o desenvolvimento foi adotada uma estrutura modular, separando as etapas de treinamento, otimização e inferência em partes independentes,Inicialmente os modelos estavam sendo salvos no diretório de execução do programa. Esse comportamento foi corrigido utilizando caminhos relativos ao diretório do próprio script (__file__), garantindo que os artefatos fossem sempre gerados na pasta do projeto, independentemente do diretório em que o script fosse executado.

Também foi utilizada a estratégia de EarlyStopping para evitar overfitting e reduzir o tempo de treinamento.

### 6️⃣ Exemplo de Inferência

Rodando inferência em 5 amostras usando model.tflite:
Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4

Nas cinco amostras avaliadas, o modelo classificou corretamente todos os dígitos apresentados, demonstrando que a conversão para TensorFlow Lite preservou o desempenho do modelo treinado.
