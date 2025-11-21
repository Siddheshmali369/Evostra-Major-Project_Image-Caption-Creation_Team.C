# Flickr8k Image Captioning Project - Deep Learning (CNN-LSTM)

This project focuses on building an **Image Caption Generator** that bridges the gap between **Computer Vision** and **Natural Language Processing (NLP)**. The goal is to build a model that "sees" an image and generates a coherent, grammatically correct description in English. The project implements an Encoder-Decoder architecture using **InceptionV3** for feature extraction and **LSTMs** for text generation, trained on the **Flickr8k dataset**.

-----

## Table of Contents

1.  [Understanding the Problem](#understanding-the-problem)
2.  [Dataset Information](#dataset-information)
3.  [Architecture Design](#architecture-design)
4.  [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5.  [Model Training](#model-training)
6.  [Evaluation and Results](#evaluation-and-results)
7.  [Inference Techniques](#inference-techniques)
8.  [Technologies Used](#technologies-used)
9.  [How to Run](#how-to-run)

-----

## Understanding the Problem

Image captioning requires a model to understand visual content and translate it into a sequence of words. This project solves two main challenges:

1.  **Visual Understanding**: Extracting meaningful features from an image (colors, objects, actions).
2.  **Sequential Modeling**: Generating a sentence word-by-word, where the next word depends on the previous context and the image content.

-----

## Dataset Information

The model is trained on the **Flickr8k Dataset**, a standard benchmark for sentence-based image description.

  - **Total Images**: 8,091
  - **Captions**: 40,455 (5 captions per image)
  - **Source**: Flickr (Images of people and dogs performing actions are common)
  - **Splits**:
      - **Training**: 6,000 images
      - **Validation**: 1,000 images
      - **Testing**: ~1,091 images

The raw dataset consists of a folder of `.jpg` images and a text file (`captions.txt`) mapping image filenames to their descriptions.

-----

## Architecture Design

The project uses a **Merge-Model Architecture**:

### 1\. The Encoder (Computer Vision)

  - **Model**: InceptionV3 (Pre-trained on ImageNet)
  - **Strategy**: Transfer Learning. The final classification layer is removed.
  - **Output**: A **2048-dimensional vector** representing the visual features of the image.

### 2\. The Decoder (NLP)

  - **Input**: Sequence of partial captions.
  - **Layers**:
      - **Embedding Layer**: Converts word indices to dense vectors (256-dim).
      - **LSTM Layer**: A Long Short-Term Memory unit (256 units) that handles sequence dependency.

### 3\. The Merge

  - The output of the Image Encoder and the Text Decoder are added together.
  - Passed through a **Dense Layer** (256 units).
  - **Final Output**: A Softmax layer over the entire vocabulary (\~8,574 words) to predict the probability of the next word.

-----

## Data Preprocessing Pipeline

To handle the dataset efficiently without crashing RAM (a common issue in Colab), a custom pipeline was built:

### Image Processing

  - **Resizing**: All images resized to `299x299` (InceptionV3 input standard).
  - **Feature Serialization**: Images were processed once through InceptionV3, and the resulting 2048-vectors were saved into **TFRecord files** (`train.tfrecord`, `val.tfrecord`, `test.tfrecord`). This allows for high-speed streaming during training.

### Text Processing

  - **Cleaning**: Converted to lowercase, removed punctuation, removed numbers.
  - **Tokenization**: Created a vocabulary of **8,574 unique words**.
  - **Padding**: Sequences padded to a maximum length of **31 words**.
  - **Input-Output Pairs**:
      - Input: `[startseq, A, dog, is]`
      - Target: `running`

-----

## Model Training

The model was trained on Google Colab using GPU acceleration.

  - **Loss Function**: Categorical Cross-Entropy.
  - **Optimizer**: Adam (Learning Rate: 0.001 with `ReduceLROnPlateau`).
  - **Batch Size**: 64.
  - **Epochs**: Trained for 12 epochs (stopped early due to convergence).
  - **Callbacks**: `ModelCheckpoint` was used to save weights after every epoch to prevent data loss.

-----

## Evaluation and Results

The model was evaluated on a held-out Test Set of \~1,000 images using the **BLEU Score** (Bilingual Evaluation Understudy) metric.

### Quantitative Analysis

We achieved a **BLEU-4 score \> 0.10**, which passes the standard baseline for CNN-LSTM models on this dataset.

| Metric | Score | Meaning |
| :--- | :--- | :--- |
| **BLEU-1** | **0.435** | 43.5% of generated words match the reference captions. |
| **BLEU-4** | **0.101** | The model generates fluent 4-word phrases, passing the coherency baseline. |

### Qualitative Analysis

  - **Success**: The model excels at identifying "dogs", "people", "grass", and "running".
  - **Limitations**: Without an Attention Mechanism, the model sometimes struggles with small details (e.g., missing a "ball" in a dog's mouth) or surreal images (confusing abstract textures with "crowds").

-----

## Inference Techniques

Two methods were implemented to generate captions from the trained model:

1.  **Greedy Search**:

      - Picks the single word with the highest probability at each step.
      - *Result*: Fast, but often produces repetitive or simple sentences.

2.  **Beam Search (K=5)**:

      - Keeps track of the top 5 most likely sequences at every step.
      - *Result*: Produced significantly better grammar and more natural sentence structures.
      - *Trade-off*: Computationally expensive (took \~40 mins to evaluate the full test set vs \~3 mins for Greedy).

-----

## Technologies Used

### Core Libraries

  - **TensorFlow / Keras**: Model building and training.
  - **InceptionV3**: Pre-trained CNN for feature extraction.
  - **NLTK**: Calculating BLEU scores.
  - **Pandas / NumPy**: Data manipulation.
  - **Matplotlib**: Visualizing images and captions.

### Tools

  - **Google Colab**: Development environment (T4 GPU).
  - **Google Drive**: Persistent storage for models and checkpoints.
  - **TFRecords**: Efficient data serialization format.

-----

## How to Run

### Prerequisites

1.  **Google Account**: To access Google Colab and Drive.
2.  **Dataset**: Download `flickr8k` dataset (archive.zip).

### Execution Steps

1.  **Setup**: Upload `archive.zip` to a folder named `flickr8k_project` in Google Drive.
2.  **Mount Drive**: Run the initial cells to mount Drive and unzip data.
3.  **Preprocessing**: Run the extraction cells to generate `.tfrecord` files and `tokenizer.pkl`.
4.  **Training**: Execute the training loop. It will automatically save `model_final.keras` to Drive.
5.  **Inference**:
      - Load the saved model.
      - Upload a custom image using the provided widget.
      - View the generated caption (Beam Search enabled).

-----

**Project**: Image Captioning with Deep Learning  
**Dataset**: Flickr8k  
**Status**: Completed

*This project is for educational purposes.*
