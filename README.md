# 🧠 AI Next Word Prediction using LSTM

An AI-powered **Next Word Prediction** web application built using **TensorFlow**, **Keras**, and **Streamlit**. The model is trained on the **WikiText** dataset using an LSTM (Long Short-Term Memory) neural network to predict the most probable next word given a sequence of input words.

---

## 🚀 Live Demo

🔗 **Coming Soon**

---

## 📸 Application Preview

> Add screenshots of your Streamlit application here.

| Home Page | Prediction |
|-----------|------------|
| ![Home]("Screenshot 2026-08-04 172419.png") | ![Prediction]([Screenshot 2026-08-04 171622.png](https://github.com/mohakmogs/Next_Word_Predictor-Lstm_model/blob/main/Screenshot%202026-08-04%20171622.png)) |

---

## ✨ Features

- 🧠 LSTM-based Language Model
- 📖 Trained on the WikiText Dataset
- 🔤 Predicts the Next Word from User Input
- ⚡ Fast Inference using TensorFlow
- 🎲 Top-k Sampling for Better Predictions
- 📱 Interactive Streamlit Web Application
- 💾 Model Serialization using Pickle
- 🎯 Clean and Responsive User Interface

---

## 🛠️ Tech Stack

- Python
- TensorFlow
- Keras
- NumPy
- Streamlit
- Pickle

---

## 📂 Project Structure

```
Next-Word-Prediction/
│
├── app.py
├── requirements.txt
├── tokenizer.pkl
├── max_len.pkl
├── wikitext_ac18_model.h5
├── README.md
└── images/
    ├── home.png
    └── output.png
```

---

## 🧠 Model Architecture

```
Input Text
     │
Tokenizer
     │
Padding
     │
Embedding Layer (128)
     │
LSTM Layer (256)
     │
Dropout (0.2)
     │
Dense Layer (Softmax)
     │
Predicted Next Word
```

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Dataset | WikiText |
| Vocabulary Size | 20,000 |
| Embedding Dimension | 128 |
| LSTM Units | 256 |
| Dropout | 0.2 |
| Batch Size | 256 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Sparse Categorical Crossentropy |

---

## 📊 Dataset

This project uses the **WikiText** dataset, a high-quality collection of Wikipedia articles widely used for Natural Language Processing and Language Modeling research.

### Dataset Statistics

- 100,000 cleaned text samples used
- Vocabulary limited to 20,000 most frequent words
- Maximum sequence length: 50 tokens

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/next-word-prediction-lstm.git

cd next-word-prediction-lstm
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 💡 Example

**Input**

```
Artificial intelligence
```

**Generated Output**

```
Artificial intelligence has become one of the most important technologies in modern computing.
```

---

## 📈 Future Improvements

- Bidirectional LSTM
- Transformer-based Language Model
- Beam Search Decoding
- Attention Mechanism
- GPT-style Text Generation
- Hugging Face Deployment
- Docker Support

---

## 👨‍💻 Author

**Mohak Pandey**

Computer Science Engineering Student

📧 Email: *your-email@example.com*

🔗 GitHub: https://github.com/mohakmogs

🔗 LinkedIn: *Add your LinkedIn profile*

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub.

It motivates me to build more AI and Machine Learning projects.

---

## 📜 License

This project is licensed under the MIT License.
