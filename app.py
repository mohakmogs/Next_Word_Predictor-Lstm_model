import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🧠",
    layout="centered"
)

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("wikitext_ac18_model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_maxlen():
    with open("max_len.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()
max_len = load_maxlen()

# Reverse dictionary
index_to_word = {v: k for k, v in tokenizer.word_index.items()}


# ---------------- Predictor ----------------
def predictor(model, tokenizer, text, max_len,
              temperature=0.8,
              top_k=5):

    text = text.lower()

    seq = tokenizer.texts_to_sequences([text])[0]

    seq = pad_sequences(
        [seq],
        maxlen=max_len,
        padding="pre"
    )

    pred = model.predict(seq, verbose=0)[0]

    # Ignore OOV token
    if len(pred) > 1:
        pred[1] = 0

    pred = np.log(pred + 1e-10) / temperature
    pred = np.exp(pred)
    pred /= np.sum(pred)

    top_indices = np.argpartition(pred, -top_k)[-top_k:]

    top_probs = pred[top_indices]
    top_probs /= np.sum(top_probs)

    pred_index = np.random.choice(top_indices, p=top_probs)

    return index_to_word.get(pred_index, "")


# ---------------- Generate Text ----------------
def generate_text(seed_text, n_words):

    generated = seed_text

    for _ in range(n_words):

        next_word = predictor(
            model,
            tokenizer,
            generated,
            max_len
        )

        if next_word == "":
            break

        generated += " " + next_word

    return generated


# ---------------- UI ----------------

st.title("🧠 Next Word Predictor")

st.markdown(
"""
Generate text using an **LSTM Language Model**
trained on the **WikiText Dataset**.
"""
)

seed = st.text_input(
    "Enter starting text",
    placeholder="Example: Artificial intelligence"
)

num_words = st.slider(
    "Number of words to generate",
    1,
    30,
    10
)

temperature = st.slider(
    "Creativity (Temperature)",
    0.5,
    1.5,
    0.8,
    0.1
)

if st.button("🚀 Generate"):

    if seed.strip() == "":
        st.warning("Please enter some text.")
    else:

        with st.spinner("Generating..."):

            output = seed

            for _ in range(num_words):

                next_word = predictor(
                    model,
                    tokenizer,
                    output,
                    max_len,
                    temperature=temperature,
                    top_k=5
                )

                if next_word == "":
                    break

                output += " " + next_word

        st.success("Generated Text")

        st.markdown("### ✨ Output")

        st.write(output)

st.markdown("---")

st.caption(
    "Built with TensorFlow • LSTM • Streamlit"
)