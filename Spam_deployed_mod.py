import streamlit as st
import tensorflow_text
import math
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Spam Classification", page_icon=":tada:")

def load_spam_model(model_path):
    with st.spinner("Loading model..."):
        loaded_model = load_model(model_path)
    return loaded_model

def predict_spam(text, model):
    score = model.predict([text])[0]
    return score

def main():
    st.title("Spam Classification")
    model_path = "model"
    loaded_model = load_spam_model(model_path)

    text_input = st.text_input("Enter the Text").lower()
    st.write("---")
    
    if st.button('PREDICT'):
        with st.spinner("Please wait..."):
            score = predict_spam(text_input, loaded_model)
        if score < 0.5:
            st.subheader("Ham")
        else:
            st.subheader("Spam")

if __name__ == "__main__":
    main()
