import json
import streamlit as st
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    pipeline,
)

model_card = 'fine-tuned-emotion-classification-model'
model = DistilBertForSequenceClassification.from_pretrained(model_card)
tokenizer = DistilBertTokenizer.from_pretrained(model_card)


def main():
    emotions_labels = read_json()
    st.set_page_config(
        page_title="Emotions classification",
        page_icon="😁",
    )
    st.subheader("Emotions classification 😢😃😍🤬🙈😱")
    text = st.text_input("Introduce your text here")
    st_classify_button(text, emotions_labels)


def st_classify_button(text, emotions_labels):
    if st.button("Classify", disabled=not text):
        with st.spinner('Loading emotion...'):
            emotion_number, score = classify_emotion(text)

        emotion_dict = emotions_labels[emotion_number]
        for emotion, emoji in emotion_dict.items():
            st.write(f"The emotion is {emotion.capitalize()} {emoji} with {(score*100):.2f}%")
        st.balloons()


def classify_emotion(text):
    generator_pipeline = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer
    )
    # [{'label': 'LABEL_0', 'score': 0.9996743202209473}]
    result = generator_pipeline(text)
    emotion = result[0]['label'].split('_')[-1]
    score = result[0]['score']
    return emotion, score


def read_json(path="./streamlit/emotions.json"):
    with open(path, 'r') as file:
        emotions_labels = json.load(file)
    return emotions_labels


if __name__ == "__main__":
    main()
