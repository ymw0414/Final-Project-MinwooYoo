import streamlit as st
from inference import predict

# Page settings
st.set_page_config(page_title="Slant Classifier", page_icon="📰")

st.title("Newspaper Slant Classifier")

# Text input
text = st.text_area("Enter text")

# Button action
if st.button("Classify"):
    if len(text.strip()) == 0:
        st.warning("Please enter text.")
    else:
        # Run prediction
        score = predict(text)
        dem = score[0]
        rep = score[1]

        st.subheader("Result")

        # Compare scores and show final prediction
        if dem > rep:
            st.success(f"Predicted: Democrat ({dem:.4f})")
        else:
            st.success(f"Predicted: Republican ({rep:.4f})")

        # Show raw probabilities
        st.write("Raw scores:", score)
