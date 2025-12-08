import streamlit as st
from inference import predict

# Page setup
st.set_page_config(page_title="Slant Classifier", page_icon="📰")

st.title("Newspaper Slant Classifier")

# Input field
text = st.text_area("Enter text")

# Button
if st.button("Classify"):
    if len(text.strip()) == 0:
        st.warning("Please enter text.")
    else:
        # Run prediction
        scores = predict(text)
        dem = scores[0]
        rep = scores[1]

        # Display result
        st.subheader("Result")
        if dem > rep:
            st.success(f"Predicted: Democrat ({dem:.4f})")
        else:
            st.success(f"Predicted: Republican ({rep:.4f})")

        # Display raw outputs
        st.write("Raw scores:", scores)
