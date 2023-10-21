import streamlit as st
from transformers import pipeline

# Initialize the classifier model
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# Create a Streamlit web app
st.set_page_config(
    page_title="Emotion Detection",
    page_icon=":bar_chart:",
    layout="centered",  # Center the content
)

# Define CSS style
st.markdown(
    """
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        padding: 10px 20px;
        border: none;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #86D8DB;
    }
    .stApp {
        background-color: #73D9C8;  /* Background color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Define title with emoji and image
st.title("🎭 Emotion Detection")
st.markdown("Choose the input type and enter a sentence or upload an image to classify emotions.")

# Create a toggle button to select input type
input_type = st.radio("Select Input Type", ("Text", "Image"))

# Create a text input box for user input
if input_type == "Text":
    user_input = st.text_area("Enter a sentence:")
    uploaded_image = None
else:
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    user_input = ""

# Create an 'Analyze' button
if st.button("Analyze"):
    with st.spinner("Analyzing..."):
        if input_type == "Text" and user_input:
            # Perform the classification for text input
            model_outputs = classifier(user_input)
            st.subheader("Emotion Classification Results (Text):")
        elif input_type == "Image" and uploaded_image is not None:
            # Display the uploaded image and perform the classification
            st.image(uploaded_image, use_column_width=True, caption="Uploaded Image")
            model_outputs = classifier("Analyze this image.")
            st.subheader("Emotion Classification Results (Image):")
        else:
            st.warning("Please enter a sentence or upload an image to analyze.")

        for label_info in model_outputs[0]:
            label = label_info["label"]
            score = label_info["score"]
            st.write(f"- {label}: {score:.4f}")

    # Add a 'Clear' button to reset the input
    if st.button("Clear"):
        user_input = ""
        uploaded_image = None
