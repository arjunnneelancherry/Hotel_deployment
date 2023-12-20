import streamlit as st
import joblib
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Load the pre-trained models
tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
final_logreg_model = joblib.load('final_logreg_model.joblib')

# Function to predict sentiment
def predict_sentiment(review):
    review_tfidf = tfidf_vectorizer.transform([review])
    prediction_probs = final_logreg_model.predict_proba(review_tfidf)[0]
    prediction = np.argmax(prediction_probs)
    confidence = prediction_probs[prediction]
    return prediction, confidence

# Function to generate Word Cloud
def generate_wordcloud(review):
    if review and any(c.isalpha() for c in review):  # Check if there are any alphabetical characters in the input
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(review)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("No words to generate a word cloud.")

# Streamlit app
def main():
    # Set page configuration to full screen
    st.set_page_config(layout="wide")

    # Create a layout with two columns
    col1, col2 = st.columns(2)

    # Load and display the image in the first column with a specific width
    hotel_image = 'hotel_image.png'
    col1.image(hotel_image, width=145)

    # Display the title in the second column
    col2.title("HotelSentinel: Review Sentiment Analyzer")

    # User input for review
    review_input = st.text_area("Enter your review:")

    # Feedback form using st.form and st.form_submit_button
    with st.form(key='feedback_form'):
        # Make prediction when the user clicks the button
        if st.form_submit_button("Predict"):
            if not review_input:
                st.warning("Please enter a review.")
            else:
                sentiment_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
                prediction, confidence = predict_sentiment(review_input)
                sentiment_label = sentiment_mapping[prediction]

                # Display sentiment emoji
                sentiment_emoji = get_sentiment_emoji(sentiment_label)

                # Display predicted sentiment with confidence
                st.success(f"Predicted Sentiment: {sentiment_label} {sentiment_emoji}")
                st.info(f"Confidence: {confidence:.2%}")

                # Check if the review is not empty before generating Word Cloud
                if review_input.strip():
                    generate_wordcloud(review_input)
                else:
                    st.warning("No words to generate a word cloud.")

        # Feedback input box
        feedback = st.text_input("Your feedback (optional):")

        # Submit button for feedback
        submit_button = st.form_submit_button("Submit Feedback")
        if submit_button and feedback:
            st.session_state.feedback_submitted = True  # Mark feedback as submitted
            st.success("Thanks for your feedback!")

    

# Function to get sentiment emoji
def get_sentiment_emoji(sentiment_label):
    if sentiment_label == 'Positive':
        return "😃"
    elif sentiment_label == 'Neutral':
        return "😐"
    elif sentiment_label == 'Negative':
        return "😞"

# Run the app
if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)  # Disable the warning
    main()
