import streamlit as st
import joblib
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NLP Sentiment Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #f8fafc 0%,
            #eef2ff 50%,
            #f8fafc 100%
        );
    }

    /* Main title */
    .main-title {
        font-size: 48px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 5px;
        color: #1e293b;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #64748b;
        margin-bottom: 35px;
    }

    /* Cards */
    .card {
        padding: 25px;
        border-radius: 18px;
        background: rgba(255,255,255,0.85);
        box-shadow: 0 8px 30px rgba(15,23,42,0.08);
        border: 1px solid rgba(148,163,184,0.2);
        margin-bottom: 20px;
    }

    /* Prediction */
    .prediction {
        text-align: center;
        padding: 30px;
        border-radius: 20px;
        background: white;
        box-shadow: 0 10px 35px rgba(15,23,42,0.10);
    }

    .prediction-label {
        font-size: 18px;
        color: #64748b;
    }

    .prediction-value {
        font-size: 42px;
        font-weight: 800;
        color: #1e293b;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 14px;
        padding: 30px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_models():

    model = joblib.load("model.pkl")
    vectorizer = joblib.load("tfidf.pkl")

    return model, vectorizer


try:

    model, vectorizer = load_models()

    model_loaded = True

except Exception as e:

    model_loaded = False

    st.error("❌ Model files could not be loaded.")

    st.code(str(e))

    st.info(
        "Make sure model.pkl and tfidf.pkl are present "
        "in the same folder as app.py."
    )


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):

    text = str(text)

    # lowercase
    text = text.lower()

    # remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text
    )

    # remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🧠 NLP Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered sentiment prediction using TF-IDF and Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Project Information")

    st.write(
        """
        **NLP Sentiment Analysis**

        This application converts text into numerical
        TF-IDF features and uses a trained machine
        learning model to predict sentiment.
        """
    )

    st.divider()

    st.subheader("🛠️ Technology")

    st.write("🐍 Python")
    st.write("🎈 Streamlit")
    st.write("🔤 TF-IDF")
    st.write("🤖 Machine Learning")
    st.write("📦 Scikit-learn")
    st.write("💾 Joblib")

    st.divider()

    st.subheader("📌 Supported Input")

    st.write(
        "English and Unicode/Hindi text can be entered."
    )


# =========================================================
# MODEL STATUS
# =========================================================

if model_loaded:

    st.success("🟢 Model loaded successfully")

else:

    st.stop()


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("📝 Enter Your Review")

review = st.text_area(
    "Type your review below:",
    placeholder=(
        "Example: This product is excellent. "
        "I really enjoyed using it!"
    ),
    height=180,
    label_visibility="collapsed"
)

# Example buttons

st.caption("💡 Try an example:")

col1, col2, col3 = st.columns(3)

with col1:

    positive_example = st.button(
        "😊 Positive Example",
        use_container_width=True
    )

with col2:

    negative_example = st.button(
        "😞 Negative Example",
        use_container_width=True
    )

with col3:

    hindi_example = st.button(
        "🇮🇳 Hindi Example",
        use_container_width=True
    )

if positive_example:

    review = (
        "This product is amazing and excellent. "
        "I really loved it!"
    )

if negative_example:

    review = (
        "This product is terrible and disappointing. "
        "I completely hated it."
    )

if hindi_example:

    review = "यह उत्पाद बहुत अच्छा है और मुझे बहुत पसंद आया।"

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# PREDICTION
# =========================================================

predict_button = st.button(
    "🚀 Analyze Sentiment",
    type="primary",
    use_container_width=True
)


if predict_button:

    if not review or not review.strip():

        st.warning(
            "⚠️ Please enter a review before predicting."
        )

    else:

        # Clean text
        cleaned_review = clean_text(review)

        # Convert to TF-IDF
        review_tfidf = vectorizer.transform(
            [cleaned_review]
        )

        # Prediction
        prediction = model.predict(
            review_tfidf
        )[0]

        # Probability
        probability = None

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                review_tfidf
            )[0]

            probability = max(probabilities) * 100


        # =================================================
        # RESULT
        # =================================================

        st.divider()

        st.subheader("🎯 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.markdown(
                '<div class="prediction">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="prediction-label">'
                'Predicted Sentiment'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="prediction-value">'
                f'{prediction}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with result_col2:

            st.markdown(
                '<div class="prediction">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="prediction-label">'
                'Model Confidence'
                '</div>',
                unsafe_allow_html=True
            )

            if probability is not None:

                st.markdown(
                    f'<div class="prediction-value">'
                    f'{probability:.2f}%'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.progress(
                    int(probability)
                )

            else:

                st.markdown(
                    '<div class="prediction-value">'
                    'N/A'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # =================================================
        # SENTIMENT MESSAGE
        # =================================================

        prediction_text = str(prediction).lower()

        if (
            "positive" in prediction_text
            or prediction_text == "1"
        ):

            st.success(
                "😊 The model detected a positive sentiment."
            )

        elif (
            "negative" in prediction_text
            or prediction_text == "0"
        ):

            st.error(
                "😞 The model detected a negative sentiment."
            )

        else:

            st.info(
                f"🔎 Predicted class: {prediction}"
            )


        # =================================================
        # TEXT DETAILS
        # =================================================

        st.subheader("🔍 Analysis Details")

        detail1, detail2, detail3 = st.columns(3)

        with detail1:

            st.metric(
                "Characters",
                len(review)
            )

        with detail2:

            st.metric(
                "Words",
                len(review.split())
            )

        with detail3:

            st.metric(
                "TF-IDF Features",
                review_tfidf.shape[1]
            )


        # =================================================
        # PROBABILITY DISTRIBUTION
        # =================================================

        if hasattr(model, "predict_proba"):

            st.subheader(
                "📊 Prediction Probability"
            )

            probability_df = pd.DataFrame(
                {
                    "Sentiment": model.classes_,
                    "Probability": probabilities
                }
            )

            probability_df["Probability"] = (
                probability_df["Probability"] * 100
            )

            st.bar_chart(
                probability_df.set_index(
                    "Sentiment"
                )
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        🧠 NLP Sentiment Analysis |
        Built with Python + TF-IDF + Machine Learning + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
