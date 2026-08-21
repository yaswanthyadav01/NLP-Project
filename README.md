# NLP-Project
# NLP Sentiment Analysis Project

## 📌 Project Overview

This project is a **Natural Language Processing (NLP) Sentiment Analysis** application that analyzes customer reviews and classifies them into three sentiment categories:

* **Positive**
* **Neutral**
* **Negative**

The project uses review **title, body, and rating** information. The rating is converted into a sentiment label, after which the review text is cleaned and converted into numerical features using **TF-IDF**.

Multiple machine-learning classification algorithms are trained and evaluated to identify the best-performing model.

---

## 🎯 Project Objective

The main objectives of this project are:

1. Perform Exploratory Data Analysis (EDA) on customer reviews.
2. Clean and preprocess textual data.
3. Convert text into numerical features using TF-IDF.
4. Build multiple machine-learning classification models.
5. Compare model performance using Accuracy and F1 Score.
6. Select the best-performing model.
7. Save the trained model and TF-IDF vectorizer.
8. Deploy the sentiment-analysis application using Streamlit.

---

## 📊 Dataset

The project uses an Excel dataset containing review information.

Important columns used in the project include:

* `title` — Review title
* `body` — Review text
* `rating` — Customer rating

The project combines the title and body into a single review field.

```python
df['Review'] = (
    df['title'].astype(str) + ' ' + df['body'].astype(str)
)
```

---

## 🏷️ Sentiment Classification

The customer rating is converted into three sentiment classes.

| Rating | Sentiment |
| ------ | --------- |
| 1–2    | Negative  |
| 3      | Neutral   |
| 4–5    | Positive  |

The project implements this using a `rating_to_sentiment()` function.

---

## 🧹 Text Preprocessing

The review text goes through several preprocessing steps:

* Convert text to lowercase
* Remove URLs
* Remove unwanted punctuation
* Preserve Hindi/Devanagari characters
* Remove extra spaces
* Calculate word counts

The preprocessing function supports Hindi Unicode characters using the Devanagari Unicode range.

---

## 🔢 TF-IDF Feature Extraction

The cleaned review text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The project uses:

```python
TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)
```

This configuration uses both:

* Unigrams — single words
* Bigrams — two-word combinations

The TF-IDF vectorizer is trained only on the training data and then used to transform the test data.

---

## 🤖 Machine Learning Models

The project trains and compares four machine-learning algorithms:

### 1. Logistic Regression

```python
LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)
```

### 2. Multinomial Naive Bayes

```python
MultinomialNB()
```

### 3. Linear Support Vector Machine

```python
LinearSVC(
    class_weight='balanced'
)
```

### 4. Random Forest

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)
```

---

## 📈 Model Evaluation

The models are evaluated using:

* Accuracy
* F1 Score
* Classification Report
* Confusion Matrix

The project compares the models using a results table containing Accuracy and weighted F1 Score.

The model with the highest weighted F1 Score is intended to be selected as the best model.

---

## 📊 Exploratory Data Analysis

The project performs several EDA operations, including:

* Dataset shape
* Dataset information
* Missing-value analysis
* Duplicate-value analysis
* Statistical summary
* Rating distribution
* Sentiment distribution
* Rating vs sentiment
* Review word-count distribution
* Positive review word cloud
* Negative review word cloud

---

## ☁️ Word Clouds

Word clouds are generated separately for:

* Positive reviews
* Negative reviews

This helps identify frequently occurring words associated with different sentiments.

---

## 💾 Saved Model Files

After training, the project saves two important files:

### Trained ML model

```text
model.pkl
```

### TF-IDF vectorizer

```text
tfidf.pkl
```

These files can be loaded later by the Streamlit application without retraining the model.

The project saves them using `joblib`.

```python
joblib.dump(best_model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
```

---

## 🌐 Streamlit Deployment

The trained model can be deployed as a web application using **Streamlit**.

Recommended project structure:

```text
NLP-Project/
│
├── app.py
├── requirements.txt
├── model.pkl
├── tfidf.pkl
├── project_(1).py
└── README.md
```

---

## 📦 Requirements

Create a file named:

```text
requirements.txt
```

Add:

```text
streamlit
pandas
numpy
scikit-learn
joblib
nltk
openpyxl
wordcloud
seaborn
matplotlib
```

The `joblib` package is required because the trained model and TF-IDF vectorizer are saved and loaded using Joblib.

---

## 🚀 Run the Project Locally

### Step 1 — Clone the repository

```bash
git clone https://github.com/yaswanthyadav01/NLP-Project.git
```

### Step 2 — Open the project

```bash
cd NLP-Project
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Deploy on Streamlit Cloud

1. Upload the project to GitHub.
2. Make sure `app.py` is in the repository.
3. Make sure `requirements.txt` is in the repository root.
4. Make sure `model.pkl` and `tfidf.pkl` are uploaded.
5. Open Streamlit Cloud.
6. Select your GitHub repository.
7. Select the `main` branch.
8. Select:

```text
app.py
```

as the main file.
9. Deploy the application.

---

## 🛠️ Technologies Used

| Technology   | Purpose                        |
| ------------ | ------------------------------ |
| Python       | Programming language           |
| Pandas       | Data processing                |
| NumPy        | Numerical operations           |
| Matplotlib   | Data visualization             |
| Seaborn      | Statistical visualization      |
| NLTK         | NLP processing                 |
| WordCloud    | Text visualization             |
| Scikit-learn | Machine learning               |
| TF-IDF       | Text feature extraction        |
| Joblib       | Model serialization            |
| Streamlit    | Web application and deployment |
| GitHub       | Source-code management         |

---

## 📁 Project Files

```text
app.py
```

Streamlit application.

```text
project_(1).py
```

Main NLP training and analysis code.

```text
model.pkl
```

Saved trained machine-learning model.

```text
tfidf.pkl
```

Saved TF-IDF vectorizer.

```text
requirements.txt
```

Python dependencies required for deployment.

```text
README.md
```

Project documentation.

---

## 🔄 Project Workflow

```text
Dataset
   ↓
Data Loading
   ↓
EDA
   ↓
Missing Value Handling
   ↓
Rating → Sentiment
   ↓
Text Combination
   ↓
Text Cleaning
   ↓
Train/Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Machine Learning Models
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Save model.pkl
   ↓
Save tfidf.pkl
   ↓
Streamlit Application
   ↓
Sentiment Prediction
```

---

## 🎯 Expected Application

The final Streamlit application can allow a user to enter a review such as:

```text
The product quality is excellent and I really enjoyed using it.
```

The application processes the text using the saved TF-IDF vectorizer and sends the resulting features to the saved machine-learning model.

The predicted result can then be displayed as:

```text
Predicted Sentiment: Positive
```

---

## ⚠️ Important Deployment Notes

Make sure the following files are available in the GitHub repository:

```text
app.py
requirements.txt
model.pkl
tfidf.pkl
```

The dependency file must be named exactly:

```text
requirements.txt
```

Not:

```text
requirement.txt
```

Also, `scikit-learn` is the package name in `requirements.txt`, while Python imports it using `sklearn`.

For example:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```

and:

```text
scikit-learn
```

in `requirements.txt`.

---

## 👨‍💻 Author

**Yaswanth Yadav**

GitHub:

`yaswanthyadav01/NLP-Project`

---

## ⭐ Conclusion

This project demonstrates an end-to-end NLP sentiment-analysis pipeline, starting from customer-review data and ending with a deployable machine-learning application.

The system combines:

**Data Analysis → Text Preprocessing → TF-IDF → Machine Learning → Model Evaluation → Model Saving → Streamlit Deployment**

to provide an automated sentiment-classification solution.
