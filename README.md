# 🎬 IMDB Movie Reviews - Sentiment Analysis Benchmarking

An end-to-end Machine Learning and Natural Language Processing (NLP) repository benchmarking various text embedding and vectorization techniques (**TF-IDF, Word2Vec, and BERT Transformers**) to classify text sentiments over 50,000 highly polar IMDB movie reviews.

---

## 📊 Performance Matrix & Benchmarking
This repository evaluates traditional statistical approaches against modern deep transfer learning paradigms to optimize text sequencing profiles:

| Vectorization Strategy | Model Class Architecture | Key Advantage | Target Accuracy |
| :--- | :--- | :--- | :--- |
| **TF-IDF** | Statistical Linear Classifier | Lightweight, Extremely Fast Computation | ~88% - 89% |
| **Word2Vec** | Neural Word Embeddings (Continuous) | Captures Semantic/Contextual Vector Proximity | ~86% - 87% |
| **BERT (Transformers)** | Deep Bidirectional Attention Networks | State-of-the-Art Contextual Sequence Awareness | **~93%++** |

---

## 🚀 Key Architectural Pipeline

1. **Text Normalization Engine:** Dynamic text cleaning pipelines filtering HTML tags, trailing regex tokens, punctuation matrices, and stopword maps.
2. **Multi-Model Embeddings:** Implements multi-tier comparison architectures transitioning from sparse frequency models to dense transformer matrices.
3. **Deep Sequence Modeling:** Evaluates execution workflows to benchmark traditional machine learning backends against deep learning structural attention profiles.
4. **Diagnostic Analytics:** Detailed performance plotting tracking overfitting loops via loss charts, precision-recall evaluation, and confusion matrices.

---

## 📂 Repository Structure
```text
imdb_sentiment_analysis/
├── imdb_sentiment_analysis.ipynb   # Modular Jupyter Notebook containing the full execution pipeline
└── README.md                        # Professional documentation hub
🛠️ Reproduction & Local Setup
1. Environment Cloning
Bash
git clone [https://github.com/delowarhossaincse63/imdb_sentiment_analysis.ipynb.git](https://github.com/delowarhossaincse63/imdb_sentiment_analysis.ipynb.git)
cd imdb_sentiment_analysis.ipynb
2. Dependency Extraction
Initialize a localized environment and install core computational libraries:

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install critical NLP frameworks
pip install numpy pandas scikit-learn nltk gensim torch transformers notebook
3. Executing Notebook
Open the development environment server and step through the functional programming layers:

Bash
jupyter notebook imdb_sentiment_analysis.ipynb
📡 Pipeline Architecture Details
🔹 Preprocessing Tokenizer Layer
Python
# Cleans and standardizes raw unstructured strings
def advanced_text_cleaning(raw_text):
    # 1. Strip HTML wrappers
    # 2. Tokenize and normalize text case maps
    # 3. Filter alphanumeric sequences and NLTK standard stop vocabularies
    return cleaned_tokens
🔹 Sample Production Inference Hook
Python
# Evaluates custom client-facing text inputs through the optimal trained backend
def live_predict_sentiment(custom_review_string: str):
    # Runs the raw pipeline matrix mapping over the transformer tokenizer
    prediction_probability = optimal_model.predict(custom_review_string)
    sentiment = "Positive" if prediction_probability > 0.5 else "Negative"
    return {"sentiment": sentiment, "confidence_score": float(prediction_probability)}
🤝 Contribution & Optimization
Feel free to open structural issues or fork the architecture to append higher-order transformer variants like RoBERTa or DeBERTa architectures.

📄 License
This benchmark environment is open-sourced under the MIT License.
