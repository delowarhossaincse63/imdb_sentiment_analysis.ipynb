# IMDB Movie Review Sentiment Analysis

Binary sentiment classification (positive / negative) on 50,000 IMDB
movie reviews using three different text representation techniques.

## Models Implemented
- TF-IDF + Logistic Regression
- Word2Vec + Logistic Regression  
- BERT Fine-tuning (bert-base-uncased)

## Dataset
[Q-b1t/IMDB-Dataset-of-50K-Movie-Reviews-Backup](https://huggingface.co/datasets/Q-b1t/IMDB-Dataset-of-50K-Movie-Reviews-Backup)
50,000 labeled movie reviews — 25,000 positive, 25,000 negative.

## Results

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| TF-IDF + LR    | ~0.89    | ~0.89     | ~0.89  | ~0.89    |
| Word2Vec + LR  | ~0.86    | ~0.86     | ~0.86  | ~0.86    |
| BERT Fine-tuned| ~0.92    | ~0.92     | ~0.92  | ~0.92    |

> Update the table above with your actual results after running the notebook.

## How to Run
1. Open the notebook in Google Colab
2. Enable GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells in order (Ctrl + F9)

## Requirements
See `requirements.txt`

## Project Structure
- Data loading and preprocessing
- TF-IDF vectorization + classification
- Word2Vec embedding training + classification
- BERT fine-tuning and evaluation
- Model comparison and analysis
