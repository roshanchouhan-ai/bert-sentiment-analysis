# IMDb Sentiment Analysis

An end-to-end sentiment analysis project using the IMDb movie review dataset. The project compares classical NLP techniques with neural word embeddings and a fine-tuned BERT model, then deploys the final model through a FastAPI service with Docker and automated testing.

## Overview

The project evaluates:

* Bag of Words
* TF-IDF
* Word2Vec
* FastText
* Doc2Vec
* BERT

The final BERT model is exposed through a FastAPI API supporting single-review and CSV batch inference.

## Project Highlights

* Exploratory data analysis and fixed train/validation/test split
* Comparison of multiple NLP representations
* Fine-tuned BERT for binary sentiment classification
* Model evaluation using accuracy, precision, recall, and F1
* BERT error analysis
* Single and batch inference
* Pydantic request validation
* Logging and exception handling
* Automated testing with pytest
* Dockerized API

## Dataset

The IMDb dataset contains movie reviews labeled as `positive` or `negative`.

Dataset split:

* 72% training
* 8% validation
* 20% testing

The same fixed split was used across experiments, with the test set reserved for final evaluation.

## Model Comparison

Results on the fixed test set:

| Model        |   Accuracy |  Precision |     Recall |         F1 |
| ------------ | ---------: | ---------: | ---------: | ---------: |
| Bag of Words |     90.28% |     89.90% |     90.84% |     90.37% |
| TF-IDF       |     89.59% |     88.56% |     91.02% |     89.77% |
| Doc2Vec      |     89.33% |     89.82% |     88.81% |     89.31% |
| Word2Vec     |     88.19% |     87.91% |     88.67% |     88.29% |
| FastText     |     88.19% |     87.97% |     88.59% |     88.28% |
| **BERT**     | **92.49%** | **92.46%** | **92.59%** | **92.52%** |

These results reflect this particular experimental setup and are not intended as universal comparisons between the techniques.

## BERT

Configuration:

* Maximum sequence length: `256`
* Epochs: `2`
* Learning rate: `2e-5`
* Batch size: `8`

Final test performance:

* Accuracy: **92.49%**
* Precision: **92.46%**
* Recall: **92.59%**
* F1: **92.52%**

The trained model is stored locally at:

```text
models/bert_imdb_80_20_split_final/
```

The model files are excluded from Git because of their size.

## BERT Error Analysis

Confusion matrix:

```text
[[4564, 376],
 [ 369, 4608]]
```

Review-length analysis showed:

* 43.30% of test reviews contained more than 256 tokens.
* 58.93% of model errors involved reviews longer than 256 tokens.
* Error rate for reviews >256 tokens: 10.22%.
* Error rate for reviews ≤256 tokens: 5.44%.

This shows an association between longer reviews and higher error rates in this experiment, but does not establish sequence truncation as the direct cause.

Other error patterns included:

* Mixed or conflicting sentiment
* Sarcasm and irony
* Nuanced reviews
* Misleading phrases without broader context
* Potentially ambiguous labels

## Project Structure

```text
sentiment-analysis-imdb/
├── config/
│   └── settings.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── bert_imdb_80_20_split_final/
├── notebooks/
│   ├── 01_eda_and_data_split.ipynb
│   ├── 02_bow_and_tfidf.ipynb
│   ├── 03_word2vec_fasttext_doc2vec.ipynb
│   ├── 04_bert.ipynb
│   └── 05_model_comparison.ipynb
├── results/
├── src/
│   ├── api.py
│   ├── batch_inference.py
│   ├── inference.py
│   └── logging_config.py
├── tests/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## API

### Health Check

```http
GET /health
```

```json
{
  "status": "healthy"
}
```

### Single Prediction

```http
POST /predict
```

Request:

```json
{
  "text": "This movie was absolutely fantastic!"
}
```

Response:

```json
{
  "sentiment": "Positive",
  "confidence": 0.9994
}
```

### CSV Batch Prediction

```http
POST /predict/csv
```

Upload a CSV containing a `review` column.

The API returns the original reviews with:

* `predicted_sentiment`
* `confidence`

## Running Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd sentiment-analysis-imdb
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the trained model

Place the model at:

```text
models/bert_imdb_80_20_split_final/
```

### 5. Start the API

```bash
python -m uvicorn src.api:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Running with Docker

Build the image:

```bash
docker build -t imdb-sentiment-api .
```

Run the container from PowerShell:

```powershell
docker run --rm -p 8000:8000 -v "${PWD}\models:/app/models" imdb-sentiment-api
```

The model directory is mounted into the container rather than copied into the Docker image.

## Testing

Run the complete test suite:

```bash
python -m pytest
```

Current result:

```text
21 passed, 1 warning
```

Tests cover:

* API endpoints
* Request validation
* Error handling
* CSV upload and validation
* Batch prediction
* Inference logic

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* FastAPI
* Pydantic
* Pandas
* NumPy
* Pytest
* Docker
* Jupyter Notebook

## Future Improvements

* GitHub Actions CI
* Cloud deployment
* Model hosting and automated model download
* API authentication
* Performance monitoring
* Further error analysis
* Experiments with longer sequence lengths
