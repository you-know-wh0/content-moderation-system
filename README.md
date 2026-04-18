# Content Moderation System

A text moderation API that checks whether a message should be approved, rejected, or sent for human review.

It is built with `FastAPI`, `DistilBERT`, and `ONNX Runtime`. The goal is to combine simple rules with machine learning so moderation is both fast and practical.

## What This Project Does

The API reads a piece of text and returns a moderation decision.

Possible results:

- `APPROVED` if the text looks safe
- `REJECTED` if the text looks harmful
- `HUMAN_REVIEW` if the system is not fully confident

This makes the project useful as a simple backend service for platforms that need to filter user-generated content such as comments, chats, or reviews.

## How It Works

The moderation flow has three steps:

1. `Blacklist check`
   The system first checks for clearly harmful words or phrases. If it finds one, the text is rejected immediately.

2. `Whitelist check`
   The system then checks for trusted safe phrases. If a safe phrase matches and no blacklist rule matched earlier, the text is approved.

3. `AI model check`
   If the text is not handled by the rule-based checks, the API sends it to an ONNX version of a DistilBERT model. The model predicts how toxic the text is.

If the model is unsure, the text is sent to `HUMAN_REVIEW` instead of forcing an unsafe decision.

## Main Features

- `POST /moderate` endpoint for moderation
- Rule-based filtering with blacklist and whitelist logic
- ONNX-based DistilBERT inference for text classification
- Human review queue for uncertain cases
- SQLite storage for moderation history
- Analytics endpoint for simple moderation stats

## Project Structure

```text
content-moderation-system/
├── app/
│   ├── main.py
│   ├── review_store.py
│   ├── toxic_model.onnx
│   └── train.csv
├── scripts/
│   ├── benchmark.py
│   └── evaluate.py
├── tests/
│   └── test_api.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## API Endpoints

### `POST /moderate`

This is the main endpoint.

Request:

```json
{
  "text": "I love this project"
}
```

Example response:

```json
{
  "text": "I love this project",
  "status": "APPROVED",
  "decision_source": "whitelist",
  "toxicity_score": 0.0,
  "ai_confidence": 1.0,
  "requires_human_review": false,
  "matched_rule": "love this project",
  "label_scores": {
    "non_toxic": 1.0,
    "toxic": 0.0
  },
  "latency_ms": 0.214
}
```

### Other useful endpoints

- `GET /health` checks if the service is running
- `GET /analytics` shows basic moderation statistics
- `GET /reviews/pending` shows items waiting for human review
- `POST /reviews/{review_id}` lets you resolve a review item
- `GET /docs` opens the interactive FastAPI Swagger docs

## How To Run The Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

If you also want the local testing tools:

```bash
pip install -r requirements-dev.txt
```

### 2. Start the API

```bash
uvicorn app.main:app --reload
```

The app will usually start at:

```text
http://127.0.0.1:8000
```

### 3. Test the API

You can use `curl`:

```bash
curl -X POST http://127.0.0.1:8000/moderate \
  -H "Content-Type: application/json" \
  -d '{"text":"I love this project"}'
```

Or open:

```text
http://127.0.0.1:8000/docs
```

and test everything from the browser.

## Running Tests

```bash
python3 -m unittest discover -s tests
```

## Benchmark And Evaluation

To measure latency:

```bash
python3 scripts/benchmark.py --iterations 25
```

To evaluate the model on sample data:

```bash
python3 scripts/evaluate.py --limit 200
```

These scripts help you generate numbers like latency, accuracy, precision, recall, and F1 score.

## Configuration

You can change the project behavior with environment variables:

- `MODEL_PATH`
- `MODERATION_DB_PATH`
- `TOKENIZER_PATH`
- `TOKENIZER_NAME`
- `TOKENIZER_LOCAL_ONLY`
- `LOWER_REVIEW_THRESHOLD`
- `UPPER_REVIEW_THRESHOLD`
- `ORT_PROVIDERS`
- `BANNED_PHRASES`
- `WHITELIST_PHRASES`
- `LOG_LEVEL`
- `MODERATION_API_KEY` (optional)

If `MODERATION_API_KEY` is set, these endpoints require an API key:

- `POST /moderate`
- `GET /predict`
- `GET /analytics`
- `GET /reviews/pending`
- `POST /reviews/{review_id}`

Pass the key with either `X-API-Key` or `Authorization: Bearer <key>`.



If anyone has any queries about this project, feel free to reach out to me! 
