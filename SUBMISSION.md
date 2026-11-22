# Project 07 – GRU Sentiment Analysis API

**GitHub repository:** https://github.com/faridhasan2020/sentiment-api

**Model used:** GRU sentiment classifier trained on IMDB

**Implemented API endpoints:**
- `GET /` – API name and short description
- `GET /health` – server status
- `POST /predict` – accepts `{"text": "..."}` and returns `{"label": "POSITIVE" | "NEGATIVE", "probability": <float>}`
