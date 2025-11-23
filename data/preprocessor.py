import json, re, logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("preprocessor")

DATA_DIR = Path("data")
NEWS_JSON = DATA_DIR / "news.json"
ENRICHED_JSON = DATA_DIR / "enriched_data.json"

sid = SentimentIntensityAnalyzer()

# ---------- TEXT CLEANING ----------
def clean_text(t: str) -> str:
    if not isinstance(t, str):
        return ""

    t = t.lower().strip()
    t = re.sub(r"http\S+", "", t)         # remove URLs
    t = re.sub(r"[$][A-Za-z]+", "", t)    # remove $TSLA tickers
    t = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", t)  # remove emojis/symbols
    t = re.sub(r"\s+", " ", t)
    return t.strip()

# ---------- WEIGHT FUNCTIONS ----------
def recency_weight(timestamp: str) -> float:
    """
    Recent news gets more weight.
    """
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", ""))
        hours_old = (datetime.utcnow() - dt).total_seconds() / 3600
        # 1.0 weight if < 4 hrs old, decays after
        return max(0.3, min(1.0, 1.0 - (hours_old / 48)))
    except:
        return 1.0

def relevance_weight(text: str, ticker: str) -> float:
    """
    Weight based on how relevant the article is to the ticker.
    """
    t = text.lower()
    score = 1.0
    score += 0.2 if ticker.lower() in t else 0
    score += 0.1 * len(re.findall(r"\b(ai|market|earnings|growth)\b", t))
    return min(score, 2.0)

def classify_sentiment(score: float) -> str:
    if score >= 0.2: return "bullish"
    if score <= -0.2: return "bearish"
    return "neutral"

# ---------- MAIN ----------
def main(ticker: str = "AAPL"):
    if not NEWS_JSON.exists():
        log.warning("news.json not found; skipping")
        return

    with NEWS_JSON.open() as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        log.warning("unexpected news.json shape")
        return

    df = pd.DataFrame(raw)

    # unify text (headline + summary + source)
    df["text_raw"] = (
        df["headline"].fillna("") + " " +
        df["summary"].fillna("") + " " +
        df["source"].fillna("")
    )

    # clean text
    df["text"] = df["text_raw"].map(clean_text)

    # vader sentiment
    df["sentiment"] = df["text"].map(lambda t: sid.polarity_scores(t)["compound"])

    # weights
    df["recency_weight"] = df["timestamp"].map(recency_weight) if "timestamp" in df else 1.0
    df["relevance_weight"] = df["text"].map(lambda t: relevance_weight(t, ticker))

    # final score
    df["final_score"] = (df["sentiment"] * df["recency_weight"] * df["relevance_weight"]).round(4)

    # tags
    df["label"] = df["final_score"].map(classify_sentiment)

    out = df.to_dict(orient="records")

    with ENRICHED_JSON.open("w") as f:
        json.dump(out, f, indent=2)

    log.info(f"Enriched {len(out)} rows → {ENRICHED_JSON}")

if __name__ == "__main__":
    main()