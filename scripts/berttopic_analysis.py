import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import PROCESSED_DIR
from nltk.corpus import stopwords
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MIN_TOPIC_SIZE = 15
NR_TOPICS = "auto"
RANDOM_STATE = 42

_STOPWORDS = stopwords.words("english")

CUSTOM_STOPWORDS = {
    "climate",
    "change",
    "global",
    "warming",
    "climate_change",
    "global_warming",
}


def load_processed() -> pd.DataFrame:
    path = PROCESSED_DIR / "data.csv"
    df = pd.read_csv(path)
    return df


def build_model() -> BERTopic:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_TOPIC_SIZE,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        ngram_range=(1, 2),
        stop_words=list(set(_STOPWORDS).union(CUSTOM_STOPWORDS)),
        min_df=5,
    )

    representation_model = MaximalMarginalRelevance(diversity=0.3)

    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,
        nr_topics=NR_TOPICS,
        calculate_probabilities=True,
        language="english",
        verbose=True,
    )

    return model


def evaluate(model: BERTopic, docs: list[str]) -> dict:
    topic_words = []
    for topic_id in model.get_topics():
        if topic_id == -1:
            continue
        words = [w for w, _ in model.get_topic(topic_id)]
        topic_words.append(words)

    if not topic_words:
        return {}

    tokenised = [doc.split() for doc in docs]
    dictionary = Dictionary(tokenised)

    cv_model = CoherenceModel(
        topics=topic_words,
        texts=tokenised,
        dictionary=dictionary,
        coherence="c_v",
    )
    cv = cv_model.get_coherence()

    all_words = [word for topic in topic_words for word in topic]
    diversity = len(set(all_words)) / len(all_words) if all_words else 0

    topic_info = model.get_topic_info()
    outlier_row = topic_info[topic_info["Topic"] == -1]
    total = topic_info["Count"].sum()
    outlier_pct = (
        (outlier_row["Count"].values[0] / total * 100) if len(outlier_row) else 0
    )

    print(f"Number of topics: {len(topic_words)}")
    print(f"Coherence: {cv:.4f}")
    print(f"Topic Diversity: {diversity:.4f}")
    print(f"Outlier tweets: {outlier_pct:.1f}%")

    return {"c_v": cv, "diversity": diversity, "outlier_pct": outlier_pct}


def print_topics(model: BERTopic):
    info = model.get_topic_info()
    print(info[info["Topic"] != -1][["Topic", "Count", "Name"]].to_string(index=False))

    for topic_id in model.get_topics():
        if topic_id == -1:
            continue
        words = [w for w, _ in model.get_topic(topic_id)]
        print(f"Topic {topic_id:>3}: {', '.join(words)}")


def reduce_outliers(
    model: BERTopic,
    docs: list[str],
    topics: list[int],
    probabilities,
    threshold: float = 0.1,
) -> list[int]:

    new_topics = model.reduce_outliers(
        docs,
        topics,
        probabilities=probabilities,
        strategy="probabilities",
        threshold=threshold,
    )
    original_outliers = topics.count(-1)
    new_outliers = new_topics.count(-1)
    print(f"\nOutlier reduction: {original_outliers} → {new_outliers} outliers")
    return new_topics


def sentiment_topic_cross(df: pd.DataFrame):
    label_map = {-1: "Anti", 0: "Neutral", 1: "Pro", 2: "News"}
    df["sentiment_label"] = df["sentiment"].map(label_map)

    cross = pd.crosstab(
        df["bertopic_topic"],
        df["sentiment_label"],
        normalize="index",
    ).round(3)

    cross = cross[cross.index != -1]

    print(cross.to_string())

    cross.plot(kind="bar", figsize=(14, 6), colormap="Set2")
    plt.title("Sentiment Distribution per BERTopic Topic")
    plt.xlabel("Topic")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "bertopic_sentiment_cross.png", dpi=150)
    plt.show()

    return cross


def save_visualisations(model: BERTopic, docs: list[str], topics: list[int]):
    fig1 = model.visualize_topics()
    fig1.write_html(str(PROCESSED_DIR / "bertopic_topics.html"))

    fig2 = model.visualize_barchart(top_n_topics=10)
    fig2.write_html(str(PROCESSED_DIR / "bertopic_barchart.html"))

    fig3 = model.visualize_heatmap()
    fig3.write_html(str(PROCESSED_DIR / "bertopic_heatmap.html"))


def main():
    df = load_processed()
    docs = df["tweet"].tolist()

    model = build_model()

    topics, probabilities = model.fit_transform(docs)
    df["bertopic_topic"] = topics

    print_topics(model)
    metrics = evaluate(model, docs)
    topics = reduce_outliers(model, docs, topics, probabilities, threshold=0.1)
    df["bertopic_topic"] = topics
    model.update_topics(docs, topics=topics)
    cross = sentiment_topic_cross(df)
    save_visualisations(model, docs, topics)
    df[["tweet", "sentiment", "bertopic_topic"]].to_csv(
        PROCESSED_DIR / "bertopic_results.csv", index=False
    )
    model.save(
        str(PROCESSED_DIR / "bertopic_model"),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=EMBEDDING_MODEL,
    )


if __name__ == "__main__":
    main()
