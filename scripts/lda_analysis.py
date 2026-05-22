import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from config import PROCESSED_DIR
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel, Phrases
from gensim.models.phrases import Phraser
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models

_STOPWORDS = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

NUM_TOPICS = 10
PASSES = 10
RANDOM_STATE = 42
MIN_TOKEN_LEN = 3
MIN_TOKENS = 3
TOPIC_RANGE = range(2, 21)


def load_ds() -> pd.DataFrame:
    path = PROCESSED_DIR / "data.csv"
    df = pd.read_csv(path)
    return df


def tokenise(text: str) -> list[str]:
    tokens = text.split()
    tokens = [
        t
        for t in tokens
        if t not in _STOPWORDS and len(t) >= MIN_TOKEN_LEN and t.isalpha()
    ]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def build_bigrams(tokenised: list[list[str]]) -> list[list[str]]:
    bigram = Phrases(
        tokenised,
        min_count=5,
        threshold=10,
    )

    bigram_mod = Phraser(bigram)
    tokenised = [bigram_mod[doc] for doc in tokenised]
    return tokenised


def build_corpus(df: pd.DataFrame):
    dictionary = corpora.Dictionary(df["tokens"])
    dictionary.filter_extremes(no_below=5, no_above=0.4, keep_n=50_000)
    corpus = [dictionary.doc2bow(tokens) for tokens in df["tokens"]]

    df = df.copy()
    df["bow"] = corpus
    df = df[df["bow"].apply(len) > 0].copy()

    corpus = df["bow"].tolist()
    return dictionary, list(corpus), df


def find_optimal_topics(
    corpus,
    dictionary,
    tokenised: list[list[str]],
    topic_range=TOPIC_RANGE,
) -> int:
    scores = []

    for k in topic_range:
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=k,
            passes=PASSES,
            alpha="auto",
            eta="auto",
            random_state=RANDOM_STATE,
        )
        cm = CoherenceModel(
            model=model,
            texts=tokenised,
            dictionary=dictionary,
            coherence="c_v",
        )
        score = cm.get_coherence()
        scores.append((k, score))
        print(f"K={k:>3} - Coherence: {score:.4f}")

    ks, cvs = zip(*scores)
    plt.figure(figsize=(8, 4))
    plt.plot(ks, cvs, marker="o")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("LDA Topic Coherence")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PROCESSED_DIR / "lda_coherence.png", dpi=150)
    plt.show()

    best_k = max(scores, key=lambda x: x[1])[0]
    print(f"Best K: {best_k}")
    return best_k


def train_lda(corpus, dictionary, num_topics: int) -> LdaModel:
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=PASSES,
        alpha="auto",
        eta="auto",
        chunksize=2000,
        random_state=RANDOM_STATE,
    )
    return model


def evaluate(model: LdaModel, corpus, dictionary, tokenised):
    cv = CoherenceModel(
        model=model, texts=tokenised, dictionary=dictionary, coherence="c_v"
    ).get_coherence()

    npmi = CoherenceModel(
        model=model, texts=tokenised, dictionary=dictionary, coherence="c_npmi"
    ).get_coherence()

    perplexity = model.log_perplexity(corpus)

    print(f"Coherence: {cv:.4f}")
    print(f"Coherence(NPMI): {npmi:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    return {"c_v": cv, "npmi": npmi, "perplexity": perplexity}


def print_topics(model: LdaModel, num_words: int = 10):
    for idx, topic in model.print_topics(
        num_topics=model.num_topics, num_words=num_words
    ):
        print(f"Topic {idx:>2}: {topic}\n")


def assign_topics(model: LdaModel, corpus) -> list[int]:
    topics = []
    for bow in corpus:
        dist = model.get_document_topics(bow)
        dominant = max(dist, key=lambda x: x[1])[0] if dist else -1
        topics.append(dominant)
    return topics


def visualise(model: LdaModel, corpus, dictionary):
    vis = pyLDAvis.gensim_models.prepare(model, corpus, dictionary)
    out = PROCESSED_DIR / "lda_vis.html"
    pyLDAvis.save_html(vis, str(out))


def main():
    df = load_ds()

    df["tokens"] = df["tweet"].apply(tokenise)
    tokenised = df["tokens"].tolist()
    tokenised = build_bigrams(tokenised)
    df["tokens"] = tokenised
    dictionary, corpus, df = build_corpus(df)
    tokenised = df["tokens"].tolist()
    best_k = find_optimal_topics(
        corpus, dictionary, tokenised, topic_range=range(5, 25, 5)
    )

    model = train_lda(corpus, dictionary, num_topics=best_k)
    metrics = evaluate(model, corpus, dictionary, tokenised)
    print_topics(model)

    df["lda_topic"] = assign_topics(model, corpus)
    df[["tweet", "sentiment", "lda_topic"]].to_csv(
        PROCESSED_DIR / "lda_results.csv", index=False
    )
    model.save(str(PROCESSED_DIR / "lda_model"))
    print("\nModel and results saved.")
    visualise(model, corpus, dictionary)


if __name__ == "__main__":
    main()
