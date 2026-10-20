"""
model_training.py
Train an article-title engagement classifier (High vs. Low) and save artifacts.

Usage:
    python model_training.py                      # uses synthetic data
    python model_training.py --csv my_data.csv    # uses your own data

CSV format: a `title` column plus EITHER a binary `label` column (1 = high)
OR an `engagement` column (e.g. likes + retweets), which is split at the median.
"""
import argparse
import json
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import LinearSVC

SEED = 42


# ----------------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------------
def generate_synthetic_dataset(n: int = 6000, seed: int = SEED) -> pd.DataFrame:
    """Simulate article titles with a noisy engagement score.

    NOTE: the label is driven by hand-written heuristics (listicles, power words,
    trending topics, ...) plus noise, so metrics on this data show the pipeline
    works but do NOT predict real-world performance. Use --csv for real data.
    """
    rng = random.Random(seed)
    topics = ["Python", "JavaScript", "React Hooks", "TypeScript", "Rust", "Docker",
              "Kubernetes", "Machine Learning", "ChatGPT", "SQL", "Git", "CSS",
              "Next.js", "Data Science", "Linux", "AWS", "Cybersecurity", "Go"]
    hot = {"ChatGPT": 1.0, "React Hooks": 0.7, "Rust": 0.6, "Machine Learning": 0.6,
           "Python": 0.5, "TypeScript": 0.4, "Next.js": 0.4}

    hi_templates = [
        "{n} {t} Tips Every Developer Should Know",
        "{n} {t} Mistakes You're Probably Making",
        "How I Learned {t} in {n} Days",
        "Stop Doing This in {t}: {n} Better Ways",
        "{n} {t} Tricks That Will Blow Your Mind",
        "The Ultimate {t} Cheat Sheet",
        "Why Every Developer Needs to Learn {t} Today",
        "{t}: The Secret Nobody Tells You",
        "I Built a {t} App in {n} Hours (Here's What Happened)",
        "{n} Hidden {t} Features You'll Love",
    ]
    lo_templates = [
        "A Technical Overview of {t}",
        "Notes on {t} Implementation Details",
        "Release Notes for {t} Version {n}.{n}",
        "Proceedings of the {t} Workshop",
        "An Analysis of {t} Performance Characteristics",
        "Documentation Update: {t} Configuration",
        "Meeting Minutes: {t} Working Group",
        "Revisiting {t} Architecture Decisions",
        "On the Semantics of {t}",
        "Quarterly {t} Maintenance Report",
    ]
    power = ["Best", "Easy", "Simple", "Powerful", "Fast", "Free", "Amazing"]

    rows = []
    for _ in range(n):
        t = rng.choice(topics)
        num = rng.choice([3, 4, 5, 5, 7, 7, 8, 9, 10, 10, 12, 15, 20])
        high_style = rng.random() < 0.5
        tpl = rng.choice(hi_templates if high_style else lo_templates)
        title = tpl.format(t=t, n=num)
        if rng.random() < 0.2:
            title = f"{rng.choice(power)} {title}"

        score = (1.2 if high_style else -1.2) + hot.get(t, 0.0)
        score += 0.3 if any(p in title for p in power) else 0.0
        score += rng.gauss(0, 1.0)  # irreducible noise -> realistic, <100% accuracy
        rows.append((title, score))

    df = pd.DataFrame(rows, columns=["title", "score"])
    # Simulated engagement count (log-normal-ish), then median split
    df["engagement"] = np.exp(df["score"] + 4).round().astype(int)
    df["label"] = (df["engagement"] > df["engagement"].median()).astype(int)
    return df[["title", "engagement", "label"]]


def load_dataset(csv_path: str | None) -> pd.DataFrame:
    if not csv_path:
        return generate_synthetic_dataset()
    df = pd.read_csv(csv_path)
    if "title" not in df.columns:
        raise ValueError("CSV must contain a 'title' column.")
    if "label" not in df.columns:
        if "engagement" not in df.columns:
            raise ValueError("CSV needs a 'label' or 'engagement' column.")
        df["label"] = (df["engagement"] > df["engagement"].median()).astype(int)
    df = df.dropna(subset=["title", "label"]).drop_duplicates(subset="title")
    df["title"] = df["title"].astype(str).str.strip()
    return df



# 2. Training

def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),            # unigrams + bigrams ("react hooks")
        sublinear_tf=True,             # 1 + log(tf) damping
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",  # keep numbers / 1-char tokens ("10", "c")
        min_df=2,
        max_df=0.95,
    )


def main(csv_path: str | None) -> None:
    df = load_dataset(csv_path)
    print(f"Dataset: {len(df)} titles | high-engagement share: {df['label'].mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["title"], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED
    )

    # Fit vectorizer on training data only (no leakage)
    vectorizer = build_vectorizer()
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)
    print(f"TF-IDF features: {Xtr.shape[1]:,}")

    # Candidate 1: LinearSVC (tuned C), calibrated to give probabilities
    svc_search = GridSearchCV(
        LinearSVC(random_state=SEED), {"C": [0.1, 0.3, 1, 3]},
        scoring="f1", cv=5, n_jobs=-1,
    ).fit(Xtr, y_train)
    svc = CalibratedClassifierCV(LinearSVC(C=svc_search.best_params_["C"], random_state=SEED), cv=5)

    # Candidate 2: Random Forest
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED)

    candidates = {"LinearSVC (calibrated)": svc, "RandomForest": rf}
    cv_scores = {}
    for name, est in candidates.items():
        cv_scores[name] = cross_val_score(est, Xtr, y_train, cv=5, scoring="f1", n_jobs=-1).mean()
        print(f"  CV F1 | {name:<24} {cv_scores[name]:.4f}")

    best_name = max(cv_scores, key=cv_scores.get)
    model = candidates[best_name].fit(Xtr, y_train)
    print(f"\nSelected model: {best_name}")

    # Evaluate on held-out test set
    y_pred = model.predict(Xte)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}\n")
    print("Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Low", "High"], digits=3))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(cm, index=["Actual Low", "Actual High"],
                       columns=["Pred Low", "Pred High"]))

    # Save artifacts
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(model, "model.pkl")
    with open("metrics.json", "w") as f:
        json.dump({
            "model": best_name,
            "accuracy": round(acc, 4),
            "f1_high": round(f1_score(y_test, y_pred), 4),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "confusion_matrix": cm.tolist(),
            "synthetic_data": csv_path is None,
        }, f, indent=2)
    print("\nSaved: vectorizer.pkl, model.pkl, metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to your own dataset (optional)")
    main(parser.parse_args().csv)
