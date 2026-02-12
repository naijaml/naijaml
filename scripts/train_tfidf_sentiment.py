#!/usr/bin/env python3
"""Train TF-IDF + Logistic Regression sentiment classifier on NaijaSenti.

Outputs a lightweight model (~2-5MB) that can be used without sklearn at runtime.

Usage:
    uv run python scripts/train_tfidf_sentiment.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    print("=" * 60)
    print("TF-IDF + Logistic Regression Sentiment Classifier")
    print("=" * 60)

    # Import sklearn (only needed for training)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import classification_report, accuracy_score
        import numpy as np
    except ImportError:
        print("Training requires scikit-learn:")
        print("  pip install scikit-learn")
        sys.exit(1)

    from naijaml.data import load_dataset

    # 1. Load data
    print("\n[1/5] Loading NaijaSenti dataset...")

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for lang in ["yor", "hau", "ibo", "pcm"]:
        train_data = load_dataset("naijasenti", lang=lang, split="train")
        test_data = load_dataset("naijasenti", lang=lang, split="test")

        for item in train_data:
            train_texts.append(item["text"])
            train_labels.append(item["label"])

        for item in test_data:
            test_texts.append(item["text"])
            test_labels.append(item["label"])

        print(f"  {lang}: {len(train_data)} train, {len(test_data)} test")

    print(f"\nTotal: {len(train_texts)} train, {len(test_texts)} test")
    print(f"Labels: {Counter(train_labels)}")

    # 2. Build TF-IDF features
    print("\n[2/5] Building TF-IDF features...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),      # Unigrams and bigrams
        max_features=50000,      # Limit vocabulary size
        min_df=3,                # Ignore rare terms
        max_df=0.95,             # Ignore very common terms
        sublinear_tf=True,       # Use log(tf) - helps with long documents
        strip_accents=None,      # Keep diacritics!
        lowercase=True,
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  Train matrix: {X_train.shape}")

    # 3. Train Logistic Regression
    print("\n[3/5] Training Logistic Regression...")

    # Map labels to integers
    label2id = {"positive": 0, "neutral": 1, "negative": 2}
    id2label = {v: k for k, v in label2id.items()}

    y_train = np.array([label2id[l] for l in train_labels])
    y_test = np.array([label2id[l] for l in test_labels])

    model = LogisticRegression(
        C=1.0,                   # Regularization strength
        max_iter=1000,
        solver="lbfgs",
        verbose=1,
    )

    model.fit(X_train, y_train)

    # 4. Evaluate
    print("\n[4/5] Evaluating...")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {accuracy:.1%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["positive", "neutral", "negative"]))

    # Test examples
    print("\nTest Examples:")
    examples = [
        "This film too sweet!",
        "I no like am at all",
        "E dey okay sha",
        "Na rubbish be this",
        "Mo nifẹ rẹ gan",
        "Wannan fim din yana da kyau",
    ]

    for text in examples:
        vec = vectorizer.transform([text])
        probs = model.predict_proba(vec)[0]
        pred_id = np.argmax(probs)
        print(f"  \"{text}\"")
        print(f"    → {id2label[pred_id]} ({probs[pred_id]:.0%})")

    # 5. Export model
    print("\n[5/5] Exporting model...")

    output_path = Path(__file__).parent.parent / "src" / "naijaml" / "nlp" / "sentiment_model.json"

    # Export vocabulary (term -> index) - convert numpy int64 to int
    vocab = {k: int(v) for k, v in vectorizer.vocabulary_.items()}

    # Export IDF weights
    idf = vectorizer.idf_.tolist()

    # Export logistic regression coefficients
    # Shape: (n_classes, n_features)
    coef = model.coef_.tolist()
    intercept = model.intercept_.tolist()

    model_data = {
        "vocab": vocab,
        "idf": idf,
        "coef": coef,
        "intercept": intercept,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "accuracy": round(accuracy, 4),
        "ngram_range": [1, 2],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nModel saved to: {output_path}")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Vocabulary: {len(vocab)} terms")

    print("\n" + "=" * 60)
    print(f"Done! Accuracy: {accuracy:.1%}, Size: {size_mb:.1f}MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
