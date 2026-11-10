"""
SMS Spam Detector - Naive Bayes classifier built from scratch.
Uses ONLY the Python standard library (no pip install needed).

Usage:
    python spam_detector.py                  # train, evaluate, run demo
    python spam_detector.py "your message"   # classify one message
    python spam_detector.py --interactive    # type messages one by one
"""
import csv
import math
import os
import random
import re
import sys
from collections import Counter

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "messages.csv")


def tokenize(text):
    """Lowercase the text and split it into words."""
    return re.findall(r"[a-z0-9']+", text.lower())


def load_data(path=DATA_PATH):
    with open(path, newline="", encoding="utf-8") as f:
        return [(row["label"], row["message"]) for row in csv.DictReader(f)]


def train_test_split(data, test_ratio=0.25, seed=42):
    data = data[:]
    random.Random(seed).shuffle(data)
    cut = int(len(data) * (1 - test_ratio))
    return data[:cut], data[cut:]


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing
        self.word_counts = {"spam": Counter(), "ham": Counter()}
        self.class_counts = Counter()
        self.vocab = set()

    def fit(self, data):
        for label, text in data:
            self.class_counts[label] += 1
            for w in tokenize(text):
                self.word_counts[label][w] += 1
                self.vocab.add(w)
        return self

    def _log_score(self, label, words):
        total_docs = sum(self.class_counts.values())
        score = math.log(self.class_counts[label] / total_docs)
        total_words = sum(self.word_counts[label].values())
        denom = total_words + self.alpha * len(self.vocab)
        for w in words:
            score += math.log((self.word_counts[label][w] + self.alpha) / denom)
        return score

    def predict(self, text):
        """Return (label, spam_probability)."""
        words = tokenize(text)
        s = self._log_score("spam", words)
        h = self._log_score("ham", words)
        m = max(s, h)
        p_spam = math.exp(s - m) / (math.exp(s - m) + math.exp(h - m))
        return ("spam" if p_spam >= 0.5 else "ham"), p_spam

    def top_spam_words(self, n=8):
        ranked = sorted(
            self.vocab,
            key=lambda w: (self.word_counts["spam"][w] + 1) / (self.word_counts["ham"][w] + 1),
            reverse=True,
        )
        return ranked[:n]


def evaluate(model, test):
    tp = fp = tn = fn = 0
    for label, text in test:
        pred, _ = model.predict(text)
        if label == "spam" and pred == "spam": tp += 1
        elif label == "ham" and pred == "spam": fp += 1
        elif label == "ham" and pred == "ham": tn += 1
        else: fn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total}


def build_model():
    data = load_data()
    train, test = train_test_split(data)
    model = NaiveBayes().fit(train)
    return model, train, test


def show(model, text):
    label, p = model.predict(text)
    icon = "🚨 SPAM" if label == "spam" else "✅ NOT SPAM (ham)"
    print(f"  {icon}  (spam probability: {p:.1%})\n  Message: {text}")


def main():
    model, train, test = build_model()
    args = sys.argv[1:]

    if args and args[0] == "--interactive":
        print("Type a message and press Enter (empty line to quit).")
        while True:
            text = input("> ").strip()
            if not text:
                break
            show(model, text)
        return

    if args:
        show(model, " ".join(args))
        return

    print("=" * 60)
    print(" SMS SPAM DETECTOR  (Naive Bayes from scratch)")
    print("=" * 60)
    print(f"Training messages : {len(train)}")
    print(f"Testing messages  : {len(test)}")
    print(f"Vocabulary size   : {len(model.vocab)} words\n")

    m = evaluate(model, test)
    print("--- Evaluation on unseen test data ---")
    print(f"Accuracy  : {m['accuracy']:.1%}")
    print(f"Precision : {m['precision']:.1%}")
    print(f"Recall    : {m['recall']:.1%}")
    print(f"F1 score  : {m['f1']:.1%}\n")
    print("Confusion matrix:")
    print(f"                 Pred SPAM   Pred HAM")
    print(f"  Actual SPAM    {m['tp']:>8}   {m['fn']:>8}")
    print(f"  Actual HAM     {m['fp']:>8}   {m['tn']:>8}\n")
    print("Top spam-indicating words:", ", ".join(model.top_spam_words()), "\n")

    print("--- Demo predictions ---")
    for msg in [
        "Congratulations! You won a free prize. Click now to claim",
        "Hey, are you coming to college tomorrow?",
        "Urgent: your account is blocked. Verify immediately",
        "Please send me the project notes",
    ]:
        show(model, msg)
        print()


if __name__ == "__main__":
    main()
