# 📩 SMS Spam Detector (Naive Bayes from Scratch)

A machine learning project that classifies messages as **spam** or **ham (not spam)**.
The Multinomial Naive Bayes algorithm is implemented **from scratch in pure Python** — no external libraries needed.

## Features
- Text tokenization and vocabulary building
- Naive Bayes with Laplace smoothing (log probabilities)
- Train/test split and evaluation: accuracy, precision, recall, F1, confusion matrix
- Command-line prediction and interactive mode
- Unit tests

## Requirements
Python 3.8+ only. **No pip install required.**

## How to run
```bash
python spam_detector.py                         # train + evaluate + demo
python spam_detector.py "You won a free prize, click now"   # classify one message
python spam_detector.py --interactive           # type your own messages
python -m unittest -v                           # run tests
```

## Project structure
```
spam-detector/
├── spam_detector.py        # model + CLI
├── test_spam_detector.py   # unit tests
├── data/messages.csv       # labelled dataset (add more rows to improve accuracy)
├── requirements.txt
└── README.md
```

## How it works
For a message, the model compares `P(spam) * Π P(word | spam)` against `P(ham) * Π P(word | ham)` and picks the higher one.

## Ideas to extend
- Replace the dataset with the full UCI SMS Spam Collection
- Add a Flask/Streamlit web UI
- Compare with scikit-learn's `MultinomialNB`
