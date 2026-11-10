import unittest
from spam_detector import NaiveBayes, tokenize, build_model, evaluate


class TestSpamDetector(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(tokenize("Hello, WORLD!"), ["hello", "world"])

    def test_spam_message(self):
        model, _, _ = build_model()
        self.assertEqual(model.predict("Free prize! Click now to claim cash")[0], "spam")

    def test_ham_message(self):
        model, _, _ = build_model()
        self.assertEqual(model.predict("Are you coming to class tomorrow")[0], "ham")

    def test_accuracy_reasonable(self):
        model, _, test = build_model()
        self.assertGreaterEqual(evaluate(model, test)["accuracy"], 0.8)


if __name__ == "__main__":
    unittest.main()
