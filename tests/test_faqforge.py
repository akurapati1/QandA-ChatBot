import tempfile
import unittest
from pathlib import Path

from faqforge.bot import FAQBot, load, save
from faqforge.generate import _question_from_heading, generate, generate_folder

DOCS = Path(__file__).parents[1] / "samples" / "docs"
PAIRS = generate_folder(DOCS)


class Generation(unittest.TestCase):
    def test_questions_from_sample_docs(self):
        self.assertEqual([p.question for p in PAIRS], [
            "What is the refund policy?", "How do I change my plan?", "What is two-factor authentication?", "How does data export work?",
            "What are the supported browsers?", "How do I reset my password?", "What are API keys?"])

    def test_heading_rules(self):
        for heading, body, want in [("Setting up SSO", "", "How do I set up SSO?"), ("Why is my export empty", "", "Why is my export empty?"),
                                    ("Pricing", "Plans start at $10.", "What is the pricing?"), ("Managing your team", "", "How do I manage my team?"),
                                    ("Already a question?", "", "Already a question?")]:
            self.assertEqual(_question_from_heading(heading, body), want)

    def test_numbered_steps_keep_line_breaks_and_dedupe(self):
        pairs = generate("## Changing your plan\n1. Open Settings.\n2. Click Change.\n\n## Changing your plan\nduplicate", "x")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].answer, "1. Open Settings.\n2. Click Change.")

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            save(PAIRS, Path(d) / "faq.json")
            self.assertEqual(load(Path(d) / "faq.json"), PAIRS)


class Bot(unittest.TestCase):
    bot = FAQBot(PAIRS)

    def test_paraphrases_hit_the_right_answer(self):
        for q, want in [("how can I get my money back", "refund"), ("enable 2FA", "Two-factor"), ("which browsers work", "Chrome"),
                        ("I forgot my password", "Forgot password"), ("what is an API key", "credentials")]:
            r = self.bot.ask(q)
            self.assertEqual(r.kind, "answer", q)
            self.assertIn(want, r.text, q)

    def test_confidence_gates(self):
        self.assertEqual(self.bot.ask("do you have a mobile app").kind, "fallback")
        self.assertEqual(self.bot.ask("").kind, "fallback")
        r = self.bot.ask("change my subscription")
        self.assertEqual(r.kind, "suggest")
        self.assertEqual(r.suggestions[0].question, "How do I change my plan?")

    def test_feedback_shifts_ranking_and_persists(self):
        with tempfile.TemporaryDirectory() as d:
            bot = FAQBot(PAIRS, log_dir=d)
            before = bot.ask("export my data").confidence
            for _ in range(3):
                bot.feedback(4, True)
            self.assertGreater(bot.ask("export my data").confidence, before)
            self.assertEqual(FAQBot(PAIRS, log_dir=d).votes, {"4": 3})   # reloaded from disk

    def test_gap_report_counts_repeated_unanswered_questions(self):
        with tempfile.TemporaryDirectory() as d:
            bot = FAQBot(PAIRS, log_dir=d)
            for q in ["Do you have a mobile app?", "do you have a mobile app", "Is there dark mode?"]:
                bot.ask(q)
            bot.ask("how can I get my money back")   # answered: must not be logged
            self.assertEqual(bot.gaps(), [("do you have a mobile app", 2), ("is there dark mode", 1)])


if __name__ == "__main__":
    unittest.main()
