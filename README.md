# FAQForge

Support teams answer the same questions every day, and help-center articles are rarely written as Q&A. **FAQForge turns your documentation into an FAQ,
serves it with a chatbot that knows when it does not know, and tells you which questions your docs fail to answer.**

```text
$ python -m faqforge build samples/docs -o faq.json
 1. What is the refund policy?
 2. How do I change my plan?
 3. What is two-factor authentication?
 4. How does data export work?
 5. What are the supported browsers?
 6. How do I reset my password?
 7. What are API keys?
-> 7 Q&A pairs written to samples/faq.json

$ python -m faqforge ask samples/faq.json "how can I get my money back"
[answer 0.48] You can request a refund within 30 days of purchase. Refunds are returned to the original payment method w
$ python -m faqforge ask samples/faq.json "enable 2FA"
[answer 0.5] Two-factor authentication is an extra security step that asks for a code from your phone when you sign in. 
$ python -m faqforge ask samples/faq.json "do you have a mobile app"
[fallback 0.0] I do not have an answer for that yet. I have noted it for the team - would you like to talk to a person?
$ python -m faqforge ask samples/faq.json "change my subscription"
[suggest 0.44] I am not sure I understood. Did you mean one of these?
  - How do I change my plan?
```

## How it works
1. **Generate** - from Markdown/text: headings become questions (`Changing your plan` -> *How do I change my plan?*, `Supported browsers` -> *What are the supported browsers?*),
   numbered procedures keep their steps, definitions ("API keys are credentials...") become *What are API keys?*, existing questions are kept, duplicates removed.
2. **Match** - TF-IDF cosine over question (weighted 3x) + answer, light synonym map (`money back` ~ refund, `2FA` ~ two-factor) and plural stemming.
3. **Gate by confidence**
   | Confidence | Behaviour |
   | --- | --- |
   | >= 0.45 | answer |
   | 0.20 - 0.45 | "Did you mean...?" with the closest questions |
   | < 0.20 | polite fallback / hand-off and the question is logged |
4. **Learn** - `bot.feedback(id, helpful)` nudges ranking (bounded +-0.1, persisted to `feedback.json`); `python -m faqforge gaps faq.json` lists the **most-asked unanswered
   questions** - a to-do list for your docs.

## Run
```bash
python -m faqforge chat faq.json          # interactive; writes unanswered.jsonl / feedback.json to --logs
python -m faqforge gaps faq.json
python -m unittest discover -s tests -t . -v      # 8 tests, no dependencies
```
Deterministic and offline by design: answers are always your own text, never generated, so the bot cannot hallucinate policy. The original LangChain/Streamlit prototype is kept in [`legacy/`](legacy/).
