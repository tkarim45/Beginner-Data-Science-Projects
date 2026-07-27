# Chatbot (Rule-Based)

A beginner-level **NLP-fundamentals** project: a retrieval-based intent chatbot. It matches a user
message to the closest known intent with TF-IDF cosine similarity and replies with that intent's
response, with a confidence threshold so it falls back gracefully on out-of-scope messages.

## Problem Statement

Build a customer-service chatbot without any neural network: given a user message, identify its
*intent* (greeting, hours, payments, returns, account, …) and respond appropriately, or fall back
to a safe "I didn't understand" when confidence is low.

*This is an NLP-fundamentals project and does not use the classification template, it has its own
`RuleBasedChatbot` class and a build/evaluate notebook flow.*

## Dataset

- **Source**: a hand-crafted `intents.json` (in `data/`), **12 intents, 77 example patterns**,
  each intent with several example phrasings and canned responses.

> Dataset note: the checklist's Cornell Movie-Dialog corpus is built for open-ended dialogue, not
> intent classification; a compact intents file is the standard, transparent basis for a *rule-based*
> chatbot and keeps the project self-contained.

| File | Contents |
|---|---|
| `data/intents.json` | `tag`, `patterns[]`, `responses[]` per intent |

## Project Structure

```
Chatbot Rule Based/
├── 01_eda.ipynb              # Intents, patterns per intent, examples
├── 02_chatbot.ipynb          # Build bot, demo conversation, fallback, matching
├── 03_evaluation.ipynb       # Held-out intent accuracy + error analysis
├── utils.py                  # load_intents, RuleBasedChatbot, evaluate
├── requirements.txt
├── README.md
└── data/intents.json
```

Run notebooks in order: `01` → `02` → `03`.

## How It Works

TF-IDF (unigrams + bigrams) vectorizes all example patterns. For a user message, cosine similarity
finds the nearest pattern; its intent's response is returned. Below a similarity threshold (0.2) the
bot returns a fallback instead of guessing.

## Results

All figures produced by executing the notebooks, not assumed.

- **Held-out intent accuracy: 0.458** (30% of patterns held out, 24 test phrasings).
- On **seen / closely-matching phrasings** the bot is reliable, the demo correctly handles
  greetings, hours, payments, and password resets with high similarity (0.6 to 1.0).

## Key Findings

- **The bot is reliable on familiar phrasings but brittle on novel ones**, with only ~6 patterns per
  intent, a paraphrase sharing few words with any stored pattern gets mis-routed (e.g. "return my
  order" matching *shipping*), which is why held-out accuracy is ~0.46.
- **The confidence threshold is essential**, it turns low-similarity guesses into a safe fallback,
  preventing confidently-wrong answers to gibberish or out-of-scope queries.
- **Rule/retrieval chatbots are fast, transparent, and controllable**, you can read exactly why an
  intent was chosen (notebook 02 shows the top matching patterns), but they don't generalise to
  unseen vocabulary.
- **Takeaway**: scaling this approach means more patterns per intent or learned intent embeddings, 
  the natural step up to modern NLU.

## Tech Stack

- pandas, numpy, matplotlib
- scikit-learn

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook 01_eda.ipynb
```
