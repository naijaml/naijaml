# Tokenizer Benchmark: The Cost of AI in Nigerian Languages

How much more do Nigerian developers pay to use AI APIs — simply because of the language they build in?

We benchmarked 7 major tokenizers across 5 languages to find out.

## Key Findings

**Yoruba costs up to 3.5x more to process than English.** Same meaning, same API, different price.

| Language | GPT-4o Cost (₦/1M tokens) | Multiplier vs English |
|----------|---------------------------|----------------------|
| English  | ₦3,375                    | 1.00x                |
| Pidgin   | ₦3,613                    | 1.07x                |
| Hausa    | ₦4,892                    | 1.45x                |
| Igbo     | ₦5,201                    | 1.54x                |
| Yoruba   | ₦8,084                    | 2.40x                |

The English word "telephone" is 1 token. The Yoruba equivalent "ẹ̀rọ ìbánisọ̀rọ̀" is 17 tokens on Mistral's tokenizer. Same word. 17x the cost.

## Tokenizers Tested

- **GPT-4** (cl100k_base)
- **GPT-4o** (o200k_base)
- **Llama 3** (Meta)
- **Gemma 2** (Google)
- **Mistral v0.3**
- **BERT Multilingual**
- **XLM-RoBERTa**

## Languages

- English (baseline)
- Yoruba
- Hausa
- Igbo
- Nigerian Pidgin

## Full Matrix: Token Ratio vs English

|          | GPT-4 | GPT-4o | Llama 3 | Gemma 2 | Mistral | BERT-ML | XLM-R |
|----------|-------|--------|---------|---------|---------|---------|-------|
| Yoruba   | 3.4x  | 2.4x   | 3.1x    | 2.9x    | 3.5x    | 2.4x    | 2.8x  |
| Igbo     | 2.4x  | 1.5x   | 2.3x    | 2.3x    | 2.5x    | 2.2x    | 2.3x  |
| Hausa    | 1.8x  | 1.4x   | 1.8x    | 1.6x    | 1.9x    | 1.5x    | 1.3x  |
| Pidgin   | 1.1x  | 1.1x   | 1.1x    | 1.1x    | 1.1x    | 1.1x    | 1.0x  |

1.0x = same cost as English. Higher = more expensive.

## Worst Tokenized Words

| Word | Language | English | Tokens | Tokenizer | Ratio |
|------|----------|---------|--------|-----------|-------|
| ẹ̀rọ ìbánisọ̀rọ̀ | Yoruba | telephone | 17 | Mistral | 17x |
| pápá ọkọ̀ òfuurufú | Yoruba | airport | 15 | GPT-4 | 15x |
| ilé ẹ̀kọ́ gíga | Yoruba | university | 13 | Mistral | 13x |
| iná mọ̀nàmọ́ná | Yoruba | electricity | 11 | Mistral | 11x |
| mmịrị ọzụzọ | Igbo | rain | 10 | GPT-4 | 10x |

## Methodology

- 100 entries per language x 5 languages x 7 tokenizers = **3,500 data points**
- Categories: greetings, nouns, verbs, sentences, names, places, proverbs, fintech, culture
- All entries are matched by meaning across languages
- Cost calculated using GPT-4o pricing ($2.50/1M input tokens) at ₦1,350/$1

## Files

- `benchmark_words.json` — Input dataset: 100 entries with matched meanings across 5 languages
- `results.json` — Full results with all 3,500 data points and aggregate stats
- `tokenizer_benchmark.py` — Benchmark script (requires `tiktoken`, `transformers`, `numpy`)

## Run It Yourself

```bash
pip install tiktoken transformers sentencepiece protobuf numpy
python tokenizer_benchmark.py
```

**Note:** Gemma 2 and Llama 3 are gated models on HuggingFace. You'll need to accept their license agreements on HuggingFace and set your token:

```python
import os
os.environ["HF_TOKEN"] = "hf_..."
```

The script will skip any tokenizer it can't load and run the rest.

## Why This Matters

Two developers build the same AI chatbot. One in London, one in Lagos. Same API, same product, same users. Lagos pays 2.4x more — not because of pricing tiers, but because tokenizers weren't trained on Nigerian language data.

This benchmark is part of [NaijaML](https://github.com/naijaml/naijaml), an open-source Python toolkit building ML infrastructure for Nigerian languages.

```bash
pip install naijaml
```
