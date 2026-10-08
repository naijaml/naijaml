<p align="center">
  <h1 align="center">NaijaML</h1>
  <p align="center"><strong>Sovereign ML infrastructure for Nigeria.</strong></p>
  <p align="center">Production-ready NLP tools for Yoruba, Hausa, Igbo, and Nigerian Pidgin.<br>Works on CPU. Works offline. No GPU required.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/naijaml/"><img alt="PyPI" src="https://img.shields.io/pypi/v/naijaml"></a>
  <a href="https://pypi.org/project/naijaml/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/naijaml"></a>
  <a href="https://github.com/naijaml/naijaml/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/naijaml/naijaml"></a>
  <a href="https://huggingface.co/naijaml"><img alt="HuggingFace" src="https://img.shields.io/badge/🤗-HuggingFace-yellow"></a>
</p>

---

Standard NLP tools don't work for Nigeria. Tokenizers strip Yoruba diacritics. NER models don't recognize Nigerian names or states. Sentiment tools think Pidgin is broken English. Preprocessing libraries flag "sha" and "sef" as misspellings.

NaijaML is an open-source Python library that fixes this — built for the real constraints of developing ML in Nigeria: limited compute, intermittent connectivity, expensive bandwidth, and 500+ languages that the global ML ecosystem ignores.

```bash
pip install naijaml
```

## Quick Start

### Yoruba Diacritizer

```python
from naijaml.nlp import diacritize_yoruba, diacritize_yoruba_dot_below

diacritize_yoruba_dot_below("Ojo lo si oja")
# → 'Ọjọ lo si ọja'  (dot-below only, no tones)

diacritize_yoruba("Ojo lo si oja lana")
# → 'Ọjọ́ ló sí ọjà lànà'  (full tonal restoration)

# Dot-below: 93.3% word accuracy | Full tonal: 80.3% word accuracy (MENYO-20k test set)
# Both use a 12.6MB model, auto-downloaded on first use.
# Offline on first use, dot-below falls back to a 6.4MB bundled model (85.7%).
```

### Igbo Diacritizer

```python
from naijaml.nlp import diacritize_igbo

diacritize_igbo("Kedu ka i mere")
# → 'Kedụ ka ị mere'

# 92.3% word accuracy (MasakhaNER 2.0 Igbo test) | 4.9MB model | CPU only
```

### Language Detection

```python
from naijaml.nlp import detect_language

detect_language("Bawo ni, se daadaa ni?")   # → 'yor'
detect_language("Ina kwana?")                # → 'hau'
detect_language("Kedu ka ị mere?")           # → 'ibo'
detect_language("How far, wetin dey happen?") # → 'pcm'

# 5 languages: Yoruba, Hausa, Igbo, Pidgin, English | 96.4% accuracy on NaijaSenti test tweets
```

### Sentiment Analysis

```python
from naijaml.nlp import analyze_sentiment

analyze_sentiment("This film too sweet!")
# → {'label': 'positive', 'confidence': 0.64, ...}

analyze_sentiment("I no like am at all")
# → {'label': 'negative', 'confidence': 0.54, ...}

analyze_sentiment("Wannan fim din yana da kyau")  # Hausa
# → {'label': 'positive', 'confidence': 0.81, ...}

# Works across Yoruba, Hausa, Igbo, and Pidgin
```

### Load Nigerian Datasets

```python
from naijaml.data import load_dataset

# NaijaSenti — Sentiment in 4 Nigerian languages
data = load_dataset("naijasenti", lang="yor", split="train")
# → 8,522 Yoruba samples, 14,172 Hausa, 10,192 Igbo, 5,121 Pidgin

# MasakhaNER — Named Entity Recognition
ner_data = load_dataset("masakhaner", lang="hau", split="train")
# → Tags: PER, ORG, LOC, DATE

# MasakhaNEWS — News Classification
news = load_dataset("masakhanews", lang="pcm", split="train")
# → Categories: business, entertainment, health, politics, sports, technology

# 7 datasets total | Downloads once, cached offline
```

### Text Preprocessing

```python
from naijaml.nlp import mask_pii, is_pidgin_particle

# Mask Nigerian PII patterns
mask_pii("Call me on 08012345678 or email me@example.com")
# → 'Call me on [PHONE] or [EMAIL]'
# Detects: +234 numbers, 080x/070x/090x, BVN, NIN, emails

# Pidgin-aware — preserves particles other tools strip
is_pidgin_particle("sha")   # → True
is_pidgin_particle("sef")   # → True
is_pidgin_particle("abeg")  # → True
```

### Nigerian Constants

```python
from naijaml.utils.constants import STATES, BANKS, format_naira, get_telco

STATES["Lagos"]              # → 'Ikeja'
BANKS["Guaranty Trust Bank"]  # → '058'
format_naira(1500000)        # → '₦1,500,000.00'
get_telco("08031234567")     # → 'MTN'
```

### Tokenizer

```python
from naijaml.nlp import Tokenizer

tok = Tokenizer("yoruba")
tokens = tok.encode("Ọjọ́ àìkú")
text = tok.decode(tokens)  # Perfect roundtrip

# Or use the unified tokenizer for all 4 languages
tok = Tokenizer("naija")
tok.encode("Ẹ kú àbọ̀")      # Yoruba
tok.encode("Kedụ ka ị mere")  # Igbo
tok.encode("Ina kwana?")      # Hausa

# 63% fewer tokens than GPT-4 for Yoruba | 100% diacritic preservation
```

## Features

| Feature | Status | Accuracy / Efficiency | Model Size |
|---------|--------|----------------------|------------|
| Tokenizer (Yoruba) | ✅ | 63% fewer tokens vs GPT-4, 45% vs AfriBERTa | 560KB |
| Tokenizer (Igbo) | ✅ | 50% fewer tokens vs GPT-4, 40% vs AfriBERTa | 550KB |
| Tokenizer (Hausa) | ✅ | 31% fewer tokens vs GPT-4, 18% vs AfriBERTa | 420KB |
| Tokenizer (Pidgin) | ✅ | 14% fewer tokens vs GPT-4 | 510KB |
| Tokenizer (Unified) | ✅ | All 4 languages | 400KB |
| Language Detection | ✅ | 96.4% accuracy (NaijaSenti test tweets), 93.1% (MasakhaNEWS test headlines) | 29.6MB |
| Yoruba Diacritizer (full tonal) | ✅ | 80.3% word accuracy (MENYO-20k test) | 12.6MB |
| Yoruba Diacritizer (dot-below) | ✅ | 93.3% word accuracy (MENYO-20k test) | 12.6MB, or 6.4MB bundled fallback (85.7%) |
| Igbo Diacritizer | ✅ | 92.3% word accuracy (MasakhaNER 2.0 Igbo test) | 4.9MB |
| Sentiment Analysis | ✅ | 71.5% accuracy (NaijaSenti test) | 4.3MB |
| Dataset Loaders (7 datasets) | ✅ | — | — |
| Text Preprocessing & PII Masking | ✅ | — | — |
| Nigerian Constants (states, banks, telcos) | ✅ | — | — |

**~48MB bundled, 13MB downloaded on first use.** Everything runs on CPU. No GPU required.

## Design Philosophy

**CPU-first.** Every feature works on a laptop with 4GB RAM. GPU makes things faster but is never required. 95% of African AI talent has no meaningful GPU access — NaijaML is built for them.

**Offline-capable.** Small models ship with the package; larger ones auto-download from [HuggingFace](https://huggingface.co/naijaml/naijaml-models) on first use and cache locally. After first run, everything works without internet.

**Minimal dependencies.** Core package needs only `numpy`, `requests`, `tqdm`, and `tokenizers`. We don't pull in PyTorch if we don't need it.

**Honest metrics.** We report real accuracy numbers, not cherry-picked results. The sentiment model is 72%, not 95%. The Yoruba diacritizer gets dot-below right for 93% of words but full tonal marks for only 80%. We tell you upfront.

**Nigerian context.** Examples use Nigerian names, cities, and data. PII masking handles Nigerian phone formats and national ID numbers. Currency is in Naira, not dollars.

## Models

| Model | Size | Approach |
|-------|------|----------|
| Tokenizers (5 models) | 2.4MB total | BPE trained on dedicated Nigerian language corpora |
| Language Detection | 29.6MB | Naive Bayes + char n-grams (1-4) + language features |
| Yoruba Diacritizer (full) | 12.6MB | Word-level lookup + Viterbi decoding |
| Yoruba Diacritizer (dot-below) | 12.6MB | Word-level model with tones dropped; 6.4MB syllable-based k-NN bundled as offline fallback |
| Igbo Diacritizer | 4.9MB | Syllable-based k-NN |
| Sentiment Analysis | 4.3MB | TF-IDF + Logistic Regression |

## Limitations

We believe in transparency. Here's what NaijaML can't do yet:

- **Yoruba tones:** On the MENYO-20k test set (6,524 multi-domain sentences), dot-below restoration (ọ, ẹ, ṣ) gets 93.3% of words right and full tonal diacritization (à, á, è, é) gets 80.3%, using Viterbi decoding. Only 6.5% of whole sentences come out fully correct with tones. Many of the remaining errors are due to contextual ambiguity where even native speakers sometimes disagree on tones.
- **Already-diacritized input:** Both Yoruba diacritizers keep marks already in the input. The full diacritizer leaves a word that carries a tone mark exactly as given, and adds tones to a word that has only dot-below (e.g. `Ọjọ` → `Ọjọ́`). A word written with no marks at all is always treated as unmarked, so an all-mid-tone word can still gain marks.
- **Sentiment accuracy:** 71.5% on the NaijaSenti test set (17,654 tweets): Igbo 76.9%, Yoruba 72.0%, Hausa 70.8%, Pidgin 67.1%. Good enough for trend analysis, not for production decisions on individual texts. Optional transformer models coming soon.
- **Pidgin sentiment leans negative:** On Pidgin tweets the model finds 92.5% of negatives but only 45.1% of positives and 1.6% of neutrals (7 of 431). The Pidgin training data has just 72 neutral tweets out of 5,121, so treat a Pidgin result as positive-or-negative only. The model is not trained on English.
- **Pidgin vs English:** Pidgin is an English-based creole, so code-mixed texts can be ambiguous. The detector requires Pidgin-specific markers (e.g., "dey", "wetin", "abeg") to classify as Pidgin — English-like text without markers defaults to English. Pidgin recall is 94.6% on NaijaSenti test tweets but only 41.3% on MasakhaNEWS test headlines, where most Pidgin headlines are read as English. English recall on those headlines is 99.9%.
- **Igbo diacritics:** The Igbo diacritizer restores dot-below vowels (ị, ọ, ụ) only, not tone marks. On the MasakhaNER 2.0 Igbo test set (2,181 sentences) it gets 92.3% of words right, against 68.1% for leaving the text unmarked, and 15.7% of whole sentences.

### Reproducing these numbers

```bash
python scripts/eval_heldout.py              # Yoruba diacritizers on the MENYO-20k test set
python scripts/eval_heldout_igbo.py         # Igbo diacritizer on the MasakhaNER 2.0 Igbo test set
python scripts/eval_heldout_langdetect.py   # language detection on the NaijaSenti and MasakhaNEWS test sets
python scripts/eval_heldout_sentiment.py    # sentiment on the NaijaSenti test set, per language and class
python scripts/evaluate_all.py              # quick checks on small curated fixtures for every module
```

The `eval_heldout*` scripts download their test sets on first run. `eval_heldout.py` excludes the 109 test sentences that also occur in the diacritizer's training data. The fixture sets are small (tens of samples), so treat `evaluate_all.py` as a regression check, not a benchmark.

## Tokenizer Benchmark

We benchmarked against GPT-4 (tiktoken), AfriBERTa, and AfroXLMR on Nigerian languages:

### Token Efficiency (fewer = better)

| Language | GPT-4 | AfriBERTa | AfroXLMR | **NaijaML** |
|----------|:-----:|:---------:|:--------:|:-----------:|
| Yoruba | baseline | +45% | +12% | **+63%** |
| Igbo | baseline | +40% | -1% | **+50%** |
| Hausa | baseline | +18% | +14% | **+31%** |
| Pidgin | baseline | -1% | — | **+14%** |

### Diacritic Handling (critical difference)

| Input | GPT-4 | AfriBERTa | **NaijaML** |
|-------|:-----:|:---------:|:-----------:|
| `ọ́` (compound) | 2 tokens | 2 tokens | **1 token** |
| `ẹ̀` (compound) | 3 tokens | 2 tokens | **1 token** |
| `Ẹ kú àbọ̀` | 8 tokens | 5 tokens | **3 tokens** |

Other tokenizers **split diacritics** because they weren't trained on enough Nigerian data. NaijaML keeps them together.

### Speed (batch encoding)

| Tokenizer | Speed | vs GPT-4 |
|-----------|------:|:--------:|
| **NaijaML (Rust)** | 3.8M tok/s | **2.5x faster** |
| GPT-4 (tiktoken) | 1.7M tok/s | baseline |
| AfriBERTa | 0.4M tok/s | 4x slower |

See the full analysis in [`benchmarks/`](benchmarks/).

## Roadmap

- Hausa diacritizer
- More dataset loaders (MENYO-20k, NollySenti, AfriQA, MasakhaPOS)
- Optional transformer models via `pip install naijaml[transformers]`
- Named Entity Recognition for Nigerian entities
- Speech-to-text for Nigerian languages

## Contributing

We need people who know Nigerian languages, Nigerian data, and Nigerian problems — ML engineers, linguists, data scientists, and domain experts in fintech, agritech, and health.

```bash
git clone https://github.com/naijaml/naijaml.git
cd naijaml
pip install -e ".[dev]"
pytest tests/ -v
```

## Links

- [PyPI](https://pypi.org/project/naijaml/)
- [GitHub](https://github.com/naijaml/naijaml)
- [HuggingFace](https://huggingface.co/naijaml)

## Acknowledgments

Built with data and research from [Masakhane](https://www.masakhane.io/), [HausaNLP](https://hausanlp.github.io/), and the African NLP community.

## License

The library code is Apache 2.0.

The model files are statistical tables built from third-party datasets, and those datasets carry their own licences. Whether a model inherits the terms of its training data is not settled, and the model files have not been given a separate licence yet. If you redistribute the models or use them commercially, check the terms below.

| Model | Shipped | Training data | Dataset licence |
|-------|---------|---------------|-----------------|
| Yoruba diacritizers (`word_diacritic_model.json`, `diacritic_model.json`, `dot_below_model.json`) | Downloaded; `dot_below_model.json` is bundled | [`bumie-e/Yoruba-diacritics-vs-non-diacritics`](https://huggingface.co/datasets/bumie-e/Yoruba-diacritics-vs-non-diacritics) | GPL-3.0 |
| Sentiment (`sentiment_model.json`) | Bundled | [`HausaNLP/NaijaSenti-Twitter`](https://huggingface.co/datasets/HausaNLP/NaijaSenti-Twitter) | CC BY-NC-SA 4.0 |
| Language detection (`lang_model.json`) | Bundled | NaijaSenti, [MasakhaNEWS](https://huggingface.co/datasets/masakhane/masakhanews), [MasakhaNER 2](https://huggingface.co/datasets/masakhane/masakhaner2), NollySenti | CC BY-NC-SA 4.0 (NaijaSenti), AFL-3.0 (MasakhaNEWS, MasakhaNER 2) |
| Igbo diacritizer (`igbo_diacritic_model.json`) | Bundled | [`Tommy0201/JW300_Igbo_To_Eng`](https://huggingface.co/datasets/Tommy0201/JW300_Igbo_To_Eng), MasakhaNEWS | None declared (JW300 mirror), AFL-3.0 (MasakhaNEWS) |

Licences are the ones declared on each dataset's Hugging Face page on 8 October 2026.
