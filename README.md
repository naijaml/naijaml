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

# Dot-below: 97.5% accuracy | 6.4MB bundled
# Full tonal: 90.0% accuracy | 12.6MB auto-downloaded on first use
```

### Igbo Diacritizer

```python
from naijaml.nlp import diacritize_igbo

diacritize_igbo("Kedu ka i mere")
# → 'Kedụ ka ị mere'

# 95.2% accuracy | 4.9MB model | CPU only
```

### Language Detection

```python
from naijaml.nlp import detect_language

detect_language("Bawo ni, se daadaa ni?")   # → 'yor'
detect_language("Ina kwana?")                # → 'hau'
detect_language("Kedu ka ị mere?")           # → 'ibo'
detect_language("How far, wetin dey happen?") # → 'pcm'

# 5 languages: Yoruba, Hausa, Igbo, Pidgin, English | ~95% accuracy
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

## Features

| Feature | Status | Accuracy | Model Size |
|---------|--------|----------|------------|
| Language Detection | ✅ | ~95% | 1.8MB |
| Yoruba Diacritizer (full tonal) | ✅ | 90.0% word | 12.6MB |
| Yoruba Diacritizer (dot-below) | ✅ | 97.5% char | 6.4MB |
| Igbo Diacritizer | ✅ | 95.2% | 4.9MB |
| Sentiment Analysis | ✅ | 72% | 4.3MB |
| Dataset Loaders (7 datasets) | ✅ | — | — |
| Text Preprocessing & PII Masking | ✅ | — | — |
| Nigerian Constants (states, banks, telcos) | ✅ | — | — |

**17MB bundled, 13MB downloaded on first use.** Everything runs on CPU. No GPU required.

## Design Philosophy

**CPU-first.** Every feature works on a laptop with 4GB RAM. GPU makes things faster but is never required. 95% of African AI talent has no meaningful GPU access — NaijaML is built for them.

**Offline-capable.** Small models ship with the package; larger ones auto-download from [HuggingFace](https://huggingface.co/naijaml/naijaml-models) on first use and cache locally. After first run, everything works without internet.

**Minimal dependencies.** Core package needs only `numpy`, `requests`, `tqdm`. We don't pull in PyTorch if we don't need it.

**Honest metrics.** We report real accuracy numbers, not cherry-picked results. The sentiment model is 72%, not 95%. The Yoruba diacritizer handles dot-below at 97.5% but full tonal is 90%. We tell you upfront.

**Nigerian context.** Examples use Nigerian names, cities, and data. PII masking handles Nigerian phone formats and national ID numbers. Currency is in Naira, not dollars.

## Models

| Model | Size | Approach |
|-------|------|----------|
| Language Detection | 1.8MB | Naive Bayes + char n-grams |
| Yoruba Diacritizer (full) | 12.6MB | Word-level lookup + Viterbi decoding |
| Yoruba Diacritizer (dot-below) | 6.4MB | Syllable-based k-NN |
| Igbo Diacritizer | 4.9MB | Syllable-based k-NN |
| Sentiment Analysis | 4.3MB | TF-IDF + Logistic Regression |

## Limitations

We believe in transparency. Here's what NaijaML can't do yet:

- **Yoruba tones:** Dot-below restoration (ọ, ẹ, ṣ) is 97.5% accurate. Full tonal diacritization (à, á, è, é) is 90% word accuracy using Viterbi decoding — remaining errors are due to contextual ambiguity where even native speakers sometimes disagree on tones.
- **Sentiment accuracy:** 72% on Twitter data. Good enough for trend analysis, not for production decisions on individual texts. Optional transformer models coming soon.
- **Pidgin vs English:** Short texts can be ambiguous between Pidgin and informal English. The detector works best on sentences of 5+ words.

## Tokenizer Benchmark

We benchmarked 7 major AI tokenizers (GPT-4, GPT-4o, Llama 3, Gemma 2, Mistral, BERT, XLM-RoBERTa) on Nigerian languages. The results:

| Language | Avg Token Ratio vs English |
|----------|:-:|
| Yoruba | **3.14x** |
| Igbo | **2.30x** |
| Hausa | **1.75x** |
| Pidgin | **1.05x** |

Yoruba text costs 3x more to process than English with most tokenizers. GPT-4o's newer tokenizer performs best (1.69x); Mistral performs worst (2.47x). See the full analysis with interactive charts in [`benchmarks/`](benchmarks/).

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

Apache 2.0
