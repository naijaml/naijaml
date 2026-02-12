# NaijaML

Python library for Nigerian language NLP. Supports Yorùbá, Hausa, Igbo, and Nigerian Pidgin.

```bash
pip install naijaml
```

## What's Inside

| Feature | What it does | Accuracy |
|---------|--------------|----------|
| Language Detection | Identify yor/hau/ibo/pcm/eng | ~95% |
| Yorùbá Diacritizer | Restore ọ, ẹ, ṣ marks | 97.5% |
| Igbo Diacritizer | Restore ị, ọ, ụ marks | 95.2% |
| Sentiment Analysis | Classify pos/neg/neutral | 72% |
| Dataset Loaders | NaijaSenti, MasakhaNER, MasakhaNEWS | - |
| Text Preprocessing | PII masking, Pidgin-aware cleaning | - |

All models run on CPU. No GPU required. Total size: ~17MB.

## Quick Start

### Language Detection

```python
from naijaml.nlp import detect_language

detect_language("Bawo ni, se daadaa ni?")
# → 'yor'

detect_language("Ina kwana?")
# → 'hau'

detect_language("Kedu ka ị mere?")
# → 'ibo'

detect_language("How far, wetin dey happen?")
# → 'pcm'
```

### Yorùbá Diacritizer

```python
from naijaml.nlp import diacritize_dot_below

diacritize_dot_below("Ojo lo si oja")
# → 'Ọjọ lo si ọja'

diacritize_dot_below("Ese pupo fun iranlowo re")
# → 'Ẹsẹ pupo fun iranlọwọ rẹ'
```

### Igbo Diacritizer

```python
from naijaml.nlp import diacritize_igbo

diacritize_igbo("Kedu ka i mere")
# → 'Kedụ ka ị mere'

diacritize_igbo("Daalu nne")
# → 'Daalu nne'
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
```

### Dataset Loaders

```python
from naijaml.data import load_dataset

# NaijaSenti - Sentiment Analysis (Hausa, Igbo, Yorùbá, Pidgin)
data = load_dataset("naijasenti", lang="yor", split="train")
# → [{'text': 'Ọjọ́ yìí dára gan!', 'label': 'positive'}, ...]
# 8,522 Yorùbá samples, 14,172 Hausa, 10,192 Igbo, 5,121 Pidgin

# MasakhaNER - Named Entity Recognition (Hausa, Igbo, Yorùbá)
ner_data = load_dataset("masakhaner", lang="hau", split="train")
# → [{'tokens': ['Shugaba', 'Tinubu', 'ya', ...], 'ner_tags': ['B-PER', 'I-PER', 'O', ...]}, ...]
# Tags: PER (person), ORG (organization), LOC (location), DATE

# MasakhaNEWS - News Classification (Hausa, Igbo, Yorùbá, Pidgin)
news = load_dataset("masakhanews", lang="pcm", split="train")
# → [{'text': '...', 'label': 'sports', 'headline': '...', 'url': '...'}, ...]
# Categories: business, entertainment, health, politics, sports, technology
```

### Text Preprocessing

```python
from naijaml.nlp import mask_pii, is_pidgin_particle

# Mask Nigerian phone numbers, emails, BVN, NIN
mask_pii("Call me on 08012345678 or email me@example.com")
# → 'Call me on [PHONE] or [EMAIL]'

# Check Pidgin particles (words often stripped by other NLP tools)
is_pidgin_particle("sha")  # → True
is_pidgin_particle("sef")  # → True
is_pidgin_particle("abeg") # → True
```

### Nigerian Constants

```python
from naijaml.utils.constants import STATES, BANKS, format_naira, get_telco

# All 36 states + FCT
STATES["Lagos"]  # → 'Ikeja'

# Nigerian banks
BANKS["GTBank"]  # → '058'

# Format Naira
format_naira(1500000)  # → '₦1,500,000.00'

# Identify telco from phone number
get_telco("08031234567")  # → 'MTN'
```

## Design Philosophy

- **CPU-first**: Everything works on a laptop with 4GB RAM
- **Minimal dependencies**: Core package needs only `numpy`, `requests`, `tqdm`
- **Offline-capable**: Models cached locally after first download
- **Honest metrics**: We report real accuracy numbers, not cherry-picked results

## Models

All models are lightweight and included in the package:

| Model | Size | Approach |
|-------|------|----------|
| Language Detection | 1.8MB | Naive Bayes + char n-grams |
| Yorùbá Diacritizer | 6.4MB | Syllable-based k-NN |
| Igbo Diacritizer | 4.9MB | Syllable-based k-NN |
| Sentiment Analysis | 4.3MB | TF-IDF + Logistic Regression |

## Limitations

Be aware of current limitations:

- **Yorùbá tones**: Dot-below restoration is 97.5% accurate, but full tonal diacritization is ~77% due to contextual ambiguity
- **Sentiment**: 72% accuracy on Twitter data. Production use cases may want the optional transformer model (coming soon)
- **Pidgin-English**: Short texts can be ambiguous between Pidgin and English

## Coming Soon

- [ ] Hausa diacritizer
- [ ] More dataset loaders (MENYO-20k, NollySenti, AfriQA, MasakhaPOS)
- [ ] Optional transformer models via `pip install naijaml[nlp]`
- [ ] Named Entity Recognition wrapper
- [ ] Speech-to-text for Nigerian languages

## Contributing

We welcome contributions from the Nigerian and global ML community.

```bash
# Clone and install dev dependencies
git clone https://github.com/naijaml/naijaml.git
cd naijaml
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## License

MIT

## Acknowledgments

Built with data from [Masakhane](https://www.masakhane.io/), [HausaNLP](https://hausanlp.github.io/), and the African NLP community.
