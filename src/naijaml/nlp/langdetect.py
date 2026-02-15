"""Nigerian language detection using Naive Bayes classifier.

Detects Yorùbá (yor), Hausa (hau), Igbo (ibo), Nigerian Pidgin (pcm),
and English (eng) from text using a Multinomial Naive Bayes classifier
trained on character n-gram features from real Nigerian NLP datasets.

This is a lightweight, CPU-only implementation with no heavy dependencies.
"""
from __future__ import annotations

import json
import logging
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES: List[str] = ["yor", "hau", "ibo", "pcm", "eng"]

from naijaml.utils.download import get_model_path

# Bundled model (ships with pip install)
_BUNDLED_MODEL_PATH = Path(__file__).parent / "lang_model.json"

# Cached model (loaded lazily)
_MODEL: Optional["NaiveBayesLangDetector"] = None

# N-gram sizes to use (bigrams and trigrams)
_NGRAM_SIZES = [2, 3]

# Minimum n-gram count to include in model (prune rare n-grams)
_MIN_NGRAM_COUNT = 2


# =============================================================================
# Text normalization and feature extraction
# =============================================================================

def _normalize_text(text: str) -> str:
    """Normalize text for n-gram extraction."""
    # NFC normalization for consistent diacritic handling
    text = unicodedata.normalize("NFC", text)
    # Lowercase
    text = text.lower()
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    return text


def _extract_ngrams(text: str) -> Counter:
    """Extract character n-grams from text as a Counter."""
    text = _normalize_text(text)
    ngrams: Counter = Counter()
    for n in _NGRAM_SIZES:
        for i in range(len(text) - n + 1):
            ngram = text[i:i + n]
            ngrams[ngram] += 1
    return ngrams


# =============================================================================
# Naive Bayes Language Detector
# =============================================================================

class NaiveBayesLangDetector:
    """Multinomial Naive Bayes classifier for language detection.

    Uses character n-gram features with Laplace smoothing.

    The classifier computes:
        P(lang | text) ∝ P(lang) × ∏ P(ngram | lang)

    Using log probabilities for numerical stability:
        log P(lang | text) = log P(lang) + Σ log P(ngram | lang)
    """

    def __init__(self, alpha: float = 1.0, uniform_priors: bool = True):
        """Initialize the detector.

        Args:
            alpha: Laplace smoothing parameter (default 1.0).
            uniform_priors: If True, use uniform priors instead of data priors.
                           This helps when training data is imbalanced.
        """
        self.alpha = alpha
        self.uniform_priors = uniform_priors
        self.log_priors: Dict[str, float] = {}
        self.log_likelihoods: Dict[str, Dict[str, float]] = {}
        self.vocab: Set[str] = set()
        self._default_log_likelihood: Dict[str, float] = {}

    def fit(self, texts: List[str], labels: List[str]) -> "NaiveBayesLangDetector":
        """Train the classifier on labeled text data.

        Args:
            texts: List of text samples.
            labels: List of language labels (same length as texts).

        Returns:
            self for method chaining.
        """
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")

        # Count samples per language for priors
        label_counts: Counter = Counter(labels)
        total_samples = len(labels)
        num_classes = len(label_counts)

        # Compute log priors (uniform or from data)
        if self.uniform_priors:
            # Uniform priors: each language equally likely a priori
            self.log_priors = {
                lang: math.log(1.0 / num_classes)
                for lang in label_counts
            }
        else:
            self.log_priors = {
                lang: math.log(count / total_samples)
                for lang, count in label_counts.items()
            }

        # Aggregate n-gram counts per language
        ngram_counts: Dict[str, Counter] = {lang: Counter() for lang in label_counts}
        total_ngrams: Dict[str, int] = {lang: 0 for lang in label_counts}

        for text, label in zip(texts, labels):
            ngrams = _extract_ngrams(text)
            ngram_counts[label].update(ngrams)
            total_ngrams[label] += sum(ngrams.values())
            self.vocab.update(ngrams.keys())

        # Prune rare n-grams (keep only those with count >= _MIN_NGRAM_COUNT across all languages)
        global_counts: Counter = Counter()
        for lang_counts in ngram_counts.values():
            global_counts.update(lang_counts)

        self.vocab = {ng for ng, count in global_counts.items() if count >= _MIN_NGRAM_COUNT}

        vocab_size = len(self.vocab)
        logger.info("Vocabulary size after pruning: %d", vocab_size)

        # Compute log likelihoods with Laplace smoothing
        self.log_likelihoods = {}
        self._default_log_likelihood = {}

        for lang in label_counts:
            self.log_likelihoods[lang] = {}
            denominator = total_ngrams[lang] + self.alpha * vocab_size

            # Default log-likelihood for unseen n-grams
            self._default_log_likelihood[lang] = math.log(self.alpha / denominator)

            for ngram in self.vocab:
                count = ngram_counts[lang].get(ngram, 0)
                self.log_likelihoods[lang][ngram] = math.log(
                    (count + self.alpha) / denominator
                )

        logger.info("Trained on %d samples across %d languages",
                    total_samples, len(label_counts))

        return self

    def predict(self, text: str) -> str:
        """Predict the most likely language for the given text.

        Args:
            text: Input text to classify.

        Returns:
            Language code ('yor', 'hau', 'ibo', 'pcm', 'eng').
        """
        scores = self._compute_log_posteriors(text)
        return max(scores, key=scores.get)

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Get probability distribution over languages.

        Args:
            text: Input text to classify.

        Returns:
            Dict mapping language codes to probabilities (sum to 1.0).
        """
        log_scores = self._compute_log_posteriors(text)

        # Convert log probabilities to probabilities using log-sum-exp trick
        max_log = max(log_scores.values())
        exp_scores = {lang: math.exp(score - max_log) for lang, score in log_scores.items()}
        total = sum(exp_scores.values())

        return {lang: score / total for lang, score in exp_scores.items()}

    def _compute_log_posteriors(self, text: str) -> Dict[str, float]:
        """Compute unnormalized log posterior for each language."""
        ngrams = _extract_ngrams(text)

        scores = {}
        for lang, log_prior in self.log_priors.items():
            log_likelihood = 0.0
            lang_likelihoods = self.log_likelihoods[lang]
            default_ll = self._default_log_likelihood[lang]

            for ngram, count in ngrams.items():
                # Get log-likelihood for this n-gram (or default for unseen)
                ll = lang_likelihoods.get(ngram, default_ll)
                log_likelihood += count * ll

            scores[lang] = log_prior + log_likelihood

        return scores

    def save(self, path: Path) -> None:
        """Save the trained model to a JSON file.

        Args:
            path: Path to save the model.
        """
        data = {
            "alpha": self.alpha,
            "uniform_priors": self.uniform_priors,
            "log_priors": self.log_priors,
            "log_likelihoods": self.log_likelihoods,
            "vocab": list(self.vocab),
            "default_log_likelihood": self._default_log_likelihood,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "NaiveBayesLangDetector":
        """Load a trained model from a JSON file.

        Args:
            path: Path to the model file.

        Returns:
            Loaded NaiveBayesLangDetector instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        detector = cls(
            alpha=data.get("alpha", 1.0),
            uniform_priors=data.get("uniform_priors", True),
        )
        detector.log_priors = data["log_priors"]
        detector.log_likelihoods = data["log_likelihoods"]
        detector.vocab = set(data.get("vocab", []))
        detector._default_log_likelihood = data.get("default_log_likelihood", {})

        # If default_log_likelihood not saved, compute from scratch
        if not detector._default_log_likelihood:
            for lang in detector.log_priors:
                # Use a very low probability for unseen n-grams
                detector._default_log_likelihood[lang] = -15.0

        return detector


# =============================================================================
# Training data collection
# =============================================================================

def _get_english_corpus() -> List[str]:
    """Get a built-in English corpus for training."""
    # English sentences from Wikipedia/news style text
    return [
        "The weather is quite pleasant today.",
        "I would like to order some food please.",
        "What is your name? My name is John.",
        "She went to the market this morning.",
        "The children are playing in the garden.",
        "Thank you very much for your help.",
        "We will travel to Lagos next week.",
        "Please give me a glass of water.",
        "I am going to the market to buy some groceries for dinner tonight.",
        "The quick brown fox jumps over the lazy dog.",
        "Nigeria is a beautiful country with diverse cultures.",
        "They have been working on this project for several months now.",
        "Would you please help me with this task? I really appreciate your assistance.",
        "The meeting will be held tomorrow afternoon at three o'clock.",
        "She has already finished her homework and is now watching television.",
        "The government announced new economic policies yesterday.",
        "Education is very important for the development of any nation.",
        "The hospital provides excellent healthcare services to patients.",
        "Technology has transformed the way we communicate with each other.",
        "Climate change is one of the greatest challenges facing humanity.",
        "The company reported strong financial results this quarter.",
        "Scientists have made a breakthrough discovery in cancer research.",
        "The football match was very exciting and ended in a draw.",
        "Many people are now working remotely from their homes.",
        "The restaurant serves delicious food at affordable prices.",
        "Children need proper nutrition for healthy growth and development.",
        "The university has a world-class library with millions of books.",
        "Traffic congestion is a major problem in big cities.",
        "Renewable energy sources are becoming increasingly popular.",
        "The museum displays artifacts from ancient civilizations.",
        "Good communication skills are essential in the workplace.",
        "The bank offers competitive interest rates on savings accounts.",
        "Social media has changed how people share information.",
        "The economy is expected to grow by five percent next year.",
        "Regular exercise is important for maintaining good health.",
        "The president addressed the nation in a televised speech.",
        "Environmental protection should be a priority for all governments.",
        "Online shopping has become very convenient for consumers.",
        "The airline offers direct flights to major international destinations.",
        "Fresh fruits and vegetables are essential for a balanced diet.",
        "The stock market experienced significant volatility this week.",
        "Public transportation is an efficient way to travel in urban areas.",
        "The film received excellent reviews from critics and audiences.",
        "Job opportunities are increasing in the technology sector.",
        "The hotel provides excellent amenities for business travelers.",
        "Water conservation is crucial in regions facing drought.",
        "The school has implemented new safety measures for students.",
        "International cooperation is needed to address global challenges.",
        "The construction of the new bridge will take two years.",
        "Healthcare workers deserve our gratitude and support.",
    ]


def _collect_training_data(max_per_lang: int = 500) -> Tuple[List[str], List[str]]:
    """Collect training data from cached NaijaML datasets + fallback.

    Always includes fallback data to ensure all 5 languages are represented.
    Balances data by limiting samples per language.

    Args:
        max_per_lang: Maximum samples per language to prevent imbalance.

    Returns:
        Tuple of (texts, labels) lists.
    """
    import random
    from naijaml.data import load_dataset
    from naijaml.data.cache import is_cached

    # Collect all data per language
    lang_data: Dict[str, List[str]] = {lang: [] for lang in SUPPORTED_LANGUAGES}

    # Get fallback data first
    fallback_texts, fallback_labels = _get_fallback_training_data()
    for text, label in zip(fallback_texts, fallback_labels):
        lang_data[label].append(text)
    logger.info("Loaded fallback data")

    # Try to augment with cached datasets
    for lang in ["yor", "hau", "ibo", "pcm"]:
        # Try NaijaSenti (primary source)
        for split in ["train", "validation", "test"]:
            if is_cached("naijasenti", lang, split):
                try:
                    data = load_dataset("naijasenti", lang=lang, split=split)
                    lang_data[lang].extend(item["text"] for item in data)
                except Exception as e:
                    logger.debug("Failed to load naijasenti %s/%s: %s", lang, split, e)

        # Try MasakhaNEWS
        for split in ["train", "validation", "test"]:
            if is_cached("masakhanews", lang, split):
                try:
                    data = load_dataset("masakhanews", lang=lang, split=split)
                    lang_data[lang].extend(item["text"] for item in data)
                except Exception as e:
                    logger.debug("Failed to load masakhanews %s/%s: %s", lang, split, e)

        # Try MasakhaNER (tokens -> text)
        if lang in ["yor", "hau", "ibo"]:
            for split in ["train", "validation", "test"]:
                if is_cached("masakhaner", lang, split):
                    try:
                        data = load_dataset("masakhaner", lang=lang, split=split)
                        for item in data:
                            if "tokens" in item:
                                lang_data[lang].append(" ".join(item["tokens"]))
                    except Exception as e:
                        logger.debug("Failed to load masakhaner %s/%s: %s", lang, split, e)

    # Balance data by sampling max_per_lang from each language
    texts: List[str] = []
    labels: List[str] = []

    for lang, lang_texts in lang_data.items():
        if len(lang_texts) > max_per_lang:
            # Random sample to balance
            random.seed(42)  # Reproducible
            sampled = random.sample(lang_texts, max_per_lang)
        else:
            sampled = lang_texts

        texts.extend(sampled)
        labels.extend([lang] * len(sampled))
        logger.info("Using %d samples for %s (total available: %d)",
                    len(sampled), lang, len(lang_texts))

    return texts, labels


def _get_fallback_training_data() -> Tuple[List[str], List[str]]:
    """Get fallback training data when no datasets are cached."""
    texts = []
    labels = []

    # Yorùbá samples with diacritics (50+ samples)
    yor_texts = [
        "ọjọ́ dára púpọ̀ ẹ kú iṣẹ́ o",
        "mo fẹ́ràn rẹ púpọ̀ ọrẹ mi",
        "báwo ni ọjà ṣe wà lónìí",
        "ó ti lọ sí ilé ìwé láàárọ̀ yìí",
        "ẹ̀ṣẹ̀ púpọ̀ fún oúnjẹ náà",
        "kí ni orúkọ rẹ mo ń pè mí ní adé",
        "àwọn ọmọdé ń ṣeré ní ọ̀nà",
        "ẹ jọ̀wọ́ ẹ fún mi ní omi",
        "olúwa a bukun fún ẹ ẹ ṣeun púpọ̀",
        "mo ń lọ sí ọjà láti ra oúnjẹ",
        "ọmọ mi ti ṣe dáadáa ní ilé ẹ̀kọ́",
        "àwa ń gbé ní ilẹ̀ yorùbá",
        "ẹ wá sí ilé wa láti jẹun",
        "ṣé o ti rí ọmọ náà lónìí",
        "ìwọ ni mo fẹ́ràn jù lọ",
        "ọrẹ mi dára púpọ̀ láti bá mi sọ̀rọ̀",
        "ẹ má bínú mo ti pẹ́ díẹ̀",
        "nígbà tí mo dé ilé ó ti lọ",
        "àwọn ará ìlú náà ń ṣiṣẹ́ dáadáa",
        "ọba àti ìjọba ń ṣe iṣẹ́ rere",
        "oúnjẹ yìí dùn gan ẹ ṣe é dáadáa",
        "mo ń kọ́ èdè yorùbá ní ilé ẹ̀kọ́",
        "àwọn akẹ́kọ̀ọ́ ń kàwé ní ilé ìkàwé",
        "ọjọ̀ kan mo máa lo sí ibi iṣẹ́",
        "ìyá mi ti ṣe oúnjẹ àárọ̀",
        "bàbá mi ń ṣiṣẹ́ ní ilé iṣẹ́",
        "ẹ̀gbọ́n mi ń gbé ní ìlú lagos",
        "àbúrò mi ti lọ sí ilé ẹ̀kọ́ gíga",
        "a ó rí ara wa lọ́la ẹ máa rìn dáadáa",
        "ọ̀rọ̀ yìí dára púpọ̀ mo gbọ́ ọ dáadáa",
        "àwa Yorùbá ń gbé ní gúúsù ìwọ̀ oòrùn nàìjíríà",
        "ìṣẹ̀lẹ̀ náà ṣẹlẹ̀ ní ọjọ́ ọ̀sẹ̀ tó kọjá",
        "ó dára kí a máa bá ara wa sọ̀rọ̀",
        "ẹ̀rọ yìí ṣiṣẹ́ dáadáa gan ni",
        "mo ń wá iṣẹ́ ní ilú náà",
        "àwọn oníṣòwò ń ta ọjà ní ọjà",
        "ọmọdé kò gbọdọ̀ máa ṣe bẹ́ẹ̀",
        "olùkọ́ ń kọ́ àwọn akẹ́kọ̀ọ́ ní yàrá",
        "ìròyìn dùn mo gbọ́ ẹ ṣeun",
        "ó ṣe pàtàkì láti kọ́ èdè míì",
    ]
    texts.extend(yor_texts)
    labels.extend(["yor"] * len(yor_texts))

    # Hausa samples (50+ samples)
    hau_texts = [
        "ina kwana yaya aiki",
        "na gode sosai allah ya kara mana",
        "wannan abinci yana da dadi kwarai",
        "yaushe za ka zo gidanmu",
        "mun tafi kasuwa da safe",
        "yara suna wasa a waje",
        "allah ya ba mu lafiya",
        "kin san hausa i na san hausa",
        "barka da zuwa najeriya",
        "ina son koyon sabuwar fasaha domin inganta aikina",
        "mutane da yawa suna magana da hausa",
        "yau muna da taro a ofis",
        "abincin nan yana da tsami sosai",
        "ban ji ba sai ka sake fadinsa",
        "muna farin ciki sosai da zuwanka",
        "ina fata za ka iya zuwa taron",
        "yaya iyalinka suna lafiya",
        "mun shirya komai don bikin",
        "wannan shine abin da muke so",
        "za mu tafi gobe da safe",
        "na karanta littafi mai kyau",
        "yana da muhimmanci sosai",
        "suna aiki tare a ofis",
        "mun kammala aikin da wuri",
        "ina bukatar taimakonka yanzu",
        "yarinya tana wasa da abokanta",
        "malam yana koyar da dalibai",
        "ina magana da shi kullum",
        "mun yi tafiya zuwa kano",
        "gidan yake kusa da kasuwa",
        "za su zo nan gobe ko yau",
        "muna cin abinci da karfe shida",
        "ina aiki a ofis na gwamnati",
        "yarinya tana karatu a makaranta",
        "muna sauraron labarai a rediyo",
        "sun dawo daga aikin hajji",
        "muna biki a wannan makon",
        "ina tsammanin zai zo",
        "mun gama duk ayyukan yau",
        "za mu sake ganinsu gobe",
        "na sayi sabon waya a kasuwa",
        "muna bukatar ruwa mai yawa",
        "abincin da ta dafa yana da dadi",
        "za mu tashi da wuri gobe",
        "sun iso garin jiya da dare",
        "ina son in koyi harshen turanci",
        "muna zaune a gidan baban mu",
        "yara suna zuwa makaranta kullum",
        "na ga wannan a talabijin jiya",
        "muna kiran allah don taimako",
    ]
    texts.extend(hau_texts)
    labels.extend(["hau"] * len(hau_texts))

    # Igbo samples (50+ samples)
    ibo_texts = [
        "nnọọ kedu ka ị mere",
        "ahụrụ m gị n'anya nwanne m",
        "ọ nọ n'ụlọ akwụkwọ ugbu a",
        "anyị ga aga ahịa echi",
        "nri a dị ụtọ nke ukwuu",
        "kedụ aha gị aha m bụ chidi",
        "ụmụaka na egwu egwu n'ụzọ",
        "chukwu gozie gị daalụ",
        "ọ bụ onye igbo o si n'enugu",
        "biko nye m mmiri",
        "anyị nọ n'obodo anyị",
        "ihe a dị mma nke ukwuu",
        "ọ na agụ akwụkwọ n'ụlọ",
        "gịnị ka ị chọrọ",
        "nke a bụ ezi okwu",
        "anyị ga akwụrụ gị ụgwọ",
        "ọ bịara n'ụbọchị gara aga",
        "m chọrọ ịhụ ya taa",
        "ọ dị mma nke ukwuu",
        "ị ga abịa echi",
        "nwoke ahụ nọ n'ụlọ ya",
        "nwanyị ahụ na azụ ahịa",
        "ụmụ nwoke na egwu bọọlụ",
        "ọ na ekwu okwu ọma",
        "anyị na arụ ọrụ ugbu a",
        "o nwere ego ole",
        "m ga eje obodo a",
        "ha na eri nri ehihie",
        "ọ bụ nwanne m nwoke",
        "m ga akwụ gị ụgwọ gị",
        "ndị mmadụ na anọ n'ụlọ",
        "ọ na akwadebe maka ọrụ ya",
        "anyị na ele ihe ngosi taa",
        "m mụtara asụsụ igbo",
        "ha biara site n'obodo ọzọ",
        "ọ chọrọ ịmụ asụsụ ọhụụ",
        "anyị na eje ozi",
        "ọ kụrụ ụka ahụ",
        "m ji ego zụta ya",
        "ọ masịrị m nke ukwuu",
        "anyị na eri nri anyị",
        "ọ bịara ụlọ anyị",
        "m hụrụ ya n'ụzọ",
        "ọ nọ n'ime ụlọ ahụ",
        "ha ga abịa echi",
        "anyị na ele egwuregwu",
        "ọ bụ ezigbo mmadụ",
        "ọ na asụ bekee",
        "m nwere ike ime ya",
        "anyị nọ n'ebe ahụ",
    ]
    texts.extend(ibo_texts)
    labels.extend(["ibo"] * len(ibo_texts))

    # Nigerian Pidgin samples (70+ samples) - distinctive from English
    pcm_texts = [
        "wetin dey happen for this country",
        "wetin dey happen na",
        "wetin dey sup for here",
        "i no fit shout abeg e don tire me",
        "this food sweet die who cook am",
        "make we go that place tomorrow",
        "na so life be nothing we fit do",
        "you too dey form calm down abeg",
        "omo this thing no easy at all o",
        "i wan chop hunger dey catch me",
        "how far na long time no see",
        "no wahala everything go dey alright",
        "dem don chop belle full",
        "who dey there abeg help me carry this load",
        "you sabi wetin you dey talk so",
        "make una hear word jor",
        "shey you see am wetin e come dey do",
        "wahala dey o my broda i no fit shout",
        "oya dey go jeje no go cause any kasala",
        "na you go suffer am no be me",
        "e no concern me at all abeg",
        "the thing don spoil finish",
        "abeg no vex i no mean am",
        "wetin be your own sef",
        "i dey come wait for me",
        "na im talk am for me",
        "you don see am before",
        "make i yarn you something",
        "e get as e be for dis side",
        "no be small thing o",
        "i no sabi wetin you wan",
        "dem no dey see us o",
        "wetin you carry come",
        "abeg make you no vex",
        "na wa for you o",
        "i dey go house now",
        "you dey craze abi",
        "na so dem take do am",
        "e sweet me die",
        "who tell you that one",
        "i go see you later",
        "make we dey go abeg",
        "no be lie na true",
        "you too sabi book",
        "dem say na you do am",
        "i no go gree for that",
        "na here we dey o",
        "you no go believe am",
        "wetin concern me for that matter",
        "abeg leave me make i rest",
        "na only god fit help us",
        "e don tey wey we see",
        # More distinctive Pidgin with "for" patterns
        "na for this country wahala full",
        "wetin happen for that side na",
        "dem dey do am for everywhere",
        "e don happen for this our area",
        "na for here we dey stay",
        "wetin concern you for this matter",
        "na for who this thing be",
        "make we comot for here abeg",
        "i wan go for that place",
        "dem dey wait for us there",
        # More wetin patterns
        "wetin you dey find for here",
        "wetin you wan make i do",
        "wetin happen to your phone",
        "wetin be the matter sef",
        "wetin you go tell am",
        # More dey patterns
        "i dey here since morning",
        "dem dey come now now",
        "which kain thing you dey do",
        "where you dey go like this",
        "why you dey act like that",
    ]
    texts.extend(pcm_texts)
    labels.extend(["pcm"] * len(pcm_texts))

    # English samples
    eng_texts = _get_english_corpus()
    texts.extend(eng_texts)
    labels.extend(["eng"] * len(eng_texts))

    return texts, labels


# =============================================================================
# Model loading
# =============================================================================

def _get_model() -> NaiveBayesLangDetector:
    """Get the language detection model (loading from file or training)."""
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    # Try bundled model first, then HF download
    try:
        if _BUNDLED_MODEL_PATH.exists():
            model_path = _BUNDLED_MODEL_PATH
        else:
            model_path = get_model_path("lang_model.json")
        _MODEL = NaiveBayesLangDetector.load(model_path)
        logger.debug("Loaded pre-trained model from %s", model_path)
        return _MODEL
    except Exception as e:
        logger.warning("Failed to load lang model: %s", e)

    # Train a new model
    logger.info("Training new language detection model...")

    try:
        texts, labels = _collect_training_data()
    except Exception as e:
        logger.debug("Could not collect from datasets: %s", e)
        texts, labels = _get_fallback_training_data()

    if len(texts) < 10:
        # Use fallback if we didn't get enough data
        texts, labels = _get_fallback_training_data()

    _MODEL = NaiveBayesLangDetector()
    _MODEL.fit(texts, labels)

    return _MODEL


# =============================================================================
# Public API
# =============================================================================

def detect_language(text: str) -> Optional[str]:
    """Detect the language of Nigerian text.

    Uses a Naive Bayes classifier trained on character n-grams from
    real Nigerian language data. Supports Yorùbá (yor), Hausa (hau),
    Igbo (ibo), Nigerian Pidgin (pcm), and English (eng).

    Args:
        text: Input text to classify.

    Returns:
        Language code ('yor', 'hau', 'ibo', 'pcm', 'eng') or None for empty text.

    Example:
        >>> detect_language("Ọjọ́ náà dára púpọ̀")
        'yor'
        >>> detect_language("Wetin dey happen?")
        'pcm'
        >>> detect_language("The weather is nice today")
        'eng'
    """
    if not text or not text.strip():
        return None

    model = _get_model()
    return model.predict(text)


def detect_language_with_confidence(text: str) -> Tuple[Optional[str], float]:
    """Detect language with confidence score.

    Args:
        text: Input text to classify.

    Returns:
        Tuple of (language_code, confidence) where confidence is 0.0 to 1.0.

    Example:
        >>> lang, conf = detect_language_with_confidence("Ọjọ́ dára púpọ̀")
        >>> print(f"{lang}: {conf:.2f}")
        yor: 0.85
    """
    if not text or not text.strip():
        return None, 0.0

    model = _get_model()
    probs = model.predict_proba(text)
    best_lang = max(probs, key=probs.get)

    return best_lang, probs[best_lang]


def detect_all_languages(text: str) -> Dict[str, float]:
    """Get probability scores for all supported languages.

    Args:
        text: Input text to classify.

    Returns:
        Dict mapping language codes to probability scores (sum to 1.0).

    Example:
        >>> scores = detect_all_languages("Wetin dey happen?")
        >>> print(scores)
        {'yor': 0.05, 'hau': 0.10, 'ibo': 0.05, 'pcm': 0.70, 'eng': 0.10}
    """
    if not text or not text.strip():
        return {lang: 1.0 / len(SUPPORTED_LANGUAGES) for lang in SUPPORTED_LANGUAGES}

    model = _get_model()
    probs = model.predict_proba(text)

    # Ensure all supported languages are in the result
    result = {lang: 0.0 for lang in SUPPORTED_LANGUAGES}
    result.update(probs)

    # Renormalize in case of mismatch
    total = sum(result.values())
    if total > 0:
        result = {lang: score / total for lang, score in result.items()}
    else:
        result = {lang: 1.0 / len(SUPPORTED_LANGUAGES) for lang in SUPPORTED_LANGUAGES}

    return result


def train_and_save_model(path: Optional[Path] = None) -> Path:
    """Train a new model from datasets and save it.

    Call this after downloading datasets to create an optimized model.

    Args:
        path: Path to save the model. Defaults to bundled location.

    Returns:
        Path where the model was saved.
    """
    global _MODEL

    if path is None:
        from naijaml.utils.download import get_models_cache_dir
        path = get_models_cache_dir() / "lang_model.json"

    # Try to collect training data from datasets
    try:
        texts, labels = _collect_training_data()
    except Exception as e:
        logger.warning("Could not collect from datasets: %s", e)
        texts, labels = _get_fallback_training_data()

    if len(texts) < 10:
        texts, labels = _get_fallback_training_data()

    # Train new model
    model = NaiveBayesLangDetector()
    model.fit(texts, labels)
    model.save(path)

    # Update cached model
    _MODEL = model

    return path


# Backwards compatibility alias
train_and_save_profiles = train_and_save_model
