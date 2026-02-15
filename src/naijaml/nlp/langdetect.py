"""Nigerian language detection using Naive Bayes classifier.

Detects Yorùbá (yor), Hausa (hau), Igbo (ibo), Nigerian Pidgin (pcm),
and English (eng) from text using a Multinomial Naive Bayes classifier
trained on character n-gram features from real Nigerian NLP datasets.

Features:
- Character n-grams (1-4) for broad language patterns
- Character-set detection (diacritics as strong language signals)
- Pidgin vocabulary features for Pidgin vs English disambiguation

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

# N-gram sizes to use (unigrams through quad-grams)
_NGRAM_SIZES = [1, 2, 3, 4]

# Minimum n-gram count to include in model (prune rare n-grams)
_MIN_NGRAM_COUNT = 2

# =============================================================================
# Language-specific character sets and vocabulary
# =============================================================================

# Characters that are near-definitive signals for specific languages
_YORUBA_CHARS = set("ẹọṣẸỌṢ")
_YORUBA_TONE_CHARS = set("áàéèíìóòúùÁÀÉÈÍÌÓÒÚÙ")
_IGBO_CHARS = set("ịụỊỤ")  # ọ is shared with Yoruba, ị and ụ are Igbo-specific
_HAUSA_CHARS = set("ɓɗƙƁƊƘ")

# Pidgin-exclusive vocabulary (words that strongly signal Pidgin, not English)
_PIDGIN_WORDS = {
    "dey", "wetin", "abeg", "sef", "wahala", "sha", "shey", "pikin",
    "una", "naim", "abi", "jare", "shaa", "ehen", "ehn", "chop",
    "gist", "joor", "oya", "comot", "yarn", "kain", "wey", "dem",
    "sabi", "palava", "vex", "jeje", "kasala", "anyhow", "omo",
    "bros", "sista", "boda", "madam", "oga", "waka", "dodo",
    "suya", "na", "no be", "no fit", "make we", "i dey", "e don",
}

# Pidgin bigrams (two-word patterns) that are strong signals
_PIDGIN_BIGRAMS = {
    # "dey" constructions (progressive/habitual marker)
    "no dey", "i dey", "dey do", "dey go", "dey come",
    "you dey", "we dey", "dem dey", "e dey", "go dey",
    # "no" negation (without auxiliary verb — Pidgin grammar)
    "no fit", "no be", "no go",
    # "make" constructions (subjunctive/imperative)
    "make we", "make i", "make dem",
    # "na" constructions (copula/focus marker)
    "na im", "na so", "na wa",
    # "don" constructions (perfective aspect)
    "e don", "dem don",
    # Question/exclamation patterns
    "wetin dey", "wetin be", "abeg no",
    # Locative
    "for here", "for there",
    # Verb + "am" (object pronoun = "it/him/her" — Pidgin grammar)
    "cook am", "tell am", "carry am", "kill am", "give am",
    "do am", "see am", "chop am", "leave am", "bring am",
    "buy am", "sell am", "take am", "put am", "get am",
    # Adjective/verb + "die" (intensifier = "very/extremely")
    "sweet die", "fine die", "cold die", "hot die",
    "hungry die", "tire die", "hard die",
    # "no" + adjective (Pidgin negation without copula)
    "no easy", "no good", "no sweet",
}


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
    ngrams = Counter()  # type: Counter
    for n in _NGRAM_SIZES:
        for i in range(len(text) - n + 1):
            ngram = text[i:i + n]
            ngrams[ngram] += 1
    return ngrams


def _detect_char_features(text: str) -> Dict[str, int]:
    """Detect language-specific character features.

    Returns a dict of binary feature names to counts.
    These features are injected into the n-gram feature vector.
    """
    text_norm = unicodedata.normalize("NFC", text)
    features = {}  # type: Dict[str, int]

    # Count language-specific characters
    yor_count = sum(1 for c in text_norm if c in _YORUBA_CHARS)
    yor_tone_count = sum(1 for c in text_norm if c in _YORUBA_TONE_CHARS)
    igbo_count = sum(1 for c in text_norm if c in _IGBO_CHARS)
    hau_count = sum(1 for c in text_norm if c in _HAUSA_CHARS)

    # Add as features (scaled to be significant in the n-gram model)
    if yor_count > 0:
        features["__FEAT_YOR_DOTBELOW__"] = yor_count * 5
    if yor_tone_count > 0:
        features["__FEAT_YOR_TONE__"] = yor_tone_count * 3
    if igbo_count > 0:
        features["__FEAT_IBO_DOTBELOW__"] = igbo_count * 5
    if hau_count > 0:
        features["__FEAT_HAU_HOOK__"] = hau_count * 5

    # Pidgin vocabulary detection (strip punctuation so "Omo," matches "omo")
    words = set(w.strip(".,!?;:'\"()-") for w in text_norm.lower().split())
    pidgin_hits = len(words & _PIDGIN_WORDS)
    if pidgin_hits >= 1:
        features["__FEAT_PCM_VOCAB__"] = pidgin_hits * 5

    # Pidgin bigram detection
    text_lower = text_norm.lower()
    pidgin_bigram_hits = sum(1 for bg in _PIDGIN_BIGRAMS if bg in text_lower)
    if pidgin_bigram_hits >= 1:
        features["__FEAT_PCM_BIGRAM__"] = pidgin_bigram_hits * 5

    return features


def _extract_features(text: str) -> Counter:
    """Extract all features: n-grams + character-set + vocabulary features."""
    features = _extract_ngrams(text)
    char_features = _detect_char_features(text)
    features.update(char_features)
    return features


# =============================================================================
# Naive Bayes Language Detector
# =============================================================================

class NaiveBayesLangDetector:
    """Multinomial Naive Bayes classifier for language detection.

    Uses character n-gram features with Laplace smoothing, augmented with
    character-set detection and vocabulary features.

    The classifier computes:
        P(lang | text) ∝ P(lang) × ∏ P(feature | lang)

    Using log probabilities for numerical stability:
        log P(lang | text) = log P(lang) + Σ log P(feature | lang)
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
        self.log_priors = {}  # type: Dict[str, float]
        self.log_likelihoods = {}  # type: Dict[str, Dict[str, float]]
        self.vocab = set()  # type: Set[str]
        self._default_log_likelihood = {}  # type: Dict[str, float]

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
        label_counts = Counter(labels)  # type: Counter
        total_samples = len(labels)
        num_classes = len(label_counts)

        # Compute log priors (uniform or from data)
        if self.uniform_priors:
            self.log_priors = {
                lang: math.log(1.0 / num_classes)
                for lang in label_counts
            }
        else:
            self.log_priors = {
                lang: math.log(count / total_samples)
                for lang, count in label_counts.items()
            }

        # Aggregate feature counts per language
        feature_counts = {lang: Counter() for lang in label_counts}  # type: Dict[str, Counter]
        total_features = {lang: 0 for lang in label_counts}  # type: Dict[str, int]

        for text, label in zip(texts, labels):
            features = _extract_features(text)
            feature_counts[label].update(features)
            total_features[label] += sum(features.values())
            self.vocab.update(features.keys())

        # Prune rare features (keep only those with count >= _MIN_NGRAM_COUNT across all languages)
        global_counts = Counter()  # type: Counter
        for lang_counts in feature_counts.values():
            global_counts.update(lang_counts)

        self.vocab = {f for f, count in global_counts.items() if count >= _MIN_NGRAM_COUNT}

        # Always keep special features even if rare
        special_features = {f for f in global_counts if f.startswith("__FEAT_")}
        self.vocab.update(special_features)

        vocab_size = len(self.vocab)
        logger.info("Vocabulary size after pruning: %d", vocab_size)

        # Compute log likelihoods with Laplace smoothing
        self.log_likelihoods = {}
        self._default_log_likelihood = {}

        for lang in label_counts:
            self.log_likelihoods[lang] = {}
            denominator = total_features[lang] + self.alpha * vocab_size

            # Default log-likelihood for unseen features
            self._default_log_likelihood[lang] = math.log(self.alpha / denominator)

            for feature in self.vocab:
                count = feature_counts[lang].get(feature, 0)
                self.log_likelihoods[lang][feature] = math.log(
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
        features = _extract_features(text)

        scores = {}
        for lang, log_prior in self.log_priors.items():
            log_likelihood = 0.0
            lang_likelihoods = self.log_likelihoods[lang]
            default_ll = self._default_log_likelihood[lang]

            for feature, count in features.items():
                ll = lang_likelihoods.get(feature, default_ll)
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
                detector._default_log_likelihood[lang] = -15.0

        return detector


# =============================================================================
# Training data collection
# =============================================================================

def _get_english_corpus() -> List[str]:
    """Get a built-in English corpus for training.

    Large enough to balance against thousands of Pidgin samples,
    since English and Pidgin share vocabulary and the model needs
    sufficient English examples to learn the distinction.
    """
    return [
        # News / formal
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
        # More news / current affairs
        "The central bank has raised interest rates to combat rising inflation.",
        "A delegation of foreign diplomats arrived in the capital for talks.",
        "The agriculture minister announced new subsidies for smallholder farmers.",
        "Police have arrested three suspects in connection with the robbery.",
        "The newly elected governor has pledged to improve road infrastructure.",
        "Thousands of students graduated from the university this weekend.",
        "The court has ordered a retrial after new evidence was presented.",
        "Health officials confirmed the outbreak has been contained.",
        "The telecommunications company announced expansion to rural communities.",
        "A new report highlights the growing gap between rich and poor.",
        "The minister of finance presented the annual budget to parliament.",
        "Scientists warn that deforestation is accelerating at an alarming rate.",
        "The election commission has announced the date for local government polls.",
        "A trade agreement between the two countries was signed yesterday.",
        "The national football team qualified for the continental championship.",
        "Experts recommend investing in education to reduce unemployment.",
        "The power company has promised to address frequent blackouts.",
        "A magnitude five earthquake was recorded off the eastern coast.",
        "The president has appointed new members to the economic advisory council.",
        "Several major roads were closed due to flooding after heavy rains.",
        # Business / technology
        "The startup raised ten million dollars in its latest funding round.",
        "Artificial intelligence is transforming how businesses operate globally.",
        "The quarterly earnings report exceeded analyst expectations significantly.",
        "Cloud computing has become essential for modern enterprise infrastructure.",
        "The merger between the two banks will create the largest financial institution.",
        "Cybersecurity threats have increased dramatically in the past year.",
        "The new smartphone features an improved camera and longer battery life.",
        "Investors are increasingly interested in sustainable and green technologies.",
        "The supply chain disruptions have caused significant delays in manufacturing.",
        "Digital payment platforms are gaining widespread adoption across Africa.",
        "The company plans to hire five hundred new employees by the end of the year.",
        "Machine learning algorithms are being used to detect fraud in banking.",
        "The tech conference attracted over ten thousand attendees from around the world.",
        "Remote work tools have seen a massive surge in demand since the pandemic.",
        "The electric vehicle market is projected to double in size by next year.",
        "Blockchain technology is being explored for secure land registry systems.",
        "The software update addresses several critical security vulnerabilities.",
        "Data privacy regulations are becoming stricter across many jurisdictions.",
        "The robotics laboratory has developed a new prototype for warehouse automation.",
        "Venture capital investments in African startups reached a record high.",
        # Sports
        "The defending champions were eliminated in the quarterfinal round.",
        "The young athlete broke the national record in the hundred meter sprint.",
        "The coach announced the final squad for the upcoming tournament.",
        "The match was suspended due to poor weather conditions.",
        "The team has won five consecutive games and sits atop the league table.",
        "The transfer window closes at midnight and several deals are pending.",
        "The Olympic committee has selected the host city for the next games.",
        "The boxer won the title fight by unanimous decision of the judges.",
        "Tennis rankings were updated after the completion of the grand slam.",
        "The cricket team declared their innings at three hundred and fifty runs.",
        # Everyday / conversational
        "Could you please pass me the salt from the other end of the table?",
        "I think we should leave early to avoid the traffic on the highway.",
        "Have you seen the new documentary about marine conservation?",
        "The grocery store is closed on Sundays but opens early on Monday.",
        "My sister is studying medicine at the university in the capital.",
        "We need to schedule a meeting with the entire project team.",
        "The library has extended its opening hours during the exam period.",
        "I forgot my umbrella at home and it started raining on my way.",
        "The new shopping mall has a wide variety of stores and restaurants.",
        "We should plan our vacation before all the good hotels are booked.",
        "The plumber came to fix the leaking pipe in the bathroom.",
        "She received a scholarship to study abroad for her master's degree.",
        "The neighbors are having a party and the music is quite loud.",
        "I have been trying to reach him by phone but he is not answering.",
        "The traffic was terrible this morning because of the road construction.",
        "We are planning to renovate the kitchen and add new cabinets.",
        "The flight was delayed by three hours due to mechanical issues.",
        "He volunteered to help organize the community fundraising event.",
        "The doctor recommended drinking more water and getting enough sleep.",
        "They moved to a new apartment closer to the children's school.",
        # Science / academic
        "The research team published their findings in a peer-reviewed journal.",
        "Genetic studies have revealed new insights about human migration patterns.",
        "The experiment demonstrated that the hypothesis was statistically significant.",
        "Mathematical models can help predict the spread of infectious diseases.",
        "The conference proceedings include papers from researchers in twenty countries.",
        "Archaeological excavations uncovered artifacts dating back several centuries.",
        "The vaccine trial showed a ninety percent efficacy rate in preventing infection.",
        "Satellite imagery reveals the extent of glacier retreat over the past decade.",
        "The chemistry department received a grant for advanced materials research.",
        "Computational methods have accelerated drug discovery in pharmaceutical research.",
        # Education
        "The curriculum has been updated to include digital literacy skills.",
        "Students are required to complete a capstone project before graduation.",
        "The scholarship program supports talented students from disadvantaged backgrounds.",
        "Teacher training workshops focus on innovative classroom techniques.",
        "The school board approved the construction of two new primary schools.",
        "Academic performance has improved significantly since the new program started.",
        "The literacy rate has increased due to government investment in education.",
        "Examination results will be released at the end of the month.",
        "The university offers online courses for working professionals.",
        "Parents are encouraged to participate actively in their children's education.",
        # Health / lifestyle
        "A balanced diet combined with regular exercise promotes overall wellness.",
        "The clinic offers free screening for common chronic diseases.",
        "Mental health awareness campaigns are reaching more people every year.",
        "Proper handwashing technique is one of the most effective ways to prevent illness.",
        "The new fitness center has state of the art equipment and personal trainers.",
        "Adequate sleep is essential for cognitive function and emotional well-being.",
        "Nutritionists recommend eating at least five servings of fruits daily.",
        "The community health program provides vaccinations for children under five.",
        "Meditation and mindfulness practices can help reduce stress and anxiety.",
        "The hospital introduced a new patient management system to reduce wait times.",
        # Culture / entertainment
        "The art gallery is hosting an exhibition of contemporary African paintings.",
        "The bestselling novel has been translated into more than thirty languages.",
        "The music festival lineup includes both local and international artists.",
        "The documentary won several awards at the international film festival.",
        "Traditional crafts are being preserved through community workshop programs.",
        "The theater company will perform Shakespeare's plays throughout the month.",
        "The photography competition received over two thousand entries this year.",
        "The cultural heritage site has been nominated for preservation status.",
        "The animation studio released its first feature length film last week.",
        "The poetry reading event attracted a diverse and enthusiastic audience.",
        # Short informal English (to distinguish from Pidgin)
        "What time does the meeting start?",
        "I will be there in ten minutes.",
        "Can you send me the report by Friday?",
        "That sounds like a great idea.",
        "I am not sure if that is correct.",
        "Let me know when you are ready.",
        "Do you have any questions about the assignment?",
        "I appreciate your help with this matter.",
        "The project deadline has been extended by one week.",
        "We should discuss this further at tomorrow's meeting.",
        "I completely agree with your assessment of the situation.",
        "The results were better than we had originally expected.",
        "Please review the document and provide your feedback.",
        "I will follow up with them about the pending request.",
        "The workshop has been rescheduled to next Thursday afternoon.",
        "It was a pleasure meeting you at the conference last week.",
        "We need to reconsider our approach to this particular problem.",
        "The committee will announce their decision within two weeks.",
        "I suggest we explore alternative solutions before making a final choice.",
        "Thank you for bringing this important issue to our attention.",
    ]


def _collect_training_data(max_per_lang: int = 5000) -> Tuple[List[str], List[str]]:
    """Collect training data from NaijaML datasets + fallback.

    Pulls from multiple domains (tweets, news, reviews) for robustness.
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
    lang_data = {lang: [] for lang in SUPPORTED_LANGUAGES}  # type: Dict[str, List[str]]

    # Get fallback data first (always included as baseline)
    fallback_texts, fallback_labels = _get_fallback_training_data()
    for text, label in zip(fallback_texts, fallback_labels):
        lang_data[label].append(text)
    logger.info("Loaded fallback data")

    # Load from multiple datasets for domain diversity
    for lang in ["yor", "hau", "ibo", "pcm"]:
        # NaijaSenti (tweets — primary source, short text)
        for split in ["train", "validation"]:
            if is_cached("naijasenti", lang, split):
                try:
                    data = load_dataset("naijasenti", lang=lang, split=split)
                    lang_data[lang].extend(item["text"] for item in data if item.get("text"))
                except Exception as e:
                    logger.debug("Failed to load naijasenti %s/%s: %s", lang, split, e)

        # MasakhaNEWS (news articles — longer text)
        for split in ["train", "validation"]:
            if is_cached("masakhanews", lang, split):
                try:
                    data = load_dataset("masakhanews", lang=lang, split=split)
                    lang_data[lang].extend(item["text"] for item in data if item.get("text"))
                except Exception as e:
                    logger.debug("Failed to load masakhanews %s/%s: %s", lang, split, e)

        # NollySenti (movie reviews — different domain)
        for split in ["train", "validation"]:
            if is_cached("nollysenti", lang, split):
                try:
                    data = load_dataset("nollysenti", lang=lang, split=split)
                    lang_data[lang].extend(item["text"] for item in data if item.get("text"))
                except Exception as e:
                    logger.debug("Failed to load nollysenti %s/%s: %s", lang, split, e)

        # MasakhaNER (NER sentences — yet another domain)
        if lang in ["yor", "hau", "ibo"]:
            for split in ["train", "validation"]:
                if is_cached("masakhaner", lang, split):
                    try:
                        data = load_dataset("masakhaner", lang=lang, split=split)
                        for item in data:
                            if "tokens" in item:
                                lang_data[lang].append(" ".join(item["tokens"]))
                    except Exception as e:
                        logger.debug("Failed to load masakhaner %s/%s: %s", lang, split, e)

    # English from MasakhaNEWS — load directly from HuggingFace since our loader
    # doesn't support 'eng'. Use individual sentences from articles to get diverse,
    # shorter training samples that overlap in style with tweets/conversational text.
    try:
        from datasets import load_dataset as hf_load_dataset
        for split in ["train", "validation"]:
            try:
                ds = hf_load_dataset("masakhane/masakhanews", "eng", split=split)
                for item in ds:
                    # Use headline (short, tweet-length)
                    headline = item.get("headline", "")
                    if headline and len(headline) > 20:
                        lang_data["eng"].append(headline)
                    # Split article into individual sentences for shorter,
                    # more diverse training samples
                    text = item.get("text", "")
                    if text:
                        for sentence in text.replace("\n", ". ").split(". "):
                            sentence = sentence.strip()
                            if len(sentence) > 30 and len(sentence) < 300:
                                lang_data["eng"].append(sentence)
                logger.info("Loaded English from MasakhaNEWS %s: %d items",
                           split, len(ds))
            except Exception as e:
                logger.debug("Failed to load MasakhaNEWS eng/%s: %s", split, e)
    except ImportError:
        logger.debug("datasets library not available for English data")

    # Filter Pidgin training data: only keep texts with STRONG Pidgin markers.
    # NaijaSenti "Pidgin" tweets are often heavily code-mixed with English,
    # which teaches the model that any informal English text is Pidgin.
    # Requiring multiple markers keeps genuinely Pidgin text and discards
    # English-dominant code-mixed text.
    if lang_data.get("pcm"):
        filtered_pcm = []
        for text in lang_data["pcm"]:
            text_lower = text.lower()
            words = set(w.strip(".,!?;:'\"()-") for w in text_lower.split())
            pcm_word_hits = len(words & _PIDGIN_WORDS)
            pcm_bigram_hits = sum(1 for bg in _PIDGIN_BIGRAMS if bg in text_lower)
            total_markers = pcm_word_hits + pcm_bigram_hits
            if total_markers >= 2:  # Need at least 2 Pidgin markers
                filtered_pcm.append(text)
        logger.info("Filtered Pidgin: %d -> %d (kept texts with 2+ Pidgin markers)",
                    len(lang_data["pcm"]), len(filtered_pcm))
        lang_data["pcm"] = filtered_pcm

    # Balance data by sampling max_per_lang from each language
    texts = []  # type: List[str]
    labels = []  # type: List[str]

    for lang, lang_texts in lang_data.items():
        # Deduplicate
        seen = set()  # type: Set[str]
        unique_texts = []
        for t in lang_texts:
            t_stripped = t.strip()
            if t_stripped and t_stripped not in seen:
                seen.add(t_stripped)
                unique_texts.append(t_stripped)

        if len(unique_texts) > max_per_lang:
            random.seed(42)
            sampled = random.sample(unique_texts, max_per_lang)
        else:
            sampled = unique_texts

        texts.extend(sampled)
        labels.extend([lang] * len(sampled))
        logger.info("Using %d samples for %s (total available: %d)",
                    len(sampled), lang, len(unique_texts))

    return texts, labels


def _get_fallback_training_data() -> Tuple[List[str], List[str]]:
    """Get fallback training data when no datasets are cached."""
    texts = []
    labels = []

    # Yorùbá samples with diacritics
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

    # Hausa samples
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

    # Igbo samples
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

    # Nigerian Pidgin samples — distinctive from English
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
        "wetin you dey find for here",
        "wetin you wan make i do",
        "wetin happen to your phone",
        "wetin be the matter sef",
        "wetin you go tell am",
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
    prediction = model.predict(text)

    # Pidgin/English disambiguation: Pidgin is an English-based creole, so
    # their n-gram distributions overlap almost entirely. The NB model alone
    # cannot reliably separate them. We apply the linguistic constraint that
    # Pidgin requires positive evidence (Pidgin-specific vocabulary/grammar).
    # If the model predicts Pidgin but the text contains no Pidgin markers,
    # reclassify using the next-best language (usually English).
    if prediction == "pcm":
        text_lower = text.lower()
        words = set(w.strip(".,!?;:'\"()-") for w in text_lower.split())
        has_pcm_word = bool(words & _PIDGIN_WORDS)
        has_pcm_bigram = any(bg in text_lower for bg in _PIDGIN_BIGRAMS)
        if not has_pcm_word and not has_pcm_bigram:
            # No Pidgin markers — use second-best prediction
            scores = model._compute_log_posteriors(text)
            sorted_langs = sorted(scores, key=scores.get, reverse=True)
            prediction = sorted_langs[1]  # Second-best

    return prediction


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

    # Use detect_language which includes Pidgin/English disambiguation
    lang = detect_language(text)
    model = _get_model()
    probs = model.predict_proba(text)

    return lang, probs.get(lang, 0.0)


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
