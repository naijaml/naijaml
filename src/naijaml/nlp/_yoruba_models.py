"""Yorùbá diacritizer models: syllable k-NN, word-level lookup with Viterbi, and dot-below only."""
from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from naijaml.nlp._yoruba_text import (
    normalize_yoruba,
    strip_diacritics,
    syllabify,
    syllabify_text,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Diacritizer model
# =============================================================================

class YorubaDiacritizer:
    """Syllable-based k-NN diacritizer for Yorùbá.

    Uses context (surrounding syllables) to predict the diacritized form
    of each syllable.
    """

    def __init__(self):
        """Initialize an empty diacritizer."""
        # syllable_undiac -> {context -> {diacritized -> count}}
        self.syllable_db: Dict[str, Dict[Tuple, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        # syllable_undiac -> {diacritized -> count} (context-free fallback)
        self.syllable_freq: Dict[str, Counter] = defaultdict(Counter)
        # Total syllables seen
        self.total_syllables = 0

    def train(self, diacritized_texts: List[str]) -> "YorubaDiacritizer":
        """Train the diacritizer on diacritized Yorùbá texts.

        Args:
            diacritized_texts: List of properly diacritized Yorùbá sentences.

        Returns:
            self for method chaining.
        """
        for text in diacritized_texts:
            tokens = syllabify_text(text)

            # Extract words and syllabify them
            words_syllables = []
            for token, is_word in tokens:
                if is_word:
                    syls = syllabify(token)
                    words_syllables.extend(syls)
                else:
                    # Keep separators as context markers
                    if token.strip():  # Non-whitespace separator
                        words_syllables.append(token)

            # Build context-aware database
            for i, syl in enumerate(words_syllables):
                if not syl or not any(c.isalpha() for c in syl):
                    continue

                undiac = strip_diacritics(syl).lower()

                # Context: previous and next syllables
                prev_syl = words_syllables[i - 1] if i > 0 else "<START>"
                next_syl = words_syllables[i + 1] if i + 1 < len(words_syllables) else "<END>"

                prev_undiac = strip_diacritics(prev_syl).lower() if prev_syl not in ("<START>", "<END>") else prev_syl
                next_undiac = strip_diacritics(next_syl).lower() if next_syl not in ("<START>", "<END>") else next_syl

                context = (prev_undiac, next_undiac)

                # Store with context
                self.syllable_db[undiac][context][syl.lower()] += 1

                # Store frequency (context-free)
                self.syllable_freq[undiac][syl.lower()] += 1
                self.total_syllables += 1

        logger.info("Trained on %d syllables, %d unique undiacritized forms",
                    self.total_syllables, len(self.syllable_freq))

        return self

    def predict_syllable(
        self,
        syllable: str,
        prev_context: str = "<START>",
        next_context: str = "<END>",
    ) -> str:
        """Predict the diacritized form of a syllable.

        Args:
            syllable: Undiacritized syllable.
            prev_context: Previous syllable (undiacritized) or <START>.
            next_context: Next syllable (undiacritized) or <END>.

        Returns:
            Most likely diacritized form.
        """
        undiac = strip_diacritics(syllable).lower()
        original_case = syllable[0].isupper() if syllable else False

        # Try exact context match
        context = (prev_context.lower(), next_context.lower())
        if undiac in self.syllable_db and context in self.syllable_db[undiac]:
            result = self.syllable_db[undiac][context].most_common(1)[0][0]
            return result.capitalize() if original_case else result

        # Try partial context (prev only)
        for ctx, counts in self.syllable_db.get(undiac, {}).items():
            if ctx[0] == context[0]:
                result = counts.most_common(1)[0][0]
                return result.capitalize() if original_case else result

        # Try partial context (next only)
        for ctx, counts in self.syllable_db.get(undiac, {}).items():
            if ctx[1] == context[1]:
                result = counts.most_common(1)[0][0]
                return result.capitalize() if original_case else result

        # Fall back to most frequent (context-free)
        if undiac in self.syllable_freq:
            result = self.syllable_freq[undiac].most_common(1)[0][0]
            return result.capitalize() if original_case else result

        # Unknown syllable - return as-is
        return syllable

    def diacritize(self, text: str) -> str:
        """Restore diacritics to undiacritized Yorùbá text.

        Args:
            text: Undiacritized Yorùbá text.

        Returns:
            Text with diacritics restored.

        Example:
            >>> diacritizer.diacritize("Ojo dara")
            'Ọjọ́ dára'
        """
        tokens = syllabify_text(text)
        result = []

        # First pass: collect all syllables with positions
        all_syllables = []  # (syllable, token_idx, syl_idx, is_word)

        for token_idx, (token, is_word) in enumerate(tokens):
            if is_word:
                syls = syllabify(token)
                for syl_idx, syl in enumerate(syls):
                    all_syllables.append((syl, token_idx, syl_idx, True))
            else:
                all_syllables.append((token, token_idx, 0, False))

        # Second pass: predict diacritics with context
        predicted = []
        word_syllables = [(s, i) for i, (s, _, _, is_word) in enumerate(all_syllables) if is_word]

        for idx, (syl, token_idx, syl_idx, is_word) in enumerate(all_syllables):
            if not is_word:
                predicted.append(syl)
                continue

            # Find position in word_syllables
            word_idx = next(i for i, (_, orig_idx) in enumerate(word_syllables) if orig_idx == idx)

            # Get context from word syllables only
            prev_ctx = strip_diacritics(word_syllables[word_idx - 1][0]).lower() if word_idx > 0 else "<START>"
            next_ctx = strip_diacritics(word_syllables[word_idx + 1][0]).lower() if word_idx + 1 < len(word_syllables) else "<END>"

            pred_syl = self.predict_syllable(syl, prev_ctx, next_ctx)
            predicted.append(pred_syl)

        # Third pass: reconstruct words
        result = []
        current_word = []
        last_token_idx = -1

        for i, (syl, token_idx, syl_idx, is_word) in enumerate(all_syllables):
            if is_word:
                if token_idx != last_token_idx:
                    if current_word:
                        result.append("".join(current_word))
                        current_word = []
                current_word.append(predicted[i])
                last_token_idx = token_idx
            else:
                if current_word:
                    result.append("".join(current_word))
                    current_word = []
                result.append(predicted[i])
                last_token_idx = -1

        if current_word:
            result.append("".join(current_word))

        return "".join(result)

    def save(self, path: Path) -> None:
        """Save the trained model to JSON.

        Args:
            path: Path to save the model.
        """
        # Convert defaultdicts and tuples to JSON-serializable format
        data = {
            "syllable_db": {
                undiac: {
                    f"{ctx[0]}|{ctx[1]}": dict(counts)
                    for ctx, counts in contexts.items()
                }
                for undiac, contexts in self.syllable_db.items()
            },
            "syllable_freq": {
                undiac: dict(counts)
                for undiac, counts in self.syllable_freq.items()
            },
            "total_syllables": self.total_syllables,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Saved diacritizer model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "YorubaDiacritizer":
        """Load a trained model from JSON.

        Args:
            path: Path to the model file.

        Returns:
            Loaded YorubaDiacritizer instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls()

        # Reconstruct data structures
        for undiac, contexts in data.get("syllable_db", {}).items():
            for ctx_str, counts in contexts.items():
                prev_ctx, next_ctx = ctx_str.split("|", 1)
                context = (prev_ctx, next_ctx)
                model.syllable_db[undiac][context] = Counter(counts)

        for undiac, counts in data.get("syllable_freq", {}).items():
            model.syllable_freq[undiac] = Counter(counts)

        model.total_syllables = data.get("total_syllables", 0)

        return model


# =============================================================================
# Word-Level Diacritizer (Primary Approach)
# =============================================================================

class WordLevelDiacritizer:
    """Word-level lookup diacritizer for Yorùbá.

    Uses direct word-to-word mapping for known words, achieving ~83% accuracy
    with 99% vocabulary coverage. Falls back to syllable-based approach for
    unknown words.

    This approach significantly outperforms syllable-only methods because:
    1. Many words are ambiguous at syllable level but unambiguous at word level
    2. Word-level captures lexical patterns that syllables miss
    3. High-frequency words dominate real text, and they're well-covered
    """

    def __init__(self, min_word_freq: int = 5, min_bigram_freq: int = 3,
                 max_candidates: int = 5):
        """Initialize the word-level diacritizer.

        Args:
            min_word_freq: Minimum frequency threshold for including a word
                          in the lookup table. Words seen fewer times are
                          handled by syllable fallback. Default 5.
            min_bigram_freq: Minimum frequency threshold for including a bigram
                            in the bigram lookup. Default 3.
            max_candidates: Maximum number of candidate diacritized forms to
                           keep per ambiguous word for Viterbi decoding. Default 5.
        """
        self.min_word_freq = min_word_freq
        self.min_bigram_freq = min_bigram_freq
        self.max_candidates = max_candidates
        # undiacritized_word -> diacritized_word (most common form)
        self.word_map: Dict[str, str] = {}
        # bigram context: "prev_undiac\tcurrent_undiac" -> diacritized form
        # Only stores entries where bigram prediction differs from unigram
        self.bigram_map: Dict[str, str] = {}
        self.bigram_count: int = 0
        # Statistics
        self.total_words_trained = 0
        self.vocab_size = 0
        # Syllable fallback model (lazy loaded)
        self._syllable_model: Optional[YorubaDiacritizer] = None
        # Viterbi decoding data structures
        # Top-K candidates per ambiguous word: undiac -> [(diac_form, log_prob), ...]
        self.word_candidates: Dict[str, List[Tuple[str, float]]] = {}
        # Transition log-probs: diac_prev -> {diac_current -> log_prob}
        self.transition_probs: Dict[str, Dict[str, float]] = {}
        # Unigram log-probs (back-off for unseen transitions)
        self.unigram_log_probs: Dict[str, float] = {}
        self.has_viterbi: bool = False

    def train(
        self,
        diacritized_texts: List[str],
        undiacritized_texts: Optional[List[str]] = None,
    ) -> "WordLevelDiacritizer":
        """Train the word-level diacritizer.

        Args:
            diacritized_texts: List of properly diacritized Yorùbá sentences.
            undiacritized_texts: Optional list of corresponding undiacritized
                                sentences. If None, will be generated by
                                stripping diacritics from diacritized_texts.

        Returns:
            self for method chaining.
        """
        # Build word frequency table: undiac -> {diac -> count}
        word_lookup: Dict[str, Counter] = defaultdict(Counter)
        # Build bigram frequency table: "prev\tcurrent" -> {diac -> count}
        bigram_lookup: Dict[str, Counter] = defaultdict(Counter)

        for i, diac_text in enumerate(diacritized_texts):
            # Get undiacritized version
            if undiacritized_texts is not None and i < len(undiacritized_texts):
                undiac_text = undiacritized_texts[i]
            else:
                undiac_text = strip_diacritics(diac_text)

            diac_words = diac_text.split()
            undiac_words = undiac_text.split()

            prev_undiac = None
            for undiac, diac in zip(undiac_words, diac_words):
                undiac_lower = undiac.lower()
                diac_lower = diac.lower()

                word_lookup[undiac_lower][diac_lower] += 1
                self.total_words_trained += 1

                # Collect bigrams
                if prev_undiac is not None:
                    bigram_key = prev_undiac + "\t" + undiac_lower
                    bigram_lookup[bigram_key][diac_lower] += 1

                prev_undiac = undiac_lower

        # Build word map: keep only words above frequency threshold
        for undiac, diac_counts in word_lookup.items():
            total_count = sum(diac_counts.values())
            if total_count >= self.min_word_freq:
                # Use the most common diacritized form
                self.word_map[undiac] = diac_counts.most_common(1)[0][0]

        self.vocab_size = len(self.word_map)

        # Build bigram map: only store disambiguating bigrams
        # (where bigram prediction differs from unigram default)
        for bigram_key, diac_counts in bigram_lookup.items():
            total_count = sum(diac_counts.values())
            if total_count < self.min_bigram_freq:
                continue

            best_bigram = diac_counts.most_common(1)[0][0]
            current_undiac = bigram_key.split("\t", 1)[1]
            unigram_default = self.word_map.get(current_undiac)

            # Only store if bigram disagrees with unigram
            if unigram_default is not None and best_bigram != unigram_default:
                self.bigram_map[bigram_key] = best_bigram

        self.bigram_count = len(self.bigram_map)

        # =================================================================
        # Build Viterbi data structures
        # =================================================================

        # 1. Build word_candidates: top-K candidates per ambiguous word
        for undiac, diac_counts in word_lookup.items():
            total_count = sum(diac_counts.values())
            if total_count < self.min_word_freq:
                continue
            candidates = diac_counts.most_common()
            if len(candidates) > 1:
                # Ambiguous word — store top-K with log probabilities
                top_k = candidates[:self.max_candidates]
                self.word_candidates[undiac] = [
                    (normalize_yoruba(form), math.log(count / total_count))
                    for form, count in top_k
                ]

        # 2. Build diacritized bigram counts (transitions between diac forms)
        diac_bigram_counts: Dict[str, Counter] = defaultdict(Counter)
        diac_unigram_counts: Counter = Counter()

        for i, diac_text in enumerate(diacritized_texts):
            diac_words = diac_text.split()
            prev_diac = None
            for diac in diac_words:
                diac_lower = normalize_yoruba(diac.lower())
                diac_unigram_counts[diac_lower] += 1
                if prev_diac is not None:
                    diac_bigram_counts[prev_diac][diac_lower] += 1
                prev_diac = diac_lower

        # Collect all diacritized forms that Viterbi can produce
        known_diac_forms = set(self.word_map.values())
        for candidates in self.word_candidates.values():
            for form, _ in candidates:
                known_diac_forms.add(form)

        # 3. Compute unigram log-probs (only for forms Viterbi can produce)
        total_unigram = sum(diac_unigram_counts.values())
        if total_unigram > 0:
            for form in known_diac_forms:
                count = diac_unigram_counts.get(form, 0)
                if count > 0:
                    self.unigram_log_probs[form] = math.log(count / total_unigram)

        # 4. Compute transition log-probs (sparse, only between known forms)
        # Keep only top-N transitions per source to bound model size.
        max_transitions_per_source = 10

        ambiguous_forms = set()
        for candidates in self.word_candidates.values():
            for form, _ in candidates:
                ambiguous_forms.add(form)

        for prev_diac, next_counts in diac_bigram_counts.items():
            if prev_diac not in known_diac_forms:
                continue
            # Only store transitions involving at least one ambiguous form
            relevant = [
                (nd, c) for nd, c in next_counts.items()
                if nd in known_diac_forms and (
                    prev_diac in ambiguous_forms or nd in ambiguous_forms
                )
            ]
            if not relevant:
                continue
            # Keep top-N by count
            relevant.sort(key=lambda x: x[1], reverse=True)
            relevant = relevant[:max_transitions_per_source]
            total_from_prev = sum(next_counts.values())
            transitions = {}
            for next_diac, count in relevant:
                transitions[next_diac] = math.log(count / total_from_prev)
            if transitions:
                self.transition_probs[prev_diac] = transitions

        self.has_viterbi = True

        logger.info(
            "Trained word-level model: %d words in vocabulary (min_freq=%d), "
            "%d disambiguating bigrams (min_freq=%d), "
            "%d total word occurrences, "
            "%d ambiguous words with candidates, "
            "%d transition entries",
            self.vocab_size,
            self.min_word_freq,
            self.bigram_count,
            self.min_bigram_freq,
            self.total_words_trained,
            len(self.word_candidates),
            len(self.transition_probs),
        )

        return self

    def _get_syllable_fallback(self) -> YorubaDiacritizer:
        """Get the syllable-based fallback model."""
        if self._syllable_model is None:
            from naijaml.nlp.diacritizer import _get_model

            self._syllable_model = _get_model()
        return self._syllable_model

    def diacritize_word(self, word: str, prev_word: Optional[str] = None) -> str:
        """Diacritize a single word.

        Args:
            word: Undiacritized word.
            prev_word: Previous undiacritized word (lowercase) for bigram context.
                      If provided, bigram lookup is tried first.

        Returns:
            Diacritized word.
        """
        word_lower = word.lower()
        original_case = word[0].isupper() if word else False

        # Try bigram lookup first (prev_word + current_word)
        if prev_word is not None and self.bigram_map:
            bigram_key = prev_word + "\t" + word_lower
            if bigram_key in self.bigram_map:
                result = self.bigram_map[bigram_key]
                if original_case:
                    result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()
                return result

        # Try word-level lookup
        if word_lower in self.word_map:
            result = self.word_map[word_lower]
            # Preserve original capitalization
            if original_case:
                result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()
            return result

        # Fall back to syllable-based approach for unknown words
        fallback = self._get_syllable_fallback()
        return fallback.diacritize(word)

    def diacritize(self, text: str) -> str:
        """Restore diacritics to undiacritized Yorùbá text.

        Uses Viterbi decoding when available for globally optimal sequences,
        falling back to greedy bigram+unigram decoding for old models.

        Args:
            text: Undiacritized Yorùbá text.

        Returns:
            Text with diacritics restored.

        Example:
            >>> diacritizer.diacritize("Ojo dara pupo")
            'Ọjọ́ dára púpọ̀'
        """
        if not self.has_viterbi:
            return self._diacritize_greedy(text)

        text = normalize_yoruba(text)
        tokens = syllabify_text(text)

        # Split into sentences at punctuation boundaries, run Viterbi per sentence
        _sentence_end = frozenset(".!?;")
        result = []
        all_tokens = list(tokens)

        # We'll build the result by walking through tokens and replacing
        # word tokens with Viterbi output
        word_indices = []  # indices into all_tokens that are words
        for i, (token, is_word) in enumerate(all_tokens):
            if is_word:
                word_indices.append(i)

        # Group word indices into sentences (split at sentence-end punctuation)
        sentences = []  # each is a list of indices into all_tokens
        current_sentence = []
        for i, (token, is_word) in enumerate(all_tokens):
            if is_word:
                current_sentence.append(i)
            elif any(ch in _sentence_end for ch in token):
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
        if current_sentence:
            sentences.append(current_sentence)

        # Run Viterbi on each sentence and store results
        viterbi_results = {}  # index -> diacritized form
        for sent_indices in sentences:
            word_tokens = [all_tokens[i][0] for i in sent_indices]
            decoded = self._viterbi_decode(word_tokens)
            for idx, diac_form in zip(sent_indices, decoded):
                viterbi_results[idx] = diac_form

        # Reconstruct the full text
        for i, (token, is_word) in enumerate(all_tokens):
            if is_word and i in viterbi_results:
                diac = viterbi_results[i]
                # Restore original capitalization
                if token and token[0].isupper():
                    diac = diac[0].upper() + diac[1:] if len(diac) > 1 else diac.upper()
                result.append(diac)
            elif is_word:
                # Word not in any sentence group (shouldn't happen)
                result.append(self.diacritize_word(token))
            else:
                result.append(token)

        return "".join(result)

    def get_coverage(self, text: str) -> Tuple[int, int]:
        """Get word-level coverage statistics for text.

        Args:
            text: Text to analyze.

        Returns:
            Tuple of (known_words, total_words).
        """
        tokens = syllabify_text(text)
        known = 0
        total = 0

        for token, is_word in tokens:
            if is_word:
                total += 1
                if token.lower() in self.word_map:
                    known += 1

        return known, total

    def get_bigram_stats(self, text: str) -> Tuple[int, int, int]:
        """Get bigram hit statistics for text.

        Args:
            text: Text to analyze.

        Returns:
            Tuple of (known_words, bigram_hits, total_words).
        """
        tokens = syllabify_text(text)
        known = 0
        bigram_hits = 0
        total = 0

        _sentence_end = frozenset(".!?;")
        prev_word = None  # type: Optional[str]

        for token, is_word in tokens:
            if is_word:
                total += 1
                word_lower = token.lower()
                if prev_word is not None and self.bigram_map:
                    bigram_key = prev_word + "\t" + word_lower
                    if bigram_key in self.bigram_map:
                        bigram_hits += 1
                        known += 1
                        prev_word = word_lower
                        continue
                if word_lower in self.word_map:
                    known += 1
                prev_word = word_lower
            else:
                if any(ch in _sentence_end for ch in token):
                    prev_word = None

        return known, bigram_hits, total

    def _viterbi_decode(self, tokens: List[str]) -> List[str]:
        """Decode a sequence of undiacritized word tokens using Viterbi.

        Uses a hybrid approach: the curated bigram_map provides strong
        constraints for known word pairs, while transition probabilities
        handle unseen pairs. This combines the precision of targeted
        bigram overrides with global sequence optimization.

        Args:
            tokens: List of undiacritized word tokens.

        Returns:
            List of diacritized word tokens (lowercase).
        """
        if not tokens:
            return []

        n = len(tokens)
        tokens_lower = [t.lower() for t in tokens]

        # Build candidate list per position
        candidates = []  # type: List[List[Tuple[str, float]]]
        for word_lower in tokens_lower:
            if word_lower in self.word_candidates:
                # Ambiguous word — multiple candidates with emission log-probs
                candidates.append(self.word_candidates[word_lower])
            elif word_lower in self.word_map:
                # Unambiguous known word — single candidate
                form = self.word_map[word_lower]
                candidates.append([(form, 0.0)])  # log(1.0) = 0
            else:
                # Unknown word — syllable fallback, single candidate
                fallback = self._get_syllable_fallback()
                form = fallback.diacritize(word_lower)
                candidates.append([(form.lower(), 0.0)])

        # Pre-compute bigram_map bonuses per position.
        # bigram_map maps "prev_undiac\tcurr_undiac" -> diac_form.
        # If a bigram_map entry exists, we give a large bonus to that candidate.
        bigram_bonus = 5.0  # strong bonus for curated bigram predictions
        bigram_targets = {}  # type: Dict[int, str]  # position -> target diac form
        for t in range(1, n):
            bigram_key = tokens_lower[t - 1] + "\t" + tokens_lower[t]
            if bigram_key in self.bigram_map:
                bigram_targets[t] = self.bigram_map[bigram_key]

        # Viterbi forward pass
        K0 = len(candidates[0])
        score = [None] * n  # type: List[Optional[List[float]]]
        backptr = [None] * n  # type: List[Optional[List[int]]]

        # Initialize: position 0
        score[0] = [candidates[0][k][1] for k in range(K0)]
        backptr[0] = [0] * K0

        # Default back-off penalty for unseen transitions
        default_log_prob = -10.0

        for t in range(1, n):
            Kt = len(candidates[t])
            Kt_prev = len(candidates[t - 1])
            score[t] = [0.0] * Kt
            backptr[t] = [0] * Kt

            for k in range(Kt):
                curr_form = candidates[t][k][0]
                emission = candidates[t][k][1]

                # Apply bigram_map bonus if this form matches
                if t in bigram_targets and curr_form == bigram_targets[t]:
                    emission += bigram_bonus

                best_score = float("-inf")
                best_prev = 0

                for j in range(Kt_prev):
                    prev_form = candidates[t - 1][j][0]
                    # Transition: P(curr_form | prev_form)
                    trans = default_log_prob
                    if prev_form in self.transition_probs:
                        trans = self.transition_probs[prev_form].get(
                            curr_form,
                            self.unigram_log_probs.get(curr_form, default_log_prob),
                        )
                    elif curr_form in self.unigram_log_probs:
                        trans = self.unigram_log_probs[curr_form]

                    s = score[t - 1][j] + trans + emission
                    if s > best_score:
                        best_score = s
                        best_prev = j

                score[t][k] = best_score
                backptr[t][k] = best_prev

        # Backtrace
        result = [""] * n
        # Find best final state
        best_final = 0
        best_final_score = float("-inf")
        for k in range(len(candidates[n - 1])):
            if score[n - 1][k] > best_final_score:
                best_final_score = score[n - 1][k]
                best_final = k

        result[n - 1] = candidates[n - 1][best_final][0]
        k = best_final
        for t in range(n - 2, -1, -1):
            k = backptr[t + 1][k]
            result[t] = candidates[t][k][0]

        return result

    def _diacritize_greedy(self, text: str) -> str:
        """Greedy left-to-right diacritization (bigram + unigram fallback).

        This is the original decoding strategy, used when Viterbi data
        is not available (backward compatibility with old models).

        Args:
            text: Undiacritized Yorùbá text.

        Returns:
            Text with diacritics restored.
        """
        text = normalize_yoruba(text)
        tokens = syllabify_text(text)
        result = []

        _sentence_end = frozenset(".!?;")
        prev_word = None  # type: Optional[str]

        for token, is_word in tokens:
            if is_word:
                result.append(self.diacritize_word(token, prev_word=prev_word))
                prev_word = token.lower()
            else:
                result.append(token)
                if any(ch in _sentence_end for ch in token):
                    prev_word = None

        return "".join(result)

    def save(self, path: Path) -> None:
        """Save the trained model to JSON.

        Args:
            path: Path to save the model.
        """
        data = {
            "min_word_freq": self.min_word_freq,
            "min_bigram_freq": self.min_bigram_freq,
            "word_map": self.word_map,
            "bigram_map": self.bigram_map,
            "bigram_count": self.bigram_count,
            "total_words_trained": self.total_words_trained,
            "vocab_size": self.vocab_size,
        }

        if self.has_viterbi:
            data["viterbi"] = {
                "word_candidates": {
                    k: [[form, prob] for form, prob in v]
                    for k, v in self.word_candidates.items()
                },
                "transition_probs": self.transition_probs,
                "unigram_log_probs": self.unigram_log_probs,
                "max_candidates": self.max_candidates,
            }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)

        logger.info("Saved word-level diacritizer to %s (%.1f KB)",
                    path, path.stat().st_size / 1024)

    @classmethod
    def load(cls, path: Path) -> "WordLevelDiacritizer":
        """Load a trained model from JSON.

        Args:
            path: Path to the model file.

        Returns:
            Loaded WordLevelDiacritizer instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        viterbi_data = data.get("viterbi")
        model = cls(
            min_word_freq=data.get("min_word_freq", 5),
            min_bigram_freq=data.get("min_bigram_freq", 3),
            max_candidates=viterbi_data.get("max_candidates", 5) if viterbi_data else 5,
        )
        model.word_map = data.get("word_map", {})
        model.bigram_map = data.get("bigram_map", {})
        model.bigram_count = data.get("bigram_count", len(model.bigram_map))
        model.total_words_trained = data.get("total_words_trained", 0)
        model.vocab_size = data.get("vocab_size", len(model.word_map))

        # Load Viterbi data if present (backward compatible with old models)
        if viterbi_data:
            model.word_candidates = {
                k: [(form, prob) for form, prob in v]
                for k, v in viterbi_data.get("word_candidates", {}).items()
            }
            model.transition_probs = viterbi_data.get("transition_probs", {})
            model.unigram_log_probs = viterbi_data.get("unigram_log_probs", {})
            model.has_viterbi = True

        return model


class DotBelowDiacritizer:
    """Syllable-based diacritizer that ONLY restores dot-below characters.

    Restores ọ, ẹ, ṣ without adding tonal marks. This is a simpler task
    with higher accuracy than full diacritization.

    The hypothesis is that dot-below (ọ, ẹ, ṣ) is much more predictable
    than tonal marks because:
    1. Dot-below represents phonemic distinctions (different vowels)
    2. Tones are more context-dependent and ambiguous
    """

    def __init__(self):
        """Initialize an empty dot-below diacritizer."""
        # syllable_undiac -> {context -> {with_dots -> count}}
        self.syllable_db: Dict[str, Dict[Tuple, Counter]] = defaultdict(
            lambda: defaultdict(Counter)
        )
        # syllable_undiac -> {with_dots -> count} (context-free fallback)
        self.syllable_freq: Dict[str, Counter] = defaultdict(Counter)
        # Total syllables seen
        self.total_syllables = 0

    def train(self, diacritized_texts: List[str]) -> "DotBelowDiacritizer":
        """Train the dot-below diacritizer on diacritized Yorùbá texts.

        Training process:
        1. For each text, strip tonal marks but KEEP dot-below (expected output)
        2. Strip ALL diacritics (input)
        3. Learn mapping from input to expected output

        Args:
            diacritized_texts: List of properly diacritized Yorùbá sentences.

        Returns:
            self for method chaining.
        """
        for text in diacritized_texts:
            # Get text with only dot-below (no tones) - this is our target
            target_text = strip_diacritics(text, tones_only=True)
            tokens = syllabify_text(target_text)

            # Extract words and syllabify them
            words_syllables = []
            for token, is_word in tokens:
                if is_word:
                    syls = syllabify(token)
                    words_syllables.extend(syls)
                else:
                    if token.strip():
                        words_syllables.append(token)

            # Build context-aware database
            for i, syl in enumerate(words_syllables):
                if not syl or not any(c.isalpha() for c in syl):
                    continue

                # Input: fully stripped (no diacritics at all)
                undiac = strip_diacritics(syl).lower()

                # Context: previous and next syllables (also stripped)
                prev_syl = words_syllables[i - 1] if i > 0 else "<START>"
                next_syl = words_syllables[i + 1] if i + 1 < len(words_syllables) else "<END>"

                prev_undiac = strip_diacritics(prev_syl).lower() if prev_syl not in ("<START>", "<END>") else prev_syl
                next_undiac = strip_diacritics(next_syl).lower() if next_syl not in ("<START>", "<END>") else next_syl

                context = (prev_undiac, next_undiac)

                # Store with context (target is syllable with dot-below only)
                self.syllable_db[undiac][context][syl.lower()] += 1

                # Store frequency (context-free)
                self.syllable_freq[undiac][syl.lower()] += 1
                self.total_syllables += 1

        logger.info("Trained dot-below model on %d syllables, %d unique undiacritized forms",
                    self.total_syllables, len(self.syllable_freq))

        return self

    def predict_syllable(
        self,
        syllable: str,
        prev_context: str = "<START>",
        next_context: str = "<END>",
    ) -> str:
        """Predict the dot-below form of a syllable (no tones).

        Args:
            syllable: Undiacritized syllable.
            prev_context: Previous syllable (undiacritized) or <START>.
            next_context: Next syllable (undiacritized) or <END>.

        Returns:
            Syllable with dot-below characters restored (no tones).
        """
        undiac = strip_diacritics(syllable).lower()
        original_case = syllable[0].isupper() if syllable else False

        # Try exact context match
        context = (prev_context.lower(), next_context.lower())
        if undiac in self.syllable_db and context in self.syllable_db[undiac]:
            result = self.syllable_db[undiac][context].most_common(1)[0][0]
            return result.capitalize() if original_case else result

        # Try partial context (prev only)
        for ctx, counts in self.syllable_db.get(undiac, {}).items():
            if ctx[0] == context[0]:
                result = counts.most_common(1)[0][0]
                return result.capitalize() if original_case else result

        # Try partial context (next only)
        for ctx, counts in self.syllable_db.get(undiac, {}).items():
            if ctx[1] == context[1]:
                result = counts.most_common(1)[0][0]
                return result.capitalize() if original_case else result

        # Fall back to most frequent (context-free)
        if undiac in self.syllable_freq:
            result = self.syllable_freq[undiac].most_common(1)[0][0]
            return result.capitalize() if original_case else result

        # Unknown syllable - return as-is
        return syllable

    def diacritize(self, text: str) -> str:
        """Restore ONLY dot-below characters to undiacritized Yorùbá text.

        Does NOT add tonal marks. Use this for reliable diacritization
        when accuracy is more important than completeness.

        Args:
            text: Undiacritized Yorùbá text.

        Returns:
            Text with dot-below characters (ọ, ẹ, ṣ) restored, no tones.

        Example:
            >>> dot_below_diacritizer.diacritize("Ojo dara")
            'Ọjọ dara'  # Note: no tones, just ọ
        """
        tokens = syllabify_text(text)
        result = []

        # First pass: collect all syllables with positions
        all_syllables = []

        for token_idx, (token, is_word) in enumerate(tokens):
            if is_word:
                syls = syllabify(token)
                for syl_idx, syl in enumerate(syls):
                    all_syllables.append((syl, token_idx, syl_idx, True))
            else:
                all_syllables.append((token, token_idx, 0, False))

        # Second pass: predict dot-below with context
        predicted = []
        word_syllables = [(s, i) for i, (s, _, _, is_word) in enumerate(all_syllables) if is_word]

        for idx, (syl, token_idx, syl_idx, is_word) in enumerate(all_syllables):
            if not is_word:
                predicted.append(syl)
                continue

            word_idx = next(i for i, (_, orig_idx) in enumerate(word_syllables) if orig_idx == idx)

            prev_ctx = strip_diacritics(word_syllables[word_idx - 1][0]).lower() if word_idx > 0 else "<START>"
            next_ctx = strip_diacritics(word_syllables[word_idx + 1][0]).lower() if word_idx + 1 < len(word_syllables) else "<END>"

            pred_syl = self.predict_syllable(syl, prev_ctx, next_ctx)
            predicted.append(pred_syl)

        # Third pass: reconstruct words
        result = []
        current_word = []
        last_token_idx = -1

        for i, (syl, token_idx, syl_idx, is_word) in enumerate(all_syllables):
            if is_word:
                if token_idx != last_token_idx:
                    if current_word:
                        result.append("".join(current_word))
                        current_word = []
                current_word.append(predicted[i])
                last_token_idx = token_idx
            else:
                if current_word:
                    result.append("".join(current_word))
                    current_word = []
                result.append(predicted[i])
                last_token_idx = -1

        if current_word:
            result.append("".join(current_word))

        return "".join(result)

    def save(self, path: Path) -> None:
        """Save the trained model to JSON."""
        data = {
            "syllable_db": {
                undiac: {
                    f"{ctx[0]}|{ctx[1]}": dict(counts)
                    for ctx, counts in contexts.items()
                }
                for undiac, contexts in self.syllable_db.items()
            },
            "syllable_freq": {
                undiac: dict(counts)
                for undiac, counts in self.syllable_freq.items()
            },
            "total_syllables": self.total_syllables,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info("Saved dot-below model to %s", path)

    @classmethod
    def load(cls, path: Path) -> "DotBelowDiacritizer":
        """Load a trained model from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model = cls()

        for undiac, contexts in data.get("syllable_db", {}).items():
            for ctx_str, counts in contexts.items():
                prev_ctx, next_ctx = ctx_str.split("|", 1)
                context = (prev_ctx, next_ctx)
                model.syllable_db[undiac][context] = Counter(counts)

        for undiac, counts in data.get("syllable_freq", {}).items():
            model.syllable_freq[undiac] = Counter(counts)

        model.total_syllables = data.get("total_syllables", 0)

        return model
