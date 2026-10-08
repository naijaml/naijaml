"""Tests for Nigerian text preprocessing module."""
from __future__ import annotations


class TestNormalizeUnicode:
    """Test Unicode normalization."""

    def test_nfc_default(self):
        from naijaml.nlp import normalize_unicode
        # NFC should compose characters
        result = normalize_unicode("ọjọ́")
        assert isinstance(result, str)

    def test_empty_string(self):
        from naijaml.nlp import normalize_unicode
        assert normalize_unicode("") == ""

    def test_ascii_unchanged(self):
        from naijaml.nlp import normalize_unicode
        assert normalize_unicode("hello world") == "hello world"


class TestStripDiacritics:
    """Test diacritic stripping."""

    def test_yoruba_diacritics(self):
        from naijaml.nlp import strip_diacritics
        result = strip_diacritics("Ojó lo sí ọjà lánà")
        assert "ọ" not in result
        assert "á" not in result
        assert "ó" not in result

    def test_plain_text_unchanged(self):
        from naijaml.nlp import strip_diacritics
        assert strip_diacritics("hello world") == "hello world"

    def test_empty(self):
        from naijaml.nlp import strip_diacritics
        assert strip_diacritics("") == ""


class TestCleanSocialMedia:
    """Test social media cleaning."""

    def test_removes_urls(self):
        from naijaml.nlp import clean_social_media
        result = clean_social_media("Check https://t.co/abc")
        assert "https" not in result

    def test_removes_mentions(self):
        from naijaml.nlp import clean_social_media
        result = clean_social_media("@user hello")
        assert "@user" not in result
        assert "hello" in result

    def test_keeps_hashtags_by_default(self):
        from naijaml.nlp import clean_social_media
        result = clean_social_media("Great #Nollywood film")
        assert "#Nollywood" in result

    def test_removes_hashtags_when_asked(self):
        from naijaml.nlp import clean_social_media
        result = clean_social_media("Great #Nollywood film", remove_hashtags=True)
        assert "#Nollywood" not in result

    def test_reduces_repeated_chars(self):
        from naijaml.nlp import clean_social_media
        result = clean_social_media("Greaaaaat!")
        assert "aaaa" not in result

    def test_lowercase(self):
        from naijaml.nlp import clean_social_media
        result = clean_social_media("HELLO World", lowercase=True)
        assert result == "hello world"


class TestExtractHashtags:
    """Test hashtag extraction."""

    def test_extracts_hashtags(self):
        from naijaml.nlp import extract_hashtags
        result = extract_hashtags("Great #Nollywood #film")
        assert "Nollywood" in result
        assert "film" in result

    def test_no_hashtags(self):
        from naijaml.nlp import extract_hashtags
        assert extract_hashtags("No hashtags here") == []


class TestExtractMentions:
    """Test mention extraction."""

    def test_extracts_mentions(self):
        from naijaml.nlp import extract_mentions
        result = extract_mentions("cc @user1 @user2")
        assert "user1" in result
        assert "user2" in result

    def test_no_mentions(self):
        from naijaml.nlp import extract_mentions
        assert extract_mentions("No mentions here") == []


class TestMaskPII:
    """Test PII masking."""

    def test_masks_phone(self):
        from naijaml.nlp import mask_pii
        result = mask_pii("Call 08012345678")
        assert "[PHONE]" in result
        assert "08012345678" not in result

    def test_masks_email(self):
        from naijaml.nlp import mask_pii
        result = mask_pii("Email user@example.com")
        assert "[EMAIL]" in result
        assert "user@example.com" not in result

    def test_masks_bvn(self):
        from naijaml.nlp import mask_pii
        result = mask_pii("BVN: 22123456789")
        assert "[BVN]" in result

    def test_no_pii_unchanged(self):
        from naijaml.nlp import mask_pii
        text = "Lagos is a city"
        assert mask_pii(text) == text

    def test_custom_masks(self):
        from naijaml.nlp import mask_pii
        result = mask_pii("Call 08012345678", phone_mask="***")
        assert "***" in result

    def test_naira_not_masked_by_default(self):
        from naijaml.nlp import mask_pii
        result = mask_pii("Price is ₦5,000")
        assert "₦5,000" in result

    def test_naira_masked_when_enabled(self):
        from naijaml.nlp import mask_pii
        result = mask_pii("Price is ₦5,000", mask_naira=True)
        assert "[AMOUNT]" in result


class TestFindPhones:
    """Test phone number finder."""

    def test_finds_standard_number(self):
        from naijaml.nlp import find_phones
        result = find_phones("Call 08012345678")
        assert len(result) >= 1

    def test_no_phones(self):
        from naijaml.nlp import find_phones
        assert find_phones("No phone here") == []


class TestFindNairaAmounts:
    """Test Naira amount finder."""

    def test_finds_naira_symbol(self):
        from naijaml.nlp import find_naira_amounts
        result = find_naira_amounts("Price is ₦5,000")
        assert len(result) >= 1

    def test_no_amounts(self):
        from naijaml.nlp import find_naira_amounts
        assert find_naira_amounts("No money here") == []


class TestNormalizeNairaSymbol:
    """Test Naira symbol normalization."""

    def test_ngn_to_naira(self):
        from naijaml.nlp import normalize_naira_symbol
        result = normalize_naira_symbol("NGN5000")
        assert "₦" in result

    def test_already_normalized(self):
        from naijaml.nlp import normalize_naira_symbol
        assert "₦" in normalize_naira_symbol("₦5000")


class TestCleanNigerianText:
    """Test all-in-one cleaning function."""

    def test_full_pipeline(self):
        from naijaml.nlp import clean_nigerian_text
        text = "@user Check https://t.co/abc Ọjọ́ is great!"
        result = clean_nigerian_text(text)
        assert "@user" not in result
        assert "https" not in result
        assert "Ọjọ́" in result

    def test_with_pii_masking(self):
        from naijaml.nlp import clean_nigerian_text
        text = "Call 08012345678"
        result = clean_nigerian_text(text, mask_pii_data=True)
        assert "[PHONE]" in result


class TestPidginHandling:
    """Test Pidgin particle handling."""

    def test_is_pidgin_particle(self):
        from naijaml.nlp import is_pidgin_particle
        assert is_pidgin_particle("sha") is True
        assert is_pidgin_particle("abeg") is True
        assert is_pidgin_particle("computer") is False

    def test_preserve_pidgin_particles(self):
        from naijaml.nlp import preserve_pidgin_particles
        result = preserve_pidgin_particles("The film sha too sweet", ["the", "sha", "too"])
        assert "sha" in result

    def test_get_pidgin_particles(self):
        from naijaml.nlp import get_pidgin_particles
        particles = get_pidgin_particles()
        assert isinstance(particles, set)
        assert "abeg" in particles


class TestPidginNegationNormalization:
    """Test Nigerian Pidgin negation normalization."""

    def test_specific_no_too_bad_rule_runs_before_generic_rule(self):
        from naijaml.nlp.preprocess import normalize_pidgin_negation

        assert normalize_pidgin_negation("E no too bad sha") == "it is good sha"

    def test_docstring_examples_match_actual_output(self):
        from naijaml.nlp.preprocess import normalize_pidgin_negation

        assert normalize_pidgin_negation("This thing no bad at all") == "This thing very good"
        assert normalize_pidgin_negation("E no sweet me") == "it is bad me"

    def test_no_go_lie_keeps_spacing_clean(self):
        from naijaml.nlp.preprocess import normalize_pidgin_negation

        assert normalize_pidgin_negation("I no go lie this thing good") == "I honestly this thing good"
