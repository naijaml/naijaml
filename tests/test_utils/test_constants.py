"""Tests for Nigerian constants module."""
from __future__ import annotations

import pytest


class TestStates:
    """Test Nigerian states data."""

    def test_state_count(self):
        from naijaml.utils.constants import STATES
        assert len(STATES) == 37  # 36 states + FCT

    def test_fct_exists(self):
        from naijaml.utils.constants import STATES
        assert "FCT" in STATES
        assert STATES["FCT"] == "Abuja"

    def test_lagos(self):
        from naijaml.utils.constants import STATES
        assert "Lagos" in STATES
        assert STATES["Lagos"] == "Ikeja"

    def test_all_capitals_non_empty(self):
        from naijaml.utils.constants import STATES
        for state, capital in STATES.items():
            assert capital, f"State {state} has empty capital"

    def test_state_names_sorted(self):
        from naijaml.utils.constants import STATE_NAMES
        assert STATE_NAMES == sorted(STATE_NAMES)


class TestLGAs:
    """Test LGA data."""

    def test_all_states_have_lgas(self):
        from naijaml.utils.constants import LGAS, STATES
        for state in STATES:
            assert state in LGAS, f"Missing LGA data for {state}"

    def test_total_lga_count(self):
        from naijaml.utils.constants import LGAS
        total = sum(len(v) for v in LGAS.values())
        assert total == 774

    def test_lagos_lgas(self):
        from naijaml.utils.constants import LGAS
        assert "Lagos" in LGAS
        assert len(LGAS["Lagos"]) == 20

    def test_kano_most_lgas(self):
        from naijaml.utils.constants import LGAS
        assert len(LGAS["Kano"]) == 44

    def test_fct_lgas(self):
        from naijaml.utils.constants import LGAS
        assert "FCT" in LGAS
        assert len(LGAS["FCT"]) == 6

    def test_lga_values_are_lists(self):
        from naijaml.utils.constants import LGAS
        for state, lgas in LGAS.items():
            assert isinstance(lgas, list), f"LGAs for {state} is not a list"
            assert len(lgas) > 0, f"LGAs for {state} is empty"


class TestBanks:
    """Test Nigerian banks data."""

    def test_major_banks_exist(self):
        from naijaml.utils.constants import BANKS
        major = ["Access Bank", "Guaranty Trust Bank", "Zenith Bank",
                 "First Bank", "United Bank for Africa"]
        for bank in major:
            assert bank in BANKS, f"Missing bank: {bank}"

    def test_digital_banks_exist(self):
        from naijaml.utils.constants import BANKS
        digital = ["Kuda Bank", "OPay", "PalmPay", "Moniepoint"]
        for bank in digital:
            assert bank in BANKS, f"Missing digital bank: {bank}"

    def test_bank_codes_non_empty(self):
        from naijaml.utils.constants import BANKS
        for bank, code in BANKS.items():
            assert code, f"Bank {bank} has empty code"

    def test_bank_names_sorted(self):
        from naijaml.utils.constants import BANK_NAMES
        assert BANK_NAMES == sorted(BANK_NAMES)


class TestTelcos:
    """Test telecom operators data."""

    def test_all_four_telcos(self):
        from naijaml.utils.constants import TELCOS
        expected = {"MTN", "Airtel", "Glo", "9mobile"}
        assert set(TELCOS.keys()) == expected

    def test_telcos_have_prefixes(self):
        from naijaml.utils.constants import TELCOS
        for telco, info in TELCOS.items():
            assert "prefixes" in info, f"Telco {telco} missing prefixes"
            assert len(info["prefixes"]) > 0, f"Telco {telco} has no prefixes"

    def test_prefixes_format(self):
        from naijaml.utils.constants import TELCOS
        for telco, info in TELCOS.items():
            for prefix in info["prefixes"]:
                assert prefix.startswith("0"), f"Prefix {prefix} for {telco} doesn't start with 0"
                assert len(prefix) == 4, f"Prefix {prefix} for {telco} isn't 4 digits"


class TestPhoneValidation:
    """Test phone number validation."""

    def test_valid_mtn(self):
        from naijaml.utils.constants import is_valid_phone
        assert is_valid_phone("08031234567") is True

    def test_valid_airtel(self):
        from naijaml.utils.constants import is_valid_phone
        assert is_valid_phone("08021234567") is True

    def test_valid_glo(self):
        from naijaml.utils.constants import is_valid_phone
        assert is_valid_phone("08051234567") is True

    def test_valid_9mobile(self):
        from naijaml.utils.constants import is_valid_phone
        assert is_valid_phone("08091234567") is True

    def test_valid_international(self):
        from naijaml.utils.constants import is_valid_phone
        assert is_valid_phone("+2348031234567") is True

    def test_invalid_short(self):
        from naijaml.utils.constants import is_valid_phone
        assert is_valid_phone("12345") is False

    def test_invalid_format(self):
        from naijaml.utils.constants import is_valid_phone
        assert is_valid_phone("not a number") is False


class TestPhoneNormalization:
    """Test phone number normalization."""

    def test_local_to_international(self):
        from naijaml.utils.constants import normalize_phone
        assert normalize_phone("08031234567") == "+2348031234567"

    def test_already_international(self):
        from naijaml.utils.constants import normalize_phone
        assert normalize_phone("+2348031234567") == "+2348031234567"

    def test_without_plus(self):
        from naijaml.utils.constants import normalize_phone
        assert normalize_phone("2348031234567") == "+2348031234567"

    def test_invalid_returns_none(self):
        from naijaml.utils.constants import normalize_phone
        assert normalize_phone("12345") is None


class TestGetTelco:
    """Test telco identification."""

    def test_mtn(self):
        from naijaml.utils.constants import get_telco
        assert get_telco("08031234567") == "MTN"

    def test_airtel(self):
        from naijaml.utils.constants import get_telco
        assert get_telco("08021234567") == "Airtel"

    def test_glo(self):
        from naijaml.utils.constants import get_telco
        assert get_telco("08051234567") == "Glo"

    def test_9mobile(self):
        from naijaml.utils.constants import get_telco
        assert get_telco("08091234567") == "9mobile"

    def test_international_format(self):
        from naijaml.utils.constants import get_telco
        assert get_telco("+2348031234567") == "MTN"


class TestNairaFormatting:
    """Test Naira formatting utilities."""

    def test_basic_format(self):
        from naijaml.utils.constants import format_naira
        assert format_naira(1500000) == "₦1,500,000.00"

    def test_without_kobo(self):
        from naijaml.utils.constants import format_naira
        assert format_naira(1500000, include_kobo=False) == "₦1,500,000"

    def test_zero(self):
        from naijaml.utils.constants import format_naira
        assert format_naira(0) == "₦0.00"

    def test_small_amount(self):
        from naijaml.utils.constants import format_naira
        assert format_naira(99.99) == "₦99.99"


class TestNairaParsing:
    """Test Naira parsing."""

    def test_parse_naira_symbol(self):
        from naijaml.utils.constants import parse_naira
        assert parse_naira("₦1,500,000.00") == 1500000.0

    def test_parse_ngn(self):
        from naijaml.utils.constants import parse_naira
        assert parse_naira("NGN 50,000") == 50000.0

    def test_parse_invalid(self):
        from naijaml.utils.constants import parse_naira
        assert parse_naira("not money") is None


class TestBVNNIN:
    """Test BVN and NIN validation."""

    def test_valid_bvn(self):
        from naijaml.utils.constants import is_valid_bvn
        assert is_valid_bvn("22123456789") is True

    def test_invalid_bvn_wrong_prefix(self):
        from naijaml.utils.constants import is_valid_bvn
        assert is_valid_bvn("33123456789") is False

    def test_valid_nin(self):
        from naijaml.utils.constants import is_valid_nin
        assert is_valid_nin("12345678901") is True

    def test_invalid_nin_short(self):
        from naijaml.utils.constants import is_valid_nin
        assert is_valid_nin("1234") is False
