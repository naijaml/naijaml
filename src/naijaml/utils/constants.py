"""Nigerian constants: states, LGAs, banks, telcos, and regex patterns.

Provides reference data and utilities for Nigerian-specific validation and formatting.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# =============================================================================
# Nigerian States and Capitals (36 states + FCT)
# =============================================================================

STATES: Dict[str, str] = {
    "Abia": "Umuahia",
    "Adamawa": "Yola",
    "Akwa Ibom": "Uyo",
    "Anambra": "Awka",
    "Bauchi": "Bauchi",
    "Bayelsa": "Yenagoa",
    "Benue": "Makurdi",
    "Borno": "Maiduguri",
    "Cross River": "Calabar",
    "Delta": "Asaba",
    "Ebonyi": "Abakaliki",
    "Edo": "Benin City",
    "Ekiti": "Ado Ekiti",
    "Enugu": "Enugu",
    "FCT": "Abuja",
    "Gombe": "Gombe",
    "Imo": "Owerri",
    "Jigawa": "Dutse",
    "Kaduna": "Kaduna",
    "Kano": "Kano",
    "Katsina": "Katsina",
    "Kebbi": "Birnin Kebbi",
    "Kogi": "Lokoja",
    "Kwara": "Ilorin",
    "Lagos": "Ikeja",
    "Nasarawa": "Lafia",
    "Niger": "Minna",
    "Ogun": "Abeokuta",
    "Ondo": "Akure",
    "Osun": "Osogbo",
    "Oyo": "Ibadan",
    "Plateau": "Jos",
    "Rivers": "Port Harcourt",
    "Sokoto": "Sokoto",
    "Taraba": "Jalingo",
    "Yobe": "Damaturu",
    "Zamfara": "Gusau",
}

STATE_NAMES: List[str] = sorted(STATES.keys())

# =============================================================================
# Local Government Areas (774 LGAs)
# Subset of major LGAs per state - full list would be 774 entries
# =============================================================================

LGAS: Dict[str, List[str]] = {
    "Lagos": [
        "Agege", "Ajeromi-Ifelodun", "Alimosho", "Amuwo-Odofin", "Apapa",
        "Badagry", "Epe", "Eti-Osa", "Ibeju-Lekki", "Ifako-Ijaiye",
        "Ikeja", "Ikorodu", "Kosofe", "Lagos Island", "Lagos Mainland",
        "Mushin", "Ojo", "Oshodi-Isolo", "Shomolu", "Surulere",
    ],
    "Kano": [
        "Dala", "Fagge", "Gwale", "Kano Municipal", "Nassarawa", "Tarauni",
        "Ungogo", "Kumbotso", "Gezawa", "Warawa", "Dawakin Tofa",
    ],
    "Rivers": [
        "Port Harcourt", "Obio-Akpor", "Eleme", "Oyigbo", "Okrika",
        "Bonny", "Degema", "Asari-Toru", "Akuku-Toru", "Abua/Odual",
    ],
    "Oyo": [
        "Ibadan North", "Ibadan South-West", "Ibadan South-East", "Ibadan North-West",
        "Ibadan North-East", "Akinyele", "Egbeda", "Ona Ara", "Oluyole", "Ido",
    ],
    "FCT": [
        "Abaji", "Abuja Municipal", "Bwari", "Gwagwalada", "Kuje", "Kwali",
    ],
    "Kaduna": [
        "Kaduna North", "Kaduna South", "Chikun", "Igabi", "Zaria", "Sabon Gari",
    ],
    "Anambra": [
        "Awka North", "Awka South", "Onitsha North", "Onitsha South", "Nnewi North",
        "Nnewi South", "Idemili North", "Idemili South", "Ogidi", "Aguata",
    ],
}

# =============================================================================
# Nigerian Banks
# =============================================================================

BANKS: Dict[str, str] = {
    # Traditional banks
    "Access Bank": "044",
    "Citibank": "023",
    "Ecobank": "050",
    "Fidelity Bank": "070",
    "First Bank": "011",
    "First City Monument Bank": "214",
    "Globus Bank": "103",
    "Guaranty Trust Bank": "058",
    "Heritage Bank": "030",
    "Keystone Bank": "082",
    "Polaris Bank": "076",
    "Providus Bank": "101",
    "Stanbic IBTC": "221",
    "Standard Chartered": "068",
    "Sterling Bank": "232",
    "SunTrust Bank": "100",
    "Titan Trust Bank": "102",
    "Union Bank": "032",
    "United Bank for Africa": "033",
    "Unity Bank": "215",
    "Wema Bank": "035",
    "Zenith Bank": "057",
    # Digital/Fintech banks
    "Kuda Bank": "090267",
    "OPay": "100004",
    "PalmPay": "100033",
    "Moniepoint": "100022",
    "Carbon": "100026",
    "Sparkle": "100269",
    "VFD Microfinance Bank": "090110",
}

BANK_NAMES: List[str] = sorted(BANKS.keys())

# =============================================================================
# Telecom Operators
# =============================================================================

TELCOS: Dict[str, Dict[str, any]] = {
    "MTN": {
        "prefixes": ["0803", "0806", "0703", "0706", "0813", "0816", "0810", "0814", "0903", "0906", "0913", "0916"],
    },
    "Airtel": {
        "prefixes": ["0802", "0808", "0708", "0812", "0701", "0901", "0902", "0907", "0912"],
    },
    "Glo": {
        "prefixes": ["0805", "0807", "0705", "0815", "0811", "0905", "0915"],
    },
    "9mobile": {
        "prefixes": ["0809", "0818", "0817", "0909", "0908"],
    },
}

TELCO_NAMES: List[str] = sorted(TELCOS.keys())

# =============================================================================
# Regex Patterns
# =============================================================================

# Nigerian phone number: 11 digits starting with 0, or with +234/234 prefix
PHONE_PATTERN = re.compile(
    r"^(?:0|\+?234)"  # Start with 0, 234, or +234
    r"[789][01]\d"     # Second digit 7/8/9, third digit 0/1, fourth any digit
    r"\d{7}$"          # Remaining 7 digits
)

# More permissive pattern for finding phones in text (allows spaces, dashes)
PHONE_PATTERN_LOOSE = re.compile(
    r"(?:0|\+?234)[\s\-]?"
    r"[789][01]\d[\s\-]?"
    r"\d{3}[\s\-]?"
    r"\d{4}"
)

# Bank Verification Number (BVN): 11 digits starting with 22
BVN_PATTERN = re.compile(r"^22\d{9}$")

# National Identification Number (NIN): 11 digits
NIN_PATTERN = re.compile(r"^\d{11}$")

# Nigerian bank account number: 10 digits (NUBAN)
NUBAN_PATTERN = re.compile(r"^\d{10}$")

# Naira amounts in text: ₦1,000 or N1000 or NGN 1,000
NAIRA_PATTERN = re.compile(
    r"(?:₦|NGN|N)\s?"
    r"[\d,]+(?:\.\d{2})?"
)

# =============================================================================
# Utility Functions
# =============================================================================

def format_naira(amount: float, include_kobo: bool = True) -> str:
    """Format a number as Nigerian Naira currency.

    Args:
        amount: The amount to format.
        include_kobo: Whether to include decimal places (kobo).

    Returns:
        Formatted string like "₦1,500,000.00".

    Example:
        >>> format_naira(1500000)
        '₦1,500,000.00'
        >>> format_naira(1500000, include_kobo=False)
        '₦1,500,000'
    """
    if include_kobo:
        return "₦{:,.2f}".format(amount)
    return "₦{:,}".format(int(amount))


def parse_naira(text: str) -> Optional[float]:
    """Parse a Naira amount from text.

    Args:
        text: String containing a Naira amount (e.g., "₦1,500,000.00").

    Returns:
        Float value, or None if parsing fails.

    Example:
        >>> parse_naira("₦1,500,000.00")
        1500000.0
        >>> parse_naira("NGN 50,000")
        50000.0
    """
    cleaned = re.sub(r"[₦NGN\s,]", "", text)
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_valid_phone(phone: str) -> bool:
    """Check if a string is a valid Nigerian phone number.

    Args:
        phone: Phone number string to validate.

    Returns:
        True if valid Nigerian phone number format.

    Example:
        >>> is_valid_phone("08012345678")
        True
        >>> is_valid_phone("+2348012345678")
        True
        >>> is_valid_phone("12345")
        False
    """
    cleaned = re.sub(r"[\s\-]", "", phone)
    return bool(PHONE_PATTERN.match(cleaned))


def normalize_phone(phone: str) -> Optional[str]:
    """Normalize a Nigerian phone number to international format.

    Args:
        phone: Phone number in any common Nigerian format.

    Returns:
        Phone number in +234XXXXXXXXXX format, or None if invalid.

    Example:
        >>> normalize_phone("08012345678")
        '+2348012345678'
        >>> normalize_phone("234-801-234-5678")
        '+2348012345678'
    """
    cleaned = re.sub(r"[\s\-]", "", phone)
    if not PHONE_PATTERN.match(cleaned):
        return None
    if cleaned.startswith("+234"):
        return cleaned
    if cleaned.startswith("234"):
        return "+" + cleaned
    if cleaned.startswith("0"):
        return "+234" + cleaned[1:]
    return None


def get_telco(phone: str) -> Optional[str]:
    """Identify the telecom operator from a Nigerian phone number.

    Args:
        phone: Nigerian phone number.

    Returns:
        Telco name ('MTN', 'Airtel', 'Glo', '9mobile') or None.

    Example:
        >>> get_telco("08031234567")
        'MTN'
        >>> get_telco("08021234567")
        'Airtel'
    """
    cleaned = re.sub(r"[\s\-]", "", phone)
    # Normalize to 0xxx format for prefix matching
    if cleaned.startswith("+234"):
        cleaned = "0" + cleaned[4:]
    elif cleaned.startswith("234"):
        cleaned = "0" + cleaned[3:]

    prefix = cleaned[:4]
    for telco, info in TELCOS.items():
        if prefix in info["prefixes"]:
            return telco
    return None


def is_valid_bvn(bvn: str) -> bool:
    """Check if a string is a valid BVN format.

    Args:
        bvn: Bank Verification Number to validate.

    Returns:
        True if valid BVN format (11 digits starting with 22).
    """
    return bool(BVN_PATTERN.match(bvn))


def is_valid_nin(nin: str) -> bool:
    """Check if a string is a valid NIN format.

    Args:
        nin: National Identification Number to validate.

    Returns:
        True if valid NIN format (11 digits).
    """
    return bool(NIN_PATTERN.match(nin))


# =============================================================================
# Nigerian Pidgin (Naija) Particles and Common Words
# =============================================================================

# These are discourse particles, intensifiers, and function words that are
# unique to Nigerian Pidgin and should be preserved during text cleaning.
# Standard NLP tools often strip these as "noise" or "errors".

PIDGIN_PARTICLES: set = {
    # Discourse particles (sentence modifiers)
    "sha",       # anyway, though (softener)
    "sef",       # even, self (emphasis)
    "abeg",      # please (politeness marker)
    "abi",       # or, right? (question tag)
    "shey",      # isn't it? (question tag)
    "shebi",     # right? (confirmation)
    "na",        # copula/focus marker
    "dey",       # progressive marker / to be
    "no",        # negation
    "wey",       # relative pronoun (that/which)
    "oya",       # let's go, come on
    "jare",      # please (Yoruba origin)
    "jor",       # please (variant)
    "biko",      # please (Igbo origin)
    "walahi",    # I swear (Hausa origin)
    "wallahi",   # I swear (variant)

    # Intensifiers and modifiers
    "well",      # very, really
    "die",       # extremely (to die for)
    "gidigba",   # seriously, plenty
    "scatter",   # extremely
    "proper",    # properly, really
    "sharp",     # quickly
    "quick",     # quickly

    # Common Pidgin verbs/words often stripped
    "dey",       # be/is (progressive)
    "chop",      # eat
    "gist",      # chat/gossip
    "vex",       # angry
    "japa",      # run away/emigrate
    "sabi",      # know
    "wan",       # want
    "go",        # future marker
    "don",       # perfect marker
    "fit",       # can/able

    # Pronouns and determiners
    "wetin",     # what
    "weda",      # whether
    "una",       # you (plural)
    "dem",       # them/they
    "am",        # him/her/it
    "im",        # his/her

    # Greetings and expressions
    "howfar",    # hello, how are you
    "ehen",      # I see, okay
    "ehn",       # really? (question)
    "kai",       # exclamation
    "chei",      # exclamation
    "wahala",    # trouble
    "palava",    # problem
    "yawa",      # trouble
    "kolo",      # crazy
    "mumu",      # fool
    "oga",       # boss
    "madam",     # ma'am
    "bros",      # brother
    "sista",     # sister
    "pikin",     # child
    "baba",      # father/old man
    "mama",      # mother
}

# Common Pidgin multi-word expressions
PIDGIN_EXPRESSIONS: set = {
    "how far",
    "no wahala",
    "no vex",
    "e go be",
    "na so",
    "na wa",
    "no be",
    "wey dey",
    "for where",
    "make we",
    "i no",
    "you no",
    "e no",
    "dem no",
    "no dey",
}
