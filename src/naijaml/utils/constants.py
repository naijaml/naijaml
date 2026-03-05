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
    "Abia": [
        "Aba North", "Aba South", "Arochukwu", "Bende", "Ikwuano",
        "Isiala Ngwa North", "Isiala Ngwa South", "Isuikwuato", "Obi Ngwa",
        "Ohafia", "Osisioma Ngwa", "Ugwunagbo", "Ukwa East", "Ukwa West",
        "Umuahia North", "Umuahia South", "Umu Nneochi",
    ],  # 17
    "Adamawa": [
        "Demsa", "Fufore", "Ganye", "Girei", "Gombi", "Guyuk", "Hong",
        "Jada", "Lamurde", "Madagali", "Maiha", "Mayo-Belwa", "Michika",
        "Mubi North", "Mubi South", "Numan", "Shelleng", "Song", "Toungo",
        "Yola North", "Yola South",
    ],  # 21
    "Akwa Ibom": [
        "Abak", "Eastern Obolo", "Eket", "Esit Eket", "Essien Udim",
        "Etim Ekpo", "Etinan", "Ibeno", "Ibesikpo Asutan", "Ibiono Ibom",
        "Ika", "Ikono", "Ikot Abasi", "Ikot Ekpene", "Ini", "Itu", "Mbo",
        "Mkpat Enin", "Nsit Atai", "Nsit Ibom", "Nsit Ubium", "Obot Akara",
        "Okobo", "Onna", "Oron", "Oruk Anam", "Udung Uko", "Ukanafun",
        "Uruan", "Urue-Offong/Oruko", "Uyo",
    ],  # 31
    "Anambra": [
        "Aguata", "Anambra East", "Anambra West", "Anaocha", "Awka North",
        "Awka South", "Ayamelum", "Dunukofia", "Ekwusigo", "Idemili North",
        "Idemili South", "Ihiala", "Njikoka", "Nnewi North", "Nnewi South",
        "Ogbaru", "Onitsha North", "Onitsha South", "Orumba North",
        "Orumba South", "Oyi",
    ],  # 21
    "Bauchi": [
        "Alkaleri", "Bauchi", "Bogoro", "Damban", "Darazo", "Dass",
        "Gamawa", "Ganjuwa", "Giade", "Itas/Gadau", "Jama'are", "Katagum",
        "Kirfi", "Misau", "Ningi", "Shira", "Tafawa Balewa", "Toro",
        "Warji", "Zaki",
    ],  # 20
    "Bayelsa": [
        "Brass", "Ekeremor", "Kolokuma/Opokuma", "Nembe", "Ogbia",
        "Sagbama", "Southern Ijaw", "Yenagoa",
    ],  # 8
    "Benue": [
        "Ado", "Agatu", "Apa", "Buruku", "Gboko", "Guma", "Gwer East",
        "Gwer West", "Katsina-Ala", "Konshisha", "Kwande", "Logo",
        "Makurdi", "Obi", "Ogbadibo", "Ohimini", "Oju", "Okpokwu",
        "Oturkpo", "Tarka", "Ukum", "Ushongo", "Vandeikya",
    ],  # 23
    "Borno": [
        "Abadam", "Askira/Uba", "Bama", "Bayo", "Biu", "Chibok",
        "Damboa", "Dikwa", "Gubio", "Guzamala", "Gwoza", "Hawul", "Jere",
        "Kaga", "Kala/Balge", "Konduga", "Kukawa", "Kwaya Kusar", "Mafa",
        "Magumeri", "Maiduguri", "Marte", "Mobbar", "Monguno", "Ngala",
        "Nganzai", "Shani",
    ],  # 27
    "Cross River": [
        "Abi", "Akamkpa", "Akpabuyo", "Bakassi", "Bekwarra", "Biase",
        "Boki", "Calabar Municipal", "Calabar South", "Etung", "Ikom",
        "Obanliku", "Obubra", "Obudu", "Odukpani", "Ogoja", "Yakurr",
        "Yala",
    ],  # 18
    "Delta": [
        "Aniocha North", "Aniocha South", "Bomadi", "Burutu", "Ethiope East",
        "Ethiope West", "Ika North East", "Ika South", "Isoko North",
        "Isoko South", "Ndokwa East", "Ndokwa West", "Okpe", "Oshimili North",
        "Oshimili South", "Patani", "Sapele", "Udu", "Ughelli North",
        "Ughelli South", "Ukwuani", "Uvwie", "Warri North", "Warri South",
        "Warri South West",
    ],  # 25
    "Ebonyi": [
        "Abakaliki", "Afikpo North", "Afikpo South", "Ebonyi", "Ezza North",
        "Ezza South", "Ikwo", "Ishielu", "Ivo", "Izzi", "Ohaozara",
        "Ohaukwu", "Onicha",
    ],  # 13
    "Edo": [
        "Akoko-Edo", "Egor", "Esan Central", "Esan North-East",
        "Esan South-East", "Esan West", "Etsako Central", "Etsako East",
        "Etsako West", "Igueben", "Ikpoba-Okha", "Oredo", "Orhionmwon",
        "Ovia North-East", "Ovia South-West", "Owan East", "Owan West",
        "Uhunmwonde",
    ],  # 18
    "Ekiti": [
        "Ado-Ekiti", "Efon", "Ekiti East", "Ekiti South-West", "Ekiti West",
        "Emure", "Gbonyin", "Ido-Osi", "Ijero", "Ikere", "Ikole",
        "Ilejemeje", "Irepodun/Ifelodun", "Ise/Orun", "Moba", "Oye",
    ],  # 16
    "Enugu": [
        "Aninri", "Awgu", "Enugu East", "Enugu North", "Enugu South",
        "Ezeagu", "Igbo-Etiti", "Igbo-Eze North", "Igbo-Eze South",
        "Isi-Uzo", "Nkanu East", "Nkanu West", "Nsukka", "Oji River",
        "Udenu", "Udi", "Uzo-Uwani",
    ],  # 17
    "FCT": [
        "Abaji", "Abuja Municipal", "Bwari", "Gwagwalada", "Kuje", "Kwali",
    ],  # 6
    "Gombe": [
        "Akko", "Balanga", "Billiri", "Dukku", "Funakaye", "Gombe",
        "Kaltungo", "Kwami", "Nafada", "Shongom", "Yamaltu/Deba",
    ],  # 11
    "Imo": [
        "Aboh Mbaise", "Ahiazu Mbaise", "Ehime Mbano", "Ezinihitte",
        "Ideato North", "Ideato South", "Ihitte/Uboma", "Ikeduru",
        "Isiala Mbano", "Isu", "Mbaitoli", "Ngor-Okpala", "Njaba",
        "Nkwerre", "Nwangele", "Obowo", "Oguta", "Ohaji/Egbema",
        "Okigwe", "Onuimo", "Orlu", "Orsu", "Oru East", "Oru West",
        "Owerri Municipal", "Owerri North", "Owerri West",
    ],  # 27
    "Jigawa": [
        "Auyo", "Babura", "Biriniwa", "Birnin Kudu", "Buji", "Dutse",
        "Gagarawa", "Garki", "Gumel", "Guri", "Gwaram", "Gwiwa",
        "Hadejia", "Jahun", "Kafin Hausa", "Kaugama", "Kazaure",
        "Kiri Kasama", "Kiyawa", "Maigatari", "Malam Madori", "Miga",
        "Ringim", "Roni", "Sule Tankarkar", "Taura", "Yankwashi",
    ],  # 27
    "Kaduna": [
        "Birnin Gwari", "Chikun", "Giwa", "Igabi", "Ikara", "Jaba",
        "Jema'a", "Kachia", "Kaduna North", "Kaduna South", "Kagarko",
        "Kajuru", "Kaura", "Kauru", "Kubau", "Kudan", "Lere", "Makarfi",
        "Sabon Gari", "Sanga", "Soba", "Zangon Kataf", "Zaria",
    ],  # 23
    "Kano": [
        "Ajingi", "Albasu", "Bagwai", "Bebeji", "Bichi", "Bunkure",
        "Dala", "Dambatta", "Dawakin Kudu", "Dawakin Tofa", "Doguwa",
        "Fagge", "Gabasawa", "Garko", "Garun Mallam", "Gaya", "Gezawa",
        "Gwale", "Gwarzo", "Kabo", "Kano Municipal", "Karaye", "Kibiya",
        "Kiru", "Kumbotso", "Kunchi", "Kura", "Madobi", "Makoda",
        "Minjibir", "Nasarawa", "Rano", "Rimin Gado", "Rogo", "Shanono",
        "Sumaila", "Takai", "Tarauni", "Tofa", "Tsanyawa", "Tudun Wada",
        "Ungogo", "Warawa", "Wudil",
    ],  # 44
    "Katsina": [
        "Bakori", "Batagarawa", "Batsari", "Baure", "Bindawa", "Charanchi",
        "Dan Musa", "Dandume", "Danja", "Daura", "Dutsi", "Dutsin-Ma",
        "Faskari", "Funtua", "Ingawa", "Jibia", "Kafur", "Kaita",
        "Kankara", "Kankia", "Katsina", "Kurfi", "Kusada", "Mai'Adua",
        "Malumfashi", "Mani", "Mashi", "Matazu", "Musawa", "Rimi",
        "Sabuwa", "Safana", "Sandamu", "Zango",
    ],  # 34
    "Kebbi": [
        "Aleiro", "Arewa Dandi", "Argungu", "Augie", "Bagudo",
        "Birnin Kebbi", "Bunza", "Dandi", "Fakai", "Gwandu", "Jega",
        "Kalgo", "Koko/Besse", "Maiyama", "Ngaski", "Sakaba", "Shanga",
        "Suru", "Wasagu/Danko", "Yauri", "Zuru",
    ],  # 21
    "Kogi": [
        "Adavi", "Ajaokuta", "Ankpa", "Bassa", "Dekina", "Ibaji", "Idah",
        "Igalamela-Odolu", "Ijumu", "Kabba/Bunu", "Kogi", "Lokoja",
        "Mopa-Muro", "Ofu", "Ogori/Magongo", "Okehi", "Okene",
        "Olamaboro", "Omala", "Yagba East", "Yagba West",
    ],  # 21
    "Kwara": [
        "Asa", "Baruten", "Edu", "Ekiti", "Ifelodun", "Ilorin East",
        "Ilorin South", "Ilorin West", "Irepodun", "Isin", "Kaiama",
        "Moro", "Offa", "Oke Ero", "Oyun", "Pategi",
    ],  # 16
    "Lagos": [
        "Agege", "Ajeromi-Ifelodun", "Alimosho", "Amuwo-Odofin", "Apapa",
        "Badagry", "Epe", "Eti-Osa", "Ibeju-Lekki", "Ifako-Ijaiye",
        "Ikeja", "Ikorodu", "Kosofe", "Lagos Island", "Lagos Mainland",
        "Mushin", "Ojo", "Oshodi-Isolo", "Shomolu", "Surulere",
    ],  # 20
    "Nasarawa": [
        "Akwanga", "Awe", "Doma", "Karu", "Keana", "Keffi", "Kokona",
        "Lafia", "Nasarawa", "Nasarawa Eggon", "Obi", "Toto", "Wamba",
    ],  # 13
    "Niger": [
        "Agaie", "Agwara", "Bida", "Borgu", "Bosso", "Chanchaga",
        "Edati", "Gbako", "Gurara", "Katcha", "Kontagora", "Lapai",
        "Lavun", "Magama", "Mariga", "Mashegu", "Mokwa", "Muya",
        "Paikoro", "Rafi", "Rijau", "Shiroro", "Suleja", "Tafa",
        "Wushishi",
    ],  # 25
    "Ogun": [
        "Abeokuta North", "Abeokuta South", "Ado-Odo/Ota", "Ewekoro",
        "Ifo", "Ijebu East", "Ijebu North", "Ijebu North East", "Ijebu Ode",
        "Ikenne", "Imeko Afon", "Ipokia", "Obafemi Owode", "Odeda",
        "Odogbolu", "Ogun Waterside", "Remo North", "Sagamu",
        "Yewa North", "Yewa South",
    ],  # 20
    "Ondo": [
        "Akoko North-East", "Akoko North-West", "Akoko South-East",
        "Akoko South-West", "Akure North", "Akure South", "Ese Odo",
        "Idanre", "Ifedore", "Ilaje", "Ile Oluji/Okeigbo", "Irele",
        "Odigbo", "Okitipupa", "Ondo East", "Ondo West", "Ose", "Owo",
    ],  # 18
    "Osun": [
        "Aiyedade", "Aiyedire", "Atakumosa East", "Atakumosa West",
        "Boluwaduro", "Boripe", "Ede North", "Ede South", "Egbedore",
        "Ejigbo", "Ife Central", "Ife East", "Ife North", "Ife South",
        "Ifedayo", "Ifelodun", "Ila", "Ilesa East", "Ilesa West",
        "Irepodun", "Irewole", "Isokan", "Iwo", "Obokun", "Odo-Otin",
        "Ola Oluwa", "Olorunda", "Oriade", "Orolu", "Osogbo",
    ],  # 30
    "Oyo": [
        "Afijio", "Akinyele", "Atiba", "Atisbo", "Egbeda",
        "Ibadan North", "Ibadan North-East", "Ibadan North-West",
        "Ibadan South-East", "Ibadan South-West", "Ibarapa Central",
        "Ibarapa East", "Ibarapa North", "Ido", "Irepo", "Iseyin",
        "Itesiwaju", "Iwajowa", "Kajola", "Lagelu", "Ogbomosho North",
        "Ogbomosho South", "Ogo Oluwa", "Olorunsogo", "Oluyole",
        "Ona Ara", "Orelope", "Ori Ire", "Oyo East", "Oyo West",
        "Saki East", "Saki West", "Surulere",
    ],  # 33
    "Plateau": [
        "Barkin Ladi", "Bassa", "Bokkos", "Jos East", "Jos North",
        "Jos South", "Kanam", "Kanke", "Langtang North", "Langtang South",
        "Mangu", "Mikang", "Pankshin", "Qua'an Pan", "Riyom", "Shendam",
        "Wase",
    ],  # 17
    "Rivers": [
        "Abua/Odual", "Ahoada East", "Ahoada West", "Akuku-Toru", "Andoni",
        "Asari-Toru", "Bonny", "Degema", "Eleme", "Emohua", "Etche",
        "Gokana", "Ikwerre", "Khana", "Obio/Akpor", "Ogba/Egbema/Ndoni",
        "Ogu/Bolo", "Okrika", "Omuma", "Opobo/Nkoro", "Oyigbo",
        "Port Harcourt", "Tai",
    ],  # 23
    "Sokoto": [
        "Binji", "Bodinga", "Dange Shuni", "Gada", "Goronyo", "Gudu",
        "Gwadabawa", "Illela", "Isa", "Kebbe", "Kware", "Rabah",
        "Sabon Birni", "Shagari", "Silame", "Sokoto North", "Sokoto South",
        "Tambuwal", "Tangaza", "Tureta", "Wamako", "Wurno", "Yabo",
    ],  # 23
    "Taraba": [
        "Ardo Kola", "Bali", "Donga", "Gashaka", "Gassol", "Ibi",
        "Jalingo", "Karim Lamido", "Kurmi", "Lau", "Sardauna", "Takum",
        "Ussa", "Wukari", "Yorro", "Zing",
    ],  # 16
    "Yobe": [
        "Bade", "Bursari", "Damaturu", "Fika", "Fune", "Geidam", "Gujba",
        "Gulani", "Jakusko", "Karasuwa", "Machina", "Nangere", "Nguru",
        "Potiskum", "Tarmuwa", "Yunusari", "Yusufari",
    ],  # 17
    "Zamfara": [
        "Anka", "Bakura", "Birnin Magaji/Kiyaw", "Bukkuyum", "Bungudu",
        "Gummi", "Gusau", "Kaura Namoda", "Maradun", "Maru", "Shinkafi",
        "Talata Mafara", "Tsafe", "Zurmi",
    ],  # 14
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
        "prefixes": [
            "0703", "0706", "0803", "0806", "0810", "0813", "0814", "0816",
            "0903", "0906", "0913", "0916",
        ],
    },
    "Airtel": {
        "prefixes": [
            "0701", "0702", "0708", "0802", "0808", "0812",
            "0901", "0902", "0904", "0907", "0912",
        ],
    },
    "Glo": {
        "prefixes": [
            "0705", "0805", "0807", "0811", "0815", "0905", "0915",
        ],
    },
    "9mobile": {
        "prefixes": [
            "0809", "0817", "0818", "0908", "0909",
        ],
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
