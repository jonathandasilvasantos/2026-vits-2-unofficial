"""
Text processing for VITS-2 with normalized text input.
VITS2 paper Section 3: "For the experiment with normalized texts, we
normalized the input text with simple rules using open-source
software [23]" (keithito/tacotron).
VITS2 paper Section 4.4: Demonstrates reduced dependency on phoneme
conversion, enabling a fully end-to-end single-stage approach.

Text normalization rules adapted from keithito/tacotron:
- Lowercase
- Expand numbers to words
- Expand abbreviations
- Remove unsupported characters
"""
import re
import unicodedata

# Number to words conversion
ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
TENS = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
]

# Common abbreviations
ABBREVIATIONS = {
    "mrs": "misess",
    "mr": "mister",
    "dr": "doctor",
    "st": "saint",
    "co": "company",
    "jr": "junior",
    "maj": "major",
    "gen": "general",
    "drs": "doctors",
    "rev": "reverend",
    "lt": "lieutenant",
    "hon": "honorable",
    "sgt": "sergeant",
    "capt": "captain",
    "esq": "esquire",
    "ltd": "limited",
    "col": "colonel",
    "ft": "fort",
}


def _number_to_words(n):
    """Convert integer to English words."""
    if n < 0:
        return "minus " + _number_to_words(-n)
    if n == 0:
        return "zero"
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else " " + ONES[n % 10])
    if n < 1000:
        return (
            ONES[n // 100] + " hundred"
            + ("" if n % 100 == 0 else " " + _number_to_words(n % 100))
        )
    if n < 1000000:
        return (
            _number_to_words(n // 1000) + " thousand"
            + ("" if n % 1000 == 0 else " " + _number_to_words(n % 1000))
        )
    if n < 1000000000:
        return (
            _number_to_words(n // 1000000) + " million"
            + ("" if n % 1000000 == 0 else " " + _number_to_words(n % 1000000))
        )
    return (
        _number_to_words(n // 1000000000) + " billion"
        + ("" if n % 1000000000 == 0 else " " + _number_to_words(n % 1000000000))
    )


def _expand_number(match):
    """Expand a number match to words."""
    num = int(match.group(0))
    return _number_to_words(num)


def _expand_decimal(match):
    """Expand decimal numbers."""
    integer_part = match.group(1)
    decimal_part = match.group(2)
    result = _number_to_words(int(integer_part)) + " point"
    for digit in decimal_part:
        result += " " + _number_to_words(int(digit))
    return result


def _expand_dollars(match):
    """Expand dollar amounts."""
    amount = match.group(1)
    parts = amount.split(".")
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    result = ""
    if dollars:
        result += _number_to_words(dollars) + " dollar" + ("s" if dollars != 1 else "")
    if cents:
        if dollars:
            result += " "
        result += _number_to_words(cents) + " cent" + ("s" if cents != 1 else "")
    if not result:
        result = "zero dollars"
    return result


def _expand_ordinal(match):
    """Expand ordinal numbers (1st, 2nd, etc.)."""
    num = int(match.group(1))
    word = _number_to_words(num)
    # Simple ordinal suffix
    if word.endswith("one"):
        return word[:-3] + "first"
    elif word.endswith("two"):
        return word[:-3] + "second"
    elif word.endswith("three"):
        return word[:-5] + "third"
    elif word.endswith("five"):
        return word[:-4] + "fifth"
    elif word.endswith("eight"):
        return word + "h"
    elif word.endswith("nine"):
        return word[:-1] + "th"
    elif word.endswith("twelve"):
        return word[:-2] + "fth"
    elif word.endswith("y"):
        return word[:-1] + "ieth"
    else:
        return word + "th"


def normalize_text(text):
    """
    Normalize English text following keithito/tacotron approach.
    Per VITS2 paper: simple normalization rules.
    """
    # Convert to ASCII where possible
    text = unicodedata.normalize("NFKD", text)

    # Lowercase
    text = text.lower()

    # Expand abbreviations
    for abbr, expansion in ABBREVIATIONS.items():
        pattern = re.compile(r"\b" + abbr + r"\.", re.IGNORECASE)
        text = pattern.sub(expansion, text)

    # Expand dollar amounts
    text = re.sub(r"\$(\d+(?:\.\d+)?)", _expand_dollars, text)

    # Expand decimal numbers
    text = re.sub(r"(\d+)\.(\d+)", _expand_decimal, text)

    # Expand ordinals (1st, 2nd, etc.)
    text = re.sub(r"(\d+)(?:st|nd|rd|th)", _expand_ordinal, text)

    # Expand remaining numbers
    text = re.sub(r"\d+", _expand_number, text)

    # Remove unsupported characters, keep letters, punctuation, space, apostrophe, hyphen
    text = re.sub(r"[^a-z !,.:;?\-']", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def text_to_ids(text, symbol_to_id):
    """
    Convert normalized text string to a sequence of integer IDs.
    Adds BOS and EOS tokens.
    """
    ids = [symbol_to_id["^"]]  # BOS
    for char in text:
        if char in symbol_to_id:
            ids.append(symbol_to_id[char])
    ids.append(symbol_to_id["$"])  # EOS
    return ids


def ids_to_text(ids, id_to_symbol):
    """Convert a sequence of integer IDs back to text."""
    return "".join(id_to_symbol.get(i, "") for i in ids)
