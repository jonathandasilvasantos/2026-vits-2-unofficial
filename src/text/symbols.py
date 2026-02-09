"""
Symbol set for VITS-2 with normalized text input.
VITS2 paper Section 3/4.4: Normalized text input using simple rules
via open-source software [23] (keithito/tacotron).
VITS2 key contribution: Reduces dependency on phoneme conversion,
allowing a fully end-to-end single-stage approach.
No blank token interspersion (unlike VITS).
"""

# Special symbols
PAD = "_"      # Padding symbol (index 0)
BOS = "^"      # Beginning of sentence
EOS = "$"      # End of sentence

# Character set for normalized English text
# Lowercase letters
LETTERS = list("abcdefghijklmnopqrstuvwxyz")

# Punctuation (subset used in normalized text)
PUNCTUATION = list("!,.:;? ")

# Special characters that may appear in normalized text
SPECIAL = ["-", "'"]

# Build symbol list: PAD + BOS + EOS + characters
SYMBOLS = [PAD, BOS, EOS] + LETTERS + PUNCTUATION + SPECIAL

# Symbol to index mapping
SYMBOL_TO_ID = {s: i for i, s in enumerate(SYMBOLS)}
ID_TO_SYMBOL = {i: s for i, s in enumerate(SYMBOLS)}

NUM_SYMBOLS = len(SYMBOLS)
