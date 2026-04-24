# test.py — final version
from naijaml.nlp import analyze_sentiment
from naijaml.nlp.preprocess import normalize_pidgin_negation

def analyze_with_normalization(text):
    return analyze_sentiment(normalize_pidgin_negation(text))

test_cases = [
    ("This thing no bad at all", "positive"),
    ("E no be something wey bad", "positive"),
    ("The food no bad", "positive"),
    ("This thing no good at all", "negative"),
    ("E no sweet me", "negative"),
    ("I no like am", "negative"),
    ("This film too sweet!", "positive"),
    ("Omo this thing sweet die", "positive"),
    ("The film was bad", "negative"),
    ("I like this song", "positive"),
    ("E good like that", "positive"),
    ("Abeg no vex", "negative"),
    ("E no too bad sha", "positive"),
    ("I no go lie this thing good", "positive"),
]

print("=== RETRAINED MODEL — no normalization ===")
passed = 0
for text, expected in test_cases:
    result = analyze_sentiment(text)
    status = "✅" if result['label'] == expected else "❌"
    if result['label'] == expected: passed += 1
    print(f"{status} '{text}' = {result['label']}")
print(f"\n{passed}/{len(test_cases)} correct\n")

print("=== RETRAINED MODEL + NORMALIZATION ===")
passed = 0
for text, expected in test_cases:
    result = analyze_with_normalization(text)
    status = "✅" if result['label'] == expected else "❌"
    if result['label'] == expected: passed += 1
    print(f"{status} '{text}' = {result['label']}")
print(f"\n{passed}/{len(test_cases)} correct")