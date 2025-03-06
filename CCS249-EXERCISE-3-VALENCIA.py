import wikipedia
from nltk import bigrams, trigrams
from nltk.tokenize import word_tokenize
from collections import defaultdict
# import nltk
import re
# import math

# # Download the punkt tokenizer models
# nltk.download('punkt_tab')

# Fetch the first 1000 characters of the page
page = wikipedia.page("Quantum_mechanics")
text = page.content[:1000]

print("\n\n")
print("=" * 50)
print("\n")

# Test the fetch of the first 1000 characters of the page
print(text)

print("\n")
print("=" * 50)
print("\n\n")

# Normalization
text = re.sub(r'[^\w\s]', '', text.lower())

# Tokenization
tokens = word_tokenize(text)

print("=" * 50)
print("\n")

print("Tokens:", tokens)

print("\n")
print("=" * 50)


# Vocabulary size
vocab_size = len(set(tokens))

# Train bigram model with Additive Smoothing
bigram_model = defaultdict(lambda: defaultdict(int))
for w1, w2 in bigrams(tokens):
    bigram_model[w1][w2] += 1

# Convert counts to probabilities with Additive Smoothing
alpha = 1  # Smoothing parameter
for w1 in bigram_model:
    total_count = float(sum(bigram_model[w1].values()) + alpha * vocab_size)
    for w2 in bigram_model[w1]:
        bigram_model[w1][w2] = (bigram_model[w1][w2] + alpha) / total_count

# Train trigram model with Additive Smoothing
trigram_model = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for w1, w2, w3 in trigrams(tokens):
    trigram_model[w1][w2][w3] += 1

# Convert counts to probabilities with Additive Smoothing
for w1 in trigram_model:
    for w2 in trigram_model[w1]:
        total_count = float(sum(trigram_model[w1][w2].values()) + alpha * vocab_size)
        for w3 in trigram_model[w1][w2]:
            trigram_model[w1][w2][w3] = (trigram_model[w1][w2][w3] + alpha) / total_count

print("\n\n")
print("=" * 50)
print("\n")

print("Bigram and trigram models have been trained with Additive Smoothing.")

print("\n")
print("=" * 50)


def bigram_perplexity(model, test_sentence):
    tokens = word_tokenize(test_sentence.lower())
    perplexity = 1
    N = len(tokens)
    for i in range(1, N):
        w1 = tokens[i-1]
        w2 = tokens[i]
        prob = model[w1][w2] if model[w1][w2] > 0 else 1e-6
        perplexity *= 1/prob
    return perplexity ** (1/float(N))

def trigram_perplexity(model, test_sentence):
    tokens = word_tokenize(test_sentence.lower())
    perplexity = 1
    N = len(tokens)
    for i in range(2, N):
        w1 = tokens[i-2]
        w2 = tokens[i-1]
        w3 = tokens[i]
        prob = model[w1][w2][w3] if model[w1][w2][w3] > 0 else 1e-6
        perplexity *= 1/prob
    return perplexity ** (1/float(N))

print("\n\n")
print("=" * 50)
print("\n")

# Test sentence
test_sentence = input("Test Sentence: ")

print("\n")
print("=" * 50)

# Calculate perplexity scores
bigram_score = bigram_perplexity(bigram_model, test_sentence)
trigram_score = trigram_perplexity(trigram_model, test_sentence)

print("\n\n")
print("=" * 50)
print("\n")

print(f"Bigram model perplexity -> Test Sentence \"{test_sentence}\" -> Score: {bigram_score}")
print ("\n")
print(f"Trigram model perplexity -> Test Sentence \"{test_sentence}\" -> Score: {trigram_score}")

print("\n")
print("=" * 50)
print("\n\n")