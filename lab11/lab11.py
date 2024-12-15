from striprtf.striprtf import rtf_to_text
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
import math
from collections import Counter

input_file_train = 'C:\labsCV\lab11\harry_potter.txt'
input_file_test = 'C:\labsCV\lab11\harry_potter_test.txt'

with open(input_file_train, 'r', encoding='utf-8') as file:
    rtf_content_train = file.read()
train_text = rtf_to_text(rtf_content_train)

with open(input_file_test, 'r', encoding='utf-8') as file:
    rtf_content_test = file.read()
test_text = rtf_to_text(rtf_content_test)

def get_bigrams(text):
    cleaned_text = re.sub(r'[^a-zA-Z]', '', text.lower())
    bigrams = [cleaned_text[i:i+2] for i in range(len(cleaned_text)-1)]
    return Counter(bigrams)

def save_bigram_model(model, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        for bigram, count in bigram_model.items():
            f.write(f"{bigram}: {count}\n")

def load_bigram_model(filepath):
    bigram_model = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            bigram, count = line.strip().split(': ')
            bigram_model[bigram] = int(count)
    return bigram_model

    
bigram_model = get_bigrams(train_text)

save_bigram_model(bigram_model, 'C:\labsCV\lab11\\bigram_model.txt')
print("Bigram model saved to 'bigram_model.txt'.")

loaded_bigram_model = load_bigram_model('C:\labsCV\lab11\\bigram_model.txt')
print("Bigram model loaded from 'bigram_model.txt'.")

def calculate_likelihood(text, bigram_model):
    bigrams = get_bigrams(text)
    log_likelihood = 0

    for bigram, count in bigrams.items():
        freq = bigram_model.get(bigram, 0)
        if freq > 0:
            log_likelihood += count * math.log(freq)
        else:
            log_likelihood += count * math.log(1e-10)  

    num_bigrams = sum(bigrams.values())
    if num_bigrams > 1:
        log_likelihood /= num_bigrams

    return log_likelihood

def generate_random_sentence(length_range=(50, 250), alphabet_probs=None):
    alphabet = 'abcdefghijklmnopqrstuvwxyz '
    if not alphabet_probs:
        alphabet_probs = {char: 1/len(alphabet) for char in alphabet}
    population, weights = zip(*alphabet_probs.items())
    length = random.randint(*length_range)
    return ''.join(random.choices(population, weights=weights, k=length))

# Побудова ймовірностей букв з навчального тексту
def build_alphabet_probs(text):
    cleaned_text = re.sub(r'[^a-zA-Z]', '', text.lower())
    total_chars = len(cleaned_text)
    char_counts = Counter(cleaned_text)
    return {char: count / total_chars for char, count in char_counts.items()}

alphabet_probs = build_alphabet_probs(train_text)

test_sentences = [sentence.strip() for sentence in test_text.split('.') if sentence.strip()]
fake_sentences = [generate_random_sentence(alphabet_probs=alphabet_probs) for _ in test_sentences]

real_likelihoods = [calculate_likelihood(sentence, loaded_bigram_model) for sentence in test_sentences]
fake_likelihoods = [calculate_likelihood(sentence, loaded_bigram_model) for sentence in fake_sentences]

threshold = (np.median(real_likelihoods) + np.median(fake_likelihoods)) / 2
print("Threshold: "+str(threshold))

sns.kdeplot(real_likelihoods, label='Real Sentences', fill=True, color='blue', alpha=0.5)
sns.kdeplot(fake_likelihoods, label='Fake Sentences', fill=True, color='red', alpha=0.5)
plt.axvline(threshold, color='black', linestyle='--', label='Threshold')
plt.legend()
plt.title('Distribution of Log-Likelihoods')
plt.xlabel('Log-Likelihood')
plt.ylabel('Density')
plt.show()

def check_sentence(text, threshold, bigram_model):
    likelihood = calculate_likelihood(text, bigram_model)
    print("Likelihood: "+str(likelihood))
    if likelihood > threshold:
        print("The sentence is likely REAL.")
    else:
        print("The sentence is likely FAKE.")

while True:
    user_input = input("Enter a sentence to check (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    check_sentence(user_input, threshold, loaded_bigram_model)
