import nltk
nltk.download('punkt')
nltk.download('punkt_tab') 

import re
from collections import defaultdict, Counter
import random
import string

def load_and_preprocess_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    tokens = nltk.word_tokenize(text)
    
    return tokens

def create_bigrams(tokens):
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append((tokens[i], tokens[i+1]))
    return bigrams

def create_bigram_counts(tokens):

    from_bigram_to_next_token_counts = defaultdict(lambda: defaultdict(int))
    bigrams = create_bigrams(tokens)
    
    for i in range(len(bigrams) - 1):
        current_bigram = bigrams[i]
        next_word = bigrams[i+1][1]  # The second word of the next bigram
        from_bigram_to_next_token_counts[current_bigram][next_word] += 1
        
    return from_bigram_to_next_token_counts


def create_bigram_probabilities(from_bigram_to_next_token_counts):
    from_bigram_to_next_token_probs = {}
    
    for bigram, next_token_counts in from_bigram_to_next_token_counts.items():
        total_count = sum(next_token_counts.values())
        # Calculate the probability of each next token
        from_bigram_to_next_token_probs[bigram] = {
            token: count / total_count for token, count in next_token_counts.items()
        }
        
    return from_bigram_to_next_token_probs


def sample_next_token(bigram, from_bigram_to_next_token_probs):
    if bigram not in from_bigram_to_next_token_probs:
        # If the bigram is not found, return a random token or a placeholder
        return None
    
    next_tokens = list(from_bigram_to_next_token_probs[bigram].keys())
    probabilities = list(from_bigram_to_next_token_probs[bigram].values())
    
    next_token = random.choices(next_tokens, weights=probabilities, k=1)[0]
    return next_token

def generate_text_from_bigram(start_bigram, from_bigram_to_next_token_probs, num_words=50):
    text_tokens = list(start_bigram)
    
    current_bigram = start_bigram
    
    for _ in range(num_words - 2):
        next_token = sample_next_token(current_bigram, from_bigram_to_next_token_probs)
        if not next_token:
            break
        
        text_tokens.append(next_token)
        current_bigram = (current_bigram[1], next_token)
    
    return " ".join(text_tokens)


def create_trigrams(tokens):
    trigrams = []
    for i in range(len(tokens) - 2):
        trigrams.append((tokens[i], tokens[i+1], tokens[i+2]))
    return trigrams

def create_trigram_counts(tokens):
    from_trigram_to_next_token_counts = defaultdict(lambda: defaultdict(int))
    trigrams = create_trigrams(tokens)
    
    for i in range(len(trigrams) - 1):
        current_trigram = trigrams[i]
        next_token = trigrams[i+1][2]  # The third word of the next trigram
        from_trigram_to_next_token_counts[current_trigram][next_token] += 1
        
    return from_trigram_to_next_token_counts

def create_trigram_probabilities(from_trigram_to_next_token_counts):
    from_trigram_to_next_token_probs = {}
    
    for trigram, next_token_counts in from_trigram_to_next_token_counts.items():
        total_count = sum(next_token_counts.values())
        from_trigram_to_next_token_probs[trigram] = {
            token: count / total_count for token, count in next_token_counts.items()
        }
        
    return from_trigram_to_next_token_probs

def sample_next_token_trigram(trigram, from_trigram_to_next_token_probs):
    if trigram not in from_trigram_to_next_token_probs:
        return None
    
    next_tokens = list(from_trigram_to_next_token_probs[trigram].keys())
    probabilities = list(from_trigram_to_next_token_probs[trigram].values())
    next_token = random.choices(next_tokens, weights=probabilities, k=1)[0]
    return next_token

def generate_text_from_trigram(start_trigram, from_trigram_to_next_token_probs, num_words=50):
    text_tokens = list(start_trigram)
    current_trigram = start_trigram
    
    for _ in range(num_words - 3):
        next_token = sample_next_token_trigram(current_trigram, from_trigram_to_next_token_probs)
        if not next_token:
            break
        
        text_tokens.append(next_token)
        current_trigram = (current_trigram[1], current_trigram[2], next_token)
    
    return " ".join(text_tokens)


def create_quadgrams(tokens):
    quadgrams = []
    for i in range(len(tokens) - 3):
        quadgrams.append((tokens[i], tokens[i+1], tokens[i+2], tokens[i+3]))
    return quadgrams

def create_quadgram_counts(tokens):
    from_quadgram_to_next_token_counts = defaultdict(lambda: defaultdict(int))
    quadgrams = create_quadgrams(tokens)
    
    for i in range(len(quadgrams) - 1):
        current_quadgram = quadgrams[i]
        next_token = quadgrams[i+1][3]  
        from_quadgram_to_next_token_counts[current_quadgram][next_token] += 1
        
    return from_quadgram_to_next_token_counts

def create_quadgram_probabilities(from_quadgram_to_next_token_counts):
    from_quadgram_to_next_token_probs = {}
    
    for quadgram, next_token_counts in from_quadgram_to_next_token_counts.items():
        total_count = sum(next_token_counts.values())
        from_quadgram_to_next_token_probs[quadgram] = {
            token: count / total_count for token, count in next_token_counts.items()
        }
        
    return from_quadgram_to_next_token_probs

def sample_next_token_quadgram(quadgram, from_quadgram_to_next_token_probs):
    if quadgram not in from_quadgram_to_next_token_probs:
        return None
    
    next_tokens = list(from_quadgram_to_next_token_probs[quadgram].keys())
    probabilities = list(from_quadgram_to_next_token_probs[quadgram].values())
    next_token = random.choices(next_tokens, weights=probabilities, k=1)[0]
    return next_token

def generate_text_from_quadgram(start_quadgram, from_quadgram_to_next_token_probs, num_words=50):
    text_tokens = list(start_quadgram)
    current_quadgram = start_quadgram
    
    for _ in range(num_words - 4):
        next_token = sample_next_token_quadgram(current_quadgram, from_quadgram_to_next_token_probs)
        if not next_token:
            break
        
        text_tokens.append(next_token)
        current_quadgram = (current_quadgram[1], current_quadgram[2], current_quadgram[3], next_token)
    
    return " ".join(text_tokens)

def main():
    tokens = load_and_preprocess_text('shake_text.txt')
    

    bigram_counts = create_bigram_counts(tokens)
    bigram_probs = create_bigram_probabilities(bigram_counts)
    start_bigram = ('to', 'be')
    generated_text_bigram = generate_text_from_bigram(start_bigram, bigram_probs, num_words=100)
    print("Generated Text (Bigram):")
    print(generated_text_bigram)
    

    trigram_counts = create_trigram_counts(tokens)
    trigram_probs = create_trigram_probabilities(trigram_counts)
    start_trigram = ('to', 'be', 'or') 
    generated_text_trigram = generate_text_from_trigram(start_trigram, trigram_probs, num_words=100)
    print("\nGenerated Text (Trigram):")
    print(generated_text_trigram)
    

    quadgram_counts = create_quadgram_counts(tokens)
    quadgram_probs = create_quadgram_probabilities(quadgram_counts)
    start_quadgram = ('to', 'be', 'or', 'not')  
    generated_text_quadgram = generate_text_from_quadgram(start_quadgram, quadgram_probs, num_words=100)
    print("\nGenerated Text (Quadgram):")
    print(generated_text_quadgram)

if __name__ == "__main__":
    main()
