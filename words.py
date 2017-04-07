import re
import numpy as np

MAX_WORDS = 16

# NOTE: Vocabulary must contain the start and end tokens at these positions
START_TOKEN_IDX = 0
END_TOKEN_IDX = 1

words = open('vocabulary.txt').read().split()
vocab = {}
for i, word in enumerate(words):
    vocab[i] = word
    vocab[word] = i

VOCABULARY_SIZE = len(words)
UNKNOWN_IDX = vocab['thing']


def words(indices):
    return ' '.join(vocab[i] for i in indices)


# Remove punctuation 
pattern = re.compile('[\W_]+')
def process(text):
    text = text.lower()
    text = pattern.sub(' ', text.lower())
    return text


def indices(text):
    text = process(text)
    wordlist = ('000 ' + text + ' 001').split()
    return left_pad([vocab.get(w, UNKNOWN_IDX) for w in wordlist])


def left_pad(indices):
    res = np.zeros(MAX_WORDS, dtype=int)
    res[MAX_WORDS - len(indices):] = indices
    return res

