import numpy as np
import networkx as nx
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.cluster.util import cosine_distance

def read_sentences(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Basic sentence splitting
    raw_sentences = [s.strip() for s in text.strip().split(". ") if s]

    # Clean and tokenize
    sentence_tokens = []
    for sentence in raw_sentences:
        cleaned = re.sub(r"[^a-zA-Z]", " ", sentence).lower()
        tokens = [word for word in cleaned.split() if word not in ENGLISH_STOP_WORDS]
        sentence_tokens.append(tokens)

    return raw_sentences, sentence_tokens

def sentence_similarity(sent1, sent2):
    all_words = list(set(sent1 + sent2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in sent1:
        vector1[all_words.index(word)] += 1
    for word in sent2:
        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences_tokenized):
    size = len(sentences_tokenized)
    sim_matrix = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            sim_matrix[i][j] = sentence_similarity(sentences_tokenized[i], sentences_tokenized[j])

    return sim_matrix

def summarize(file_path, top_n=3):
    original_sentences, tokenized_sentences = read_sentences(file_path)
    similarity_matrix = build_similarity_matrix(tokenized_sentences)

    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    ranked = sorted(((scores[i], s) for i, s in enumerate(original_sentences)), reverse=True)
    selected = [ranked[i][1] for i in range(min(top_n, len(ranked)))]

    print("\n🔹 Summarized Text:\n")
    for i, sent in enumerate(selected, 1):
        print(f"{i}. {sent}")

# Run summarizer
summarize("example.txt", top_n=3)

