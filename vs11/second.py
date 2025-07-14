import nltk
import scipy 
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re

def read_text(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        text = file.read()
    # Split into sentences
    sentences = text.strip().split(". ")
    cleaned_sentences = []
    for sentence in sentences:
        # Remove non-letter characters and tokenize
        cleaned = re.sub(r"[^a-zA-Z]", " ", sentence)
        tokens = cleaned.split()
        cleaned_sentences.append(tokens)
    return cleaned_sentences

def sentence_similarity(sent1, sent2, stop_words=None):
    if stop_words is None:
        stop_words = []

    # Lowercase words
    sent1 = [w.lower() for w in sent1 if w.lower() not in stop_words]
    sent2 = [w.lower() for w in sent2 if w.lower() not in stop_words]

    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        vector1[all_words.index(w)] += 1

    for w in sent2:
        vector2[all_words.index(w)] += 1

    return 1 - cosine_distance(vector1, vector2)

def gen_sim_matrix(sentences, stop_words=None):
    size = len(sentences)
    similarity_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == j:
                continue
            similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)
    return similarity_matrix

def generate_summary(file_name, top_n=5):
    stop_words = set(stopwords.words('english'))
    sentences = read_text(file_name)
    sim_matrix = gen_sim_matrix(sentences, stop_words)
    graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    summarize_text = []
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentences[i][1]))

    print("Summarized Text:\n", ". ".join(summarize_text))

# Call the function
generate_summary("example.txt", 2)
