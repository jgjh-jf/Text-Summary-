# 📝 Text Summarizer using TextRank

This project implements a simple **Text Summarization Tool** using the **TextRank algorithm**, which is inspired by Google's PageRank. It takes a plain text file as input and returns the most important sentences as a summary.

---

## 📂 Files Included

- `example.txt`: A sample input text describing human body systems.
- `first.py`: Main implementation of the TextRank-based summarizer using `scikit-learn` and `networkx`.
- `second.py`: Alternative implementation with more manual stopword processing using `nltk`.

---

## 📌 Features

- Cleans and tokenizes raw text input.
- Computes sentence similarity using cosine distance.
- Builds a similarity graph and ranks sentences using PageRank.
- Extracts the top N most relevant sentences as a summary.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
