import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))

class IRSystem:
    """TF-IDF + cosine-similarity Information Retrieval system."""

    def __init__(self, documents):
        self.raw_documents = documents
        self.stemmer = PorterStemmer()
        self.stop_words = STOP_WORDS
        self.vectorizer = TfidfVectorizer()

        self.processed_corpus = {
            doc_id: self._preprocess(text)
            for doc_id, text in self.raw_documents.items()
        }
        self.doc_ids = list(self.processed_corpus.keys())
        self.tfidf_matrix = self.vectorizer.fit_transform(
            list(self.processed_corpus.values())
        )
        self.feature_names = self.vectorizer.get_feature_names_out()

    def _preprocess(self, text):
        text = re.sub(r"[^a-zA-Z\\s]", "", text.lower())
        tokens = text.split()
        cleaned = [
            self.stemmer.stem(word)
            for word in tokens
            if word not in self.stop_words and len(word) > 1
        ]
        return " ".join(cleaned)

    def preprocessing_results(self, limit=3):
        rows = []
        for doc_id, raw in list(self.raw_documents.items())[:limit]:
            rows.append({
                "Doc ID": doc_id,
                "Raw": raw,
                "Preprocessed": self.processed_corpus[doc_id]
            })
        return pd.DataFrame(rows)

    def tfidf_summary(self):
        return {
            "unique_features": len(self.feature_names),
            "matrix_shape": self.tfidf_matrix.shape
        }

    def search(self, query, top_k=5):
        processed_q = self._preprocess(query)
        query_vector = self.vectorizer.transform([processed_q])
        scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        ranked_indices = np.argsort(scores)[::-1]

        results = []
        for rank, idx in enumerate(ranked_indices[:top_k], start=1):
            doc_id = self.doc_ids[idx]
            results.append({
                "Rank": rank,
                "Doc ID": doc_id,
                "Score": round(float(scores[idx]), 4),
                "Text": self.raw_documents[doc_id]
            })
        return processed_q, pd.DataFrame(results)

    def evaluation(self):
        raw_words = re.sub(
            r"[^a-zA-Z\\s]", "", " ".join(self.raw_documents.values()).lower()
        ).split()
        unique_raw = set(raw_words)
        processed_vocab = set(self.feature_names)
        reduction = ((len(unique_raw) - len(processed_vocab)) / len(unique_raw)) * 100
        return {
            "raw_word_count": len(raw_words),
            "unique_raw_vocabulary": len(unique_raw),
            "processed_features": len(processed_vocab),
            "vocabulary_reduction_percent": round(reduction, 2)
        }
