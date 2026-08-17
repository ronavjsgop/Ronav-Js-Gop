from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from documents import documents
from ir_system import IRSystem

OUTPUT = ROOT / "outputs"
OUTPUT.mkdir(exist_ok=True)

engine = IRSystem(documents)

print("=== OUTPUT: PREPROCESSING RESULTS ===")
print(engine.preprocessing_results(limit=3).to_string(index=False))
print()

summary = engine.tfidf_summary()
print("=== OUTPUT: TF-IDF MATRIX SUMMARY ===")
print(f"Total Unique Vocabulary Features : {summary['unique_features']}")
print(f"Matrix Dimension (Docs x Terms) : {summary['matrix_shape']}")
print()

# Export TF-IDF matrix
tfidf_df = pd.DataFrame(
    engine.tfidf_matrix.toarray(),
    index=engine.doc_ids,
    columns=engine.feature_names
)
tfidf_df.to_csv(OUTPUT / "tfidf_matrix.csv")

query1 = "access control and authentication for network security"
processed_q, results = engine.search(query1, top_k=3)

print("=== QUERY EXECUTION OUTPUT ===")
print(f"RAW QUERY : '{query1}'")
print(f"PROCESSED QUERY : '{processed_q}'")
print(results.to_string(index=False))
results.to_csv(OUTPUT / "query_results.csv", index=False)
print()

evaluation = engine.evaluation()
print("=== PIPELINE EVALUATION METRICS ===")
print(f"Total Raw Corpus Word Count : {evaluation['raw_word_count']}")
print(f"Unique Raw Vocabulary Terms : {evaluation['unique_raw_vocabulary']}")
print(f"Processed Features (TF-IDF) : {evaluation['processed_features']}")
print(f"Vocabulary Dimension Reduction: {evaluation['vocabulary_reduction_percent']:.2f}%")
print()

query2 = "malware detection using deep learning"
processed_q2, class_results = engine.search(query2, top_k=2)

print("=== CLASS SEARCH RESULTS ===")
print(class_results.to_string(index=False))
class_results.to_csv(OUTPUT / "class_search_results.csv", index=False)

# Save preprocessing output
engine.preprocessing_results(limit=10).to_csv(
    OUTPUT / "preprocessing_results.csv", index=False
)

print("\nAll output files saved in the outputs folder.")
