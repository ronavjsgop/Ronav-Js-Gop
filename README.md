# Information Retrieval Project – Cybersecurity & Network Security

## Project Title
**Information Retrieval System for Cybersecurity and Network Security Documents Using TF-IDF and Cosine Similarity**

## Objective
To build a simple Information Retrieval system that preprocesses cybersecurity documents, converts them into TF-IDF vectors, processes a user query, and ranks documents using cosine similarity.

## Folder Structure
- `data/` – corpus information
- `src/` – Python source files
- `outputs/` – generated TF-IDF matrix and search results
- `docs/` – project report/documentation
- `tests/` – test area
- `requirements.txt` – required Python libraries
- `run_project.py` – main execution file

## How to Run
1. Install Python 3.9 or later.
2. Open a terminal in this project folder.
3. Run:
   `pip install -r requirements.txt`
4. Run:
   `python run_project.py`

## IR Pipeline
Documents → Preprocessing → TF-IDF → Query Processing → Cosine Similarity → Ranking → Results

## Techniques Used
- Tokenization
- Lowercasing
- Stop-word removal
- Porter stemming
- TF-IDF
- Vector Space Model
- Cosine similarity
- Object-oriented implementation
