# CS5720 — Home Assignment 4

**Student:** Ramya Modala  
**University:** University of Central Missouri  
**Course:** CS5720 Neural Networks and Deep Learning (Fall 2026)

## Files
- `nlp_preprocessing.py`: tokenization, English stopword filtering, Porter stemming.
- `ner_spacy.py`: spaCy named-entity extraction with labels and character offsets.
- `scaled_attention.py`: NumPy scaled dot-product attention with row-wise stable softmax.
- `Home_Assignment_4_Solutions.docx`: written answers and source code.

## Setup
```bash
pip install numpy nltk spacy
python -m spacy download en_core_web_sm
```
NLTK stopwords are downloaded automatically if missing.

## Run
```bash
python nlp_preprocessing.py
python ner_spacy.py
python scaled_attention.py
```

## Results
The attention script prints weights approximately `[[0.731, 0.269], [0.269, 0.731]]` and output `[[2.076, 3.076, 4.076, 5.076], [3.924, 4.924, 5.924, 6.924]]`. NER outputs may vary across spaCy model versions.

## Submission
Push these files to GitHub, verify scripts locally, and submit the repository link in Brightspace.
