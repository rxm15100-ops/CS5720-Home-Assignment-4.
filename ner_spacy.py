"""CS5720 Home Assignment 4, Part II Q2: spaCy NER."""
import spacy

def extract_entities(sentence):
    try:
        nlp = spacy.load('en_core_web_sm')
    except OSError as exc:
        raise RuntimeError('Install model: python -m spacy download en_core_web_sm') from exc
    doc = nlp(sentence)
    for ent in doc.ents:
        print(f'{ent.text!r} | {ent.label_} | start={ent.start_char} | end={ent.end_char}')
    return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

if __name__ == '__main__':
    extract_entities(' LeBron James as the NBA player of the United States in Laker and won the NBA Final MVP in 2020.')
