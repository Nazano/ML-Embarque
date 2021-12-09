import re
import spacy
import en_core_web_sm
import fasttext

nlp = en_core_web_sm.load()

def preprocess_text(text):
    '''
    Returns a list of words of a preprocessed document
        Parameter:
            text(str): The document
        Return:
            The proprocessed text
    '''
    # Remove delimiters
    text = re.sub(r'[\r\n\t]+', ' ', text)

    # Remove all the special characters
    text = re.sub(r'\W', ' ', text)
    # Remove numbers
    text = re.sub(r'[0-9]+', ' ', text)
    # Replace multiple space by one
    text = re.sub(r' +', ' ', text)

    # Converting to Lowercase
    text = text.lower()

    # Lemmatization
    tokens = [token.lemma_ for token in nlp(text) if token.lemma_]
    # Remove french stop words
    #tokens = [word for word in tokens if word not in fr_stop]

    return " ".join(tokens)
