import spacy

# Load the large English NLP model
nlp = spacy.load('en_core_web_lg')

# Replace a token with "REDACTED" if it is a name
def replace_name_with_placeholder(token):
    if token.ent_iob != 0 and token.ent_type_ == "PERSON":
        return "[REDACTED] "
    else:
        return token.string

# Loop through all the entities in a document and check if they are names
def scrub(text):
    doc = nlp(text)
    for ent in doc.ents:
        ent.merge()
    tokens = map(replace_name_with_placeholder, doc)
    return "".join(tokens)

s = """
Abdul Kalam was an Indian aerospace scientist and politician who served 
as the 11th President of India from 2002 to 2007. Narendra Damodardas Modi 
is an Indian politician serving as the 14th and current Prime Minister of India
 since 2014. He was the Chief Minister of Gujarat from 2001 to 2014 and is the 
 Member of Parliament for Varanasi.
"""

print(scrub(s))