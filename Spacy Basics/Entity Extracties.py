import spacy

# Load the large English NLP model
nlp = spacy.load('en_core_web_lg')

# The text we want to examine
text = """Chennai , formerly known as Madras, the official name 
until 1996 is the capital of the Indian state of Tamil Nadu. 
Located on the Coromandel Coast of the Bay of Bengal, it is one of 
the largest cultural, economic and educational centres of south India. 
According to the 2011 Indian census, it is the sixth-most populous city 
and fourth-most populous urban agglomeration in India. The city together 
with the adjoining regions constitutes the Chennai Metropolitan Area, which 
is the 36th-largest urban area by population in the world. The traditional 
and de facto gateway of South India, Chennai is among the most-visited Indian 
cities by foreign tourists. It was ranked the 43rd-most visited city in the world 
for the year 2015. The Quality of Living Survey rated Chennai as the safest city 
in India. Chennai attracts 45 percent of health tourists visiting India, and 30 
to 40 percent of domestic health tourists. As such, it is termed "India's health 
capital". Chennai has the fifth-largest urban economy of India.
"""

# Parse the text with spaCy. This runs the entire pipeline.
doc = nlp(text)

# 'doc' now contains a parsed version of text. We can use it to do anything we want!
# For example, this will print out all the named entities that were detected:
for entity in doc.ents:
    print(f"{entity.text} ({entity.label_})")