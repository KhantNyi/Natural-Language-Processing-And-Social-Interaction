import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

with open ('The_Adventures_of_Sherlock_Holmes.txt', 'r') as f:
    text = f.read()
doc = nlp(text)

# Noun chunks
noun_chunks = [chunk.text.lower() for chunk in doc.noun_chunks]

noun_chunk_counts = Counter(noun_chunks).most_common(20)

print("=== Top 20 Noun Chunks ===")
for chunk, freq in noun_chunk_counts:
    print(f"{chunk} -> {freq}")
print()

# Person, ORG, Location

persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
locations = [ent.text for ent in doc.ents if ent.label_ in ("GPE", "LOC")]

top_persons = Counter(persons).most_common(10)
top_orgs = Counter(orgs).most_common(10)
top_locations = Counter(locations).most_common(10)

print("=== Top 10 PERSONS ===")
for name, count in top_persons:
    print(f"{name}: {count}")
print()

print("=== Top 10 ORGANIZATIONS ===")
for name, count in top_orgs:
    print(f"{name}: {count}")
print()

print("=== Top 10 LOCATIONS (GPE/LOC) ===")
for name, count in top_locations:
    print(f"{name}: {count}")
