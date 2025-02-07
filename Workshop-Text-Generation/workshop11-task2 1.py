import spacy
from collections import Counter

# 1) Load a more accurate model
nlp = spacy.load("en_core_web_trf")
nlp.max_length = 9999999 # Increase the max length since book 3 and 4 contain more than 1 million characters

ruler = nlp.add_pipe("entity_ruler", before="ner")


# Alias map to unify character references
CHARACTER_ALIAS = {
    "Harry": "Harry Potter",
    "Potter": "Harry Potter",
    "Mr. Potter": "Harry Potter",
    "Hermione": "Hermione Granger",
    "Granger": "Hermione Granger",
    "Miss Granger": "Hermione Granger"
}

def analyze_book(file_path):
    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        book_text = file.read()
    
    doc = nlp(book_text)
    
    # Extract person names
    raw_characters = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    # Normalize them via the alias map
    normalized_characters = []
    for ch in raw_characters:
        if ch in CHARACTER_ALIAS:
            normalized_characters.append(CHARACTER_ALIAS[ch])
        else:
            normalized_characters.append(ch)
    
    # Extract locations (GPE, LOC, FAC)
    places = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC", "FAC"]]
    
    # Count occurrences
    character_counts = Counter(normalized_characters).most_common()
    place_counts = Counter(places).most_common(15)
    
    return character_counts, place_counts

def display_results(book_title, characters, places):
    print(f"\n--- Analysis for {book_title} ---")
    print("\nLeading Characters (Top 10):")
    for name, count in characters[:10]:
        print(f"{name}: {count}")
    
    print("\nImportant Places (Top 15):")
    for place, count in places:
        print(f"{place}: {count}")

books = [
    ("Harry_Potter_1_Sorcerers_Stone.txt", "Harry Potter and the Sorcerer's Stone"),
    ("Harry_Potter_2-The_Chamber_of_Secrets.txt", "Harry Potter and the Chamber of Secrets"),
    ("Harry_Potter_3_Prisoner_of_Azkaban.txt", "Harry Potter and Prisoner of Azkaban"),
    ("Harry_Potter_4_The_Goblet_of_Fire.txt", "Harry Potter and the Goblet of Fire"),
]

for file_path, book_title in books:
    characters, places = analyze_book(file_path)
    display_results(book_title, characters, places)
