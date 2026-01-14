import sqlite3
import torch
import requests
from sentence_transformers import SentenceTransformer

model_name = "all-MiniLM-L6-v2"
print(f"Loading {model_name}...")
model = SentenceTransformer(model_name)

def get_portuguese_data():
    print("Downloading Portuguese Bible text...")

    import json
    import os
    
    if not os.path.exists("bible_pt.json"):
        print("Error: Please download 'bible_pt.json' first.")
        print("You can find one here: https://raw.githubusercontent.com/thiagobodruk/bible/master/json/pt_nvi.json")
        print("Save that raw text as 'bible_pt.json' in this folder and run again.")
        exit()

    with open("bible_pt.json", "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    verses = []
    
    for book in data:
        book_name = book["name"]
        chapters = book["chapters"]
        for chap_idx, chapter in enumerate(chapters):
            chap_num = chap_idx + 1
            for verse_idx, text in enumerate(chapter):
                verse_num = verse_idx + 1
                reference = f"{book_name} {chap_num}:{verse_num}"
                verses.append((reference, text))
                
    print(f"Loaded {len(verses)} verses in Portuguese.")
    return verses

def create_database(verses):
    print("Creating bible_pt_vectors.db...")
    conn = sqlite3.connect("bible_pt_vectors.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS verses (reference TEXT, text TEXT, embedding BLOB)")
    
    batch_size = 64
    total = len(verses)
    
    for i in range(0, total, batch_size):
        batch = verses[i:i+batch_size]
        refs = [v[0] for v in batch]
        texts = [v[1] for v in batch]
        
        embeddings = model.encode(texts, convert_to_tensor=True)
        
        for j in range(len(batch)):
            ref = refs[j]
            text = texts[j]
            emb_blob = embeddings[j].cpu().numpy().tobytes()
            c.execute("INSERT INTO verses VALUES (?, ?, ?)", (ref, text, emb_blob))
            
        conn.commit()
        print(f"Processed {i + len(batch)}/{total} verses...", end="\r")
        
    conn.close()
    print("\nSuccess! 'bible_pt_vectors.db' is ready.")

if __name__ == "__main__":
    data = get_portuguese_data()
    create_database(data)