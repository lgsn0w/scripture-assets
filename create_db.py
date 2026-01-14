import json
import sqlite3
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os


MODEL_NAME = "all-MiniLM-L6-v2"
DB_NAME = "bible_vectors.db"
BIBLE_URL = "https://raw.githubusercontent.com/thiagobodruk/bible/master/json/en_kjv.json"

def get_bible_data():
    print(f"Downloading Bible from {BIBLE_URL}...")
    r = requests.get(BIBLE_URL)
    
    try:
        data = json.loads(r.content.decode('utf-8-sig'))
    except json.JSONDecodeError:
        data = json.loads(r.content.decode('utf-8'))

    verses = []
    for book in data:
        book_name = book['name']
        for chapter_idx, chapter in enumerate(book['chapters']):
            chapter_num = chapter_idx + 1
            for verse_idx, verse_text in enumerate(chapter):
                verse_num = verse_idx + 1
                verses.append({
                    "reference": f"{book_name} {chapter_num}:{verse_num}",
                    "text": verse_text
                })
    print(f"Downloaded {len(verses)} verses.")
    return verses

def create_database(verses):
    print(f"Loading AI Model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE verses (
            id INTEGER PRIMARY KEY,
            reference TEXT,
            text TEXT,
            embedding BLOB
        )
    ''')

    print("Generating vectors and saving to DB... (This will take a while!)")
    
    batch_size = 100
    for i in tqdm(range(0, len(verses), batch_size)):
        batch = verses[i:i+batch_size]
        texts = [v['text'] for v in batch]
        
        embeddings = model.encode(texts)
        
        rows_to_insert = []
        for j, verse in enumerate(batch):
            vector_bytes = embeddings[j].astype(np.float32).tobytes()
            rows_to_insert.append((verse['reference'], verse['text'], vector_bytes))
        
        cursor.executemany('INSERT INTO verses (reference, text, embedding) VALUES (?, ?, ?)', rows_to_insert)
        conn.commit()

    conn.close()
    print(f"Success! Database saved as '{DB_NAME}'")

if __name__ == "__main__":
    bible_verses = get_bible_data()
    create_database(bible_verses)