import re

# Kamus sederhana untuk analisis sentimen lexicon-based
POSITIVE_WORDS = ['bagus', 'keren', 'mantap', 'beli', 'mau', 'suka', 'minat', 'pesan', 'top', 'murah', 'rekomendasi', 'cantik', 'wow', 'good']
NEGATIVE_WORDS = ['jelek', 'mahal', 'buruk', 'kecewa', 'lama', 'rusak', 'penipu', 'jangan', 'nggak', 'tidak', 'kurang', 'bad']

def analyze_sentiment(text):
    """
    Analisis sentimen kalimat dalam bahasa Indonesia menggunakan lexicon base sederhana.
    Return:
        score: 0 (Negatif), 1 (Netral), 2 (Positif)
        label: 'Negatif', 'Netral', 'Positif'
    """
    if not text:
        return 1, 'Netral'
        
    text = text.lower()
    # Hapus tanda baca
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    
    pos_count = sum(1 for word in words if word in POSITIVE_WORDS)
    neg_count = sum(1 for word in words if word in NEGATIVE_WORDS)
    
    if pos_count > neg_count:
        return 2, 'Positif'
    elif neg_count > pos_count:
        return 0, 'Negatif'
    else:
        return 1, 'Netral'
