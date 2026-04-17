def forward_chaining(likes, comments, shares, sentiment_val):
    """
    Menarik kesimpulan dari pemicu/fakta yang ada di depan (input).
    Aturan:
    1. Jika interaksi sangat tinggi (likes > 5000) dan sentimen Positif (2) -> Niat Beli Sangat Tinggi.
    2. Jika interaksi sedang (likes > 1000) dan sentimen Positif (2) -> Niat Beli Cukup Tinggi.
    3. Jika sentimen Negatif (0) -> Niat Beli Rendah (apapun interaksinya, karena biasanya kontroversial).
    4. Jika interaksi rendah namun sentimen Netral/Positif -> Niat Beli Rendah.
    """
    total_engagement = likes + (comments * 10) + (shares * 20)
    
    if sentiment_val == 0:
        return "Kemungkinan Beli RENDAH (Faktor Sentimen Negatif dominan)."
        
    if total_engagement > 10000 and sentiment_val == 2:
        return "Kemungkinan Beli SANGAT TINGGI (Interaksi Masif & Sentimen Positif)."
        
    if total_engagement > 3000 and sentiment_val == 2:
        return "Kemungkinan Beli TINGGI (Interaksi Bagus & Sentimen Positif)."
        
    if total_engagement > 3000 and sentiment_val == 1:
        return "Kemungkinan Beli SEDANG (Interaksi Bagus, namun Sentimen Netral)."
        
    return "Kemungkinan Beli RENDAH (Interaksi kurang memadai)."

def backward_chaining(goal_intent_high, likes, comments, shares, sentiment_val):
    """
    Backward chaining memvalidasi hasil prediksi (Goal).
    Jika goal_intent_high = True, apakah masuk akal berdasarkan rule?
    Rule untuk memvalidasi High Intent:
    - Harus tidak memiliki sentimen negatif (sentiment_val != 0).
    - Harus ada tingkat interaksi minimal (likes > 500).
    """
    if goal_intent_high:
        reasons = []
        is_valid = True
        
        if sentiment_val == 0:
            reasons.append("Terdapat sentimen negatif, seharusnya tidak memiliki Niat Beli tinggi.")
            is_valid = False
        else:
            reasons.append("Sentimen mendukung (Tidak negatif).")
            
        if likes < 500:
            reasons.append("Jumlah Like terlalu rendah untuk mendukung niat beli.")
            is_valid = False
        else:
            reasons.append("Jumlah Like memadai (>= 500).")
            
        return is_valid, reasons
    else:
        # Jika prediksi Machine Learning adalah LOW
        reasons = []
        is_valid = True
        
        if sentiment_val == 2 and likes > 5000:
            reasons.append("Sentimen sangat positif dan likes banyak, anomali jika prediksi Low.")
            is_valid = False
        elif sentiment_val == 0:
            reasons.append("Sentimen negatif menjelaskan mengapa prediksinya Low.")
        else:
            reasons.append("Kombinasi interaksi dan respon belum cukup untuk membentuk Niat Beli yang kuat.")
            
        return is_valid, reasons
