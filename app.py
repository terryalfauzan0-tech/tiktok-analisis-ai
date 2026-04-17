import streamlit as st
import pandas as pd
import plotly.express as px

from ml_pipeline import PipelineML, generate_dummy_data
from nlp_utils import analyze_sentiment
from rule_reasoning import forward_chaining, backward_chaining
from scraper_mock import scrape_tiktok_url

# Set page config
st.set_page_config(page_title="TikTok Purchase Intent Auto-Analyzer", layout="wide", page_icon="🛍️")

# Custom CSS for styling
st.markdown("""
<style>
.main {
    background-color: #0d1117;
    color: #e6edf3;
}
div.stButton > button:first-child {
    background-color: #2ea043;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
div.stButton > button:first-child:hover {
    background-color: #238636;
}
.metric-box {
    background-color: #161b22;
    border: 1px solid #30363d;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
}
h1, h2, h3 {
    color: #58a6ff;
}
</style>
""", unsafe_allow_html=True)

# Initialize ML Pipeline in Session State
if 'ml_pipeline' not in st.session_state:
    st.session_state.df = generate_dummy_data(1000)
    pipeline = PipelineML()
    metrics = pipeline.train_and_eval(st.session_state.df)
    st.session_state.pipeline = pipeline
    st.session_state.metrics = metrics

pipeline = st.session_state.pipeline
base_df = st.session_state.df

st.title("🛍️ TikTok Auto-Analyzer: Prediksi Niat Beli")
st.markdown("Cukup masukkan **Tautan Video TikTok**, AI akan menarik data komentar secara otomatis dan memprediksi berapa persen penonton yang memiliki **Niat Beli Tinggi**.")

st.markdown("---")

# Main Interface for URL Input
url_input = st.text_input("🔗 Masukkan URL Video TikTok", placeholder="https://www.tiktok.com/@username/video/1234567890")
analyze_btn = st.button("Mulai Analisis Otomatis 🚀")

if analyze_btn and url_input:
    with st.spinner("Mengakses server TikTok & mengekstrak data asli video dan komentar..."):
        # Penarikan data asli menggunakan API
        result = scrape_tiktok_url(url_input, num_comments=100)
        
    st.success("Ekstraksi data berhasil!")
    
    # Extract data
    vid_metrics = result['video_metrics']
    comments_df = result['comments_df']
    
    st.header("📊 Video Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown(f"<div class='metric-box'><h3>Total Likes</h3><h2>{vid_metrics['likes']:,}</h2></div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='metric-box'><h3>Komentar</h3><h2>{vid_metrics['comments_count']:,}</h2></div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='metric-box'><h3>Shares</h3><h2>{vid_metrics['shares']:,}</h2></div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='metric-box'><h3>Avg Watch (dtk)</h3><h2>{vid_metrics['watch_duration']}</h2></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.spinner("Menganalisis Maksud/Niat Pembelian dari 100 Komentar (NLP & ML)..."):
        # Memproses semua komentar menggunakan NLP & ML
        intent_results = []
        sentiment_labels = []
        sentiment_scores = []
        
        for index, row in comments_df.iterrows():
            # 1. Analisis Sentimen NLP
            sent_val, sent_label = analyze_sentiment(row['text'])
            sentiment_scores.append(sent_val)
            sentiment_labels.append(sent_label)
            
            # 2. Prediksi Machine Learning
            ml_pred, ml_prob = pipeline.predict_intent(
                row['likes'], row['comments'], row['shares'], row['watch_duration'], sent_val
            )
            intent_results.append("Tinggi" if ml_pred == 1 else "Rendah")
            
        comments_df['sentiment_score'] = sentiment_scores
        comments_df['sentiment_label'] = sentiment_labels
        comments_df['purchase_intent'] = intent_results
        
    st.header("🧠 Hasil Analisis AI")
    
    # Ringkasan Hasil
    high_intent_count = (comments_df['purchase_intent'] == "Tinggi").sum()
    low_intent_count = (comments_df['purchase_intent'] == "Rendah").sum()
    
    rc1, rc2 = st.columns(2)
    
    with rc1:
        # Pie chart purchase intent
        intent_counts = comments_df['purchase_intent'].value_counts().reset_index()
        intent_counts.columns = ['Purchase Intent', 'Jumlah Komentar']
        
        fig_intent = px.pie(
            intent_counts, 
            names='Purchase Intent', 
            values='Jumlah Komentar',
            title="🎯 Persentase Niat Beli Konsumen",
            color='Purchase Intent',
            color_discrete_map={"Tinggi": "#2ea043", "Rendah": "#f85149"},
            hole=0.4
        )
        st.plotly_chart(fig_intent, use_container_width=True)
        
    with rc2:
        # Pie chart sentiment
        sent_counts = comments_df['sentiment_label'].value_counts().reset_index()
        sent_counts.columns = ['Sentimen', 'Jumlah']
        
        fig_sent = px.bar(
            sent_counts, 
            x='Sentimen', 
            y='Jumlah',
            title="🗣️ Distribusi Sentimen Ulasan",
            color='Sentimen',
            color_discrete_map={"Positif": "#58a6ff", "Netral": "#8b949e", "Negatif": "#f85149"}
        )
        st.plotly_chart(fig_sent, use_container_width=True)
        
    # Detail Tabel
    st.subheader("📝 Daftar Ulasan dan Simpulan per User")
    
    # Helper to style the dataframe
    def color_intent(val):
        color = '#2ea043' if val == 'Tinggi' else '#f85149'
        return f'color: {color}'
        
    st.dataframe(
        comments_df[['username', 'text', 'sentiment_label', 'purchase_intent']]
        .style.map(color_intent, subset=['purchase_intent']),
        use_container_width=True, height=400
    )

elif analyze_btn and not url_input:
    st.warning("⚠️ Mohon masukkan tautan video TikTok terlebih dahulu!")
    
# Tambahkan info debugging
with st.expander("ℹ️ Informasi Sistem Utama (Mekanisme AI)"):
    st.markdown("""
    **Sistem ini beroperasi dengan teknologi berikut:**
    1. **Ekstraksi Data**: Script scraper mensimulasikan penarikan *batch content* (teks & metrik) dari TikTok.
    2. **NLP Lexicon**: Menganalisis kata kunci dalam komentar (*bag of words*) untuk menentukan sentimen *Positif / Negatif / Netral*.
    3. **Logistic Regression Classifier**: Dilatih atas ribuan *dummy data* untuk mencari pola relasi antara metrik _engagement_ video dan sentimen, memetakan luaran *Binary Class* (Niat Beli: Tinggi/Rendah).
    """)
