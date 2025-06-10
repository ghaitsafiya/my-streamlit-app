import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import os # Untuk memeriksa keberadaan file

# --- Bagian 1: Definisi Model (Tidak perlu diubah, tinggal di copy-paste) ---
class IndoBERT_CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
        self.conv1 = nn.Conv1d(768, 128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad(): # BERT digunakan sebagai feature extractor
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x = x.permute(0, 2, 1)  # CNN butuh format (batch, channels, seq)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # LSTM butuh (batch, seq, channels)
        _, (h_n, _) = self.lstm(x)
        logits = self.fc(h_n.squeeze(0))
        return logits

# --- Bagian 2: Fungsi Utilitas (Pre-processing, Tokenisasi, Prediksi) ---

@st.cache_resource # Gunakan cache_resource karena memuat model dan tokenizer
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    model = IndoBERT_CNN_LSTM()
    # PENTING: Anda perlu menyimpan model yang sudah dilatih
    # Jika model.pth ada di repositori, muat di sini:
    # try:
    #     model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
    #     model.eval()
    # except FileNotFoundError:
    #     st.error("File model 'model.pth' tidak ditemukan. Harap pastikan model sudah dilatih dan disimpan.")
    #     # Jika tidak ada model.pth, Anda harus melatihnya terlebih dahulu dan menyimpannya.
    #     # Untuk deployment, model harus sudah ada, tidak dilatih saat runtime aplikasi.
    model.eval() # Pastikan model dalam mode evaluasi
    return tokenizer, model

@st.cache_data # Gunakan cache_data untuk pemrosesan teks yang mungkin berulang
def clean_text(text):
    text = str(text).lower()  # lowercase
    text = re.sub(r"http\S+|www.\S+", '', text)  # hapus URL
    text = re.sub(r"\d+", '', text)  # hapus angka
    text = re.sub(r"[^\w\s]", '', text)  # hapus tanda baca
    text = re.sub(r"\s+", ' ', text).strip()  # hapus spasi ganda
    return text

@st.cache_data
def predict_hoax_or_valid(text, tokenizer, model):
    clean = clean_text(text)
    tokenized_input = tokenizer(
        clean,
        padding='max_length',
        truncation=True,
        max_length=512, # Sesuaikan dengan max_length saat training
        return_tensors='pt'
    )
    input_ids = tokenized_input['input_ids']
    attention_mask = tokenized_input['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_id = torch.argmax(probabilities, dim=1).item()

    # Asumsi: 0 = valid, 1 = hoax (sesuai mapping Anda di training)
    label_map = {0: "VALID", 1: "HOAX"}
    prediction_label = label_map[predicted_class_id]
    confidence = probabilities[0][predicted_class_id].item() * 100

    return prediction_label, confidence

# --- Bagian 3: Antarmuka Streamlit ---

st.set_page_config(layout="wide") # Opsional: Atur layout halaman
st.title("Aplikasi Deteksi Hoaks Berita")
st.write("Masukkan teks berita di bawah ini untuk mendeteksi apakah itu hoaks atau valid.")

# Muat model dan tokenizer sekali saja menggunakan caching
tokenizer, model = load_model_and_tokenizer()

# Input teks dari pengguna
user_input_text = st.text_area("Teks Berita:", height=200, help="Salin dan tempel teks berita di sini.")

# Tombol untuk prediksi
if st.button("Deteksi Berita"):
    if user_input_text:
        with st.spinner("Menganalisis teks..."):
            predicted_label, confidence = predict_hoax_or_valid(user_input_text, tokenizer, model)

        st.subheader("Hasil Deteksi:")
        if predicted_label == "HOAX":
            st.error(f"⚠️ Berita ini kemungkinan besar **HOAKS** dengan keyakinan {confidence:.2f}%")
        else:
            st.success(f"✅ Berita ini kemungkinan **VALID** dengan keyakinan {confidence:.2f}%")

        st.info("Catatan: Akurasi model bergantung pada data pelatihan dan kompleksitas teks.")
    else:
        st.warning("Mohon masukkan teks berita terlebih dahulu.")

# --- Bagian Opsional: Menampilkan Statistik (Jika Anda punya data CSV di repositori) ---
st.sidebar.header("Tentang Aplikasi Ini")
st.sidebar.write("""
Aplikasi ini menggunakan model IndoBERT + CNN-LSTM untuk mengklasifikasikan berita
sebagai hoaks atau valid berdasarkan teks yang diberikan.
""")

# Bagian untuk menampilkan distribusi data (jika Anda ingin menampilkan plot)
# Untuk bagian ini, Anda perlu memastikan hoax_dataset_with_text.csv dan kompas_dataset_with_text.csv
# tersedia di repositori GitHub Anda.
# try:
#     # Menggunakan cache_data agar tidak load ulang setiap kali interaksi
#     @st.cache_data
#     def load_and_process_data():
#         df_hoax_raw = pd.read_csv("hoax_dataset_with_text.csv")
#         df_valid_raw = pd.read_csv("kompas_dataset_with_text.csv")
#         df_hoax_raw['label'] = 'hoax'
#         df_valid_raw['label'] = 'valid'
#         df_final = pd.concat([df_hoax_raw[['teks', 'label']], df_valid_raw[['teks', 'label']]]).sample(frac=1, random_state=42).reset_index(drop=True)
#         df_final['panjang_karakter'] = df_final['teks'].apply(len)
#         df_final['jumlah_kata'] = df_final['teks'].apply(lambda x: len(str(x).split()))
#         return df_final

#     st.subheader("Visualisasi Data Pelatihan (Contoh)")
#     data_to_viz = load_and_process_data()

#     fig1, ax1 = plt.subplots(figsize=(8, 5))
#     sns.countplot(data=data_to_viz, x='label', ax=ax1)
#     ax1.set_title("Distribusi Label Hoaks vs Valid")
#     st.pyplot(fig1)

#     fig2, ax2 = plt.subplots(figsize=(10, 6))
#     sns.histplot(data=data_to_viz, x='panjang_karakter', hue='label', kde=True, bins=30, ax=ax2)
#     ax2.set_title("Distribusi Panjang Karakter Teks")
#     st.pyplot(fig2)

# except FileNotFoundError:
#     st.warning("File dataset (hoax_dataset_with_text.csv, kompas_dataset_with_text.csv) tidak ditemukan. Visualisasi data tidak dapat ditampilkan.")
