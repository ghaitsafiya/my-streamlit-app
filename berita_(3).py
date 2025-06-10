import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# Menggunakan try-except untuk LabelEncoder karena mungkin tidak dibutuhkan secara langsung
# jika hanya digunakan untuk memetakan prediksi
try:
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    st.warning("sklearn.preprocessing.LabelEncoder tidak ditemukan. Pastikan scikit-learn terinstal jika Anda memerlukannya untuk transformasi label.")

# Menggunakan try-except untuk imblearn karena mungkin tidak dibutuhkan secara langsung
# jika hanya digunakan untuk oversampling data training
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    st.warning("imblearn.over_sampling.SMOTE tidak ditemukan. Pastikan imbalanced-learn terinstal jika Anda memerlukannya.")

# Menggunakan try-except untuk TfidfVectorizer karena mungkin tidak dibutuhkan secara langsung
# jika hanya digunakan untuk preprocessing data training
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except ImportError:
    st.warning("sklearn.feature_extraction.text.TfidfVectorizer tidak ditemukan. Pastikan scikit-learn terinstal jika Anda memerlukannya.")


# --- Bagian 1: Definisi Model (Tidak diubah dari kode Anda) ---
class IndoBERT_CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('indobenchmark/indobert-base-p1')
        self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 2) # Output 2 kelas: Hoax, Valid

    def forward(self, input_ids, attention_mask):
        with torch.no_grad(): # BERT digunakan sebagai feature extractor
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x = x.permute(0, 2, 1)  # CNN butuh format (batch, channels, seq)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # LSTM butuh (batch, seq, channels)
        _, (h_n, _) = self.lstm(x) # h_n adalah hidden state terakhir dari LSTM
        logits = self.fc(h_n.squeeze(0)) # Squeeze h_n dan masukkan ke fully connected layer
        return logits

# --- Bagian 2: Fungsi Utilitas dan Caching ---

@st.cache_resource # Gunakan cache_resource karena memuat model dan tokenizer adalah operasi yang mahal
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    model = IndoBERT_CNN_LSTM()

    # --- PENTING: Ganti "model.pth" dengan nama file model Anda ---
    try:
        # Memuat model yang sudah dilatih
        # map_location=torch.device('cpu') penting untuk kompatibilitas jika dilatih di GPU
        model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')))
        model.eval() # Pastikan model dalam mode evaluasi
        st.success("Model dan tokenizer berhasil dimuat!")
    except FileNotFoundError:
        st.error("""
            File model 'model.pth' tidak ditemukan di repositori Anda.
            Aplikasi tidak dapat berjalan tanpa model yang sudah dilatih.
            Harap pastikan Anda sudah melatih model dan mengunggah file 'model.pth' ke repositori GitHub.
        """)
        st.stop() # Hentikan eksekusi jika model tidak ditemukan
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        st.stop() # Hentikan eksekusi jika ada error lain saat memuat model

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
    clean_txt = clean_text(text)
    # Tokenisasi input
    tokenized_input = tokenizer(
        clean_txt,
        padding='max_length',
        truncation=True,
        max_length=512, # Pastikan max_length konsisten dengan training Anda
        return_tensors='pt'
    )
    input_ids = tokenized_input['input_ids']
    attention_mask = tokenized_input['attention_mask']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # Pindahkan model ke device yang sesuai
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_id = torch.argmax(probabilities, dim=1).item()

    # Asumsi: 0 = valid, 1 = hoax (sesuai mapping Anda di training: 'hoax': 1, 'valid': 0)
    label_map = {0: "VALID", 1: "HOAX"}
    prediction_label = label_map[predicted_class_id]
    confidence = probabilities[0][predicted_class_id].item() * 100

    return prediction_label, confidence

# --- Bagian 3: Antarmuka Streamlit ---

st.set_page_config(layout="wide", page_title="Deteksi Hoaks Berita")
st.title("Deteksi Hoaks Berita dengan IndoBERT & CNN-LSTM")
st.write("Aplikasi ini dirancang untuk membantu mengidentifikasi apakah sebuah berita cenderung hoaks atau valid berdasarkan teks yang Anda berikan.")

# Muat model dan tokenizer sekali saja
tokenizer, model = load_model_and_tokenizer()

st.header("Masukkan Teks Berita")
user_input_text = st.text_area(
    "Salin dan tempel teks berita di sini:",
    height=250,
    help="Masukkan teks berita yang ingin Anda verifikasi keasliannya."
)

if st.button("Deteksi Hoaks"):
    if user_input_text:
        with st.spinner("Menganalisis teks berita..."):
            predicted_label, confidence = predict_hoax_or_valid(user_input_text, tokenizer, model)

        st.subheader("Hasil Deteksi:")
        if predicted_label == "HOAX":
            st.error(f"🚨 **BERITA INI KEMUNGKINAN BESAR HOAKS!**")
            st.write(f"Tingkat keyakinan model: **{confidence:.2f}%**")
            st.warning("Selalu verifikasi informasi dari berbagai sumber terpercaya.")
        else:
            st.success(f"✅ **BERITA INI KEMUNGKINAN VALID.**")
            st.write(f"Tingkat keyakinan model: **{confidence:.2f}%**")
            st.info("Meskipun demikian, disarankan untuk tetap melakukan verifikasi ulang.")
    else:
        st.warning("Mohon masukkan teks berita terlebih dahulu untuk dideteksi.")

st.markdown("---")
st.sidebar.header("Informasi")
st.sidebar.markdown("""
Aplikasi ini menggunakan model deep learning gabungan **IndoBERT** (sebagai *feature extractor*) dan arsitektur **CNN-LSTM** untuk klasifikasi teks.
""")
st.sidebar.markdown("Model dilatih untuk membedakan antara berita hoaks dan berita valid.")

# --- Bagian Opsional: Menampilkan Statistik Dataset ---
# Pastikan 'hoax_dataset_with_text.csv' dan 'kompas_dataset_with_text.csv' ada di repositori GitHub Anda
# Jika tidak ada, bagian ini akan menampilkan pesan error
st.sidebar.header("Visualisasi Data (Opsional)")

@st.cache_data
def load_and_process_data_for_viz():
    try:
        df_hoax_raw = pd.read_csv("hoax_dataset_with_text.csv")
        df_valid_raw = pd.read_csv("kompas_dataset_with_text.csv")

        df_hoax_raw['label'] = 'hoax'
        df_valid_raw['label'] = 'valid'

        # Gabungkan dan bersihkan kolom teks (opsional, tergantung kebutuhan visualisasi)
        df_final = pd.concat([
            df_hoax_raw[['teks', 'label']].dropna(),
            df_valid_raw[['teks', 'label']].dropna()
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

        df_final['panjang_karakter'] = df_final['teks'].apply(len)
        df_final['jumlah_kata'] = df_final['teks'].apply(lambda x: len(str(x).split()))

        return df_final
    except FileNotFoundError:
        return None # Mengembalikan None jika file tidak ditemukan
    except Exception as e:
        st.error(f"Kesalahan saat memuat data untuk visualisasi: {e}")
        return None

if st.sidebar.checkbox("Tampilkan Visualisasi Data"):
    data_for_viz = load_and_process_data_for_viz()
    if data_for_viz is not None:
        st.subheader("Distribusi Label Dataset")
        fig1, ax1 = plt.subplots(figsize=(7, 4))
        sns.countplot(data=data_for_viz, x='label', ax=ax1, palette='viridis')
        ax1.set_title("Distribusi Jumlah Berita Hoaks vs Valid")
        ax1.set_xlabel("Jenis Berita")
        ax1.set_ylabel("Jumlah")
        st.pyplot(fig1)

        st.subheader("Distribusi Panjang Karakter Teks")
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        sns.histplot(data=data_for_viz, x='panjang_karakter', hue='label', kde=True, bins=30, ax=ax2, palette='magma')
        ax2.set_title("Distribusi Panjang Karakter Berita (Hoaks vs Valid)")
        ax2.set_xlabel("Panjang Karakter")
        ax2.set_ylabel("Jumlah Berita")
        st.pyplot(fig2)

        st.subheader("Sebaran Jumlah Kata per Kelas")
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=data_for_viz, x='label', y='jumlah_kata', ax=ax3, palette='cividis')
        ax3.set_title("Sebaran Jumlah Kata per Kelas (Hoaks vs Valid)")
        ax3.set_xlabel("Jenis Berita")
        ax3.set_ylabel("Jumlah Kata")
        st.pyplot(fig3)
    else:
        st.sidebar.warning("File dataset tidak ditemukan untuk visualisasi.")
