# 🇮🇩 Indonesian Sentiment Analysis
### nolimit.id — Data Scientist Technical Test


---

## 📌 Task
**Classification — Sentiment Analysis**  
Classify Indonesian text into **Positive / Neutral / Negative** sentiment using transformer-based models.

---

## 📊 Dataset

| Item | Detail |
|------|--------|
| **Name** | NusaX-senti (Indonesian Sentiment Analysis) |
| **Source** | [https://huggingface.co/datasets/indonlp/NusaX-senti](https://huggingface.co/datasets/indonlp/NusaX-senti) |
| **Subset** | `ind` (Bahasa Indonesia) |
| **License** | CC-BY-SA 4.0 |
| **Language** | Bahasa Indonesia 🇮🇩 |
| **Size** | 1.000 teks (train/500 · val/100 · test/400) |
| **Labels** | `positive`, `neutral`, `negative` |
| **Annotation** | Diterjemahkan & dianotasi native speaker Indonesia; derivasi dari SmSA (IndoNLU) |
| **Format** | Parquet — kompatibel penuh dengan `datasets` library terbaru |

> **Cara perolehan data:** Dataset diambil langsung dari Hugging Face Hub menggunakan library `datasets`.  
> Format Parquet standar — tidak memerlukan `trust_remote_code` atau loading script.  
> Data bersifat publik, open-source, dan telah diverifikasi kualitasnya oleh native speaker.

---

## 🤗 Models Used

| Role | Model | Source |
|------|-------|--------|
| **Classifier** | `indobenchmark/indobert-base-p1` (IndoBERT) | [HuggingFace Hub](https://huggingface.co/indobenchmark/indobert-base-p1) |
| **Embeddings** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | [HuggingFace Hub](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| **Vector Index** | FAISS `IndexFlatIP` (cosine similarity) | [Facebook AI](https://github.com/facebookresearch/faiss) |

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE END-TO-END                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  📥 Raw Data (SmSA)                                             │
│       │                                                         │
│       ▼                                                         │
│  🔧 Preprocessing (tokenization, padding, truncation@128)       │
│       │                                                         │
│       ├──────────────────────────┐                              │
│       ▼                          ▼                              │
│  🤖 IndoBERT Fine-tune     📐 Sentence Embeddings              │
│  (Classification Head)     (MiniLM-L12-v2)                      │
│       │                          │                              │
│       ▼                          ▼                              │
│  📊 Predict Sentiment      🗂️ FAISS Index                      │
│  (pos/neu/neg + conf%)     (1000-vector corpus)                 │
│       │                          │                              │
│       └──────────────────────────┘                              │
│                    │                                            │
│                    ▼                                            │
│  📋 Output: Label + Confidence + Similar Examples               |
└─────────────────────────────────────────────────────────────────┘
```

> See `flowchart.png` for the visual diagram.

---

## 📂 Repository Structure

```
nolimit-ds-test-[nama]/
├── nolimit_sentiment_analysis.ipynb   # Main Colab notebook
├── sample_input.csv                   # 20 sample texts for local testing
├── flowchart.png                      # End-to-end pipeline diagram
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## ⚙️ Setup Instructions

### Option A: Google Colab (Recommended)
1. Klik badge **Open in Colab** di atas
2. Pilih Runtime → Change runtime type → **T4 GPU**
3. Jalankan semua sel secara berurutan (`Runtime → Run all`)

### Option B: Local Environment
```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/nolimit-ds-test-YOUR_NAME.git
cd nolimit-ds-test-YOUR_NAME

# 2. Create virtual environment (Python 3.9+)
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run notebook
jupyter notebook nolimit_sentiment_analysis.ipynb
```

> ⚠️ **GPU sangat direkomendasikan** untuk fine-tuning. Pada CPU, proses training ~3x lebih lama.


## 🔄 Contoh Output

```
[1] Input   : "Produk ini sangat bagus dan pengiriman cepat, puas banget!"
    Sentimen : ✅ POSITIVE (96.4%)
    Contoh Serupa (FAISS):
      ✅ [positive] sim=0.921 — "Kualitas produk sangat memuaskan, packaging rapi..."
      ✅ [positive] sim=0.887 — "Pengiriman super cepat, barang aman sampai..."

[2] Input   : "Sangat kecewa, barang datang rusak dan pelayanan buruk."
    Sentimen : ❌ NEGATIVE (98.1%)
    Contoh Serupa (FAISS):
      ❌ [negative] sim=0.934 — "Barang rusak waktu sampai, sangat mengecewakan..."
      ❌ [negative] sim=0.891 — "Pelayanan CS tidak responsif, komplain diabaikan..."
```

---

## 📦 Dependencies

See `requirements.txt` — key packages:
- `transformers>=4.40`
- `datasets>=2.19`
- `sentence-transformers>=2.7`
- `faiss-cpu>=1.8`
- `torch>=2.2`

---

*Built for nolimit.id Data Scientist Technical Test*
