from fastapi import APIRouter
from pydantic import BaseModel
import pandas as pd
import faiss
import re
import numpy as np
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key="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")

router = APIRouter()

df_stunting = pd.read_csv("data/stunting_dataset.csv")
df_stunting["text"] = df_stunting["text"].astype(str)

# Data MPASI
df_mpasi = pd.read_excel("data/dataset_mpasi.xlsx")
df_mpasi["text"] = df_mpasi.apply(
    lambda row: f"{row['nama_makanan']} - {row['bahan']} - {row['tekstur']} - {row['cocok_untuk']} - {row['resep']}",
    axis=1,
)

# Model embedding
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Embedding stunting
stunting_embeddings = embedding_model.encode(df_stunting["text"].tolist(), convert_to_numpy=True)
stunting_embeddings = np.array(stunting_embeddings).astype("float32")
index_stunting = faiss.IndexFlatL2(stunting_embeddings.shape[1])
index_stunting.add(stunting_embeddings)

# Embedding MPASI
mpasi_embeddings = embedding_model.encode(df_mpasi["text"].tolist(), convert_to_numpy=True)
df_mpasi["embedding"] = list(mpasi_embeddings.astype("float32"))


class MPASIQuestion(BaseModel):
    question: str
    usia_bulan: int | None = None
    user_id: int

conversation_history = {}

def extract_usia_from_text(text: str):
    m = re.search(r"(\d+)\s*bulan", text.lower())
    if m:
        return int(m.group(1))
    if text.strip().isdigit():
        return int(text.strip())
    return None

def truncate_text(text, max_words=80):
    return " ".join(text.split()[:max_words])

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )

def save_qa_to_db(pertanyaan, jawaban, status):
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute(
                "INSERT INTO chatbot (pertanyaan, jawaban, status) VALUES (%s, %s, %s)",
                (pertanyaan, jawaban, status),
            )
        conn.commit()
    finally:
        conn.close()

def save_unanswered_to_chat(pertanyaan, user_id):
    conn = get_db_connection()
    try:
        with conn.cursor() as c:
            c.execute(
                "INSERT INTO chat (chat, chatby, user_id) VALUES (%s, 'user', %s)",
                (pertanyaan, user_id),
            )
        conn.commit()
    finally:
        conn.close()

def is_relevant_to_stunting(text, threshold=0.3):
    kws = [
        "stunting", "gizi", "pertumbuhan", "berat badan", "tumbuh kembang",
        "tinggi badan", "balita", "bayi", "mpasi", "makanan pendamping asi",
        "pencegahan", "nutrisi", "asi", "makanan sehat", "kekurangan gizi",
        "website stuntaid", "stuntaid"
    ]
    if any(k in text.lower() for k in kws):
        return True
    topic_emb = embedding_model.encode(["stunting, mpasi, gizi, pertumbuhan, asi, bayi"])
    inp_emb = embedding_model.encode([text])
    sim = cosine_similarity(inp_emb, topic_emb)[0][0]
    return sim >= threshold

def _extract_message_content(choice):
    try:
        msg = getattr(choice, "message", None)
        if msg is not None:
            c = getattr(msg, "content", None)
            if c:
                return c
    except Exception:
        pass
    try:
        if isinstance(choice, dict):
            return choice.get("message", {}).get("content")
        return choice["message"]["content"]
    except Exception:
        return None

def generate_answer_stunting(question, user_id, threshold=2.5):
    if not is_relevant_to_stunting(question):
        j = "Maaf, pertanyaan Anda tidak berkaitan dengan topik stunting atau MPASI."
        save_qa_to_db(question, j, "belum terjawab")
        save_unanswered_to_chat(question, user_id)
        return j, "belum terjawab"
    qv = embedding_model.encode([question]).astype("float32")
    dist, idx = index_stunting.search(qv, 1)
    if dist[0][0] > threshold:
        j = "Maaf, saya tidak menemukan informasi yang relevan untuk pertanyaan tersebut."
        save_qa_to_db(question, j, "belum terjawab")
        save_unanswered_to_chat(question, user_id)
        return j, "belum terjawab"
    context = "\n".join(truncate_text(df_stunting.iloc[i]["text"]) for i in idx[0])
    messages = conversation_history.get(user_id, [])
    messages = messages[-5:]
    messages.append({"role": "user", "content": f"Info:\n{context}\n\nPertanyaan:\n{question}"})
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.3,
        )
        choice0 = completion.choices[0]
        content = _extract_message_content(choice0)
        if content is None:
            j = "Maaf, terjadi kesalahan saat membaca respons dari model."
            st = "belum terjawab"
        else:
            j = content.strip()
            st = "terjawab"
            messages.append({"role": "assistant", "content": j})
            conversation_history[user_id] = messages
    except Exception as e:
        j = f"Maaf, terjadi kesalahan saat menghasilkan jawaban: {e}"
        st = "belum terjawab"
    save_qa_to_db(question, j, st)
    return j, st

def generate_answer_mpasi(question, usia_bulan, user_id=None, threshold=2.5):
    f = df_mpasi[
        (df_mpasi["usia_min_bulan"] <= usia_bulan)
        & (df_mpasi["usia_max_bulan"] >= usia_bulan)
    ]
    if f.empty:
        return f"Maaf, tidak ada rekomendasi MPASI untuk usia {usia_bulan} bulan.", "belum terjawab"
    emb_f = list(f["embedding"])
    idx_f = faiss.IndexFlatL2(len(emb_f[0]))
    idx_f.add(np.array(emb_f).astype("float32"))
    qv = embedding_model.encode([question]).astype("float32")
    dist, ind = idx_f.search(qv, 1)
    if dist[0][0] > threshold:
        return "Maaf, saya tidak menemukan rekomendasi MPASI yang sesuai.", "belum terjawab"
    rec = f.iloc[ind[0][0]]
    nama = rec["nama_makanan"]
    bahan = rec["bahan"].replace("\n", " ").replace("- ", "• ")
    cara = rec["resep"].replace("\n", "\n").replace("- ", "• ")
    r = f"**{nama}** (Usia {usia_bulan} bulan)\n\n**Bahan:**\n{bahan.strip()}\n\n**Detail Bahan & Cara Membuat:**\n{cara.strip()}"
    save_qa_to_db(question, r, "terjawab")
    return r, "terjawab"

def generate_answer_with_age(question, usia_bulan=None, user_id=None):
    d = extract_usia_from_text(question)
    if d is not None:
        return generate_answer_mpasi(question, d, user_id)
    if "rekomendasi" in question.lower() and "mpasi" in question.lower():
        if usia_bulan is None:
            return "Boleh tahu usia anak Anda (dalam bulan) agar saya bisa memberikan rekomendasi MPASI yang sesuai?", "terjawab"
        return generate_answer_mpasi(question, usia_bulan, user_id)
    return generate_answer_stunting(question, user_id)

@router.post("/ask")
def mpasi(q: MPASIQuestion):
    j, s = generate_answer_with_age(q.question, q.usia_bulan, q.user_id)
    return {"jawaban": j, "status": s}
