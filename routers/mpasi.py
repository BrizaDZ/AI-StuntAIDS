from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd

router = APIRouter()

model = joblib.load("models/model_mpasi.pkl")
scaler = joblib.load("models/scaler.pkl")
le_tekstur = joblib.load("models/le_tekstur.pkl")
le_kelas = joblib.load("models/le_kelas.pkl")

df_makanan = pd.read_excel("data/dataset_mpasi.xlsx")
df_makanan["kategori_umur"] = df_makanan["usia_min_bulan"].astype(str) + "-" + df_makanan["usia_max_bulan"].astype(str)

mapping_tekstur = {
    "halus_lunak": ["halus", "halus bertekstur", "cincang halus", "lunak", "lunak bertekstur"],
    "lembut_pulen": ["lembut", "pulen", "nasi pulen, ayam empuk"],
    "padat_kasar": ["padat", "kasar"],
    "kental_berkuah": ["kental", "berkuah"]
}

def map_to_detail_tekstur(kategori):
    detail_list = mapping_tekstur.get(kategori, [kategori])
    return detail_list[0]

class FiturMakanan(BaseModel):
    kalori_kkal: float
    protein_gr: float
    lemak_gr: float
    tekstur: str
    jumlah_bahan: int = 0

class KategoriUmurInput(BaseModel):
    kategori_umur: str

@router.post("/predict")
def predict(fitur: FiturMakanan):
    fitur_dict = fitur.dict()
    tekstur_detail = map_to_detail_tekstur(fitur_dict['tekstur'])

    x = [
        fitur_dict['kalori_kkal'],
        fitur_dict['protein_gr'],
        fitur_dict['lemak_gr'],
        le_tekstur.transform([tekstur_detail])[0],
        fitur_dict['jumlah_bahan']
    ]
    x_scaled = scaler.transform([x])
    pred = model.predict(x_scaled)
    kategori = le_kelas.inverse_transform(pred)[0].replace(" bulan", "").strip()

    df_filtered = df_makanan[df_makanan["kategori_umur"] == kategori]
    sampel = df_filtered.sample(n=min(5, len(df_filtered)))
    hasil = [{"nama": row["nama_makanan"], "resep": row["resep"]} for _, row in sampel.iterrows()]

    return {
        "input": fitur_dict,
        "prediksi_kategori_umur": kategori,
        "rekomendasi": hasil
    }

@router.post("/recommend-by-age")
def recommend_by_age(input_data: KategoriUmurInput):
    kategori = input_data.kategori_umur
    df_filtered = df_makanan[df_makanan["kategori_umur"] == kategori]

    if df_filtered.empty:
        return {"message": "Kategori umur tidak ditemukan", "rekomendasi": []}

    sampel = df_filtered.sample(n=min(5, len(df_filtered)))
    hasil = [{"nama": row["nama_makanan"], "resep": row["resep"]} for _, row in sampel.iterrows()]

    return {
        "kategori_umur": kategori,
        "rekomendasi": hasil
    }