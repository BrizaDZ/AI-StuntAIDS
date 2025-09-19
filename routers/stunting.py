from fastapi import APIRouter
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

router = APIRouter()

model = joblib.load("models/model_stunting.pkl")
scaler = joblib.load("models/scaler_stunting.pkl")
encoders = joblib.load("models/encoders.pkl")

pb_laki_24 = pd.read_excel("data/pb_laki_24.xlsx")
pb_perempuan_24 = pd.read_excel("data/pb_perempuan_24.xlsx")
tb_laki_60 = pd.read_excel("data/tb_laki_60.xlsx")
tb_perempuan_60 = pd.read_excel("data/tb_perempuan_60.xlsx")

feature_names = [
    'JK', 'Usia', 'Pendapatan',
    'Apakah_ibu_hanya_memberikan_asi_saja_kepada_anak_tanpa_makanan_pendamping_selama_anak_berusia_6_bulan?',
    'Apakah_ibu_memberikan_makan_pendamping_kepada_anak_ketika_anak_berumur_6_bulan?',
    'Apakah_anak_sudah_diberikan_imunisasi_lengkap?',
    'Apakah_porsi_makan_ibu_ketika_hamil_lebih_sedikit_dibandingkan_ketika_tidak_hamil?]',
    'Si_bayi_diberikan_ASI_saja_selama_6_bulan',
    'Bayi_setiap_bulan_dilakukan_penimbangan_berat_badan_di_posyandu/puskesmas',
    'Menggunakan_air_bersih_untuk_kegiatan_sehari-hari',
    'Mencuci_tangan_dengan_air_bersih_dan_sabun_sebelum_makan_dan_setelah_buang_air_besar',
    'Untuk_BAB_dan_BAK_menggunakan_jamban_yang_bersih_dam_sehat',
    'Pemberantasan_nyamuk_dilakukan_setiap_seminggu_satu_kali_\n(Pemberantasan_nyamuk_adalah_tindakan_mengurangi_dan_membunuh_nyamuk_yang_bias_menyebabkan_penyakit_demam_berdarah)',
    'Buah_dan_sayur_dikonsumsi_setiap_hari',
    'Setiap_hari_meluangkan_waktu_untuk_melakukan_aktivitas_fisik_atau_olahraga',
    'Merokok_di_dalam_rumah',
    'TB'
]

class StuntingInput(BaseModel):
    data: dict

def calculate_zscore(age_in_months: int, gender: int, height_cm: float) -> float:
    if age_in_months < 24:
        df = pb_laki_24 if gender == 0 else pb_perempuan_24
    else:
        df = tb_laki_60 if gender == 0 else tb_perempuan_60
    row = df[df['Umur (bulan)'] == age_in_months]
    if row.empty:
        raise ValueError(f"Umur {age_in_months} tidak ditemukan dalam tabel referensi.")
    median = row['Median'].values[0]
    minus_2sd = row['-2 SD'].values[0]
    plus_2sd = row['+2 SD'].values[0]
    std_dev = (plus_2sd - minus_2sd) / 4
    z_score = (height_cm - median) / std_dev
    return round(z_score, 2)

@router.post("/predict")
def predict_stunting(payload: StuntingInput):
    input_dict = payload.data
    input_data = []

    for feature in feature_names:
        val = input_dict.get(feature)
        if feature in encoders:
            val = encoders[feature].transform([val])[0]
        else:
            val = float(val)
        input_data.append(val)

    input_array = np.array(input_data).reshape(1, -1)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0]

    age = int(input_array[0][1])
    gender = int(input_array[0][0])
    height = float(input_array[0][-1])

    try:
        z_score = calculate_zscore(age, gender, height)
    except Exception as e:
        z_score = str(e)

    return {
        "prediction": prediction,
        "z_score": z_score
    }
