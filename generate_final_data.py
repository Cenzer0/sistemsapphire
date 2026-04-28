import pandas as pd
import numpy as np

# Data Spesifikasi dari Brosur
# Tipe, LB, LT_Std, KT, KM, R_Tamu, Carport, Harga_Dasar
ref_data = [
    ['Tahap 4', '33/77', 33, 77, 2, 1, 9.0, 12.0, 416500000],
    ['Tahap 4', '38/80', 38, 80, 2, 1, 9.0, 13.5, 426000000],
    ['Tahap 4', '38/84', 38, 84, 2, 1, 9.0, 12.0, 439000000],
    ['Tahap 3', '47/112', 47, 112, 2, 1, 10.5, 13.5, 567000000],
    ['Tahap 3', '67/112', 67, 112, 3, 2, 12.25, 15.0, 698500000]
]

rows = []
# Total 541 Unit (Tahap 3: 380, Tahap 4: 161)
for i in range(1, 542):
    if i <= 380:
        d = ref_data[3] if i % 2 == 0 else ref_data[4] # Campuran Tipe 47 & 67
        no_kav = f"T3-{str(i).zfill(3)}"
    else:
        d = ref_data[i % 3] # Campuran Tipe 33, 38/80, 38/84
        no_kav = f"T4-{str(i-380).zfill(3)}"
    
    # Simulasi Variasi Luas Tanah & Posisi sesuai Pricelist
    lt_tambahan = np.random.choice([0, 2, 5, 15, 30]) if np.random.rand() > 0.7 else 0
    lt_riil = d[3] + lt_tambahan
    posisi = np.random.choice(['Standard', 'Main Road', 'Hook'], p=[0.7, 0.2, 0.1])
    
    # Hitung Harga (Harga Dasar + Kelebihan Tanah + Premi Posisi)
    harga = d[8] + (lt_tambahan * 2500000)
    if posisi == 'Main Road': harga += 15000000
    if posisi == 'Hook': harga += 25000000
    
    status = 'Terjual' if np.random.rand() < 0.85 else 'Sisa'
    
    rows.append([
        d[0], no_kav, d[1], posisi, lt_riil, d[2], d[4], d[5], 
        d[6], d[7], 'Ya', int(harga), status
    ])

cols = ['Tahap', 'No_Kavling', 'Tipe', 'Posisi', 'Luas_Tanah', 'Luas_Bangunan', 
        'Kamar_Tidur', 'Kamar_Mandi', 'Luas_R_Tamu', 'Luas_Carport', 'Smart_Lock', 'Harga_Jual', 'Status']

df = pd.DataFrame(rows, columns=cols)
df.to_csv('Data_Sapphire_Tegal_541.csv', index=False, sep=';')
print("Berhasil! File 'Data_Sapphire_Tegal_541.csv' telah dibuat di folder Anda.")