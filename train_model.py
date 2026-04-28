import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import joblib
import os
import json

# -------------------------------------------------------------------
# SISTEM PREDIKSI HARGA RUMAH - SAPPHIRE RESIDENCE TEGAL
# Artificial Neural Network (MLP) dengan Perbandingan Optimizer
# Arsitektur: Input(7) -> Hidden(64, ReLU) -> Hidden(32, ReLU) -> Hidden(16, ReLU) -> Output(1)
# -------------------------------------------------------------------

folder_path = 'models'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

try:
    # =================================================================
    # 1. LOAD DATASET
    # =================================================================
    nama_file = 'Dataset_Sapphire_Final.csv'
    print(f"Membaca dataset: {nama_file}")
    df = pd.read_csv(nama_file, sep=';')
    print(f"Total data: {len(df)} unit\n")

    # =================================================================
    # 2. ANALISIS DESKRIPTIF DATASET (Tabel 3.1 Proposal)
    # =================================================================
    print("=" * 65)
    print("ANALISIS DESKRIPTIF DATASET")
    print("=" * 65)

    # Statistik numerik
    numerik_cols = ['Luas Bangunan', 'Luas Tanah', 'Jumlah Kamar Mandi', 'Jumlah Kamar Tidur', 'Harga KPR']
    desc_stats = {}
    for col in numerik_cols:
        stats = df[col].describe()
        desc_stats[col] = {
            'count': int(stats['count']),
            'mean': round(float(stats['mean']), 2),
            'std': round(float(stats['std']), 2),
            'min': float(stats['min']),
            'max': float(stats['max']),
            'median': float(df[col].median())
        }
        print(f"\n{col}:")
        print(f"  Mean: {stats['mean']:,.2f} | Std: {stats['std']:,.2f}")
        print(f"  Min: {stats['min']:,.0f} | Max: {stats['max']:,.0f} | Median: {df[col].median():,.0f}")

    # Distribusi tipe rumah
    tipe_dist = df['Tipe'].value_counts().to_dict()
    tipe_persen = {}
    for tipe, count in tipe_dist.items():
        persen = round(count / len(df) * 100, 1)
        tipe_persen[tipe] = {'count': int(count), 'persen': persen}
        print(f"  Tipe {tipe}: {count} unit ({persen}%)")

    # Distribusi tahap
    tahap_dist = df['Tahap'].value_counts().to_dict()
    tahap_persen = {}
    for tahap, count in tahap_dist.items():
        persen = round(count / len(df) * 100, 1)
        tahap_persen[tahap] = {'count': int(count), 'persen': persen}

    # Distribusi blok
    blok_dist = df['Nama Blok'].value_counts().to_dict()
    blok_persen = {}
    for blok, count in blok_dist.items():
        blok_persen[blok] = int(count)

    deskriptif = {
        'total_data': len(df),
        'statistik': desc_stats,
        'distribusi_tipe': tipe_persen,
        'distribusi_tahap': tahap_persen,
        'distribusi_blok': blok_persen,
        'rentang_harga': {
            'min': int(df['Harga KPR'].min()),
            'max': int(df['Harga KPR'].max()),
            'mean': int(df['Harga KPR'].mean()),
            'median': int(df['Harga KPR'].median())
        }
    }

    # =================================================================
    # 3. PREPROCESSING DATA
    # =================================================================
    print("\n" + "=" * 65)
    print("PREPROCESSING DATA")
    print("=" * 65)

    # Label Encoding - Nama Blok
    le_blok = LabelEncoder()
    df['Blok_Encoded'] = le_blok.fit_transform(df['Nama Blok'])
    joblib.dump(le_blok, os.path.join(folder_path, 'le_blok.pkl'))
    print(f"Nama Blok: {len(le_blok.classes_)} kategori -> Label Encoded")

    # Label Encoding - Posisi Kavling
    le_posisi = LabelEncoder()
    df['Posisi_Encoded'] = le_posisi.fit_transform(df['Posisi Kavling'])
    joblib.dump(le_posisi, os.path.join(folder_path, 'le_posisi.pkl'))
    print(f"Posisi Kavling: {len(le_posisi.classes_)} kategori -> Label Encoded")

    # Smart Door Lock -> Binary
    df['Smart_Lock_N'] = df['Smart Door Lock'].apply(lambda x: 1 if str(x).strip() == 'Ya' else 0)
    print(f"Smart Door Lock: Binary Encoded (Ya=1, Tidak=0)")

    # Rename columns untuk konsistensi
    df.columns = [c.replace(' ', '_') for c in df.columns]

    # Seleksi fitur (sesuai Proposal Bab 3.5.2)
    features = [
        'Blok_Encoded', 'Luas_Bangunan', 'Luas_Tanah',
        'Jumlah_Kamar_Tidur', 'Jumlah_Kamar_Mandi',
        'Posisi_Encoded', 'Smart_Lock_N'
    ]
    feature_labels = [
        'Nama Blok', 'Luas Bangunan', 'Luas Tanah',
        'Jumlah Kamar Tidur', 'Jumlah Kamar Mandi',
        'Posisi Kavling', 'Smart Door Lock'
    ]

    X = df[features]

    # Normalisasi target: bagi 1.000.000 agar ANN stabil
    y = df['Harga_KPR'] / 1000000

    print(f"Fitur input: {len(features)} variabel")
    print(f"Target: Harga KPR (dinormalisasi /1.000.000)")

    # =================================================================
    # 4. UJI STATISTIK (Pembuktian H1 dan H2)
    # =================================================================
    print("\n" + "=" * 65)
    print("UJI STATISTIK - REGRESI LINEAR (OLS)")
    print("=" * 65)

    # OLS Regression untuk Uji F dan Uji t
    # Hapus kolom dengan varians nol agar OLS tidak error
    X_ols_raw = X.copy()
    zero_var_cols = X_ols_raw.columns[X_ols_raw.nunique() <= 1].tolist()
    if zero_var_cols:
        print(f"  Kolom dengan varians nol (dikeluarkan dari OLS): {zero_var_cols}")
        X_ols_raw = X_ols_raw.drop(columns=zero_var_cols)

    X_ols = sm.add_constant(X_ols_raw)
    model_ols = sm.OLS(df['Harga_KPR'], X_ols).fit()

    print(f"\n--- Uji F (Simultan / H1) ---")
    print(f"F-Statistic : {model_ols.fvalue:,.4f}")
    print(f"P-Value (F) : {model_ols.f_pvalue:.6e}")
    f_signifikan = model_ols.f_pvalue < 0.05
    print(f"Kesimpulan  : {'SIGNIFIKAN' if f_signifikan else 'TIDAK SIGNIFIKAN'} (alpha=0.05)")
    if f_signifikan:
        print("  -> H1 DITERIMA: Seluruh variabel secara simultan berpengaruh signifikan terhadap harga.")

    print(f"\n--- Uji t (Parsial / H2) ---")
    print(f"{'Variabel':<20} {'Koefisien':>15} {'t-stat':>12} {'P-Value':>12} {'Status':>12}")
    print("-" * 75)

    uji_t_results = []
    for i, feat in enumerate(features):
        label = feature_labels[i]
        if feat in zero_var_cols:
            # Variabel konstan, tidak diuji
            uji_t_results.append({
                'variabel': label,
                'koefisien': 0,
                't_statistic': 0,
                'p_value': 1.0,
                'signifikan': False,
                'keterangan': 'Nilai konstan (tidak diuji)'
            })
            print(f"{label:<20} {'N/A':>15} {'N/A':>12} {'N/A':>12} {'Konstan':>12}")
        else:
            coef = float(model_ols.params[feat])
            t_val = float(model_ols.tvalues[feat])
            p_val = float(model_ols.pvalues[feat])
            sig = "Signifikan" if p_val < 0.05 else "Tidak Sig."
            print(f"{label:<20} {coef:>15,.2f} {t_val:>12.4f} {p_val:>12.6f} {sig:>12}")
            uji_t_results.append({
                'variabel': label,
                'koefisien': round(coef, 4),
                't_statistic': round(t_val, 4),
                'p_value': p_val,
                'signifikan': p_val < 0.05
            })

    r2_ols = model_ols.rsquared
    r2_adj = model_ols.rsquared_adj
    print(f"\nR-Squared       : {r2_ols:.6f}")
    print(f"Adj. R-Squared  : {r2_adj:.6f}")

    statistik = {
        'uji_f': {
            'f_statistic': round(float(model_ols.fvalue), 4),
            'p_value': float(model_ols.f_pvalue),
            'signifikan': f_signifikan
        },
        'uji_t': uji_t_results,
        'r_squared': round(float(r2_ols), 6),
        'adj_r_squared': round(float(r2_adj), 6),
        'n_observations': int(model_ols.nobs),
        'df_model': int(model_ols.df_model),
        'df_resid': int(model_ols.df_resid)
    }

    # =================================================================
    # 5. SCALING FITUR (StandardScaler)
    # =================================================================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(folder_path, 'scaler.pkl'))
    print(f"\nStandardScaler: fit & saved")

    # =================================================================
    # 6. SPLIT DATA (80:20 - Tabel 3.3 Proposal)
    # =================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print(f"Data Training: {len(X_train)} ({len(X_train)/len(X)*100:.0f}%)")
    print(f"Data Testing : {len(X_test)} ({len(X_test)/len(X)*100:.0f}%)")

    split_info = {
        'total': len(X),
        'training': len(X_train),
        'testing': len(X_test),
        'ratio': '80:20',
        'random_state': 42
    }

    # =================================================================
    # 7. TRAINING & PERBANDINGAN OPTIMIZER (Tabel 3.4 Proposal)
    # Arsitektur: (64, 32, 16) sesuai Tabel 3.2
    # =================================================================
    print("\n" + "=" * 65)
    print("TRAINING MODEL ANN (MLP Regressor)")
    print("Arsitektur: Input(7) -> H1(64) -> H2(32) -> H3(16) -> Output(1)")
    print("=" * 65)

    optimizers = ['adam', 'sgd', 'lbfgs']
    hasil_metrik = {}

    print(f"\n{'Optimizer':<10} | {'R2 Score':<12} | {'MAE (Rp)':<18} | {'RMSE (Rp)':<18} | {'MAPE (%)':<10}")
    print("-" * 80)

    for opt in optimizers:
        model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),  # Sesuai Tabel 3.2 Proposal
            activation='relu',
            solver=opt,
            max_iter=10000,
            random_state=42,
            early_stopping=True if opt != 'lbfgs' else False,
            validation_fraction=0.1 if opt != 'lbfgs' else 0.1,
            tol=1e-6
        )

        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Metrik (kembalikan ke skala Rupiah untuk MAE dan RMSE)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test * 1000000, y_pred * 1000000)
        rmse = np.sqrt(mean_squared_error(y_test * 1000000, y_pred * 1000000))

        # MAPE - hindari division by zero
        y_test_rp = y_test * 1000000
        y_pred_rp = y_pred * 1000000
        mape = np.mean(np.abs((y_test_rp.values - y_pred_rp) / y_test_rp.values)) * 100
        akurasi = 100 - mape

        # Loss curve (hanya untuk adam dan sgd yang punya loss_curve_)
        loss_curve = []
        if hasattr(model, 'loss_curve_') and model.loss_curve_ is not None:
            loss_curve = [round(float(v), 6) for v in model.loss_curve_]

        hasil_metrik[opt.upper()] = {
            'r2': round(float(r2), 6),
            'mae': round(float(mae), 2),
            'rmse': round(float(rmse), 2),
            'mape': round(float(mape), 4),
            'akurasi': round(float(akurasi), 4),
            'loss_curve': loss_curve,
            'n_iter': int(model.n_iter_),
            'arsitektur': '64-32-16',
            'aktivasi': 'ReLU'
        }

        # Simpan model
        joblib.dump(model, os.path.join(folder_path, f'model_{opt}.pkl'))

        print(f"{opt.upper():<10} | {r2:<12.6f} | Rp {mae:<15,.0f} | Rp {rmse:<15,.0f} | {mape:<10.4f}")

    print("-" * 80)

    # Tentukan optimizer terbaik berdasarkan R2
    best_opt = max(hasil_metrik, key=lambda k: hasil_metrik[k]['r2'])
    print(f"\nOptimizer Terbaik: {best_opt} (R2 = {hasil_metrik[best_opt]['r2']:.6f})")

    # =================================================================
    # 8. SIMPAN SEMUA HASIL KE JSON
    # =================================================================
    output = {
        'deskriptif': deskriptif,
        'statistik': statistik,
        'split_data': split_info,
        'metrik_model': hasil_metrik,
        'optimizer_terbaik': best_opt,
        'arsitektur': {
            'input_layer': 7,
            'hidden_layers': [64, 32, 16],
            'output_layer': 1,
            'aktivasi': 'ReLU',
            'max_iter': 10000
        },
        'fitur': feature_labels
    }

    # Custom encoder untuk numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = os.path.join(folder_path, 'training_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\nHasil disimpan ke: {json_path}")
    print("\n[SELESAI] Semua model dan metrik berhasil disimpan.")

except Exception as e:
    import traceback
    print(f"\n[ERROR] {e}")
    traceback.print_exc()