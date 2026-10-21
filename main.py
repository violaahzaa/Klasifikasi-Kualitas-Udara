"""
Klasifikasi Tingkat Kualitas Udara dari Data Sensor IoT dengan Deep Learning

Studi Kasus: Klasifikasi Multikelas (4 kelas) dengan Neural Network

Kelas: Baik, Sedang, Tidak Sehat, Berbahaya

Alur:
1. Membangun dataset simulasi sensor kualitas udara (7 fitur, 4 kelas)
2. Membagi data (stratified) dan normalisasi fitur
3. Membangun MLP dengan Batch Normalization, Dropout, dan Softmax
4. Pelatihan dengan EarlyStopping dan ReduceLROnPlateau
5. Evaluasi: akurasi, classification report, confusion matrix (jumlah dan
   ternormalisasi), serta kurva pelatihan
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models

SEED = 7
rng = np.random.default_rng(SEED)
tf.random.set_seed(SEED)

print("=== KLASIFIKASI TINGKAT KUALITAS UDARA (SENSOR IoT) ===")

# ---------------------------------------------------------------
# 1. DATASET SIMULASI SENSOR KUALITAS UDARA
# ---------------------------------------------------------------
KELAS = ["Baik", "Sedang", "Tidak Sehat", "Berbahaya"]
FITUR = ["pm25", "pm10", "co_ppm", "no2_ppb", "suhu_c",
         "kelembapan_persen", "kecepatan_angin_ms"]

# (rata-rata per kelas) -> urutan: Baik, Sedang, Tidak Sehat, Berbahaya
PARAM = {
    "pm25":               ([12, 35, 70, 140], [8, 14, 22, 40]),
    "pm10":               ([25, 60, 110, 190], [14, 22, 32, 55]),
    "co_ppm":             ([1.0, 2.5, 5.0, 9.0], [0.8, 1.4, 2.2, 3.5]),
    "no2_ppb":            ([15, 30, 50, 80], [9, 13, 18, 26]),
    "suhu_c":             ([27, 30, 32, 33], [3, 3, 3.2, 3.5]),
    "kelembapan_persen":  ([75, 68, 62, 58], [10, 10, 10, 11]),
    "kecepatan_angin_ms": ([4.0, 2.8, 1.8, 1.0], [1.6, 1.3, 1.0, 0.8]),
}

N_PER_KELAS = [1300, 1200, 900, 600]
label = np.concatenate([np.full(n, k) for k, n in enumerate(N_PER_KELAS)])
data = {}
for nama, (mu, sd) in PARAM.items():
    data[nama] = rng.normal(np.array(mu)[label], np.array(sd)[label])
df = pd.DataFrame(data).clip(lower=0)
df["kelas"] = label
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print("Distribusi kelas:")
for k, nama in enumerate(KELAS):
    print(f"  {nama:12s}: {(df['kelas'] == k).sum()}")
print()

X = df[FITUR].values
y = df["kelas"].values

# ---------------------------------------------------------------
# 2. PEMBAGIAN DATA & NORMALISASI
# ---------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tr, y_tr, test_size=0.15, random_state=SEED, stratify=y_tr)

scaler = StandardScaler().fit(X_tr)
X_tr, X_val, X_te = (scaler.transform(a) for a in (X_tr, X_val, X_te))
print(f"Data latih: {len(X_tr)} | Validasi: {len(X_val)} | Uji: {len(X_te)}\n")

# ---------------------------------------------------------------
# 3. ARSITEKTUR MODEL
# ---------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(len(FITUR),)),
    layers.Dense(48, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(24, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(len(KELAS), activation="softmax"),
])
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# ---------------------------------------------------------------
# 4. PELATIHAN
# ---------------------------------------------------------------
hist = model.fit(
    X_tr, y_tr, epochs=100, batch_size=32,
    validation_data=(X_val, y_val), verbose=0,
    callbacks=[
        callbacks.EarlyStopping(patience=12, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
    ],
)
print(f"\nPelatihan berhenti pada epoch ke-{len(hist.history['loss'])}")

# ---------------------------------------------------------------
# 5. EVALUASI
# ---------------------------------------------------------------
pred = model.predict(X_te, verbose=0).argmax(axis=1)
print(f"\nAkurasi data uji: {accuracy_score(y_te, pred):.4f}\n")
print("--- LAPORAN KLASIFIKASI ---")
print(classification_report(y_te, pred, target_names=KELAS))

laporan = pd.DataFrame(
    classification_report(y_te, pred, target_names=KELAS, output_dict=True)
).T
laporan.round(4).to_csv("laporan_klasifikasi.csv")

cm = confusion_matrix(y_te, pred)
cm_norm = cm / cm.sum(axis=1, keepdims=True)
print("--- CONFUSION MATRIX ---")
print(pd.DataFrame(cm, index=[f"asli_{k}" for k in KELAS],
                   columns=[f"pred_{k}" for k in KELAS]))

# ---------------------------------------------------------------
# 6. GRAFIK
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].plot(hist.history["loss"], label="Latih")
ax[0].plot(hist.history["val_loss"], label="Validasi")
ax[0].set_title("Kurva Loss")
ax[0].set_xlabel("Epoch")
ax[0].legend()

ax[1].plot(hist.history["accuracy"], label="Latih")
ax[1].plot(hist.history["val_accuracy"], label="Validasi")
ax[1].set_title("Kurva Akurasi")
ax[1].set_xlabel("Epoch")
ax[1].legend()

ax[2].imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
ax[2].set_xticks(range(4), KELAS, rotation=25)
ax[2].set_yticks(range(4), KELAS)
ax[2].set_xlabel("Prediksi")
ax[2].set_ylabel("Aktual")
ax[2].set_title("Confusion Matrix (ternormalisasi)")
for i in range(4):
    for j in range(4):
        ax[2].text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.0%})", ha="center",
                   va="center", fontsize=8,
                   color="white" if cm_norm[i, j] > 0.5 else "black")
plt.tight_layout()
plt.savefig("evaluasi_model.png", dpi=150)
print("\n[SELESAI] File tersimpan: evaluasi_model.png, laporan_klasifikasi.csv")
