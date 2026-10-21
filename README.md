# Klasifikasi Tingkat Kualitas Udara dari Data Sensor IoT dengan Deep Learning

Studi kasus penerapan jaringan saraf tiruan (*Deep Learning*) untuk klasifikasi multikelas: mengenali tingkat kualitas udara **Baik, Sedang, Tidak Sehat, dan Berbahaya** dari data sensor lingkungan.

## Fitur Proyek

- **Arsitektur Neural Network**: dua *hidden layer* (48 dan 24 neuron, ReLU) dengan *Batch Normalization* dan *Dropout*, serta output *Softmax* untuk 4 kelas.
- **Dataset Simulasi**: 7 fitur sensor (PM2.5, PM10, CO, NO2, suhu, kelembapan, kecepatan angin) dengan distribusi kelas yang sedikit tidak merata.
- **Data Splitting**: pembagian data latih/uji 80:20 secara stratified, normalisasi fitur (StandardScaler), dan data validasi 15% dari data latih.
- **Callback Pelatihan**: *EarlyStopping* dan *ReduceLROnPlateau* untuk menjaga kestabilan pelatihan.
- **Evaluasi Model**: akurasi, laporan klasifikasi (precision, recall, F1), confusion matrix (jumlah dan persentase), serta kurva loss dan akurasi.

## Cara Menjalankan

```
pip install -r requirements.txt
python main.py
```

## Output

| File                      | Isi                                                  |
| ------------------------- | ---------------------------------------------------- |
| `evaluasi_model.png`      | Kurva loss, kurva akurasi, dan confusion matrix      |
| `laporan_klasifikasi.csv` | Precision, recall, F1-score per kelas                |

Catatan: dataset dibuat secara simulasi (sintetis) untuk keperluan studi kasus.
