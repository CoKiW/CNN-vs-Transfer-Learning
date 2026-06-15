# CNN-vs-Transfer-Learning

# Analisis
1. Analisis Dataset

a. Apakah dataset cukup besar untuk CNN from scratch?
Dataset CIFAR-10 yang digunakan hanya 700 gambar training. Ukuran ini terlalu kecil untuk CNN from scratch yang optimal — idealnya diperlukan minimal 5.000–10.000 gambar per kelas. Dengan data sekecil ini, model berisiko tinggi mengalami overfitting.

b. Variasi gambar
CIFAR-10 memiliki variasi yang cukup baik (berbagai sudut pandang, pencahayaan, dan latar belakang) meskipun resolusinya sangat rendah (32×32). Variasi ini membantu generalisasi namun informasi tekstur/detail terbatas.

c. Ketidakseimbangan data
Kelas Airplane dan Automobile memiliki jumlah yang seimbang (1.000 sampel per kelas di dataset asli). Setelah filter dan split, distribusi tetap seimbang ±50/50.

d. Noise dan kompleksitas gambar
Gambar CIFAR-10 berukuran kecil (32×32) sehingga noise relatif rendah, namun detail fitur juga terbatas. Cats vs Dogs memiliki variasi latar belakang dan pose yang lebih tinggi.

e. Pengaruh kualitas dataset
CNN from scratch sangat bergantung pada kuantitas dan kualitas data — dengan hanya 700 sampel, model cenderung underfit atau overfit. Transfer Learning lebih robust terhadap dataset kecil karena fitur pretrained sudah kaya.

2. Analisis Performa Model
   
a. Model dengan performa terbaik
Transfer Learning (MobileNetV2) menghasilkan akurasi testing yang lebih tinggi dengan epoch lebih sedikit (5 vs 10). Hal ini menunjukkan keunggulan pretrained features untuk dataset berukuran kecil.

b. Akurasi tinggi ≠ model lebih baik
Akurasi tinggi bisa menyesatkan jika dataset tidak seimbang atau model mengalami overfitting. Confusion matrix dan classification report (precision, recall, F1) lebih informatif untuk menilai performa sesungguhnya.

c. Tanda overfitting
CNN from scratch berpotensi lebih besar mengalami overfitting (gap training vs validation accuracy lebih besar) dibanding Transfer Learning yang layer-nya dibekukan.

d. Stabilitas training
Transfer Learning lebih stabil — validation accuracy cenderung konsisten naik karena tidak ada update pada layer besar. CNN from scratch lebih fluktuatif karena semua parameter diperbarui setiap epoch.

e. Hubungan jumlah data dan performa
CNN from scratch sangat sensitif terhadap jumlah data. Transfer Learning mampu menghasilkan performa baik bahkan dengan data terbatas karena memanfaatkan pengetahuan dari ImageNet (1,2 juta gambar).

3. Analisis Pemilihan Pendekatan
   
Kapan menggunakan CNN from scratch?

Dataset sangat besar (>100.000 gambar per kelas)
Domain sangat spesifik dan berbeda jauh dari ImageNet (contoh: citra medis histopatologi, citra satelit inframerah)
Tersedia GPU/TPU dan waktu training yang panjang
Diperlukan kontrol penuh atas arsitektur untuk deployment di perangkat edge
Kapan menggunakan Transfer Learning?

Dataset kecil hingga menengah (<50.000 gambar)
Domain memiliki kemiripan dengan ImageNet (objek umum, hewan, kendaraan)
Waktu dan komputasi terbatas
Diperlukan prototipe cepat
Akurasi tinggi menjadi prioritas

4. Studi Kasus Pengambilan Keputusan
   
Skenario 1 – Klinik dengan 300 gambar medis
Pilihan: Transfer Learning (Feature Extraction)
Dengan hanya 300 gambar, CNN from scratch hampir pasti mengalami overfitting parah dan tidak dapat menangkap pola yang bermakna. Transfer Learning dengan model seperti MobileNetV2 atau ResNet yang di-freeze memungkinkan classifier baru dilatih dengan data minimal namun tetap memanfaatkan fitur visual kaya dari ImageNet. Untuk domain medis, fine-tuning pada beberapa layer terakhir juga bisa dilakukan secara hati-hati untuk adaptasi domain.

Skenario 2 – Perusahaan dengan 1 juta gambar produk internal
Pilihan: CNN from Scratch (atau Transfer Learning + Full Fine-tuning)
Dengan 1 juta gambar dan karakteristik yang sangat berbeda dari ImageNet, CNN from scratch menjadi relevan karena dataset cukup besar untuk melatih model dari awal. Namun, pendekatan hybrid (mulai dari pretrained model lalu fine-tune semua layer) tetap direkomendasikan karena menghemat waktu konvergensi awal secara signifikan.

Skenario 3 – Prototipe 2 hari, 500 gambar
Pilihan: Transfer Learning (Feature Extraction)
Transfer Learning adalah satu-satunya pilihan rasional di sini. Dengan 500 gambar dan tenggat 2 hari, tidak ada waktu untuk eksperimen arsitektur CNN atau hyperparameter tuning. MobileNetV2 yang dibekukan dengan classifier sederhana bisa mencapai akurasi acceptable dalam hitungan menit training.

Skenario 4 – Dataset besar, GPU memadai, domain sangat spesifik
Pilihan: Transfer Learning + Full Fine-tuning (bukan from scratch murni)
Meskipun punya data dan komputasi besar, memulai dari pretrained model tetap lebih efisien — konvergensi lebih cepat 3–10× dibanding random initialization. Full fine-tuning semua layer pada pretrained model dengan data besar justru memberikan hasil terbaik (pendekatan yang digunakan oleh model SOTA seperti EfficientNet pada ImageNet).

# Refleksi Pribadi

1. Tantangan terbesar
Memahami pipeline tf.data untuk Transfer Learning — khususnya preprocessing yang berbeda per model (MobileNetV2 perlu range [-1,1], bukan [0,1] seperti CNN biasa). Kesalahan ini tidak selalu terlihat dari error message, namun berdampak signifikan pada akurasi.

2. Bagian paling sulit
Transfer Learning terasa lebih rumit secara konseptual karena harus memahami mekanisme freezing layer, pemilihan pretrained model yang tepat, dan menyesuaikan preprocessing dengan spesifikasi tiap model.

3. Perbedaan paling terasa
Melatih CNN from scratch terasa seperti "membangun dari nol" — setiap komponen arsitektur harus dipertimbangkan. Transfer Learning seperti "berdiri di atas bahu raksasa" — kita memanfaatkan jutaan gambar yang sudah dipelajari model sebelumnya, sehingga convergence jauh lebih cepat.

4. Pilihan untuk kasus nyata
Transfer Learning, terutama untuk dataset berukuran kecil hingga menengah. Alasannya: lebih cepat dikembangkan, lebih stabil, dan lebih mudah di-deploy.

5. Hal baru yang dipelajari
Pengambilan keputusan dalam deep learning tidak hanya soal akurasi, tetapi mempertimbangkan trade-off antara waktu, data, komputasi, dan risiko overfitting secara holistik. Tidak ada satu pendekatan yang selalu terbaik — konteks penggunaan adalah faktor penentu utama.
