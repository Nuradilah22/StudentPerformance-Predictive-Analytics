# Laporan Proyek Machine Learning - Nur adilah
## Domain Proyek
Pendidikan merupakan fondasi utama dalam pembangunan manusia dan masyarakat. Salah satu indikator penting dalam keberhasilan pendidikan adalah kemampuan siswa untuk mencapai prestasi akademik yang baik. Namun, siswa sering kali menghadapi berbagai tantangan yang dapat menghambat proses belajar mereka, seperti stres, kurang tidur, metode pengajaran yang kurang sesuai, hingga minimnya keterlibatan dalam kegiatan pembelajaran. Selain itu, keragaman latar belakang siswa, kebiasaan belajar yang berbeda, serta kondisi lingkungan keluarga dan sosial turut menjadi faktor yang memengaruhi pencapaian akademis siswa.

Masalah ini penting untuk diselesaikan karena kemampuan untuk memprediksi performa akademik siswa dapat membantu institusi pendidikan melakukan intervensi lebih dini terhadap siswa yang berisiko mengalami kegagalan belajar. Dengan begitu, sekolah dapat menyusun strategi dukungan yang lebih terarah dan tepat sasaran. Dalam konteks ini, pemanfaatan pendekatan berbasis machine learning menjadi alternatif solusi yang efektif untuk mengidentifikasi pola dan tren dari data siswa.

Referensi: 
- Aljarah, I., et al. (Kaggle Dataset): Students' Academic Performance Prediction using ML Algorithms
- Hasan, R., & Khan, M. A. (2020). "Academic Performance Prediction using Machine Learning Techniques". Procedia Computer Science.
- T. M. T. Nguyen et al., “Predicting Students’ Academic Performance Using Machine Learning: A Review,” IEEE Access, vol. 10, pp. 114054–114072, 2022.

# Business Understanding
### Problem Statement
- Bagaimana cara memprediksi tingkat nilai (GradeClass) siswa berdasarkan data demografis, perilaku belajar, dan dukungan eksternal?
- Fitur atau faktor apa yang paling berpengaruh terhadap prediksi performa akademik siswa?
- Bagaimana performa model terhadap kelas minoritas (GradeClass 0 dan 1) yang biasanya lebih sulit diprediksi?

### Goals
- Membangun model klasifikasi yang mampu memprediksi kelas nilai siswa (0–4) dengan akurasi yang tinggi, sehingga dapat digunakan sebagai alat bantu pengambilan keputusan oleh institusi pendidikan.
- Mengidentifikasi fitur penting yang paling memengaruhi hasil klasifikasi.
- Menilai kemampuan model dalam menangani ketidakseimbangan kelas (imbalanced class).

### Solution Statement
1. Menggunakan tiga algoritma machine learning:
- Logistic Regression sebagai baseline
- Random Forest Classifier
- Gradient Boosting Classifier
2. Model akan dievaluasi menggunakan metrik akurasi, precision, recall, dan f1-score.
3. Pemilihan model terbaik dilakukan berdasarkan hasil evaluasi pada data uji.


# Data Understanding
Dataset yang digunakan dalam proyek ini adalah Student Performance Dataset yang tersedia secara publik di Kaggle. Dataset ini dapat diakses melalui tautan berikut: [Student-Performance-Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)

### Variabel-variabel pada tudent Performance Dataset adalah sebagai berikut:
- `StudentID` : ID Siswa yang bersifat Uniq
- `Age` : Usia Siswa
- `StudyTimeWeekly` : Total jam belajar setiap minggu
- `Absences`: Jumlah ketidakhadiran siswa selama semester
- `GPA` :  Nilai rata-rata akademik
- `Gender` :  Jenis kelamin siswa (0 = perempuan, 1 = laki-laki)
- `Ethnicity` : Kelompok etnis siswa
- `ParentalEducation` : Tingkat pendidikan tertinggi yang dicapai oleh orang tua
- `Tutoring` : Apakah siswa mengikuti bimbingan belajar tambahan (0 = tidak, 1 = ya)
- `ParentalSupport` : Apakah siswa menerima dukungan belajar dari orang tua (0 = tidak, 1 = ya)
- `Extracurricular` : Keterlibatan dalam kegiatan ekstrakurikuler sekolah (0 = tidak, 1 = ya)
- `Sports` : Partisipasi dalam kegiatan olahraga (0 = tidak, 1 = ya)
- `Music` : Keterlibatan dalam kegiatan musik (0 = tidak, 1 = ya)
- `Volunteering` : Kegiatan sukarela di luar sekolah (0 = tidak, 1 = ya)
- `GradeClass` : Kategori tingkat prestasi akademik siswa (0 = sangat rendah, ..., 4 = sangat tinggi)

### Exploratory Data Analysis
Exploratory data analysis atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

Berikut ini adalah EDA yang dilakukan:

```python 
Student.info()
```

Output:

<pre>
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2392 entries, 0 to 2391
Data columns (total 15 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   StudentID          2392 non-null   int64  
 1   Age                2392 non-null   int64  
 2   Gender             2392 non-null   int64  
 3   Ethnicity          2392 non-null   int64  
 4   ParentalEducation  2392 non-null   int64  
 5   StudyTimeWeekly    2392 non-null   float64
 6   Absences           2392 non-null   int64  
 7   Tutoring           2392 non-null   int64  
 8   ParentalSupport    2392 non-null   int64  
 9   Extracurricular    2392 non-null   int64  
 10  Sports             2392 non-null   int64  
 11  Music              2392 non-null   int64  
 12  Volunteering       2392 non-null   int64  
 13  GPA                2392 non-null   float64
 14  GradeClass         2392 non-null   float64
dtypes: float64(3), int64(12)
memory usage: 280.4 KB
</pre>

Dapat dilihat dartaset ini memiliki 2392 baris dan memiliki 15 kolom, kemudian terdapat 3 kolom dengan tipe data float64 yaitu: StudyTimeWeekly, GPA, dan GradeClass. Namun baiknya kolom GradeClass dibuah menjaid tipe data integer karena pada kolom tersebut isinya menunjukkan kelas diskrit (seperti 1.0, 2.0, dst), sehingga lebih tepat jika dikonversi ke tipe integer dan terdapat beberapa kolom yang sebaiknya diubah ke tipe data kategori (category)

Selanjutnya dengan kode dibawah ini, kita akan melihat statistik deskriptif dataset

```python 
student.describe()
```

Output:

| Statistik | StudentID | Age  | Gender | Ethnicity | ParentalEducation | StudyTimeWeekly | Absences | Tutoring | ParentalSupport | Extracurricular | Sports | Music | Volunteering | GPA  | GradeClass |
|-----------|-----------|------|--------|-----------|-------------------|-----------------|----------|----------|------------------|------------------|--------|-------|---------------|------|-------------|
| Count     | 2392.0    | 2392.0 | 2392.0 | 2392.0    | 2392.0            | 2392.0          | 2392.0   | 2392.0   | 2392.0           | 2392.0           | 2392.0 | 2392.0 | 2392.0        | 2392.0 | 2392.0      |
| Mean      | 2196.5    | 16.47 | 0.51   | 0.88      | 1.75              | 9.77            | 14.54    | 0.30     | 2.12             | 0.38             | 0.30   | 0.20  | 0.16          | 1.91 | 2.98        |
| Std       | 690.66    | 1.12  | 0.50   | 1.03      | 1.00              | 5.65            | 8.47     | 0.46     | 1.12             | 0.49             | 0.46   | 0.40  | 0.36          | 0.92 | 1.23        |
| Min       | 1001.0    | 15.0  | 0.0    | 0.0       | 0.0               | 0.00            | 0.0      | 0.0      | 0.0              | 0.0              | 0.0    | 0.0   | 0.0           | 0.0  | 0.0         |
| 25%       | 1598.75   | 15.0  | 0.0    | 0.0       | 1.0               | 5.04            | 7.0      | 0.0      | 1.0              | 0.0              | 0.0    | 0.0   | 0.0           | 1.17 | 2.0         |
| 50%       | 2196.5    | 16.0  | 1.0    | 0.0       | 2.0               | 9.71            | 15.0     | 0.0      | 2.0              | 0.0              | 0.0    | 0.0   | 0.0           | 1.89 | 4.0         |
| 75%       | 2794.25   | 17.0  | 1.0    | 2.0       | 2.0               | 14.41           | 22.0     | 1.0      | 3.0              | 1.0              | 1.0    | 0.0   | 0.0           | 2.62 | 4.0         |
| Max       | 3392.0    | 18.0  | 1.0    | 3.0       | 4.0               | 19.98           | 29.0     | 1.0      | 4.0              | 1.0              | 1.0    | 1.0   | 1.0           | 4.0  | 4.0         |

Kemudian dengan kode dibawah ini untuk melihat missing value

```python 
student.isnull().sum()
```

Output:

| Column             | Missing Values |
|--------------------|----------------|
| StudentID          | 0              |
| Age                | 0              |
| Gender             | 0              |
| Ethnicity          | 0              |
| ParentalEducation  | 0              |
| StudyTimeWeekly    | 0              |
| Absences           | 0              |
| Tutoring           | 0              |
| ParentalSupport    | 0              |
| Extracurricular    | 0              |
| Sports             | 0              |
| Music              | 0              |
| Volunteering       | 0              |
| GPA                | 0              |
| GradeClass         | 0              |

Tidak terdapat nilai kosong (missing values) pada seluruh kolom dalam dataset. Data sudah lengkap dan siap diproses lebih lanjut tanpa perlu melakukan pembersihan nilai null.

Kemudian, mengubah tipe data dengan kode ini

```python
student['GradeClass'] = student['GradeClass'].astype(int)
student['GradeClass'].unique()
```

Output:
<pre>array([2, 1, 4, 3, 0])</pre>

Dari output tersebut array([2, 1, 4, 3, 0]) menunjukkan bahwa kolom 'GradeClass' hanya memiliki lima nilai unik, yaitu 0, 1, 2, 3, dan 4. Ini mengindikasikan bahwa "kelas nilai akhir" siswa dikategorikan ke dalam lima tingkatan yang berbeda.

```python
# Mengubah tipe data 'GradeClass' ke integer
student['GradeClass'] = student['GradeClass'].astype(int)

categorical_features = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']

# Ubah tipe data jadi category
student[categorical_features] = student[categorical_features].astype('category')

student.dtypes
```

Output:

| Column             | Data Type |
|--------------------|-----------|
| StudentID          | int64     |
| Age                | int64     |
| Gender             | category  |
| Ethnicity          | category  |
| ParentalEducation  | category  |
| StudyTimeWeekly    | float64   |
| Absences           | int64     |
| Tutoring           | category  |
| ParentalSupport    | category  |
| Extracurricular    | category  |
| Sports             | category  |
| Music              | category  |
| Volunteering       | category  |
| GPA                | float64   |
| GradeClass         | int64     |

Beberapa kolom telah berhasil diubah tipe datanya pada kolom 'GradeClass' menjadi integer dan beberapa kolom lainnya seperti (Gender. Ethnicity, dll.) menjadi tipe data kategori. Ini mengkonfirmasi bahwa perubahan tipe data berhasil diterapkan, mengoptimalkan penggunaan memori dan performa, serta menyiapkan data dengan tepat untuk analisis dan pemodelan selanjutnya

Kemudian kita cek apakah ada outlier di dataset ini

```python
sns.set(style="whitegrid")

numerical_features = ['Age', 'StudyTimeWeekly', 'Absences', 'GPA']

plt.figure(figsize=(10, 8))
for i, col in enumerate(numerical_features):
  plt.subplot(2, 2, i + 1)
  sns.boxplot(x=student[col], color='skyblue')
  plt.title(f'Boxplot {col}')

plt.tight_layout()
plt.show()
```

Output:

![download](https://github.com/user-attachments/assets/0b79164b-bb3a-4f62-925b-b9816c3f0c46)

Terlihat bahwa pada keempat fitur tersebut tidak terdeteksi adanya outlier. Maka tidak diperlukan penangan untuk membersihkan outlier.

#### Visualisasi Data
- _Univariate Analysis_

  Memisahkan fitur menjadi Numerical Features dan Categorical Features:
  
  - Categorical Feature
  
  ![download](https://github.com/user-attachments/assets/248ab1d0-c175-4acf-b45c-1ddcc644b126)

  Dari visualisasi distribusi kolom kategorikal, terlihat bahwa sebagian besar variabel memiliki ketidakseimbangan distribusi. Gender cukup seimbang antara dua kategori, sementara Ethnicity
  didominasi oleh satu kategori (kode 0). ParentalEducation menunjukkan dominasi pada kategori 2, dengan kategori 4 sangat sedikit dan mungkin bisa digabung. Sebagian besar siswa tidak
  mengikuti bimbingan belajar (Tutoring), dan mayoritas mendapat dukungan orang tua dari kategori 2 dan 3. Aktivitas seperti Extracurricular, Sports, Music, dan Volunteering umumnya menunjukkan
  partisipasi rendah, yang bisa menjadi fitur penting dalam pemodelan karena distribusinya tidak merata.
  
  - Numerical Feature
  
  ![download](https://github.com/user-attachments/assets/28d326f5-eec6-44d2-9caa-4f5c1f0db00a)

  Berdasarkan visualisasi distribusi fitur numerik, dapat disimpulkan bahwa usia siswa (Age) berkisar antara 15–18 tahun dengan distribusi yang relatif merata. Rata-rata waktu belajar mingguan
  (StudyTimeWeekly) berada di kisaran 8–12 jam, dengan distribusi condong ke tengah dan minim outlier ekstrem. Jumlah absensi (Absences) tersebar dari 0 hingga 29, dengan distribusi sedikit
  miring ke kanan yang menunjukkan hanya sebagian kecil siswa yang sering bolos. Untuk nilai akademik (GPA), sebagian besar siswa memiliki GPA antara 1.0–3.0, dengan distribusi menyerupai kurva
  normal yang sedikit condong ke kanan, menandakan hanya sedikit siswa yang memiliki GPA sangat rendah atau sangat tinggi.

- _Multivariate Analysis_

  Menganalisis hubungan fitur kategorikal terhadap GPA
  
  ![download](https://github.com/user-attachments/assets/241cef31-cbc0-460b-980e-2eacf74e6cb5)

  Dari hasil analisis hubungan antara fitur kategorikal dengan GPA, terlihat bahwa siswa yang mengikuti tutoring cenderung memiliki median GPA yang lebih tinggi dibandingkan yang tidak. Dukungan orang tua (ParentalSupport) juga menunjukkan tren positif terhadap GPA, di mana semakin tinggi dukungan, nilai GPA siswa cenderung meningkat. Fitur lain seperti keterlibatan dalam kegiatan ekstrakurikuler, musik, dan kegiatan sukarelawan memang tidak menunjukkan perbedaan mencolok, namun masih terlihat adanya variasi GPA antar kategorinya. Sementara itu, fitur Gender dan Ethnicity tidak menunjukkan pengaruh signifikan terhadap GPA. Menariknya, pada fitur ParentalEducation justru terlihat adanya tren penurunan GPA pada tingkat pendidikan orang tua yang lebih tinggi, yang mungkin dipengaruhi oleh faktor eksternal lain.

  #### Scatterplot
  
  ![download](https://github.com/user-attachments/assets/9f382233-3e32-4d63-9ea1-66b971166719)

  Dari hasil analisis hubungan antar variabel, terlihat bahwa semakin sering siswa absen (Absences), maka semakin rendah nilai GPA mereka, menunjukkan adanya hubungan negatif yang cukup kuat. Selain itu, terdapat kecenderungan bahwa semakin banyak waktu yang dihabiskan untuk belajar setiap minggu (StudyTimeWeekly), GPA cenderung meningkat, meskipun hubungan ini tidak terlalu signifikan. Sementara itu, variabel GradeClass menunjukkan pola menurun seiring dengan meningkatnya GPA, yang mengindikasikan bahwa siswa dengan GPA tinggi cenderung memiliki peringkat kelas yang lebih baik.

  #### Heatmap Correlation
  
  ![download](https://github.com/user-attachments/assets/897ae006-cd6b-4a56-8c24-467b1d80461f)

  Berdasarkan analisis korelasi, GPA memiliki hubungan negatif yang sangat kuat dengan Absences (-0.92), menunjukkan bahwa semakin sering siswa absen, semakin rendah nilai GPA mereka. GradeClass juga berkorelasi negatif cukup kuat dengan GPA (-0.78), yang berarti siswa dengan nilai GPA tinggi cenderung mendapat peringkat kelas yang lebih baik (GradeClass lebih rendah). Sementara itu, Absences berkorelasi positif dengan GradeClass (0.73), mengindikasikan bahwa siswa yang sering absen cenderung memiliki peringkat kelas lebih buruk. Fitur seperti ParentalSupport, StudyTimeWeekly, dan ParentalEducation hanya menunjukkan korelasi lemah terhadap GradeClass, sehingga pengaruhnya terhadap kelulusan kemungkinan tidak signifikan secara langsung. StudentID dan Age memiliki nilai korelasi mendekati nol terhadap GradeClass, menandakan bahwa keduanya tidak berpengaruh berarti terhadap peringkat kelas siswa.

# Data Preparation

- Encoding Fitur Kategorikal
  
  Untuk melakukan proses encoding fitur kategori, salah satu teknik yang umum dilakukan adalah teknik one-hot-encoding.

  ```python
  from sklearn.preprocessing import OneHotEncoder

  # pisahkan target dan fitur
  X = student.drop(columns=["GradeClass", "StudentID"])
  y = student["GradeClass"]
  
  # 2. One-hot encoding fitur kategorikal pakai get_dummies
  categorical_features = ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring',
                          'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
  
  X = pd.get_dummies(X, columns=categorical_features, drop_first=True)
  student.head()
  ```

  Output:

  | StudentID | Age | Gender | Ethnicity | ParentalEducation | StudyTimeWeekly | Absences | Tutoring | ParentalSupport | Extracurricular | Sports | Music | Volunteering | GPA       | GradeClass |
  |-----------|-----|--------|-----------|--------------------|------------------|----------|----------|------------------|------------------|--------|--------|---------------|-----------|-------------|
  | 1001      | 17  | 1      | 0         | 2                  | 19.83           | 7        | 1        | 2                | 0                | 0      | 1      | 0             | 2.929196  | 2           |
  | 1002      | 18  | 0      | 0         | 1                  | 15.41           | 0        | 0        | 1                | 0                | 0      | 0      | 0             | 3.042915  | 1           |
  | 1003      | 15  | 0      | 2         | 3                  | 4.21            | 26       | 0        | 2                | 0                | 0      | 0      | 0             | 0.112602  | 4           |
  | 1004      | 17  | 1      | 0         | 3                  | 10.03           | 14       | 0        | 3                | 1                | 0      | 0      | 0             | 2.054218  | 3           |
  | 1005      | 17  | 1      | 0         | 2                  | 4.67            | 17       | 1        | 3                | 0                | 0      | 0      | 0             | 1.288061  | 4           |

  Dapat dilihat, dari analisis awal terlihat bahwa keikutsertaan dalam tutoring tidak selalu berdampak langsung pada GPA karena dipengaruhi faktor lain seperti absensi dan waktu belajar.
  Absensi tinggi cenderung berkorelasi dengan GPA rendah, sementara dukungan orang tua dan waktu belajar yang tinggi tampaknya berkontribusi positif terhadap performa akademik. Usia tidak
  menunjukkan pengaruh signifikan terhadap GPA. Aktivitas ekstrakurikuler seperti musik dan sukarelawan masih minim, sehingga belum cukup data untuk dianalisis lebih lanjut.

- Split Dataset

  Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. Kita perlu mempertahankan sebagian data yang ada untuk menguji
  seberapa baik generalisasi model terhadap data baru. Sebelum proses pembagian, fitur prediktor (X) dipisahkan terlebih dahulu dari variabel target (y) dengan menghapus kolom target dari
  dataset utama yang sudah dilakukan pada bagian encoding di atas. Selanjutnya, fungsi train_test_split dari library scikit-learn digunakan untuk membagi data dengan proporsi 80% sebagai data
  latih dan 20% sebagai data uji. Parameter random_state=42 disertakan agar hasil pembagian tetap konsisten saat kode dijalankan kembali. Setelah proses pembagian, jumlah data pada masing-
  masing subset dicetak untuk memastikan bahwa pembagian sudah sesuai dengan yang direncanakan. Kode berikut digunakan untuk melakukan proses ini.

  ```python
  from sklearn.model_selection import train_test_split

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```

- Standarisasi
  
  Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori.

  ```python
  from sklearn.preprocessing import StandardScaler

  numerical_features = ['Age', 'StudyTimeWeekly', 'Absences']
  scaler = StandardScaler()
  X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
  X_test[numerical_features] = scaler.transform(X_test[numerical_features])
  X_train[numerical_features].head()
  ```

  Output:

  | Index | Age      | StudyTimeWeekly | Absences  |
  |-------|----------|------------------|-----------|
  | 642   | 1.372851 | 1.468159         | 1.105915  |
  | 1752  | -0.405858| -1.276773        | 0.516509  |
  | 1401  | 0.483497 | -1.103632        | 1.223797  |
  | 2032  | 0.483497 | 1.068117         | 1.223797  |
  | 990   | -0.405858| -1.526524        | 0.516509  |

  Setelah distandardisasi menggunakan StandardScaler, nilai fitur numerik ('Age', 'StudyTimeWeekly', 'Absences') tidak lagi dalam satuan aslinya, melainkan direpresentasikan sebagai skor-z (z-
  scores) yang menunjukkan jarak setiap nilai dari rata-rata dalam satuan standar deviasi. Nilai positif berarti di atas rata-rata, negatif berarti di bawah rata-rata, dan nilai mendekati nol
  berarti dekat dengan rata-rata.

  Kemudian untuk mengecek nilai mean dan standar deviasi pada setelah proses standarisasi, jalankan kode ini:

  ```python X_train[numerical_features].describe().round(4)```

  Output:
  
  | Statistik | Age     | StudyTimeWeekly | Absences  |
  |-----------|---------|------------------|-----------|
  | Count     | 1913.0000 | 1913.0000       | 1913.0000 |
  | Mean      | -0.0000  | 0.0000          | -0.0000   |
  | Std       | 1.0003   | 1.0003          | 1.0003    |
  | Min       | -1.2952  | -1.7085         | -1.7232   |
  | 25%       | -1.2952  | -0.8401         | -0.8981   |
  | 50%       | -0.4059  | -0.0403         | 0.0450    |
  | 75%       | 0.4835   | 0.8172          | 0.8702    |
  | Max       | 1.3729   | 1.7926          | 1.6953    |

  Sekarang Fitur numrikal memiliki rata-rata mendekati nol dan standar deviasi mendekati satu, yang merupakan kondisi ideal untuk banyak algoritma machine learning yang sensitif terhadap
  skala data.
  
- Reduksi Dimensi dengan Principal Component Analysis (PCA)

  ```python
  from sklearn.decomposition import PCA

  pca = PCA(n_components=0.95)
  X_train_pca = pca.fit_transform(X_train)
  X_test_pca = pca.transform(X_test)
  
  print("Jumlah komponen PCA:", X_train_pca.shape[1])
  ```

  Setelah reduksi dimensi dengan PCA (n_components=0.95):
  PCA secara otomatis memilih 15 komponen utama yang bisa menjelaskan 95% total variasi (informasi) dalam data.

# Modeling

```python
# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_acc', 'test_acc'],
                      columns=['LogisticRegression', 'RandomForest', 'GradientBoosting'])
```

Pada tahap ini, saya membangun 3 model machine learning yaitu:

**1. Logistic Regression**

   Logistic Regression adalah algoritma klasifikasi yang relatif sederhana. Cara kerjanya menggunakan fungsi logistik untuk memprediksi probabilitas kelas target sangat intuitif. Ini membuatnya
mudah untuk dipahami, diinterpretasikan, dan di-debug. Model Logistic Regression umumnya sangat cepat untuk dilatih, terutama pada dataset berukuran menengah seperti ini. Ini memungkinkan
pengembang untuk dengan cepat mendapatkan hasil awal dan mengevaluasi kinerja dasar sebelum beralih ke model yang lebih kompleks dan memakan waktu lebih lama untuk dilatih.

   - Parameter yang digunakan yaitu (max_iter=1000)

   ```python
   from sklearn.linear_model import LogisticRegression
   lr_model = LogisticRegression(max_iter=1000)
   lr_model.fit(X_train, y_train)
   
   y_pred_lr = lr_model.predict(X_test)
   ```
**2. Random Forest**

   Random Forest adalah pilihan yang sangat baik untuk banyak masalah klasifikasi karena performanya yang kuat dan kemampuannya mengurangi overfitting, menjadikannya pilihan default yang sering digunakan di awal proyek. Random Forest memiliki kelebihan utama dalam memberikan performa prediksi yang tinggi dan efektif mengurangi risiko overfitting dengan menggabungkan prediksi dari
banyak pohon keputusan yang dilatih pada subset data dan fitur yang berbeda, serta relatif kuat terhadap outlier. Namun, kekurangannya adalah model ini cenderung kurang dapat
diinterpretasikan dibandingkan model yang lebih sederhana, membutuhkan sumber daya komputasi dan memori yang lebih besar, dan dapat memiliki bias terhadap fitur kategorikal dengan banyak level.

   - Parameter yang digunakan yaitu (random_state=42)
     
   ```python
   from sklearn.ensemble import RandomForestClassifier
   rf_model = RandomForestClassifier(random_state=42)
   rf_model.fit(X_train, y_train)
   
   y_pred_rf = rf_model.predict(X_test)
   ```

**3. Gradient Boosting Classifier**

   Gradient Boosting adalah algoritma yang sangat kuat yang dapat menghasilkan performa tinggi, tetapi membutuhkan lebih banyak perhatian pada tuning parameter untuk menghindari overfitting
dan bisa lebih lambat dalam proses pelatihannya. Adapun kelebihan berupa performa prediksi yang sangat tinggi, seringkali unggul pada banyak dataset, dan sangat baik dalam menangkap hubungan
non-linear dan interaksi antar fitur dengan secara iteratif memperbaiki kesalahan dari model sebelumnya. Namun, kekurangan utamanya adalah model ini lebih rentan terhadap overfitting jika
hyperparameter tidak di-tune dengan hati-hati, cenderung sensitif terhadap outlier, proses pelatihannya lebih lambat karena bersifat sekuensial, dan membutuhkan lebih banyak hyperparameter
tuning untuk mencapai kinerja optimal serta kurang dapat diinterpretasikan.

   - Parameter yang digunakan yaitu (random_state=42)

   ```python
  from sklearn.ensemble import GradientBoostingClassifier
  gb_model = GradientBoostingClassifier(random_state=42)
  gb_model.fit(X_train, y_train)
  
  y_pred_gb = gb_model.predict(X_test)
   ```
   
# Evaluation

Pada tahap evaluasi, digunakan beberapa metrik evaluasi untuk mengukur performa model klasifikasi yang telah dibangun, yaitu: **Precision**, **Recall**, dan **F1-Score**.

- Precision mengukur berapa banyak prediksi positif yang benar dari semua prediksi positif.

  Formula:
  
  ![443141671-d16e2027-7722-4b4c-9d30-a9cbff2de6f6](https://github.com/user-attachments/assets/8536173f-f1d9-46c6-b22d-ffbc5e514e33)

- Recall mengukur berapa banyak prediksi positif yang benar dari semua kejadian aktual positif

  Formula:

  ![443141869-80cc663d-b807-4cb7-a3d2-4271d854d8f8](https://github.com/user-attachments/assets/f4f0b3c2-d215-4e69-bdaf-73179d489597)

- F1-Score merupakan gabungan antara precision dan recall. Metrik ini berguna saat dibutuhkan keseimbangan antara keduanya.

  Formula:

  ![443142312-007c3a65-1de9-4fd8-a672-35c3c90ae4ca](https://github.com/user-attachments/assets/6e5d9df1-a66e-46c8-8396-ce5bc256ce61)

Hasil evaluasi dari 3 model tersbut yaitu:

<pre>
Logistic Regression:
Accuracy: 0.74
Precision (weighted avg): 0.71
Recall (weighted avg): 0.74
F1 Score (weighted avg): 0.72

Random Forest:
Accuracy: 0.91
Precision (weighted avg): 0.91
Recall (weighted avg): 0.91
F1 Score (weighted avg): 0.9

Logistic Regression:
Accuracy: 0.9
Precision (weighted avg): 0.9
Recall (weighted avg): 0.9
F1 Score (weighted avg): 0.9
</pre>

Berdasarkan hasil evaluasi dan metrik klasifikasi yang telah dilakukan, model yang paling optimal untuk digunakan dalam prediksi GradeClass adalah model Random Forest Anda (akibat akurasi 91%) secara langsung menjawab tujuan proyek dan berdampak positif pada business understanding.

1. Model membantu mengidentifikasi siswa yang berpotensi berjuang (GradeClass rendah) atau berpotensi sukses (GradeClass tinggi) berdasarkan faktor-faktor dalam data, yang merupakan inti tantangan institusi pendidikan.

2. Model berhasil membangun model klasifikasi multi-kelas yang akurat untuk memprediksi GradeClass, memungkinkan deteksi dini potensi performa akademik siswa

3. Adapun dampak dari setiap solusi steatment yaitu:

- Prediksi model memungkinkan guru fokus membimbing siswa berisiko, mengoptimalkan waktu mereka.
- Prediksi memungkinkan sekolah membuat program dukungan (intervensi/pengayaan) yang sangat spesifik dan tertarget pada kelompok siswa yang memerlukannya.
-  Informasi prediksi membantu orang tua lebih memahami potensi anak dan memberikan dukungan yang tepat di rumah.
