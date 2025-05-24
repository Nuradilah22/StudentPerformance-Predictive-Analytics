# Laporan Proyek Machine Learning - Nur adilah
## Domain Proyek
Pendidikan merupakan fondasi utama dalam pembangunan manusia dan masyarakat. Salah satu indikator penting dalam keberhasilan pendidikan adalah kemampuan siswa untuk mencapai prestasi akademik yang baik. Namun, siswa sering kali menghadapi berbagai tantangan yang dapat menghambat proses belajar mereka, seperti stres, kurang tidur, metode pengajaran yang kurang sesuai, hingga minimnya keterlibatan dalam kegiatan pembelajaran. Selain itu, keragaman latar belakang siswa, kebiasaan belajar yang berbeda, serta kondisi lingkungan keluarga dan sosial turut menjadi faktor yang memengaruhi pencapaian akademis siswa.

Masalah ini penting untuk diselesaikan karena kemampuan untuk memprediksi performa akademik siswa dapat membantu institusi pendidikan melakukan intervensi lebih dini terhadap siswa yang berisiko mengalami kegagalan belajar. Dengan begitu, sekolah dapat menyusun strategi dukungan yang lebih terarah dan tepat sasaran. Dalam konteks ini, pemanfaatan pendekatan berbasis machine learning menjadi alternatif solusi yang efektif untuk mengidentifikasi pola dan tren dari data siswa.

Referensi: 
- Aljarah, I., et al. (Kaggle Dataset): Students' Academic Performance Prediction using ML Algorithms
- Hasan, R., & Khan, M. A. (2020). "Academic Performance Prediction using Machine Learning Techniques". Procedia Computer Science.
- T. M. T. Nguyen et al., “Predicting Students’ Academic Performance Using Machine Learning: A Review,” IEEE Access, vol. 10, pp. 114054–114072, 2022.

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

```python Student.info()```

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

```python student.describe()```

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

```python student.isnull().sum()```

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
