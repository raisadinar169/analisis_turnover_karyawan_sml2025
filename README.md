# analisis_attrition_karyawan_sml2025
Proyek machine learning untuk memprediksi turnover karyawan berdasarkan dataset kompetisi Kaggle Tugas 1 SML A-2025
# Laporan Proyek Machine Learning 

## Prediksi Turnover Karyawan  
**Nama:**
1. Najwa Allena Eka P.S (5003231025)
2. Najwa Putri Lathifah (5003231026)
3. Raisa Dinar Izzati (5003231028)
   
---

## Daftar Isi  
- [Domain Proyek: Sumber Daya Manusia](#domain-proyek-sumber-daya-manusia)  
  - [Referensi](#referensi)  
- [Business Understanding](#business-understanding)  
  - [Problem Statements](#problem-statements)  
  - [Goals](#goals)  
- [Data Understanding](#data-understanding)  
- [Data Preparation](#data-preparation)  
- [Modelling](#modelling)  
- [Evaluation](#evaluation)  
- [Kesimpulan](#kesimpulan)  

---

## Domain Proyek: Sumber Daya Manusia  
Proyek ini berfokus pada prediksi **turnover** atau perpindahan/keluar karyawan di sebuah perusahaan menggunakan dataset kompetisi Kaggle “Tugas 1 SML A-2025”. Dataset ini mencakup profil karyawan, lingkungan kerja, dan status keluar atau tidak. :contentReference[oaicite:2]{index=2}  

### Referensi  
- Dataset: [Tugas 1 SML A-2025 – Kaggle](https://www.kaggle.com/competitions/tugas-1-sml-a-2025/data)  

---

## Business Understanding  

### Problem Statements  
Tingkat keluar-masuk karyawan (attrition) merupakan salah satu permasalahan penting bagi perusahaan karena dapat menimbulkan kerugian, baik dari segi biaya rekrutmen, pelatihan, maupun penurunan produktivitas kerja. Ketika banyak karyawan yang keluar, proses adaptasi dan efisiensi tim jadi terganggu. Oleh karena itu, perusahaan perlu memahami faktor-faktor yang memengaruhi keputusan karyawan untuk keluar dan memprediksi kemungkinan tersebut sedini mungkin agar dapat dicegah sedini mungkin.

### Goals   
- Membuat model untuk memprediksi apakah seorang karyawan akan keluar dari perusahaan atau tidak.
- Mengetahui faktor-faktor apa saja yang paling berpengaruh terhadap keputusan karyawan keluar.
- Menghasilkan model dengan performa optimal menggunakan pendekatan hyperparameter tuning (GridSearchCV) dan evaluasi berbasis metrik AUC, akurasi, presisi, recall, serta F1-score.
- Memberikan saran yang dapat membantu HR dalam menjaga karyawan tetap bertahan di perusahaan.

---

## Data Understanding  
Dataset yang digunakan pada proyek ini diperoleh dari situs [Tugas 1 SML A-2025 – Kaggle](https://www.kaggle.com/competitions/tugas-1-sml-a-2025/data). 
Dataset ini terdiri dari 1170 data karyawan yang berisi 31 variabel seperti
- id : ID unik karyawan untuk identifikasi.
- Age : Usia karyawan.
- BusinessTravel : Frekuensi perjalanan dinas karyawan (contoh: Non-Travel, Travel_Rarely, Travel_Frequently).
- DailyRate :  Gaji harian.
- Department :  Departemen tempat karyawan bekerja (contoh: Sales, Research & Development, Human Resources).
- DistanceFromHome : Jarak tempat tinggal karyawan ke kantor.
- Education : Tingkat pendidikan terakhir: 1 = Below College, 2 = College, 3 = Bachelor, 4 = Master, 5 = Doctor.
- EducationField : Bidang studi terakhir karyawan (contoh: Life Sciences, Medical, Marketing).
- EmployeeCount : Jumlah karyawan (selalu 1 dalam dataset).
- EmployeeNumber : Nomor unik karyawan dalam sistem HR.
- EnvironmentSatisfaction : Tingkat kepuasan terhadap lingkungan kerja: 1 = Low, 2 = Medium, 3 = High, 4 = Very High.
- Gender : Jenis kelamin karyawan (Male/Female).
- HourlyRate : Upah per jam.
- JobInvolvement : Tingkat keterlibatan pekerjaan: 1 = Low, 2 = Medium, 3 = High, 4 = Very High.
- JobLevel : Level jabatan karyawan.
- JobRole : Posisi/jabatan spesifik karyawan (contoh: Sales Executive, Research Scientist).
- JobSatisfaction : Tingkat kepuasan pekerjaan: 1 = Low, 2 = Medium, 3 = High, 4 = Very High.
- MaritalStatus : Status pernikahan karyawan (Single, Married, Divorced).
- MonthlyIncome : Gaji bulanan karyawan.
- MonthlyRate : Tarif bulanan karyawan.
- NumCompaniesWorked : Jumlah perusahaan tempat karyawan pernah bekerja sebelumnya.
- Over18 : Status usia di atas 18 tahun (selalu Y dalam dataset).
- OverTime : Apakah karyawan sering lembur (Yes/No).
- PercentSalaryHike : Persentase kenaikan gaji tahunan terakhir.
- PerformanceRating : Penilaian kinerja terakhir: 1 = Low, 2 = Good, 3 = Excellent, 4 = Outstanding.
- RelationshipSatisfaction : Tingkat kepuasan terhadap hubungan kerja: 1 = Low, 2 = Medium, 3 = High, 4 = Very High.
- StandardHours : Jam kerja standar (selalu 80 dalam dataset).
- StockOptionLevel : Level kepemilikan saham perusahaan.
- TotalWorkingYears : Total tahun pengalaman kerja.
- TrainingTimesLastYear : Jumlah pelatihan yang diikuti dalam setahun terakhir.
- WorkLifeBalance : Tingkat keseimbangan kerja–hidup: 1 = Bad, 2 = Good, 3 = Better, 4 = Best.
- YearsAtCompany : Total tahun bekerja di perusahaan saat ini.
- YearsInCurrentRole : Total tahun di posisi/jabatan saat ini.
- YearsSinceLastPromotion : Tahun sejak promosi terakhir.
- YearsWithCurrManager : Tahun bekerja dengan manajer saat ini.
- Attrition : Target: apakah karyawan keluar dari perusahaan (1 = Yes/ 0 = No).

---

## Data Preparation  
Langkah‐langkah yang dilakukan:  
1. Menghapus atau mengisi missing values.  
2. Encoding variabel kategorik (misal JobRole, Gender, OverTime) ke numerik atau one‐hot.  
3. Normalisasi atau standardisasi fitur numerik (misal MonthlyIncome, YearsAtCompany).  
4. Menangani class imbalance jika target “Attrition: Yes” sangat sedikit (contoh: oversampling, undersampling).  
5. Split data ke train & test (misal 80% train, 20% test) atau menggunakan cross‐validation.  

---

## Modelling  
Model yang digunakan:   
- LightGBM  
 
Menggunnakan estimasi model terbaik yaitu LightGBM dengan metrik akurasi / AUC / F1‐score terbaik.  

---

## Evaluation  
- Akurasi: 0.8814 
- Precision : 0.7083
- Recall : 0.4474
- F1 Score : 0.5484
- AUC : 0.7844  

---

## Kesimpulan
Notebook ini membangun pipeline end-to-end mulai dari preprocessing, pemilihan model, tuning hyperparameter, evaluasi performa, hingga penyimpanan model akhir dalam format .pkl. Seluruh tahapan dilakukan secara terstruktur menggunakan scikit-learn pipeline dan cross-validation untuk memastikan hasil yang konsisten dan reproducible.

Model LightGBM (LGBMClassifier) yang dikembangkan dengan preprocessing dan feature engineering berhasil memberikan performa yang cukup baik dalam memprediksi kemungkinan attrition atau keluarnya karyawan dari perusahaan. Berdasarkan hasil evaluasi, model memperoleh akurasi sebesar 0.8814, yang menunjukkan kemampuan model untuk mengklasifikasikan sebagian besar data dengan benar. Nilai precision sebesar 0.7083 mengindikasikan bahwa dari seluruh prediksi karyawan yang diperkirakan akan keluar, sekitar 70% di antaranya benar-benar keluar. Sementara itu, recall sebesar 0.4474 menunjukkan bahwa model masih melewatkan sebagian kasus karyawan yang benar-benar keluar, namun tetap memberikan keseimbangan yang cukup baik dengan nilai F1-score sebesar 0.5484. Nilai AUC sebesar 0.7844 menandakan bahwa model memiliki kemampuan yang baik dalam membedakan antara karyawan yang keluar dan yang bertahan.

Secara keseluruhan, hasil ini menunjukkan bahwa faktor-faktor seperti masa kerja karyawan (YearsAtCompany), frekuensi promosi (YearsSinceLastPromotion), pendapatan bulanan (MonthlyIncome), dan lembur (OverTime) berperan penting dalam menentukan risiko attrition. Model yang dihasilkan dapat digunakan oleh perusahaan sebagai alat bantu dalam pengambilan keputusan untuk mendeteksi lebih awal karyawan yang berpotensi keluar. Dengan demikian, langkah-langkah preventif seperti peningkatan kepuasan kerja, pemberian promosi tepat waktu, serta manajemen beban kerja dapat dilakukan untuk mengurangi tingkat turnover karyawan.
