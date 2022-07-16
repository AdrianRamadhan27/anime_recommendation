# Laporan Proyek Machine Learning - Raden Mohamad Adrian Ramadhan Hendar Wibawa
# Sistem Rekomendasi Anime berdasarkan data MyAnimeList.net
## Project Overview
Kartun Jepang atau yang biasanya dikenal dengan anime kini menjadi lebih wajar atau *mainstream* di kalangan umum. Terlebih lagi dengan pandemi COVID-19 yang belum lama terjadi, masyarakat yang terjebak di dalam rumah mulai mengeksplorasi animasi adaptasi manga buatan Jepang ini. Menurut [(Kettle & Brandon, 2022)](https://carolinanewsandreporter.cic.sc.edu/anime-makes-move-to-mainstream/#:~:text=Interest%20in%20anime%20programs%20is,doubled%20in%20the%20same%20period), interest akan program anime meningkat 33% di Amerika Serikat pada tahun 2020. Konglomerat raksasa yaitu Disney, bahkan mulai terjun ke dunia anime [(Yeung, 2022)](https://hypebeast.com/2022/4/disney-plus-anime-content-expansion-plans). Oleh karena itu, ada *demand* yang cukup besar untuk layanan streaming anime dan tentunya layanan streaming tersebut membutuhkan sistem rekomendasi yang lihai dalam membuat rekomendasi pada pengguna. 


## Business Understanding

### Problem Statements

- Berdasarkan data mengenai anime, bagaimana membuat sistem rekomendasi yang dipersonalisasi dengan teknik content-based filtering?
- Dengan data rating yang Anda miliki, bagaimana perusahaan dapat merekomendasikan anime lain yang mungkin disukai dan belum pernah ditonton oleh pengguna? 

### Goal
- Menghasilkan sejumlah rekomendasi anime yang dipersonalisasi untuk pengguna dengan teknik content-based filtering.
- Menghasilkan sejumlah rekomendasi anime yang sesuai dengan preferensi pengguna dan belum pernah ditonton sebelumnya dengan teknik collaborative filtering.


### Solution statements
- Membuat model content-based filtering yang menghitung kesamaan anime dengan cosine similarity atas genre
- Membuat model collaboraitve filtering yang mengelompokkan/mengembed anime berdasarkan rating dari pengguna

## Data Understanding
Data yang digunakan pada proyek ini berasal dari database [MyAnimeList](https://myanimelist.net/topanime.php) yang telah di web-scraping dan di upload ke Kaggle: [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) pada tahun 2020. Lebih spesifiknya, yang saya gunakan adalah file **anime.csv** dan **rating_complete.csv**. File anime.csv berisi list anime beserta identitasnya sedangkan rating_complete.csv berisi penilaian dari user atas anime yang telah di-complete (tonton sampai selesai). Jumlah anime pada anime.csv adalah 17.562 dan jumlah rating yang diberikan di rating_complete.csv adalah 57.633.278. \
Karena RAM/memori dari Google Colab terbatas, sampel data rating yang akan diambil hanyalah 50.000 dengan kode `rating = rating.sample(n=50000, random_state=27)`.
\
Cuplikan dari anime.csv.
![anime.csv](https://drive.google.com/uc?export=view&id=1o90IpCTIoRfCSukdjTh7N199jhuEAqIJ)
Variabel-variabel pada anime.csv  adalah sebagai berikut:
- MAL_ID: id anime pada MyAnimeList
- Name: judul dari anime
- Score: rata-rata skor yang telah diberikan pengguna
- Genres: kategori/genre dari anime (2 atau lebih dipisahkan koma)
- English name: judul anime dalam bahasa inggris
- Japanese name: judul anime dalam bahasa jepang
- Type: jenis rilis (Serial TV, Film, dsb)
- Episodes: jumlah episode yang dirilis
- Aired: linimasa rilis
- Premiered: musim rilis
- Producers: produser dari anime
- Studios: studio pembuat anime
- Source: sumber cerita adaptasi anime
- Duration: durasi anime per episode/total
- Rating: rating umur film/serial tv
- Ranked: peringkat berdasarkan score
- Popularity: peringkat berdasarkan jumlah penonton
- Members: jumlah pengguna yang telah menambahkan anime
- Favorites: jumlah pengguna yang memfavoritkan anime
- Watching: jumlah pengguna yang sedang menonton anime
- Completed: jumlah pengguna yang telah menamatkan anime
- On-Hold: jumlah pengguna yang sedang berhenti sejenak menonton anime
- Dropped: jumlah pengguna yang telah berhenti menonton anime
- Plan to Watch: jumlah pengguna yang berencana menonton anime
- Score-10: jumlah pengguna yang memberi nilai 10
- Score-9: jumlah pengguna yang memberi nilai 9
- Score-8: jumlah pengguna yang memberi nilai 8
- Score-7: jumlah pengguna yang memberi nilai 7
- Score-6: jumlah pengguna yang memberi nilai 6
- Score-5: jumlah pengguna yang memberi nilai 5
- Score-4: jumlah pengguna yang memberi nilai 4
- Score-3: jumlah pengguna yang memberi nilai 3
- Score-2: jumlah pengguna yang memberi nilai 2
- Score-1: jumlah pengguna yang memberi nilai 1

Cuplikan dari rating_complete.csv. \
![rating_complete.csv](https://drive.google.com/uc?export=view&id=1grZnXsX648rvY_nYL7Oi_hIxRbFAfUkt)
\
Variabel-variabel pada rating_complete.csv adalah sebagai berikut:
- user_id: id pengguna pada MyAnimeList
- anime_id: sama dengan MAL_ID, id anime pada MyAnimeList
- rating: penilaian yang diberikan pengguna

### Exploratory Data Analysis
Jumlah data pada rating_complete.csv 
\
![Rating](https://drive.google.com/uc?export=view&id=12KEvo_FS0jqQYwcFHLh7VkZYIAEJPGg_)
\
\
Distribusi genre pada data anime \
![Genre](https://drive.google.com/uc?export=view&id=1cBLF33pyUTq3fRqtXWDeRqUVtXErCgyN)
Terlihat bahwa masih ada beberapa data yang genre nya unknown. 
\
\
Deskripsi statistik rating \
![Stat Rating](https://drive.google.com/uc?export=view&id=1szfzx0ONe5sC--ThfkjsLj0bmRbTxLds)
\
\
Distribusi rating \
![Rating](https://drive.google.com/uc?export=view&id=1EGXY-XClghAffIn5SKYL8QC5FKZUORWR)

## Data Preparation
Pada data preparation, kita perlu menggabungkan data `rating` dengan nama dan genre anime berdasarkan anime_id pada rating menjadi 1 darframe `anime_rating` dengan kode berikut. 
```
anime_rating = pd.merge(rating, animes[['anime_id', 'Name', 'Genres']], on='anime_id', how='left')
```
Hasil penggabungan sebagai berikut.
![anime rating](https://drive.google.com/uc?export=view&id=1hyUS70NuOY64b2Hc0SSNQm_fjbkjDk29)
\
Setelah proses penggabungan, mari kita cek lagi datanya apakah ada missing value atau tidak. Output dari `anime_rating.isnull().sum()`: 
\
![isnull](https://drive.google.com/uc?export=view&id=1qeIDW6v0ml43SRxGz0hbZBlBCBLGgR4e)
\
Namun, jika kita mengingat kembali pada EDA, terdapat sebagian data anime yang genrenya bernilai *Unknown*. Jika kita panggil `anime_rating[anime_rating['Genres']=='Unknown']` output dari kode tersebut adalah: 
\
![isunknown](https://drive.google.com/uc?export=view&id=1lHA4Z73kx_8idOxk61Brh8tzPw9W9bDw)
\
Kita bisa menghapus data dengan genre unknown tersebut dengan kode berikut.
`anime_rating = anime_rating[anime_rating['Genres']!='Unknown']`
\
Selanjutnya kita akan mengurutkan data anime_rating berdasarkan `anime_id` agar nantinya ketika dipisahkan dan dijadikan sebuah dictionary semua data sudah terurut. Hasil pengurutannya sebagai berikut.
![sort_values](https://drive.google.com/uc?export=view&id=1QhVkm1brKu-YgkaGXiq7ZY-kDcnf5TYy)
\
Kita perlu membuat variabel preparation yang memprioritaskan jumlah anime ketimbang rating dan user sehingga kita perlu menghapus semua duplikasi dari anime_id dengan kode berikut.
```
preparation = anime_rating
preparation = preparation.drop_duplicates('placeID')
```
Hasil dari variabel preparation adalah sebagai berikut.
![preparation](https://drive.google.com/uc?export=view&id=1IjzxrD1h8Mya_eCgA7RTaMxOoHxk2jrd)
\
Lalu, kita perlu memisahkan fitur  `anime_id`, `Name`, dan `Genres` menjadi list tersendiri dengan kode berikut. 
```
anime_id = preparation['anime_id'].tolist()
anime_name = preparation['Name'].tolist()
anime_genres = preparation['Genres'].tolist()
```
Dari list-list tersebut kita akan membuat Dataframe dictionary yang memasangkan label kepada value list tersebut.
```
anime_new = pd.DataFrame({
    'id': anime_id,
    'anime_name': anime_name,
    'genres': anime_genres
})
```
Tampilan dictionary `anime_new`.
![anime_new](https://drive.google.com/uc?export=view&id=1wI9NaV05edn_M3jQG8x1IBWIoOvu-EQS)

## Modeling
Kita akan membuat 2 model pada proyek ini, yaitu content-based filtering dan collaborative filtering.
### Content-Based Filtering
Pertama-tama, kita perlu mengassign dictionary anime_new ke variabel data.
Cukup dengan `data = anime_new`.
Untuk algoritma content-based filtering ini, kita akan menggunakan TF-IDF(term frequency-inverse document frequency). TF-IDF dihitung dari perkalian frekuensi sebuah kata dalam satu dokumen dan invers frekuensi kata dalam seluruh dokumen. Kita akan menggunakan library scikit-learn untuk membuat vectorizer TF-IDF.
```
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(token_pattern=r'[^,\s][^\,]*[^,\s]*')
tfidf.fit(data['genres']) 
```
Parameter `token_pattern` diatas kita isi dengan regex `[^,\s][^\,]*[^,\s]*` yang akan memisahkan tiap kata dengan pemisah koma dan spasi setelahnya. Untuk lebih lanjutnya tentang regex dapat diselidiki di tautan  [berikut](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions). TF-IDF akan dihitung berdasarkan fitur genres. Hasil pemanggilan `tfidf.get_feature_names()` adalah berikut.
\
![tfidf feature](https://drive.google.com/uc?export=view&id=1Q0f6QykS0-4Vwe-ToIAEoQ6YLz9clOOc)
\
Setelah itu kita akan membuat matrix berdasarkan genre tersebut.
`tfidf_matrix = tfidf.fit_transform(data['genres'])`
Ukuran matrix ini, atau `tfidf_matrix.shape` akan mengembalikan nilai (6410, 43). Angka 6410 melambangkan jumlah anime dan 43 melambangkan jumlah genre.
\
Setelah itu, kita akan membuat dataframe untuk melihat tf-idf matrix. Kolom diisi dengan genre anime. Baris diisi dengan nama anime.
\
Kita bisa melihat cuplikan sampel data frame ini dengan kode berikut.
```
pd.DataFrame(
    tfidf_matrix.todense(), 
    columns=tfidf.get_feature_names(),
    index=data.anime_name
).sample(10, axis=1).sample(10, axis=0)
```
Yang akan mengeluarkan
![anime_genre_df](https://drive.google.com/uc?export=view&id=1mE_ymF9Hk58NPDim6ULg3ICwah5Oxnil)
Tabel ini menunjukkan seberapa cocok genre anime dengan sebuah anime. Angka 0 menunjukkan ketidakcocokan dan semakin mendekati 1 artinya anime itu mengandung genre tersebut.
\
Setelah mengukur kecocokan antara anime dan genre, sekarang kita akan mencoba mengukur kesamaan antar anime berdasarkan genre yang dikandungnya dengan **cosine similarity**. Cosine similarity pada Python menghitung kesamaan sebagai dot product yang dinormalisasi dari masukan sampel X dan Y. Kita akan menggunakan sklearn cosine_similarity untuk mendapatkan nilai cosinus dua vektor dalam matriks.
```
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(tfidf_matrix) 
```
Kita bisa membuat dataframe lagi yang menghubungkan baris anime dengan kolom anime lainnya. Yaitu dengan `cosine_sim_df = pd.DataFrame(cosine_sim, index=data['anime_name'], columns=data['anime_name'])`. Ukuran dari dataframe ini adalah 6410 x 6410 yaitu perkalian antara tiap anime. Jika kita ingin melihat cuplikan sampel dengan memanggil `cosine_sim_df.sample(5, axis=1).sample(10, axis=0)` akan mengeluarkan
![anime_anime_cosine](https://drive.google.com/uc?export=view&id=1Oy14EH15B_W9aqrYjKdWrCIM_KMfcTKx)
Hampir sama dengan sebelumnya, semakin mendekati 1 artinya kedua anime memiliki genre yang serupa.
\
Terakhir, kita akan membuat function yang mengembalikan top-k recommendation dengan kode berikut.
```
def anime_recommendations(nama_anime, similarity_data=cosine_sim_df, items=data[['anime_name', 'genres']], k=5):
    index = similarity_data.loc[:,nama_anime].to_numpy().argpartition(
        range(-1, -k, -1))
    closest = similarity_data.columns[index[-1:-(k+2):-1]]
    closest = closest.drop(nama_anime, errors='ignore')
 
    return pd.DataFrame(closest).merge(items).head(k)
```
Function ini akan membuat partisi dan mengambil k anime dengan nilai cosine similarity tertinggi dengan anime dari argumen. Karena nomor 1 tertinggi sudah pasti merupakan anime itu sendiri, kita perlu meng-drop anime dari argumen.
\
Sebagai percobaan, mari kita gunakan function ini untuk mencari rekomendasi anime yang serupa dengan *Naruto*. Jika kita mencoba mencari anime naruto pada dataframe, berikut merupakan keluaran genre yang dicantumkan.
\
![naruto](https://drive.google.com/uc?export=view&id=1ZdVuyafnwy1yRBa92uTWHV3fjEy1xUFg)
\
Lalu jika kita memanggil function `anime_recommendations('Naruto')` akan keluar output berikut.
![naruto_recommendation](https://drive.google.com/uc?export=view&id=1khzJVCSf5Flu3JO8TL5MqxAGRXPcw9dn)
Dapat dilihat bahwa hasil rekomendasi anime Naruto cukup akurat, mengingat sekuel ataupun spin-off dari anime Naruto pastinya akan memiliki genre yang sama dengan Naruto. 

### Collaborative Filtering
Untuk collaborative filtering ini, kita akan lebih menekankan kepada data rating dari pengguna terhadap anime yang telah ditonton. Kita perlu mendefinisikan variabel df sebagai assignment dari data rating diawal.
`df = rating`
Setelah itu, kita perlu encode id user dan id anime ke index integer. Pertama kita perlu membuat list user_id unik dari df lalu dengan list comprehension kita membuat dictionary `user_to_user_encoded` yang memiliki key digit enumerate dan id sebagai value. Kita lakukan juga sebaliknya untuk `user_encoded_to_user`.
```
user_ids = df['user_id'].unique().tolist()
user_to_anime_encoded = {x: i for i, x in enumerate(user_ids)}
user_encoded_to_user = {i: x for i, x in enumerate(user_ids)}
```
Potongan hasil encoding nya dapat dilihat pada gambar berikut.
![encoding](https://drive.google.com/uc?export=view&id=14YBbywXMnTL-vPcI8j5tagG26HD-29z7)

Lakukan hal yang sama untuk anime_id
```
anime_ids = df['anime_id'].unique().tolist()
anime_to_anime_encoded = {x: i for i, x in enumerate(anime_ids)}
anime_encoded_to_anime = {i: x for i, x in enumerate(anime_ids)}
```
Selanjutnya, kita akan menambahkan kolom baru pada `df` berdasarkan encoding ini.
```
df['user'] = df['userID'].map(user_to_user_encoded)
df['resto'] = df['placeID'].map(resto_to_resto_encoded)
```
Agar lebih mudah diproses, kita bisa rubah tipe data rating pada dataframe menjadi float.
`df['rating'] = df['rating'].values.astype(np.float32)`.
\
Lalu, kita perlu mendefinisikan beberapa variabel. Variabel tersebut diantaranya adalah jumlah user, jumlah anime, rating minimum, dan rating maximum.
```
num_users = len(user_to_user_encoded)
num_resto = len(resto_encoded_to_resto)
min_rating = min(df['rating'])
max_rating = max(df['rating'])
```
Hasilnya adalah sebagai berikut.
![variables](https://drive.google.com/uc?export=view&id=1GwJ8HQWHmnpXSR_6axVFlPZGcTGaPjyg)
\
Setelah itu, kita bisa memulai membagikan antara fitur (user_id, anime_id) dengan label (rating). Rating akan dinormalisasi dengan metode MinMax.
```
x = df[['user', 'anime']].values
y = df['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
```
Untuk pembagian antara data train dan data test, kita bisa mengacak data terlebih dahulu dengan menggunakan `pd.Dataframe.sample(frac=1)` setelah itu kita bagi dengan test size 20%.
```
df = df.sample(frac=1, random_state=42)
train_indices = int(0.8 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```
Akhirnya kita bisa mulai membuat model dengan mengadaptasi RecommenderNet dari tautan [berikut](https://keras.io/examples/structured_data/collaborative_filtering_movielens/). Lalu, kita bisa meng-compile model dengan parameter berikut.
```
model = RecommenderNet(num_users, num_anime, 50)
 
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = keras.optimizers.Adam(learning_rate=0.001),
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
```
Hasil pelatihan 100 epochs model dapat dilihat dalam histogram berikut.
![hist](https://drive.google.com/uc?export=view&id=1xBpB14lAwX1-4iCgRQM7TiV2qU3zMTG2)
\
Setelah dilatih kita bisa mencoba menampilkan top 10 recommendation dari sampel sebagai berikut.
![sample_recommend](https://drive.google.com/uc?export=view&id=1iEVo-iQqOKStz0KyN8q2WzZW-flAJg3O)
Dapat dilihat bahwa beberapa rekomendasi sudah berhasil menangkap genre yang serupa yaitu Comedy dan Parody terutama Gintama yang memiliki kedua genre dari anime yang sudah disukai pengguna. Padahal dalam pertimbangan model ini genre dari anime tidak diikutsertakan, namun pada nyatanya masih memberikan hasil yang cukup akurat.
## Evaluation
Metrik evaluasi yang digunakan pada proyek ini adalah RMSE (Root Mean Squared Error) yang mirip dengan Euclidean Distance. Formula penghitungan RMSE adalah sebagai berikut.
\
![RMSE](https://media.geeksforgeeks.org/wp-content/uploads/20200622171741/RMSE1.jpg)
\
https://www.geeksforgeeks.org
\
RMSE ini bekerja dengan menjumlahkan kuadrat selisih nilai prediksi dengan nilai asli dan merata-ratakannya, setelah itu diakarkan. Dengan itu, semakin kecil nilai RMSE artinya semakin kecil error atau kesalahan dari suatu model.
\
Hasil RMSE terakhir dari training model adalah `0.13`untuk data train dan `0.2` untuk data test/validation. Hasil ini sudah cukup baik dalam memberikan rekomendasi yang sesuai dengan preferensi pengguna sesuai dengan tujuan proyek. 

## Conclusion
Pada proses evaluasi, kita telah mendapatkan hasil metriks evaluasi yang cukup baik. Namun, karena data rating yang digunakan tidak lengkap demi kinerja Google Colab, hasil ini masih belum optimal. Tak hanya itu, untuk content-based filtering sebenarnya kita juga bisa mempertimbangkan fitur lain pada anime yang dapat membantu lebih baik mengukur kesamaan anime seperti sinopsisnya, studio pembuatnya, dan bahkan mangaka aslinya. Oleh karena itu, model dari proyek ini masih bisa dikembangkan lagi dengan resource yang lebih besar untuk membuat sistem rekomendasi anime yang baik.

## References
- Aakash, M. (2022, July 22). *Root-Mean-Square Error in R Programming*. GeeksForGeeks. https://www.geeksforgeeks.org/root-mean-square-error-in-r-programming/
- Banerjee, S. (2020, May 24). *Collaborative Filtering for Movie Recommendations*. Keras. https://keras.io/examples/structured_data/collaborative_filtering_movielens/
- Kettle, K. & Brandon A. (2022, February 9). *Anime makes move to mainstream*. Carolina News & Reporter. https://carolinanewsandreporter.cic.sc.edu/anime-makes-move-to-mainstream/#:~:text=Interest%20in%20anime%20programs%20is,doubled%20in%20the%20same%20period.
- Valdivieso H. (2020). *Anime Recommendation Database 2020* (Version 7) [Dataset]. Kaggle. https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020
- Yeung, Jeff. (2022, April 1). *Disney+ Is Looking to Enter the Anime Market*. Hypebeast. https://hypebeast.com/2022/4/disney-plus-anime-content-expansion-plans

