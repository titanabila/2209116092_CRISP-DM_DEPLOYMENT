import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from sklearn.preprocessing import StandardScaler

st.title ("IKEA Product Analysis")

url = 'Data_Cleaned.csv'
df = pd.read_csv(url)

st.subheader("Dataset")
st.write(df.head())

st.subheader("Histogram Harga")
plt.figure(figsize=(8, 6))
sns.histplot(df['price'], kde=True)
plt.title('Histogram Harga')
plt.xlabel('Harga')
plt.ylabel('Frekuensi')
price_hist = plt.gcf()  # Get current figure

# Display the histogram plot with variable name
st.pyplot(price_hist)

st.write('Mayoritas produk IKEA memiliki harga yang terjangkau, dengan sekitar 22.71% produk berada dalam kisaran harga terendah. Terdapat penurunan signifikan dalam frekuensi produk di kisaran harga menengah, yang mungkin menandakan adanya strategi penetapan harga yang lebih kompetitif. Meskipun demikian, masih ada sebagian kecil produk dengan harga tinggi, mungkin sebagai pilihan premium atau memiliki fitur khusus. IKEA memiliki strategi penetapan harga yang beragam untuk menjangkau berbagai segmen pasar. Untuk meningkatkan penjualan, IKEA dapat memantau pola pembelian dan tren harga produk untuk mengoptimalkan strategi penetapan harga.')

st.subheader("Barang yang dijual secara online")
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='sellable_online')
plt.title('Distribusi Barang yang dijual secara online')
plt.xlabel('Barang yang dijual secara online')
plt.ylabel('Jumlah')
sellable_online_plot = plt.gcf()  # Get current figure

# Display the sellable online plot with variable name
st.pyplot(sellable_online_plot)

# Calculate percentage of sellable online
sellable_online_percentage = df['sellable_online'].value_counts(normalize=True) * 100

# Display the sellable online percentage
for category, percentage in sellable_online_percentage.items():
        st.write(f"{category}: {percentage:.2f}%")

st.write('Mayoritas produk IKEA, sekitar 99.24%, tersedia untuk pembelian secara online, menandakan fokus yang kuat pada penjualan melalui platform digital. Meskipun demikian, sekitar 0.76% produk tidak tersedia untuk pembelian online, mungkin karena beberapa batasan logistik atau karakteristik produk tertentu. IKEA memiliki kesempatan untuk memperluas kehadiran pasar online dengan memperhatikan ketersediaan produk, meningkatkan pengalaman belanja online, dan menyelaraskan strategi penjualan dengan preferensi pelanggan. Dengan langkah-langkah ini, IKEA dapat memperluas jangkauan produk mereka dan meningkatkan keberadaan di pasar digital.')


st.subheader("Distribusi Tinggi Barang IKEA")
plt.figure(figsize=(8, 6))
sns.histplot(df['height'], kde=True)
plt.title('Distribusi Tinggi Barang IKEA')
plt.xlabel('Tinggi')
plt.ylabel('Frekuensi')
height_hist = plt.gcf()  # Get current figure
st.pyplot(height_hist)

st.write("Mayoritas produk IKEA memiliki tinggi yang relatif seragam, dengan tinggi terbanyak berada di sekitar 29 - 30 cm. Hal ini menunjukkan konsistensi dalam desain produk IKEA dalam hal ukuran tinggi.Lonjakan signifikan dalam tinggi produk pada kisaran 21 - 22 cm dan 57 - 58 cm mungkin disebabkan oleh variasi dalam jenis produk yang ditawarkan oleh IKEA, seperti rak buku yang lebih tinggi atau meja yang lebih rendah.Dikarenakan frekuensi produk dengan tinggi di bawah 20 cm atau di atas 60 cm relatif rendah, mungkin ada peluang bagi IKEA untuk memperluas pilihan produk dalam kisaran tinggi tertentu untuk memenuhi kebutuhan pelanggan yang lebih spesifik.")
st.write('Melakukan analisis lebih lanjut untuk memahami alasan di balik lonjakan signifikan dalam tinggi produk pada kisaran tertentu, dan memastikan bahwa variasi produk tersebut masih sesuai dengan preferensi dan kebutuhan pelanggan.Mengidentifikasi peluang untuk mengembangkan atau menyempurnakan produk dalam kategori tinggi tertentu yang saat ini memiliki frekuensi yang relatif rendah, untuk menarik pelanggan yang memiliki preferensi khusus terkait tinggi produk. Menawarkan opsi kustomisasi atau personalisasi untuk produk-produk tertentu dalam hal tinggi, untuk meningkatkan relevansi produk dengan preferensi individu pelanggan.')
# EDA for 'width' column
st.subheader("Distribusi Lebar Barang IKEA")
plt.figure(figsize=(8, 6))
sns.histplot(df['width'], kde=True)
plt.title('Distribusi Lebar Barang IKEA')
plt.xlabel('Lebar')
plt.ylabel('Frekuensi')
width_hist = plt.gcf()  # Get current figure
st.pyplot(width_hist)

st.write("Mayoritas produk IKEA memiliki lebar yang relatif seragam, dengan lebar terbanyak berada di sekitar 7 - 8 cm. Hal ini menunjukkan konsistensi dalam desain produk IKEA dalam hal ukuran lebar. Lonjakan signifikan dalam lebar produk pada kisaran 3 - 4 cm dan 8 - 9 cm mungkin disebabkan oleh variasi dalam jenis produk yang ditawarkan oleh IKEA, seperti rak buku yang lebih ramping atau lemari yang lebih lebar. Dikarenakan frekuensi produk dengan lebar di atas 10 cm relatif rendah, mungkin ada peluang bagi IKEA untuk memperluas pilihan produk dalam kategori lebar tertentu untuk memenuhi kebutuhan pelanggan yang lebih spesifik.")
st.write("Melakukan analisis lebih lanjut untuk memahami alasan di balik lonjakan signifikan dalam lebar produk pada kisaran tertentu, dan memastikan bahwa variasi produk tersebut masih sesuai dengan preferensi dan kebutuhan pelanggan. Mengidentifikasi peluang untuk mengembangkan atau menyempurnakan produk dalam kategori lebar tertentu yang saat ini memiliki frekuensi yang relatif rendah, untuk menarik pelanggan yang memiliki preferensi khusus terkait lebar produk. Menawarkan opsi kustomisasi atau personalisasi untuk produk-produk tertentu dalam hal lebar, untuk meningkatkan relevansi produk dengan preferensi individu pelanggan.")


st.subheader("Presentase Kategori Barang IKEA")
categorical_cols = ['category_Bar furniture', 'category_Beds', 'category_Bookcases & shelving units',
                    'category_Cabinets & cupboards', 'category_Café furniture', 'category_Chairs',
                    'category_Chests of drawers & drawer units', 'category_Children\'s furniture',
                    'category_Nursery furniture', 'category_Outdoor furniture', 'category_Room dividers',
                    'category_Sideboards, buffets & console tables', 'category_Sofas & armchairs',
                    'category_TV & media furniture', 'category_Tables & desks', 'category_Trolleys',
                    'category_Wardrobes']

# Calculate percentage of each category
category_percentages = {}
for col in categorical_cols:
    total_count = df[col].sum()  # Jumlah total produk dalam kategori
    category_percentages[col] = (total_count / len(df)) * 100  # Hitung persentase

# Plot the percentages
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(list(category_percentages.keys()), list(category_percentages.values()), color='skyblue')
ax.set_xlabel('Persentase (%)')
ax.set_ylabel('Kategori')
ax.set_title('Persentase Produk IKEA berdasarkan Kategori')
st.pyplot(fig)

st.write("Dari hasil analisis, terlihat bahwa kategori Tables & desks memiliki persentase tertinggi dalam produk IKEA, mencapai 16.57%. Ini menunjukkan adanya minat yang kuat terhadap produk meja dan meja kerja dari pelanggan IKEA. Sementara itu, kategori Bookcases & shelving units menduduki peringkat kedua dengan persentase 14.83%, menandakan permintaan yang signifikan terhadap rak buku dan unit rak. Selain itu, produk duduk seperti kursi dan sofa juga sangat diminati, dengan kategori Chairs dan Sofas & armchairs masing-masing mencapai sekitar 13.02% dan 11.59%.")
st.write("Melihat dari persentase tinggi pada kategori-kategori tertentu, strategi pemasaran dan penjualan bisa difokuskan pada produk-produk unggulan dalam kategori tersebut. Misalnya, dengan menekankan promosi lebih lanjut untuk produk meja dan meja kerja dalam kategori Tables & desks, atau memperluas variasi dan penawaran pada rak buku dan unit rak dalam kategori Bookcases & shelving units. Selain itu, penting juga untuk terus memantau tren pasar dan mengidentifikasi peluang baru untuk memperluas portofolio produk IKEA agar tetap relevan dengan kebutuhan dan preferensi pelanggan yang terus berubah.")

st.subheader("Agglomerative Clustering")
def load_data(file_path):
    return pd.read_csv(file_path)

# Perform Agglomerative Clustering
def perform_clustering(data, n_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    pca = PCA(n_components=2)  # Reduksi dimensi menjadi 2
    reduced_data = pca.fit_transform(scaled_data)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = model.fit_predict(reduced_data)

    return clusters, reduced_data

# Sidebar for selecting number of clusters
st.sidebar.header('Select Number of Clusters')
n_clusters = st.sidebar.slider('Jumlah Cluster', min_value=2, max_value=10, value=2, step=1)

# Load data
file_path = 'Data_Cleaned.csv'
data = load_data(file_path)

# Perform Agglomerative Clustering
clusters, reduced_data = perform_clustering(data, n_clusters)

# Add cluster information to the DataFrame
data['Cluster'] = clusters

# Display the clustering results
st.write(f"Hasil Clusters dengan jumlah {n_clusters} Clusters:")
st.write(data)

# Visualize clustered data points
plt.figure(figsize=(10, 8))
for cluster_num in range(n_clusters):
    cluster_data = reduced_data[data['Cluster'] == cluster_num]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_num}')

plt.title('Visualisasi Kluster')
plt.legend()

st.pyplot(plt.gcf())