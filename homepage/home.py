import streamlit as st

st.set_page_config(layout="wide")
st.title("Football Analytics Dashboard")



# =========================
# Clustering navigation
# =========================
CLUSTER_METHODS = {
    "GMM": {
        "image": "Picture/gmm.png",
        "description": """
        Hasil klasterisasi mengidentifikasi tiga kelompok performa tim yang berbeda pada musim 2024/2025. Klaster 1 (elips oranye) merepresentasikan tim-tim dengan performa terbaik yang memiliki tingkat kemenangan tinggi secara konsisten serta statistik serangan dan pertahanan yang sangat kuat. Tim seperti Bayern Munich, Barcelona, dan Paris Saint-Germain muncul sebagai outlier karena performa mereka yang sangat dominan, sehingga berada jauh di atas standar liga. Oleh karena itu, klaster ini dapat didefinisikan sebagai Top 6 atau Tim Elite.

        Klaster 2 (elips abu-abu) terdiri dari tim-tim papan tengah (mid-table) dengan performa yang berada di sekitar rata-rata liga, ditunjukkan oleh titik data yang terkonsentrasi di dekat pusat grafik. Tim seperti Brentford, Freiburg, dan Werder Bremen mewakili karakteristik klaster ini. Beberapa tim berada di area peralihan antar klaster, yang menunjukkan adanya tumpang tindih probabilistik berdasarkan model GMM–PCA, sehingga mengindikasikan karakteristik performa yang berada di antara dua klaster.

        Klaster 0 (elips merah) mencakup tim-tim kandidat degradasi atau tim dengan performa terendah, yang ditandai oleh variabilitas performa yang tinggi dan hasil pertandingan yang buruk secara keseluruhan. Tim seperti Montpellier, Southampton, dan Valladolid teridentifikasi sebagai outlier karena kinerja mereka yang jauh lebih lemah dibandingkan tim lain. Penyebaran klaster yang lebih luas menunjukkan ketidakstabilan performa di antara tim-tim berkinerja rendah, sehingga klaster ini dapat diinterpretasikan sebagai Relegation Candidate pada musim 2024/2025.
        """,
        "pages": {
            "Cluster 1": "clusterpage/cluster1_gmm.py",
            "Cluster 2": "clusterpage/cluster2_gmm.py",
            "Cluster 3": "clusterpage/cluster3_gmm.py",
        }
    },
    "Hierarchical": {
        "image": "Picture/hier.png",
        "description": """
        Hasil klasterisasi menggunakan Hierarchical Clustering dengan PCA membagi tim menjadi tiga kelompok performa pada musim 2024/2025. Klaster elit (klaster nol/merah) mencakup tim-tim dengan performa terbaik dan tingkat kemenangan tinggi, seperti Paris Saint-Germain, Barcelona, Real Madrid, Bayern Munich, dan Marseille, yang muncul sebagai outlier karena dominasi statistik mereka. Tim seperti Juventus, Arsenal, dan Athletic Club juga termasuk dalam klaster ini meskipun memiliki jumlah hasil seri yang relatif tinggi. Klaster ini dapat didefinisikan sebagai Top Performing Teams.

        Klaster tengah (klaster dua/abu-abu) berisi tim mid-table dengan performa rata-rata liga, ditunjukkan oleh titik data yang mengelompok di sekitar pusat grafik. Namun, terdapat banyak tim yang berada di area peralihan antar klaster, menandakan bahwa model mengidentifikasi karakteristik performa yang tumpang tindih antara tim elit dan tim berkinerja rendah.

        Klaster terbawah (klaster satu/oranye) terdiri dari tim kandidat degradasi dengan performa terendah dan variabilitas tinggi. Tim seperti Southampton dan Montpellier muncul sebagai outlier dengan kinerja paling buruk dan menempati posisi terbawah liga. Klaster ini secara jelas merepresentasikan Relegation Candidates pada musim 2024/2025.
        """,
        "pages": {
            "Cluster 1": "clusterpage/cluster1_hier.py",
            "Cluster 2": "clusterpage/cluster2_hier.py",
            "Cluster 3": "clusterpage/cluster3_hier.py",
        }
    },
    "K-Means": {
        "image": "Picture/kmeans.png",
        "description": """
        Hasil klasterisasi menggunakan K-Means dengan PCA membagi tim menjadi tiga kelompok performa pada musim 2024/2025. Klaster nol (elips merah) merepresentasikan tim-tim elit dengan performa terbaik dan konsistensi tinggi, seperti Bayern Munich, Paris Saint-Germain, Barcelona, Real Madrid, dan Marseille, yang muncul sebagai outlier karena dominasi statistik dalam aspek serangan dan pertahanan. Tim seperti Juventus, Arsenal, Napoli, Athletic Club, dan RB Leipzig juga termasuk dalam klaster ini, dengan tingkat kemenangan tinggi meskipun memiliki jumlah hasil seri yang relatif lebih besar. Klaster ini dapat didefinisikan sebagai Top Performing Teams.

        Klaster dua (elips abu-abu) terdiri dari tim papan tengah (mid-table) dengan performa rata-rata liga dan tingkat stabilitas menengah. Meskipun sebagian besar tim berada tepat di dalam elips, terdapat beberapa tim yang posisinya tumpang tindih dengan klaster lain, menunjukkan bahwa model mengidentifikasi karakteristik performa yang berada di antara dua kelompok. Hal ini mencerminkan volatilitas performa yang umum terjadi pada tim papan tengah.

        Klaster satu (elips oranye) mencakup tim kandidat degradasi dengan performa terendah pada musim 2024/2025. Tim seperti Southampton, Montpellier, dan Valladolid muncul sebagai outlier dengan statistik paling lemah dan menempati posisi terbawah liga masing-masing. Klaster ini secara jelas dapat diinterpretasikan sebagai Relegation Candidates, yaitu kelompok tim dengan risiko degradasi tertinggi pada musim tersebut.
        """,
        "pages": {
            "Cluster 1": "clusterpage/cluster1_kmeans.py",
            "Cluster 2": "clusterpage/cluster2_kmeans.py",
            "Cluster 3": "clusterpage/cluster3_kmeans.py",
        }
    }
}


# =========================
# Method selector
# =========================
method = st.selectbox(
    "Select clustering method",
    list(CLUSTER_METHODS.keys())
)

# Show method image
img_col1, img_col2, img_col3 = st.columns([1, 3, 1])

with img_col2:
    st.image(
        CLUSTER_METHODS[method]["image"],
        use_container_width=True
    )

    # Textbox / explanation below image
    st.markdown("#### Penjelasan Hasil Klasterisasi")
    st.info(CLUSTER_METHODS[method]["description"])


# =========================
# Cluster buttons (3 buttons)
# =========================
col1, col2, col3 = st.columns(3)

clusters = list(CLUSTER_METHODS[method]["pages"].items())

with col1:
    if st.button("Cluster 0", use_container_width=True):
        st.switch_page(clusters[0][1])

with col2:
    if st.button("Cluster 1", use_container_width=True):
        st.switch_page(clusters[1][1])

with col3:
    if st.button("Cluster 2", use_container_width=True):
        st.switch_page(clusters[2][1])