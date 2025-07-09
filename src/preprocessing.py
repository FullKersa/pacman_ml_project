import pandas as pd

from sklearn.preprocessing import  OneHotEncoder



def ohe_transform(dataset, subset, prefix, ohe):
    """
    Fungsi untuk melakukan encoding data kategorik
    
    Parameters:
    dataset (DataFrame): Set data yang ingin dilakukan pengkodean
    subset (str): Nama kolom yang terdapat pada data di parameter dataset
    prefix (str): Nama awalan yang akan disematkan pada kolom hasil pengkodean
    ohe (OneHotEncoder): Encoder yang sebelumnya telah dilatih oleh data kategorik khusus
    
    Returns:
    DataFrame: Dataset yang telah dilakukan pengkodean
    """
    
    # Percabangan pertama - Validasi parameter dataset
    if not isinstance(dataset, pd.DataFrame):
        raise RuntimeError("Fungsi ohe_transform: parameter dataset harus bertipe DataFrame!")
    
    # Percabangan kedua - Validasi parameter ohe
    if not isinstance(ohe, OneHotEncoder):
        raise RuntimeError("Fungsi ohe_transform: parameter ohe harus bertipe OneHotEncoder!")
    
    # Percabangan ketiga - Validasi parameter prefix
    if not isinstance(prefix, str):
        raise RuntimeError("Fungsi ohe_transform: parameter prefix harus bertipe str!")
    
    # Percabangan keempat - Validasi parameter subset
    if not isinstance(subset, str):
        raise RuntimeError("Fungsi ohe_transform: parameter subset harus bertipe str!")
    
    # Percabangan kelima - Pengecekan data pada parameter subset
    try:
        column_list = list(dataset.columns)
        column_list.index(subset)
    except ValueError:
        raise RuntimeError("Fungsi ohe_transform: parameter subset string namun data tidak ditemukan dalam daftar kolom yang terdapat pada parameter dataset.")
    
    print("Fungsi ohe_transform: parameter telah divalidasi.")
    
    # Buat duplikat dari data pada parameter dataset
    dataset = dataset.copy()
    
    # Print pesan yang menampilkan daftar nama kolom sebelum dilakukan pengkodean
    print(f"Fungsi ohe_transform: daftar nama kolom sebelum dilakukan pengkodean adalah {list(dataset.columns)}.")
    
    # Buat satu variabel bernama col_names untuk menyimpan nama kolom yang telah dikodekan
    col_names = [f"{prefix}_{col_name}" for col_name in ohe.categories_[0].tolist()]
    
    # Proses pengkodean:
    # Buat satu variabel bernama encoded
    transformed_data = ohe.transform(dataset[[subset]])
    
    # Cek apakah hasil transform adalah sparse matrix atau array
    if hasattr(transformed_data, 'toarray'):
        # Jika sparse matrix, convert ke array
        array_data = transformed_data.toarray()
    else:
        # Jika sudah array, gunakan langsung
        array_data = transformed_data
    
    encoded = pd.DataFrame(
        data=array_data,
        columns=col_names,
        index=dataset.index
    )
    
    # Proses penyatuan hasil pengkodean dengan data sebelum pengkodean
    dataset = pd.concat([dataset, encoded], axis=1)
    
    # Proses penghapusan kolom yang tidak diperlukan
    dataset.drop(columns=[subset], inplace=True)
    
    # Print pesan yang menandakan bahwa proses pengkodean telah berhasil
    print(f"Fungsi ohe_transform: daftar nama kolom setelah dilakukan pengkodean adalah {list(dataset.columns)}.")
    
    # Kembalikan dataset dari fungsi
    return dataset