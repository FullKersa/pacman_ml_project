import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

def load_data(fname):
    """
    Load data from a CSV file and print its shape.

    Parameters:
    -----------
    fname : str
        The path to the CSV file.

    Returns:
    --------
    pandas.DataFrame
        The loaded data as a pandas DataFrame.
    """
    data = pd.read_csv(fname)
    print("Data Shape:", data.shape)
    return data


def split_input_output(data, target_col):
    """
    Split a DataFrame into input features (X) and target variable (Y).

    Parameters:
    -----------
    data : pandas.DataFrame
        The full dataset including features and target.
    target_col : str
        The name of the column to be used as the target variable.

    Returns:
    --------
    X : pandas.DataFrame
        The input features (all columns except the target).
    Y : pandas.DataFrame
        The target variable as a single-column DataFrame.
    """
    X = data.drop(columns=[target_col])
    Y = data[[target_col]]

    print('ORIGINAL Data Shape:', data.shape)
    print('X Data Shape:', X.shape)
    print('Y Data Shape:', Y.shape)

    return X, Y


def split_train_test(X, y, test_size, random_state=None):
    """
    Membagi dataset menjadi data pelatihan (train) dan pengujian (test) menggunakan stratifikasi.

    Parameters:
    -----------
    X : pandas.DataFrame
        Data fitur (independent variables) yang akan digunakan untuk pelatihan dan pengujian.
    
    y : pandas.Series or pandas.DataFrame
        Target variabel (dependent variable) yang akan diprediksi.
    
    test_size : float
        Proporsi data yang akan digunakan sebagai test set. Nilai antara 0 dan 1.
    
    random_state : int or None, optional (default=None)
        Angka acak untuk memastikan hasil pembagian yang konsisten (reproducible).

    Returns:
    --------
    X_train : pandas.DataFrame
        Data fitur untuk pelatihan.

    X_test : pandas.DataFrame
        Data fitur untuk pengujian.

    y_train : pandas.Series or pandas.DataFrame
        Target variabel untuk pelatihan.

    y_test : pandas.Series or pandas.DataFrame
        Target variabel untuk pengujian.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("X train shape:", X_train.shape)
    print("X test shape:", X_test.shape)
    print("y train shape:", y_train.shape)
    print("y test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test


def serialize_data(data, path):
    """
    Serialize a Python object and save it to a file using joblib.

    Parameters:
    ----------
    data : Any
        The Python object to be serialized (e.g., a model, dataset, etc.).
    path : str
        The file path where the serialized object will be saved.

    Returns:
    -------
    None
        This function does not return anything. It saves the data to the specified file.
    """
    joblib.dump(data, path)



def deserialize_data(path):
    """
    Load (deserialize) a Python object from a file using joblib.

    Parameters:
    ----------
    path : str
        The file path where the serialized object is stored.

    Returns:
    -------
    data : Any
        The deserialized Python object retrieved from the file.
    """
    data = joblib.load(path)
    return data

