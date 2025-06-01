import pandas as pd

def load_dataset(file_path):
    """
    Load a dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing the dataset.
    
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} examples from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
def preprocess_dataset(file_path):
    """
    Preprocess the dataset by filtering out rows with missing data.
    
    Args:
        file_path: (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: Preprocessed dataset with no missing values in 'transliteration' and 'translation'.
    """

    data = load_dataset(file_path)
    if data is None:
        return None

    # Return a dataframe with only the 'transliteration' and 'translation' columns
    data = data[['transliteration', 'translation']]

    # Drop na values in 'transliteration' and 'translation' columns
    data = data.dropna(subset=['transliteration', 'translation'])

    # Filter out rows with missing data
    filtered_data = data.dropna(subset=['transliteration', 'translation'])
    filtered_data = filtered_data[filtered_data['transliteration'].apply(lambda x: isinstance(x, str))]
    filtered_data = filtered_data[filtered_data['translation'].apply(lambda x: isinstance(x, str))]
    filtered_data['sumerian'] = filtered_data['transliteration'].str.replace('\n', ' ', regex=False)
    filtered_data['english'] = filtered_data['translation'].str.replace('\n', ' ', regex=False)

    print(f"Preprocessed dataset contains {len(filtered_data)} examples")

    return filtered_data[['sumerian', 'english']]