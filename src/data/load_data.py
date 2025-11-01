import pandas as pd
import os

def load_data(file_path):
    """
    Load data from an Excel file into a pandas DataFrame.

    Parameters:
    file_path (str): The path to the Excel file.

    Returns:
    pd.DataFrame: The loaded data as a DataFrame.
    """

    # Check if the file path exists before loading
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read the Excel file into a pandas DataFrame
    # pandas automatically detects the Excel engine (.xlsx uses openpyxl)
    data = pd.read_excel(file_path)

    return data
