import pandas as pd

def load_inventory_data(filepath: str) -> pd.DataFrame:
    """Load and clean the inventory data panel"""
    df = pd.read_csv(filepath)
    # Set Industry Name as index for fast retrieval
    df.set_index('Industry Name', inplace=True)
    return df