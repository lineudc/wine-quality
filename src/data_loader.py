import pandas as pd
from src.config import RED_WINE_PATH, WHITE_WINE_PATH

def load_and_combine_data():
    """
    Loads and combines red and white wine datasets.
    
    Returns:
        pd.DataFrame: Combined dataframe with a 'type' column.
    """
    print("="*80)
    print("1. LOADING AND COMBINING DATA")
    print("="*80)
    
    # Load data
    df_red = pd.read_csv(RED_WINE_PATH, sep=';')
    df_white = pd.read_csv(WHITE_WINE_PATH, sep=';')
    
    # Add type column
    df_red['type'] = 'red'
    df_white['type'] = 'white'
    
    # Combine datasets
    df_combined = pd.concat([df_red, df_white], axis=0, ignore_index=True)
    
    print(f"\n✓ Red Wine: {len(df_red)} samples")
    print(f"✓ White Wine: {len(df_white)} samples")
    print(f"✓ Combined Dataset: {len(df_combined)} samples")
    
    return df_combined
