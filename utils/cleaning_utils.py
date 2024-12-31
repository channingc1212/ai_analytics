import pandas as pd
import numpy as np
from typing import List, Union, Dict, Any

def handle_missing_values(df: pd.DataFrame, strategy: Dict[str, str]) -> pd.DataFrame:
    """
    Handle missing values based on specified strategy
    
    Args:
        df: Input DataFrame
        strategy: Dict with column names and strategies ('mean', 'median', 'mode', 'drop', 'fill_value')
    """
    df_clean = df.copy()
    
    for column, method in strategy.items():
        if method == 'mean':
            df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
        elif method == 'median':
            df_clean[column] = df_clean[column].fillna(df_clean[column].median())
        elif method == 'mode':
            df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
        elif method == 'drop':
            df_clean = df_clean.dropna(subset=[column])
            
    return df_clean

def remove_duplicates(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """Remove duplicate rows based on specified columns"""
    return df.drop_duplicates(subset=subset, keep='first')

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to snake_case"""
    df = df.copy()
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(r'[\s\-]+', '_', regex=True)
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
    return df

def convert_dtypes(df: pd.DataFrame, conversions: Dict[str, str]) -> pd.DataFrame:
    """
    Convert column data types
    
    Args:
        df: Input DataFrame
        conversions: Dict with column names and target dtypes
    """
    df_converted = df.copy()
    
    for column, dtype in conversions.items():
        try:
            if dtype == 'datetime':
                df_converted[column] = pd.to_datetime(df_converted[column])
            else:
                df_converted[column] = df_converted[column].astype(dtype)
        except Exception as e:
            print(f"Error converting {column} to {dtype}: {str(e)}")
            
    return df_converted

def handle_outliers(df: pd.DataFrame, column: str, method: str = 'clip') -> pd.DataFrame:
    """Handle outliers in numeric columns"""
    df_clean = df.copy()
    
    if method == 'clip':
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
    elif method == 'remove':
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[
            (df_clean[column] >= Q1 - 1.5 * IQR) & 
            (df_clean[column] <= Q3 + 1.5 * IQR)
        ]
        
    return df_clean 