import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import plotly.express as px

def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize columns by their data types"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }

def get_missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Get summary of missing values in the dataset"""
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    return missing.sort_values('missing_count', ascending=False)

def generate_basic_plots(df: pd.DataFrame, column: str) -> Any:
    """Generate appropriate plots based on data type"""
    if df[column].dtype in ['int64', 'float64']:
        fig = px.histogram(df, x=column, title=f'Distribution of {column}')
    elif df[column].dtype in ['object', 'category']:
        value_counts = df[column].value_counts()
        fig = px.bar(x=value_counts.index, y=value_counts.values, 
                     title=f'Distribution of {column}')
    elif pd.api.types.is_datetime64_any_dtype(df[column]):
        # Create a time series plot
        fig = px.line(df, x=column, y=df[column].index, title=f'Time Series of {column}')
    else:
        return None
    return fig

def detect_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.Series:
    """Detect outliers in a numeric column"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[column].apply(lambda x: x < lower_bound or x > upper_bound)
    else:
        # Z-score method
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return z_scores > 3

def get_correlation_matrix(df: pd.DataFrame, method: str = 'pearson') -> pd.DataFrame:
    """Calculate correlation matrix for numeric columns"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    return df[numeric_cols].corr(method=method) 