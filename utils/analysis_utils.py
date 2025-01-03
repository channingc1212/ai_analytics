import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

def infer_temporal_columns(df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[List[str], List[str]]:
    """Identify categorical columns that might be temporal"""
    temporal_patterns = [
        # Date patterns
        r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
        r'\d{2}-\d{2}-\d{4}',  # DD-MM-YYYY
        # Time patterns
        r'\d{2}:\d{2}:\d{2}',  # HH:MM:SS
        # Month/Year patterns
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec).*\d{4}',  # Month Year
        r'Q[1-4].*\d{4}',  # Quarter Year
        r'\d{4}.*Q[1-4]',  # Year Quarter
        r'\d{4}'  # Year
    ]
    
    temporal_cols = []
    remaining_categorical = []
    
    for col in categorical_cols:
        # Skip if column is empty
        if df[col].dropna().empty:
            remaining_categorical.append(col)
            continue
            
        # Check first non-null value
        sample_val = str(df[col].dropna().iloc[0])
        
        # Try to parse as datetime
        try:
            pd.to_datetime(df[col], format='mixed')
            temporal_cols.append(col)
            continue
        except (ValueError, TypeError):
            pass
        
        # Check for temporal patterns
        is_temporal = any(df[col].astype(str).str.match(pattern).any() for pattern in temporal_patterns)
        if is_temporal:
            temporal_cols.append(col)
        else:
            remaining_categorical.append(col)
    
    return temporal_cols, remaining_categorical

def sample_dataframe(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    """Sample a DataFrame while preserving distribution
    
    Args:
        df: Input DataFrame
        max_rows: Maximum number of rows to sample (default: 1000)
    
    Returns:
        Sampled DataFrame
    """
    if len(df) > max_rows:
        return df.sample(n=max_rows, random_state=42)
    return df.copy()

def detect_data_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Detect and categorize column data types"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Identify potential temporal columns from categorical
    potential_temporal, categorical_cols = infer_temporal_columns(df, categorical_cols)
    
    # Add potential temporal columns to datetime_cols
    datetime_cols.extend(potential_temporal)
    
    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }

def get_missing_values_summary(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Get summary of missing values in the dataset with a concise description"""
    missing = pd.DataFrame({
        'column': df.columns,
        'missing_count': df.isnull().sum(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing = missing.sort_values('missing_count', ascending=False)
    
    # Generate concise summary
    cols_with_missing = missing[missing['missing_count'] > 0]
    n_cols_missing = len(cols_with_missing)
    
    if n_cols_missing == 0:
        summary = "No missing values found in any column."
    else:
        total_missing = cols_with_missing['missing_count'].sum()
        worst_col = cols_with_missing.iloc[0]
        summary = (
            f"Found {total_missing:,} missing values across {n_cols_missing} columns. "
            f"'{worst_col['column']}' has the most missing values ({worst_col['missing_count']:,}, "
            f"{worst_col['missing_percentage']:.1f}%)."
        )
    
    return missing, summary

def infer_data_purpose_with_llm(df: pd.DataFrame, llm: Optional[ChatOpenAI] = None) -> str:
    """Infer the likely purpose/domain of the dataset using LLM"""
    if llm is None:
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0
        )
    
    # Prepare dataset information for LLM (optimized for token usage)
    col_info = []
    for col in df.columns:
        # Get data type and basic stats
        dtype = str(df[col].dtype)
        if np.issubdtype(df[col].dtype, np.number):
            # For numeric columns, just show range
            stats = f"range: [{df[col].min():.1f}-{df[col].max():.1f}]"
        else:
            # For non-numeric, show unique count
            stats = f"unique: {df[col].nunique()}"
        
        # Get just one sample value
        sample = str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
        if len(sample) > 50:  # Truncate long values
            sample = sample[:47] + "..."
        
        col_info.append(f"- {col} ({dtype}, {stats}, e.g., {sample})")
    
    # Create a more concise prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """As a data analyst, analyze the dataset structure and identify its main purpose and domain. Be concise."""),
        ("user", """Dataset ({n_cols} columns):
{col_info}

Briefly describe the dataset's purpose.""")
    ])
    
    # Get LLM response
    response = llm.invoke(
        prompt.format_messages(
            n_cols=len(df.columns),
            col_info="\n".join(col_info[:20])  # Limit to first 20 columns if dataset is large
        )
    )
    
    return response.content

def create_data_summary_dashboard(df: pd.DataFrame) -> Dict[str, Any]:
    """Create a comprehensive data summary dashboard using plotly"""
    # Initialize dashboard
    dashboard = {}
    
    # 1. Data Types Distribution
    data_types = detect_data_types(df)
    type_counts = {
        'Numeric': len(data_types['numeric']),
        'Categorical': len(data_types['categorical']),
        'Temporal': len(data_types['datetime'])
    }
    dashboard['type_distribution'] = px.pie(
        values=list(type_counts.values()),
        names=list(type_counts.keys()),
        title='Distribution of Column Types'
    )
    
    # 2. Missing Values Heatmap
    missing_data = df.isnull().sum() / len(df) * 100
    dashboard['missing_heatmap'] = px.bar(
        x=missing_data.index,
        y=missing_data.values,
        title='Missing Values by Column (%)',
        labels={'x': 'Column', 'y': 'Missing (%)'}
    )
    
    # 3. Numeric Columns Summary
    if data_types['numeric']:
        fig = make_subplots(
            rows=len(data_types['numeric']),
            cols=1,
            subplot_titles=[f'Distribution of {col}' for col in data_types['numeric']]
        )
        for i, col in enumerate(data_types['numeric'], 1):
            fig.add_trace(
                go.Histogram(x=df[col], name=col),
                row=i, col=1
            )
        fig.update_layout(height=300*len(data_types['numeric']), showlegend=False)
        dashboard['numeric_distributions'] = fig
    
    # 4. Categorical Columns Summary
    if data_types['categorical']:
        categorical_plots = {}
        for col in data_types['categorical']:
            value_counts = df[col].value_counts().head(10)  # Top 10 categories
            categorical_plots[col] = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Top Categories in {col}'
            )
        dashboard['categorical_distributions'] = categorical_plots
    
    # 5. Temporal Columns Summary
    if data_types['datetime']:
        temporal_plots = {}
        for col in data_types['datetime']:
            try:
                df[col] = pd.to_datetime(df[col])
                temporal_plots[col] = px.line(
                    x=df[col],
                    y=df.index,
                    title=f'Timeline of {col}'
                )
            except:
                continue
        dashboard['temporal_distributions'] = temporal_plots
    
    # 6. Correlation Heatmap for Numeric Columns
    if len(data_types['numeric']) > 1:
        corr_matrix = get_correlation_matrix(df)
        dashboard['correlation_heatmap'] = px.imshow(
            corr_matrix,
            title='Correlation Matrix',
            labels=dict(color='Correlation')
        )
    
    return dashboard

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