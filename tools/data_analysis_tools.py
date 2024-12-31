from langchain.tools import BaseTool
from typing import Dict, List, Any, Optional
import pandas as pd
from utils.analysis_utils import (
    detect_data_types,
    get_missing_values_summary,
    generate_basic_plots,
    detect_outliers,
    get_correlation_matrix
)
from utils.cleaning_utils import (
    handle_missing_values,
    remove_duplicates,
    standardize_column_names,
    convert_dtypes,
    handle_outliers
)

class DataInfoTool(BaseTool):
    name: str = "data_info"
    description: str = "Get basic information about the dataset including rows, columns, data types and missing values. Please interpret the meaning of the data based on the column names."
    
    def _run(self, df: pd.DataFrame) -> str:
        data_types = detect_data_types(df)
        missing_summary = get_missing_values_summary(df)
        
        info = [
            f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns\n",
            "Data Types:",
            f"- Numeric columns: {', '.join(data_types['numeric'])}",
            f"- Categorical columns: {', '.join(data_types['categorical'])}",
            f"- Datetime columns: {', '.join(data_types['datetime'])}\n",
            "Missing Values Summary:",
            missing_summary.to_string()
        ]
        
        return "\n".join(info)
    
    def _arun(self, df: pd.DataFrame) -> str:
        raise NotImplementedError("Async not implemented")

class DataCleaningTool(BaseTool):
    name: str = "data_cleaning"
    description: str = "Clean the dataset by handling missing values, duplicates, and standardizing column names. You should always check if there is any data issue after cleaning."
    
    def _run(self, df: pd.DataFrame, cleaning_config: Dict[str, Any]) -> pd.DataFrame:
        # Apply cleaning steps based on configuration
        if cleaning_config.get('standardize_names', False):
            df = standardize_column_names(df)
            
        if cleaning_config.get('handle_missing'):
            df = handle_missing_values(df, cleaning_config['handle_missing'])
            
        if cleaning_config.get('remove_duplicates'):
            df = remove_duplicates(df, cleaning_config.get('duplicate_subset'))
            
        if cleaning_config.get('convert_dtypes'):
            df = convert_dtypes(df, cleaning_config['convert_dtypes'])
            
        if cleaning_config.get('handle_outliers'):
            for col, method in cleaning_config['handle_outliers'].items():
                df = handle_outliers(df, col, method)
                
        return df
    
    def _arun(self, df: pd.DataFrame, cleaning_config: Dict[str, Any]) -> pd.DataFrame:
        raise NotImplementedError("Async not implemented")

class EDATool(BaseTool):
    name: str = "exploratory_data_analysis"
    description: str = "Perform exploratory data analysis including statistical summaries and visualizations. You should also interpret the results based on the statistics and visualizations as insights summary to the user."
    
    def _run(self, df: pd.DataFrame, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        results = {}
        
        # Basic statistics
        if analysis_config.get('basic_stats', True):
            results['basic_stats'] = df.describe()
            
        # Correlation analysis
        if analysis_config.get('correlation', False):
            results['correlation'] = get_correlation_matrix(df)
            
        # Generate plots for specified columns
        if analysis_config.get('plot_columns'):
            results['plots'] = {}
            for column in analysis_config['plot_columns']:
                results['plots'][column] = generate_basic_plots(df, column)
                
        return results
    
    def _arun(self, df: pd.DataFrame, analysis_config: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Async not implemented")

# Additional tools can be added here as needed 