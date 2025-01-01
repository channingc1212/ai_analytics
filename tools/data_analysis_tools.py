from langchain.tools import BaseTool
from typing import Dict, List, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field
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

class DataFrameInput(BaseModel):
    """Input for tools that require a DataFrame"""
    data: Dict[str, List[Any]] = Field(..., description="The data in dictionary format that will be converted to DataFrame")

class DataInfoTool(BaseTool):
    name: str = "data_info"
    description: str = "Get basic information about the dataset including data types, missing values, and their summaries. Example input: {'column1': [1, 2, None], 'column2': ['A', 'B', 'C']}"
    
    def _run(self, data: Dict[str, List[Any]]) -> str:
        if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
            raise ValueError("Input data must be a dictionary with lists as values.")
        
        df = pd.DataFrame(data)
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
    
    def _arun(self, data: Dict[str, List[Any]]) -> str:
        raise NotImplementedError("Async not implemented")

class DataCleaningTool(BaseTool):
    name: str = "data_cleaning"
    description: str = "Clean the dataset by handling missing values, duplicates, outliers, and standardizing column names"
    
    def _run(self, data: Dict[str, List[Any]], standardize_names: bool = False,
             handle_missing: Optional[Dict[str, str]] = None,
             remove_duplicates: bool = False,
             convert_dtypes: Optional[Dict[str, str]] = None,
             handle_outliers: Optional[Dict[str, str]] = None) -> Dict[str, List[Any]]:
        df = pd.DataFrame(data)
        
        if standardize_names:
            df = standardize_column_names(df)
            
        if handle_missing:
            df = handle_missing_values(df, handle_missing)
            
        if remove_duplicates:
            df = remove_duplicates(df)
            
        if convert_dtypes:
            df = convert_dtypes(df, convert_dtypes)
            
        if handle_outliers:
            for col, method in handle_outliers.items():
                df = handle_outliers(df, col, method)
                
        return df.to_dict('list')
    
    def _arun(self, data: Dict[str, List[Any]], **kwargs) -> Dict[str, List[Any]]:
        raise NotImplementedError("Async not implemented")

class EDATool(BaseTool):
    name: str = "exploratory_data_analysis"
    description: str = "Perform exploratory data analysis including statistical summaries, correlation analysis, and visualizations"
    
    def _run(self, data: Dict[str, List[Any]], basic_stats: bool = True,
             correlation: bool = False, plot_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        df = pd.DataFrame(data)
        results = {}
        
        # Basic statistics
        if basic_stats:
            results['basic_stats'] = df.describe().to_dict()
            
        # Correlation analysis
        if correlation:
            results['correlation'] = get_correlation_matrix(df).to_dict()
            
        # Generate plots for specified columns
        if plot_columns:
            results['plots'] = {}
            for column in plot_columns:
                if column in df.columns:
                    results['plots'][column] = generate_basic_plots(df, column)
                
        return results
    
    def _arun(self, data: Dict[str, List[Any]], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Async not implemented")

# Additional tools can be added here as needed 