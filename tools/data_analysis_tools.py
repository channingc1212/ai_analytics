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

class CleaningConfig(BaseModel):
    """Configuration for data cleaning"""
    data: Dict[str, List[Any]] = Field(..., description="The data in dictionary format that will be converted to DataFrame")
    standardize_names: bool = Field(False, description="Whether to standardize column names")
    handle_missing: Optional[Dict[str, str]] = Field(None, description="Strategy for handling missing values")
    remove_duplicates: bool = Field(False, description="Whether to remove duplicate rows")
    convert_dtypes: Optional[Dict[str, str]] = Field(None, description="Column data type conversions")
    handle_outliers: Optional[Dict[str, str]] = Field(None, description="Strategy for handling outliers")

class AnalysisConfig(BaseModel):
    """Configuration for data analysis"""
    data: Dict[str, List[Any]] = Field(..., description="The data in dictionary format that will be converted to DataFrame")
    basic_stats: bool = Field(True, description="Whether to include basic statistics")
    correlation: bool = Field(False, description="Whether to include correlation analysis")
    plot_columns: Optional[List[str]] = Field(None, description="Columns to generate plots for")

class DataInfoTool(BaseTool):
    name: str = "data_info"
    description: str = "Get basic information about the dataset including data types and missing values"
    args_schema: type[BaseModel] = DataFrameInput
    
    def _run(self, data: Dict[str, List[Any]], **kwargs) -> str:
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
    
    def _arun(self, data: Dict[str, List[Any]], **kwargs) -> str:
        raise NotImplementedError("Async not implemented")

class DataCleaningTool(BaseTool):
    name: str = "data_cleaning"
    description: str = "Clean the dataset by handling missing values, duplicates, and standardizing column names"
    args_schema: type[BaseModel] = CleaningConfig
    
    def _run(self, **kwargs) -> Dict[str, List[Any]]:
        data = kwargs.get('data', {})
        df = pd.DataFrame(data)
        
        if kwargs.get('standardize_names', False):
            df = standardize_column_names(df)
            
        if kwargs.get('handle_missing'):
            df = handle_missing_values(df, kwargs['handle_missing'])
            
        if kwargs.get('remove_duplicates', False):
            df = remove_duplicates(df)
            
        if kwargs.get('convert_dtypes'):
            df = convert_dtypes(df, kwargs['convert_dtypes'])
            
        if kwargs.get('handle_outliers'):
            for col, method in kwargs['handle_outliers'].items():
                df = handle_outliers(df, col, method)
                
        return df.to_dict('list')
    
    def _arun(self, **kwargs) -> Dict[str, List[Any]]:
        raise NotImplementedError("Async not implemented")

class EDATool(BaseTool):
    name: str = "exploratory_data_analysis"
    description: str = "Perform exploratory data analysis including statistical summaries and visualizations"
    args_schema: type[BaseModel] = AnalysisConfig
    
    def _run(self, **kwargs) -> Dict[str, Any]:
        data = kwargs.get('data', {})
        df = pd.DataFrame(data)
        results = {}
        
        # Basic statistics
        if kwargs.get('basic_stats', True):
            results['basic_stats'] = df.describe().to_dict()
            
        # Correlation analysis
        if kwargs.get('correlation', False):
            results['correlation'] = get_correlation_matrix(df).to_dict()
            
        # Generate plots for specified columns
        if kwargs.get('plot_columns'):
            results['plots'] = {}
            for column in kwargs['plot_columns']:
                if column in df.columns:
                    results['plots'][column] = generate_basic_plots(df, column)
                
        return results
    
    def _arun(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Async not implemented")

# Additional tools can be added here as needed 