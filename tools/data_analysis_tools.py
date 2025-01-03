from langchain.tools import BaseTool
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field
from utils.analysis_utils import (
    detect_data_types,
    get_missing_values_summary,
    generate_basic_plots,
    detect_outliers,
    get_correlation_matrix,
    infer_data_purpose_with_llm,
    create_data_summary_dashboard,
    sample_dataframe
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
    description: str = """Get dataset information including purpose, structure, data types, and quality issues. Use for initial data understanding and overview."""
    
    def _run(self, data: Dict[str, List[Any]], llm: Optional[Any] = None) -> Dict[str, Any]:
        if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
            raise ValueError("Input data must be a dictionary with lists as values.")
        
        df = pd.DataFrame(data)
        
        # 1. Infer dataset purpose using LLM
        purpose = infer_data_purpose_with_llm(df, llm)
        
        # 2. Get data types with enhanced temporal detection
        data_types = detect_data_types(df)
        
        # 3. Get missing values summary
        missing_df, missing_summary = get_missing_values_summary(df)
        
        # Combine all information
        text_info = [
            "📊 Dataset Purpose:",
            purpose + "\n",
            
            "📋 Basic Information:",
            f"- Rows: {df.shape[0]:,}",
            f"- Columns: {df.shape[1]:,}\n",
            
            "🔍 Column Types:",
            f"- Numeric ({len(data_types['numeric'])}): {', '.join(data_types['numeric'][:5])}{'...' if len(data_types['numeric']) > 5 else ''}",
            f"- Categorical ({len(data_types['categorical'])}): {', '.join(data_types['categorical'][:5])}{'...' if len(data_types['categorical']) > 5 else ''}",
            f"- Temporal ({len(data_types['datetime'])}): {', '.join(data_types['datetime'][:5])}{'...' if len(data_types['datetime']) > 5 else ''}\n",
            
            "⚠️ Missing Values:",
            missing_summary
        ]
        
        return {
            "text_summary": "\n".join(text_info),
            "data_types": data_types,
            "missing_values": missing_df.to_dict('records')
        }
    
    def _arun(self, data: Dict[str, List[Any]]) -> Dict[str, Any]:
        raise NotImplementedError("Async not implemented")

class DataCleaningTool(BaseTool):
    name: str = "data_cleaning"
    description: str = """Clean dataset: handle missing values (mean/median/mode), remove duplicates, standardize names, convert types, handle outliers (clip/remove)."""
    
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
    description: str = """Analyze and visualize data patterns. Use this tool when you need to:
    - Create time series plots ('plot over time', 'trend', 'timeline')
    - Show distributions ('histogram', 'distribution', 'plot')
    - Analyze patterns ('chart', 'graph', 'visualization')
    - Compare values ('compare', 'relationship')"""
    
    def _run(self, data: Dict[str, List[Any]], llm: Optional[Any] = None,
             basic_stats: bool = False, correlation: bool = False, 
             plot_columns: Optional[List[str]] = None,
             analyze_categorical: bool = False,
             time_column: Optional[str] = None,
             value_column: Optional[str] = None,
             plot_type: Optional[str] = None,
             debug_tokens: bool = False) -> Dict[str, Any]:
        """Run analysis and create visualizations
        
        Args:
            data: Input data dictionary
            llm: Optional LLM for insights
            basic_stats: Whether to include basic statistics
            correlation: Whether to analyze correlations
            plot_columns: Columns to plot
            analyze_categorical: Whether to analyze categorical variables
            time_column: Column to use for time series x-axis
            value_column: Column to plot on y-axis
            plot_type: Type of plot to create ('line', 'bar', 'histogram', etc.)
            debug_tokens: Whether to track token usage
        """
        df = pd.DataFrame(data)
        df_sample = sample_dataframe(df)
        results = {}
        insights = []

        # Handle time series plotting
        if time_column and value_column:
            try:
                # Convert time column to datetime if needed
                if df[time_column].dtype != 'datetime64[ns]':
                    df[time_column] = pd.to_datetime(df[time_column])
                
                # Create time series plot
                plot = px.line(df, x=time_column, y=value_column,
                             title=f'{value_column} over {time_column}')
                results['plots'] = {
                    'time_series': {
                        'plot': plot,
                        'type': 'line',
                        'x': time_column,
                        'y': value_column
                    }
                }
                
                if llm:
                    # Get insights about the time series
                    stats = df.groupby(time_column)[value_column].agg(['mean', 'min', 'max']).describe()
                    trend_prompt = f"""Analyze this time series of {value_column}:
                    Average value: {stats['mean']['mean']:.2f}
                    Range: {stats['min']['min']:.2f} to {stats['max']['max']:.2f}
                    Key patterns or trends?"""
                    
                    trend_insights = llm.invoke(trend_prompt).content
                    insights.append(f"📈 Time Series Insights:\n{trend_insights}")
                
            except Exception as e:
                results['errors'] = f"Error creating time series plot: {str(e)}"
        
        # Handle regular plotting
        elif plot_columns:
            results['plots'] = {}
            for column in plot_columns[:3]:  # Limit to 3 columns
                if column not in df.columns:
                    continue
                    
                try:
                    if plot_type == 'histogram' or df[column].dtype in ['int64', 'float64']:
                        plot = px.histogram(df, x=column, title=f'Distribution of {column}')
                    elif plot_type == 'bar' or df[column].dtype in ['object', 'category']:
                        value_counts = df[column].value_counts()
                        plot = px.bar(x=value_counts.index, y=value_counts.values,
                                    title=f'Distribution of {column}')
                    else:
                        plot = generate_basic_plots(df, column)
                    
                    results['plots'][column] = {
                        'plot': plot,
                        'type': plot_type or 'auto',
                        'column': column
                    }
                    
                    if llm:
                        if df[column].dtype in ['int64', 'float64']:
                            stats = df[column].describe()
                            dist_prompt = f"""'{column}' distribution:
                            Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}
                            Range: {stats['min']:.2f} to {stats['max']:.2f}
                            Key patterns?"""
                            
                            dist_insights = llm.invoke(dist_prompt).content
                            insights.append(f"📊 {column}:\n{dist_insights}")
                            
                except Exception as e:
                    results['errors'] = results.get('errors', {})
                    results['errors'][column] = str(e)
        
        # Include other analyses if requested
        if basic_stats or correlation or analyze_categorical:
            analysis_results = super()._run(
                data=data,
                llm=llm,
                basic_stats=basic_stats,
                correlation=correlation,
                analyze_categorical=analyze_categorical,
                debug_tokens=debug_tokens
            )
            results.update(analysis_results)
        
        if insights:
            results['insights'] = "\n\n".join(insights)
        
        return results
    
    def _arun(self, data: Dict[str, List[Any]], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Async not implemented")

# Additional tools can be added here as needed 