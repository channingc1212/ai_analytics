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

class VisualizationTool(BaseTool):
    name: str = "create_visualization"
    description: str = """Create data visualizations. Use this tool when explicitly asked to:
    - Plot time series data ('plot over time', 'trend', 'timeline')
    - Create histograms or distributions ('distribution', 'histogram')
    - Generate bar charts ('bar chart', 'bar plot')
    - Show correlations ('scatter plot', 'correlation plot')
    - Compare categories ('compare', 'breakdown')"""
    
    def _run(self, data: Dict[str, List[Any]], **kwargs) -> Dict[str, Any]:
        df = pd.DataFrame(data)
        results = {'plots': {}}
        
        # Extract visualization parameters
        columns = kwargs.get('columns', [])
        plot_type = kwargs.get('plot_type', 'auto')
        time_column = kwargs.get('time_column')
        value_column = kwargs.get('value_column')
        
        # Time series plot
        if time_column and value_column:
            try:
                df[time_column] = pd.to_datetime(df[time_column])
                fig = px.line(df, x=time_column, y=value_column,
                            title=f'{value_column} over {time_column}')
                results['plots']['time_series'] = fig
            except Exception as e:
                results['errors'] = f"Error creating time series plot: {str(e)}"
        
        # Handle other plot types
        elif columns:
            for col in columns:
                if col not in df.columns:
                    continue
                    
                try:
                    if plot_type == 'histogram' or df[col].dtype in ['int64', 'float64']:
                        fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                    elif plot_type == 'bar' or df[col].dtype in ['object', 'category']:
                        value_counts = df[col].value_counts()
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                   title=f'Distribution of {col}')
                    elif plot_type == 'box':
                        fig = px.box(df, y=col, title=f'Box Plot of {col}')
                    else:
                        fig = generate_basic_plots(df, col)
                        
                    results['plots'][col] = fig
                except Exception as e:
                    results['errors'] = results.get('errors', {})
                    results['errors'][col] = str(e)
        
        return results
    
    def _arun(self, data: Dict[str, List[Any]], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Async not implemented")

class EDATool(BaseTool):
    name: str = "exploratory_data_analysis"
    description: str = """Analyze data patterns and create visualizations. Use this tool for:
    - Basic statistics and summaries
    - Correlation analysis
    - Pattern detection
    - Multiple visualizations
    - Comprehensive data insights"""
    
    def _run(self, data: Dict[str, List[Any]], **kwargs) -> Dict[str, Any]:
        df = pd.DataFrame(data)
        results = {}
        insights = []
        
        # Basic statistics
        if kwargs.get('basic_stats', True):
            stats_df = df.describe()
            results['basic_stats'] = stats_df.to_dict()
            
            if 'llm' in kwargs:
                stats_prompt = f"""Analyze these statistical summaries and provide key insights:
                {stats_df.to_string()}
                
                Focus on:
                1. Notable patterns in the data
                2. Unusual values or distributions
                3. Key statistics that stand out
                4. Potential business implications
                
                Be concise and specific."""
                
                stats_insights = kwargs['llm'].invoke(stats_prompt).content
                insights.append("📊 Statistical Insights:\n" + stats_insights)
        
        # Correlation analysis
        if kwargs.get('correlation', False):
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                corr_matrix = get_correlation_matrix(df[numeric_cols])
                results['correlation'] = corr_matrix.to_dict()
                
                # Create correlation heatmap
                fig = px.imshow(corr_matrix,
                              title='Correlation Heatmap',
                              labels=dict(color='Correlation'))
                results['plots'] = results.get('plots', {})
                results['plots']['correlation_heatmap'] = fig
                
                if 'llm' in kwargs and not corr_matrix.empty:
                    significant_corrs = []
                    for col1 in corr_matrix.columns:
                        for col2 in corr_matrix.index:
                            if col1 < col2:
                                corr = corr_matrix.loc[col2, col1]
                                if abs(corr) > 0.5:
                                    significant_corrs.append(f"{col1} vs {col2}: {corr:.2f}")
                    
                    if significant_corrs:
                        corr_prompt = f"""Analyze these significant correlations:
                        {', '.join(significant_corrs)}
                        
                        Explain:
                        1. What these correlations mean
                        2. Which relationships are most interesting
                        3. Potential business implications"""
                        
                        corr_insights = kwargs['llm'].invoke(corr_prompt).content
                        insights.append("🔗 Correlation Insights:\n" + corr_insights)
        
        # Categorical analysis with visualizations
        if kwargs.get('analyze_categorical', False):
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                results['categorical_analysis'] = {}
                results['plots'] = results.get('plots', {})
                
                for col in categorical_cols:
                    value_counts = df[col].value_counts()
                    fig = px.bar(x=value_counts.index[:10], y=value_counts.values[:10],
                               title=f'Top 10 Categories in {col}')
                    
                    results['categorical_analysis'][col] = {
                        'value_counts': value_counts.to_dict(),
                        'unique_count': len(value_counts)
                    }
                    results['plots'][f'{col}_distribution'] = fig
                    
                    if 'llm' in kwargs:
                        cat_prompt = f"""Analyze the distribution of {col}:
                        Top categories: {', '.join(value_counts.index[:5])}
                        Total unique categories: {len(value_counts)}
                        
                        Explain:
                        1. Distribution patterns
                        2. Dominant categories
                        3. Business implications"""
                        
                        cat_insights = kwargs['llm'].invoke(cat_prompt).content
                        insights.append(f"📊 Category Insights ({col}):\n" + cat_insights)
        
        # Numeric distributions with visualizations
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            results['plots'] = results.get('plots', {})
            for col in numeric_cols:
                fig = px.histogram(df, x=col,
                                 title=f'Distribution of {col}',
                                 marginal='box')  # Add box plot on the margin
                results['plots'][f'{col}_distribution'] = fig
        
        # Time series detection and visualization
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            results['plots'] = results.get('plots', {})
            for time_col in datetime_cols:
                # Find a numeric column to plot against time
                for value_col in numeric_cols:
                    fig = px.line(df, x=time_col, y=value_col,
                                title=f'{value_col} over {time_col}')
                    results['plots'][f'{value_col}_timeseries'] = fig
        
        results['insights'] = "\n\n".join(insights) if insights else None
        return results
    
    def _arun(self, data: Dict[str, List[Any]], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Async not implemented")

# Additional tools can be added here as needed 