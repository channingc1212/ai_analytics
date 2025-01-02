from langchain.tools import BaseTool
from typing import Dict, List, Any, Optional
import pandas as pd
from pydantic import BaseModel, Field
from utils.analysis_utils import (
    detect_data_types,
    get_missing_values_summary,
    generate_basic_plots,
    detect_outliers,
    get_correlation_matrix,
    infer_data_purpose_with_llm,
    create_data_summary_dashboard
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
    description: str = """Analyze data with statistics, correlations, and visualizations. Provides insights on patterns, distributions, and relationships."""
    
    def _run(self, data: Dict[str, List[Any]], llm: Optional[Any] = None,
             basic_stats: bool = True, correlation: bool = False, 
             plot_columns: Optional[List[str]] = None,
             analyze_categorical: bool = False) -> Dict[str, Any]:
        df = pd.DataFrame(data)
        results = {}
        insights = []
        
        # Basic statistics
        if basic_stats:
            # Only include numeric columns for basic stats
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if not numeric_df.empty:
                stats_df = numeric_df.describe()
                results['basic_stats'] = stats_df.to_dict()
                
                if llm:
                    # Optimize the stats summary for token usage
                    summary_stats = {
                        col: {
                            'mean': f"{stats_df.loc['mean', col]:.2f}",
                            'std': f"{stats_df.loc['std', col]:.2f}",
                            'min': f"{stats_df.loc['min', col]:.2f}",
                            'max': f"{stats_df.loc['max', col]:.2f}"
                        } for col in stats_df.columns
                    }
                    
                    stats_prompt = f"""Analyze key statistics of numeric columns:
                    {str(summary_stats)[:1000]}  # Limit the string length
                    Focus on: notable patterns, unusual values, key insights.
                    Be very concise."""
                    
                    stats_insights = llm.invoke(stats_prompt).content
                    insights.append("📊 Statistical Insights:\n" + stats_insights)
        
        # Correlation analysis
        if correlation:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                corr_matrix = get_correlation_matrix(df[numeric_cols])
                results['correlation'] = corr_matrix.to_dict()
                
                if llm and not corr_matrix.empty:
                    # Only analyze strong correlations
                    significant_corrs = [
                        f"{col1} vs {col2}: {corr_matrix.loc[col2, col1]:.2f}"
                        for col1 in corr_matrix.columns
                        for col2 in corr_matrix.index
                        if col1 < col2 and abs(corr_matrix.loc[col2, col1]) > 0.7  # Increased threshold
                    ][:5]  # Limit to top 5 correlations
                    
                    if significant_corrs:
                        corr_prompt = f"""Analyze top correlations:
                        {', '.join(significant_corrs)}
                        Explain key relationships briefly."""
                        
                        corr_insights = llm.invoke(corr_prompt).content
                        insights.append("🔗 Correlation Insights:\n" + corr_insights)
        
        # Categorical variable analysis
        if analyze_categorical:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                results['categorical_analysis'] = {}
                # Limit to first 5 categorical columns to reduce token usage
                for col in list(categorical_cols)[:5]:
                    value_counts = df[col].value_counts().head(5)  # Only top 5 categories
                    results['categorical_analysis'][col] = {
                        'value_counts': value_counts.to_dict(),
                        'unique_count': df[col].nunique(),
                        'plot': generate_basic_plots(df, col)
                    }
                    
                    if llm:
                        cat_prompt = f"""Analyze distribution of '{col}':
                        Top categories: {dict(value_counts)}
                        Total unique: {df[col].nunique()}
                        Key insights?"""
                        
                        cat_insights = llm.invoke(cat_prompt).content
                        insights.append(f"📊 Category Insights ({col}):\n" + cat_insights)
            else:
                results['categorical_analysis'] = {"message": "No categorical variables found"}
        
        # Generate plots for specified columns
        if plot_columns:
            results['plots'] = {}
            # Limit to first 3 columns to reduce complexity
            for column in plot_columns[:3]:
                if column in df.columns:
                    plot = generate_basic_plots(df, column)
                    results['plots'][column] = plot
                    
                    if llm and df[column].dtype in ['int64', 'float64']:
                        stats = df[column].describe()
                        dist_prompt = f"""Analyze '{column}' distribution:
                        Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}
                        Range: {stats['min']:.2f} to {stats['max']:.2f}
                        Key patterns?"""
                        
                        dist_insights = llm.invoke(dist_prompt).content
                        insights.append(f"📈 Distribution ({column}):\n" + dist_insights)
        
        results['insights'] = "\n\n".join(insights) if insights else None
        return results
    
    def _arun(self, data: Dict[str, List[Any]], **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Async not implemented")

# Additional tools can be added here as needed 