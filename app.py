# main app focus on user interaction and UI
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import io
from agents.ai_data_analyst_agent import DataAnalystAgent
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

# Configure plotly to use a static renderer
pio.templates.default = "plotly_white"

# Page config for wider layout
st.set_page_config(layout="wide", page_title="AI Data Analyst")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        st.session_state.agent = DataAnalystAgent(openai_api_key=openai_api_key)
    else:
        st.session_state.agent = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "current_df" not in st.session_state:
    st.session_state.current_df = None

# App title
st.title("AI Data Analyst")

def create_visualization(df: pd.DataFrame, column: str, plot_type: str = "histogram"):
    """Create visualization using multiple backends for reliability"""
    try:
        if plot_type == "histogram":
            # Try Plotly first
            try:
                fig = px.histogram(df, x=column, title=f'Distribution of {column}',
                                 template='plotly_white')
                fig.update_layout(showlegend=True, height=400)
                return {"plot": fig, "backend": "plotly"}
            except Exception as e:
                st.warning(f"Plotly visualization failed, trying Altair: {str(e)}")
                
            # Try Altair if Plotly fails
            try:
                chart = alt.Chart(df).mark_bar().encode(
                    alt.X(f'{column}:Q', bin=True),
                    y='count()',
                ).properties(title=f'Distribution of {column}')
                return {"plot": chart, "backend": "altair"}
            except Exception as e:
                st.warning(f"Altair visualization failed, trying Seaborn: {str(e)}")
                
            # Try Seaborn as last resort
            try:
                fig, ax = plt.subplots()
                sns.histplot(data=df, x=column, ax=ax)
                ax.set_title(f'Distribution of {column}')
                return {"plot": fig, "backend": "seaborn"}
            except Exception as e:
                st.error(f"All visualization attempts failed: {str(e)}")
                return None
                
        elif plot_type == "correlation":
            # Create correlation matrix
            corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
            
            # Try Plotly first
            try:
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    text=corr_matrix.round(3),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False,
                    colorscale='RdBu',
                    zmid=0
                ))
                fig.update_layout(
                    title='Correlation Heatmap',
                    height=600,
                    width=800
                )
                return {"plot": fig, "backend": "plotly"}
            except Exception as e:
                st.warning(f"Plotly correlation failed, trying Seaborn: {str(e)}")
                
            # Try Seaborn if Plotly fails
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.3f')
                ax.set_title('Correlation Heatmap')
                return {"plot": fig, "backend": "seaborn"}
            except Exception as e:
                st.error(f"All correlation visualization attempts failed: {str(e)}")
                return None
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def display_visualization(viz_data):
    """Display visualization based on its backend"""
    if viz_data is None:
        return
        
    if viz_data["backend"] == "plotly":
        st.plotly_chart(viz_data["plot"], use_container_width=True)
    elif viz_data["backend"] == "altair":
        st.altair_chart(viz_data["plot"], use_container_width=True)
    elif viz_data["backend"] == "seaborn":
        st.pyplot(viz_data["plot"])

def process_query(query: str):
    """Process user query and display results"""
    if not query:
        return
    
    # Add user message to chat history and display it
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            # Handle general visualization capability inquiries
            if any(term in query.lower() for term in ["what charts", "what plots", "what visualizations", "visualization capabilities", "plotting capabilities"]):
                capability_msg = """I can create various types of visualizations, including:

1. Time Series Plots: For visualizing trends over time (e.g., sales over time)
2. Histograms: To show the distribution of a single variable
3. Bar Charts: For comparing categorical data
4. Scatter Plots: To analyze relationships between two numerical variables
5. Correlation Heatmaps: To visualize correlations between multiple numerical variables
6. Box Plots: To show the distribution of a variable and identify outliers
7. Pie Charts: For showing proportions of categories

To create a visualization, you can:
- Ask for a specific plot type (e.g., "Show me a histogram of age")
- Request a correlation analysis (e.g., "Show correlation heatmap")
- Ask for distribution analysis (e.g., "Plot the distribution of salary")

What type of visualization would you like to see?"""
                st.session_state.messages.append({"role": "assistant", "content": capability_msg})
                st.write(capability_msg)
                return
            
            # Handle visualization requests
            if any(term in query.lower() for term in ["plot", "show", "visualize", "distribution", "correlation"]):
                df = st.session_state.current_df
                if df is not None:
                    if "correlation" in query.lower() or "heatmap" in query.lower():
                        viz_data = create_visualization(df, None, plot_type="correlation")
                        if viz_data:
                            # Add success message to chat history
                            success_msg = "Here's the correlation heatmap for the numerical variables:"
                            st.session_state.messages.append({"role": "assistant", "content": success_msg})
                            st.write(success_msg)
                            display_visualization(viz_data)
                            return
                    else:
                        # Try to identify the column to plot
                        cols = df.columns.tolist()
                        for col in cols:
                            if col.lower() in query.lower():
                                viz_data = create_visualization(df, col)
                                if viz_data:
                                    # Add success message to chat history
                                    success_msg = f"Here's the distribution plot for {col}:"
                                    st.session_state.messages.append({"role": "assistant", "content": success_msg})
                                    st.write(success_msg)
                                    display_visualization(viz_data)
                                    return
                        
                        # If no specific column found
                        st.warning("Could not identify which column to plot. Please specify the column name.")
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": "I couldn't identify which column to plot. Could you please specify the column name you'd like to visualize?"
                        })
                        return
            
            # For non-visualization queries or if visualization handling failed
            response = st.session_state.agent.analyze(query)
            
            # Handle successful response
            if response.get("success", False):
                # Add AI response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response["result"]})
                
                # Display the response text
                st.write(response["result"])
                
                # Display plots if any from the agent
                if "plots" in response and response["plots"]:
                    for plot_name, fig in response["plots"].items():
                        try:
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            # Try alternative visualization
                            if st.session_state.current_df is not None:
                                viz_data = create_visualization(st.session_state.current_df, plot_name.replace("_distribution", ""))
                                if viz_data:
                                    display_visualization(viz_data)
            else:
                error_msg = response.get("error", "An error occurred")
                suggestion = response.get("suggestion", "")
                full_msg = f"{error_msg} {suggestion}"
                st.error(full_msg)
                # Add error message to chat history
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {full_msg}"})

# Sidebar for file upload and data info
with st.sidebar:
    st.header("Data Upload")
    
    # Check for API key first
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable in your .env file.")
        st.stop()
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.current_df = df  # Store the dataframe in session state
            
            # Initialize agent if not already done
            if st.session_state.agent is None:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                st.session_state.agent = DataAnalystAgent(openai_api_key=openai_api_key)
            
            # Set data in agent
            st.session_state.agent.set_data(df)
            st.session_state.data_loaded = True
            st.success(f"Data loaded successfully! ({len(df)} rows, {len(df.columns)} columns)")
            
            # Display data preview
            with st.expander("Data Preview"):
                st.dataframe(df.head())
            
            # Display suggested actions
            st.header("Suggested Actions")
            suggestions = st.session_state.agent.get_suggested_actions()
            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggest_{suggestion}"):
                    process_query(suggestion)  # Process suggestion like a chat input
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state.data_loaded = False

# Main chat interface
if st.session_state.data_loaded:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about your data..."):
        process_query(prompt)
else:
    st.info("Please upload a CSV file to begin analysis.") 