# main app focus on user interaction and UI
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import io
from agents.ai_data_analyst_agent import DataAnalystAgent # import the agent, that will be used to analyze the data
import plotly.io as pio

# Configure plotly to use a static renderer
pio.templates.default = "plotly_white"

# Load environment variables from .env file
load_dotenv()

# Initialize session state, which is used to store the state of the app across user interactions
if 'messages' not in st.session_state:
    st.session_state.messages = [] # initialize with empty messages list
if 'agent' not in st.session_state:
    st.session_state.agent = DataAnalystAgent(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model_name=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
        temperature=float(os.getenv('TEMPERATURE', 0))
    )
if 'df' not in st.session_state:
    st.session_state.df = None
if 'error_count' not in st.session_state:
    st.session_state.error_count = 0

def display_visualizations(visualizations):
    """Display plotly visualizations in streamlit"""
    if not visualizations:
        return
    
    # Display type distribution pie chart
    if 'type_distribution' in visualizations:
        st.plotly_chart(visualizations['type_distribution'], use_container_width=True)
    
    # Display missing values heatmap
    if 'missing_heatmap' in visualizations:
        st.plotly_chart(visualizations['missing_heatmap'], use_container_width=True)
    
    # Display numeric distributions
    if 'numeric_distributions' in visualizations:
        st.plotly_chart(visualizations['numeric_distributions'], use_container_width=True)
    
    # Display categorical distributions
    if 'categorical_distributions' in visualizations:
        for col, fig in visualizations['categorical_distributions'].items():
            st.plotly_chart(fig, use_container_width=True)
    
    # Display temporal distributions
    if 'temporal_distributions' in visualizations:
        for col, fig in visualizations['temporal_distributions'].items():
            st.plotly_chart(fig, use_container_width=True)
    
    # Display correlation heatmap
    if 'correlation_heatmap' in visualizations:
        st.plotly_chart(visualizations['correlation_heatmap'], use_container_width=True)

def process_query(query: str):
    """Process user query and display results"""
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            response = st.session_state.agent.analyze(query)
            
            if response.get("success", False):
                result = response["result"]
                
                # Handle different types of results
                if isinstance(result, dict):
                    # Display text summary if available
                    if "text_summary" in result:
                        st.markdown(result["text_summary"])
                    
                    # Display plots if available
                    if "plots" in result:
                        for plot_name, plot_data in result["plots"].items():
                            if isinstance(plot_data, dict) and 'plot' in plot_data:
                                st.plotly_chart(plot_data['plot'], use_container_width=True)
                            elif hasattr(plot_data, 'update_layout'):  # It's a plotly figure
                                st.plotly_chart(plot_data, use_container_width=True)
                    
                    # Display insights if available
                    if "insights" in result:
                        st.markdown("### 🔍 Insights")
                        st.markdown(result["insights"])
                    
                    # Display any errors that occurred
                    if "errors" in result:
                        st.error("Some errors occurred during analysis:")
                        if isinstance(result["errors"], dict):
                            for k, v in result["errors"].items():
                                st.error(f"{k}: {v}")
                        else:
                            st.error(result["errors"])
                else:
                    st.markdown(result)
                
                st.session_state.messages.append({"role": "assistant", "content": str(result)})
                st.session_state.error_count = 0  # Reset error count on success
            else:
                error_msg = response.get("error", "An unknown error occurred")
                suggestion = response.get("suggestion", "")
                st.error(f"Error: {error_msg}")
                if suggestion:
                    st.info(suggestion)
                st.session_state.error_count += 1
                
                if st.session_state.error_count >= 3:
                    st.warning("Multiple errors occurred. Please try rephrasing your request or check your data.")

# Set page config, this is used to set the title, icon, and layout of the page
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("🤖 AI Data Analyst")
st.markdown("""
This application helps you analyze your data using AI. You can:
- Upload your data files
- Get insights about your data structure
- Clean and manipulate your data
- Perform exploratory data analysis
""")

# Sidebar for file upload and data info
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.agent.set_data(df) # setting the dataframe to the agent
            
            # Display basic information
            st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Show data preview in an expander
            with st.expander("Data Preview"):
                st.dataframe(df.head())
            
            # Show suggested actions
            st.header("💡 Suggested Actions")
            suggestions = [
                "Show basic statistics",
                "Analyze missing values",
                "Show data distribution",
                "Find correlations",
                "Generate data summary"
            ]
            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggest_{suggestion}"):
                    process_query(suggestion)  # Process suggestion like a chat input
            
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")

# Main chat interface
st.header("💬 Chat with your Data")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about your data..."):
    process_query(prompt) 