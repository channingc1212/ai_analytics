# main app focus on user interaction and UI
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import io
from agents.ai_data_analyst_agent import DataAnalystAgent # import the agent, that will be used to analyze the data

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
    """Process a query and update the chat"""
    if st.session_state.df is None:
        st.error("Please upload a dataset first!")
        return
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            response = st.session_state.agent.analyze(query)
            
            if response.get("success", False):
                # Display text result
                result = response["result"]
                st.markdown(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
                
                # Display visualizations if available
                if "visualizations" in response:
                    with st.expander("📊 Data Visualizations", expanded=True):
                        display_visualizations(response["visualizations"])
                
                st.session_state.error_count = 0  # Reset error count on success
            else:
                error_message = response.get("error", "An error occurred")
                suggestion = response.get("suggestion", "")
                st.error(f"{error_message}\n\n{suggestion}")
                st.session_state.error_count += 1
                
                # Add error handling buttons
                if st.session_state.error_count < 3:
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Try Again", key=f"retry_{st.session_state.error_count}"):
                            retry_response = st.session_state.agent.handle_error(
                                {"original_query": query, "error": error_message}
                            )
                            if retry_response.get("success", False):
                                st.markdown(retry_response["result"])
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": retry_response["result"]
                                })
                    with col2:
                        if st.button("Modify Request", key=f"modify_{st.session_state.error_count}"):
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "Please rephrase your request or try a different approach."
                            })
                else:
                    st.warning("Maximum retry attempts reached. Please try a different approach.")

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
    st.header("Data Upload")
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
            st.header("Suggested Actions")
            suggestions = st.session_state.agent.get_suggested_actions()
            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggest_{suggestion}"):
                    process_query(suggestion)  # Process suggestion like a chat input
            
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")

# Main chat interface
st.header("Chat with AI Data Analyst")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me about your data..."):
    process_query(prompt) 