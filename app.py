import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import os
import io
from agents.ai_data_analyst_agent import DataAnalystAgent

# Load environment variables
load_dotenv()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
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

# Set page config
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
            st.session_state.agent.set_data(df)
            
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
                    st.session_state.messages.append({"role": "user", "content": suggestion})
            
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
    if st.session_state.df is None:
        st.error("Please upload a dataset first!")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your data..."):
                response = st.session_state.agent.analyze(prompt)
                
                if response.get("success", False):
                    result = response["result"]
                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})
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
                                    {"original_query": prompt, "error": error_message}
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