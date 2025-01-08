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
from typing import Dict, Any

# Load environment variables, e.g. OPENAI_API_KEY and langsmith configuration
load_dotenv()

# Configure plotly to use a static renderer
pio.templates.default = "plotly_white"

# Page config for wider layout
st.set_page_config(layout="wide", page_title="AI Data Analyst")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback_states" not in st.session_state:
    st.session_state.feedback_states = {}  # Store feedback states for each message
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

# Create visualization using multiple backends for reliability, defaulting to Plotly and histogram
def create_visualization(df: pd.DataFrame, column: str = None, plot_type: str = "histogram"):
    """Create visualization using multiple backends for reliability"""
    try:
        if plot_type == "correlation":
            # Create correlation matrix
            corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()
            
            # Create correlation heatmap with Plotly
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    text=corr_matrix.values.round(3),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    colorscale='RdBu',
                    zmid=0
                )
            )
            fig.update_layout(
                title='Correlation Heatmap',
                height=600,
                width=800
            )
            return fig
            
        elif plot_type == "histogram" and column is not None:
            # Check if column exists
            if column not in df.columns:
                st.error(f"Column '{column}' not found in the dataset")
                return None
            
            # Create histogram with Plotly
            fig = px.histogram(
                df, 
                x=column,
                title=f'Distribution of {column}',
                template='plotly_white',
                marginal='box'  # Add a box plot on the margin
            )
            
            fig.update_layout(
                showlegend=False,
                height=500,
                xaxis_title=column,
                yaxis_title="Count",
                bargap=0.1
            )
            
            # Add mean line
            mean_value = df[column].mean()
            fig.add_vline(
                x=mean_value,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_value:.2f}",
                annotation_position="top"
            )
            
            return fig
            
        elif plot_type == "box" and column is not None:
            # Create box plot with Plotly
            fig = px.box(
                df,
                y=column,
                title=f'Box Plot of {column}',
                template='plotly_white'
            )
            fig.update_layout(
                height=400,
                showlegend=False
            )
            return fig
            
        elif plot_type == "scatter" and column is not None:
            # Create scatter plot with Plotly
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) >= 2:
                fig = px.scatter(
                    df,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=f'Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}',
                    template='plotly_white'
                )
                fig.update_layout(
                    height=500,
                    showlegend=False
                )
                return fig
            
        return None
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

def display_visualization(viz_data):
    """Display visualization based on its backend"""
    if viz_data is None:
        return
        
    # Handle plots dictionary from VisualizationTool
    if 'plots' in viz_data:
        for plot_name, plot in viz_data['plots'].items():
            if isinstance(plot, (go.Figure, px.Figure)):
                st.plotly_chart(plot, use_container_width=True)
            elif isinstance(plot, alt.Chart):
                st.altair_chart(plot, use_container_width=True)
            else:
                st.pyplot(plot)
    
    # Display any errors if present
    if 'errors' in viz_data:
        if isinstance(viz_data['errors'], dict):
            for col, error in viz_data['errors'].items():
                st.error(f"Error with {col}: {error}")
        else:
            st.error(viz_data['errors'])

def handle_feedback(message_idx: int, score: float, feedback_type: str = "user_feedback"):
    """Handle user feedback for a specific message"""
    if st.session_state.agent and st.session_state.agent.monitoring:
        message = st.session_state.messages[message_idx]
        run_id = message.get("run_id")  # We'll add this when storing messages
        if run_id:
            st.session_state.agent.monitoring.log_feedback(
                run_id=run_id,
                score=score,
                feedback_type=feedback_type,
                comment=f"User rated response with score: {score}"
            )
            # Update feedback state
            st.session_state.feedback_states[message_idx] = score
            st.success("Thank you for your feedback!")
            # Force a rerun to update the UI
            st.rerun()

def find_closest_column(query: str, columns: list) -> str:
    """Find the closest matching column name from the query
    Uses fuzzy matching to handle minor differences in naming"""
    # Normalize query and column names
    query = query.lower().strip()
    normalized_cols = {col.lower().strip(): col for col in columns}
    
    # Direct match after normalization
    for norm_col in normalized_cols:
        if norm_col in query:
            return normalized_cols[norm_col]
    
    # Handle common variations
    for norm_col in normalized_cols:
        # Handle plural/singular
        if norm_col.rstrip('s') in query or (norm_col + 's') in query:
            return normalized_cols[norm_col]
        # Handle spaces/underscores
        if norm_col.replace(' ', '_') in query or norm_col.replace('_', ' ') in query:
            return normalized_cols[norm_col]
        
    return None

def create_all_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Create distribution plots for all numerical columns"""
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    figs = []
    
    for col in numeric_cols:
        fig = create_visualization(df, column=col, plot_type="histogram")
        if fig is not None:
            figs.append((col, fig))
    
    return figs

def process_query(query: str):
    """Process user query and display results"""
    if not query:
        return
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Get AI response
    with st.spinner("Analyzing your data..."):
        # Start a new run for monitoring
        run_id = None
        if st.session_state.agent and st.session_state.agent.monitoring:
            run_context = st.session_state.agent.monitoring.start_run(
                operation="chat_response",
                dataset_name="user_data",
                tags={"query_type": "user_query"}
            )
            run_id = run_context.get('run_id')
        
        # Handle visualization requests directly
        if any(term in query.lower() for term in ["plot", "show", "visualize", "distribution", "histogram", "correlation", "heatmap"]):
            df = st.session_state.current_df
            if df is not None:
                # Handle correlation heatmap
                if "correlation" in query.lower() or "heatmap" in query.lower():
                    fig = create_visualization(df, plot_type="correlation")
                    if fig is not None:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Here's the correlation heatmap for the numerical variables:",
                            "run_id": run_id,
                            "has_plot": True,
                            "plot": fig
                        })
                        st.rerun()
                        return
                
                # Handle "show data distribution" or similar general distribution requests
                if query.lower() in ["show data distribution", "show distributions", "plot distributions"]:
                    figs = create_all_distributions(df)
                    if figs:
                        # Add a message for each distribution plot
                        for col, fig in figs:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Distribution of {col}:",
                                "run_id": run_id,
                                "has_plot": True,
                                "plot": fig
                            })
                        st.rerun()
                        return
                
                # Handle specific column visualization
                else:
                    # Try to identify the column using fuzzy matching
                    cols = df.columns.tolist()
                    matched_col = find_closest_column(query, cols)
                    
                    if matched_col:
                        fig = create_visualization(df, column=matched_col, plot_type="histogram")
                        if fig is not None:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"Here's the distribution of {matched_col}:",
                                "run_id": run_id,
                                "has_plot": True,
                                "plot": fig
                            })
                            st.rerun()
                            return
                    
                    # If no specific column found
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"I couldn't identify which column to plot. Available numerical columns are: {', '.join(numeric_cols)}. You can also say 'show data distribution' to see all distributions.",
                        "run_id": run_id,
                        "has_plot": False
                    })
                    st.rerun()
                    return
        
        # For other queries, use the agent
        response = st.session_state.agent.analyze(query)
        
        # Add AI response to chat history with run_id and plots
        message = {
            "role": "assistant",
            "content": response.get("result", ""),
            "run_id": run_id,
            "has_plot": False
        }
        
        # If response contains a plot, add it to the message
        if response.get("plot") is not None:
            message["has_plot"] = True
            message["plot"] = response["plot"]
        elif response.get("plots") is not None:
            message["has_plot"] = True
            message["plot"] = next(iter(response["plots"].values()))  # Get the first plot
            
        st.session_state.messages.append(message)
        st.rerun()

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
    # Display chat history with feedback buttons
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # Display the message content
            st.write(message["content"])
            
            # Display plot if any
            if message.get("has_plot") and message.get("plot") is not None:
                try:
                    st.plotly_chart(message["plot"], use_container_width=True)
                except Exception as e:
                    st.error(f"Error displaying plot: {str(e)}")
            
            # Only show feedback buttons for assistant messages
            if message["role"] == "assistant":
                col1, col2 = st.columns([1, 20])  # Adjust column widths as needed
                
                # Check if feedback was already given
                feedback_given = idx in st.session_state.feedback_states
                
                with col1:
                    # Thumbs up button
                    if st.button("👍", key=f"thumbs_up_{idx}", 
                               disabled=feedback_given,
                               help="This response was helpful"):
                        handle_feedback(idx, 1.0)
                    
                    # Thumbs down button
                    if st.button("👎", key=f"thumbs_down_{idx}",
                               disabled=feedback_given,
                               help="This response was not helpful"):
                        handle_feedback(idx, 0.0)
                
                # Show feedback status if given
                if feedback_given:
                    score = st.session_state.feedback_states[idx]
                    st.caption(f"{'Thanks for your feedback! 👍' if score == 1.0 else 'Thanks for your feedback! 👎'}")
    
    # Add chat input box
    if prompt := st.chat_input("Ask me about your data..."):
        process_query(prompt)
else:
    st.info("Please upload a CSV file to begin analysis.") 