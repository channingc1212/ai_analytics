from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool, Tool
from langchain_core.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from typing import List, Union, Dict, Any, Optional
import pandas as pd
from tools.data_analysis_tools import DataInfoTool, DataCleaningTool, EDATool, VisualizationTool
import re

class DataAnalystAgent:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini", temperature: float = 0):
        """Initialize the agent with specified model configuration
        
        Args:
            openai_api_key: API key for OpenAI
            model_name: Name of the model to use (default: gpt-4o-mini)
            temperature: Temperature for model responses (default: 0)
        """
        self.llm = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            openai_api_key=openai_api_key
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )
        self.max_retries = 3
        self.df = None
        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()

        # Initialize monitoring if available
        try:
            from utils.monitoring import MonitoringConfig
            self.monitoring = MonitoringConfig()
            self.llm = ChatOpenAI(
                callbacks=self.monitoring.callback_manager,
                temperature=temperature,
                model_name=model_name,
                openai_api_key=openai_api_key
            )
        except ImportError:
            self.monitoring = None

    def _setup_tools(self) -> List[BaseTool]:
        """Setup tools for the agent"""
        def wrap_tool(tool: BaseTool) -> BaseTool:
            """Wrap tool to include DataFrame data and LLM"""
            original_run = tool._run
            
            def wrapped_run(*args, **kwargs):
                # Add data if not provided
                if not args and 'data' not in kwargs and self.df is not None:
                    kwargs['data'] = self.df.to_dict('list')
                
                # Only pass llm to tools that need it
                if isinstance(tool, (DataInfoTool, EDATool)) and 'llm' not in kwargs:
                    kwargs['llm'] = self.llm
                elif not isinstance(tool, (DataInfoTool, EDATool)) and 'llm' in kwargs:
                    del kwargs['llm']  # Remove llm from kwargs for other tools
                    
                return original_run(*args, **kwargs)
            
            tool._run = wrapped_run
            return tool

        return [
            wrap_tool(DataInfoTool()),
            wrap_tool(DataCleaningTool()),
            wrap_tool(EDATool()),
            wrap_tool(VisualizationTool())
        ]

    def _setup_agent(self) -> AgentExecutor:
        """Setup the agent with tools and memory"""
        system_message = SystemMessage(
            content="""You are an AI Data Analyst that helps users understand and analyze their data.
            You can perform various tasks like data exploration, cleaning, and visualization.
            Always explain your findings in a clear and concise way.
            
            Tool Usage Guide:
            1. Data Info Tool (data_info):
               - Use for initial dataset understanding
               - Get dataset purpose and structure
               - Identify data types and quality issues
               Example queries: "Tell me about this dataset", "What kind of data do we have?"
            
            2. Data Cleaning Tool (data_cleaning):
               - Handle missing values using strategies: mean, median, mode, or custom value
               - Remove duplicate rows
               - Standardize column names
               Example queries: "Clean missing values", "Remove duplicates"
            
            3. EDA Tool (exploratory_data_analysis):
               - Generate basic statistics
               - Analyze correlations
               - Create basic visualizations
               Example queries: "Show basic statistics", "Analyze correlations"
            
            4. Visualization Tool (create_visualization):
               - Create specific plots and charts
               - Handle time series visualization
               - Generate custom visualizations
               Example queries: "Plot sales over time", "Show distribution of ages"
            
            When creating visualizations:
            1. Use the create_visualization tool for specific plot requests
            2. Identify the appropriate columns and plot type
            3. Consider the data type when choosing visualization
            4. Provide context and insights about the visualization
            
            If you're unsure about an operation or user intention, ask for clarification."""
        )
        
        # Create a prompt template
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create an agent using the OpenAI functions agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=prompt,
            tools=self.tools
        )

        # Create an agent executor
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=self.max_retries,
            early_stopping_method="generate",
            handle_parsing_errors=True,
            return_intermediate_steps=True
        )

    def analyze(self, query: str) -> Dict[str, Any]:
        """Process user query and return analysis results"""
        if self.df is None:
            return {"error": "No data has been loaded. Please upload a dataset first."}

        try:
            # Convert DataFrame to dictionary format
            data_dict = self.df.to_dict('list')
            
            # Initialize run context
            run_context = {}
            
            # Start monitoring if available
            if self.monitoring:
                run_context = self.monitoring.start_run(
                    operation="analysis",
                    dataset_name="user_data",
                    tags={
                        "query_type": "visualization" if "plot" in query.lower() else "analysis",
                        "data_size": str(len(self.df))
                    }
                )
            
            # Prepare the inputs properly for the agent
            agent_inputs = {
                "input": query,  # The main input key for the agent
                "data": data_dict  # Additional data passed as a tool input
            }
            
            result = self.agent_executor.invoke(agent_inputs)
            
            return {"success": True, "result": result["output"]}

        except Exception as e:
            if self.monitoring:
                self.monitoring.log_error(
                    run_id=run_context.get('run_id'),
                    error=e,
                    error_type="analysis_error",
                    context={"query": query}
                )
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Would you like me to try a different approach?"
            }

    def set_data(self, df: pd.DataFrame):
        """Set the dataframe for analysis"""
        self.df = df

    def get_suggested_actions(self) -> List[str]:
        """Get suggested actions based on the current state of the data"""
        suggestions = []
        if self.df is not None:
            if self.df.isnull().any().any():
                suggestions.append("Clean missing values")
            if len(self.df.select_dtypes(include=['number']).columns) > 0:
                suggestions.append("Analyze numerical distributions")
            if len(self.df.select_dtypes(include=['object']).columns) > 0:
                suggestions.append("Analyze categorical variables")
            suggestions.append("Show basic data statistics")
            suggestions.append("Generate data summary")
        return suggestions 