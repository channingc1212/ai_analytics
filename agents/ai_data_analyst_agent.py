from langchain.agents import AgentExecutor, create_openai_functions_agent # provides tools to create and manage agents
from langchain_openai import ChatOpenAI # provides tools to interact with OpenAI's API
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage # provides tools to create and manage messages
from langchain.tools import BaseTool, Tool # provides framework for the agent to use tools
from langchain_core.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate # provides tools to create and manage prompts
from langchain_core.chat_history import BaseChatMessageHistory # handles storage and retrieval of chat history
from langchain.memory import ConversationBufferMemory # provides memory modules that allow agents to remember information across interactions.
from typing import List, Union, Dict, Any, Optional # provides tools to create and manage types
import pandas as pd # provides tools to create and manage dataframes
from tools.data_analysis_tools import DataInfoTool, DataCleaningTool, EDATool # import the tools
import re

# This class is used to create an agent that can analyze data
class DataAnalystAgent:
    def __init__(self, openai_api_key: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0):
        self.llm = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            openai_api_key=openai_api_key
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
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
                temperature=0,
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
            wrap_tool(EDATool())
        ]

# Agent configuration
    def _setup_agent(self) -> AgentExecutor:
        """Setup the agent with tools and memory"""
        system_message = SystemMessage(
            content="""You are an AI Data Analyst that helps users understand and analyze their data.
            You can perform various tasks like data exploration, cleaning, and basic analysis.
            Always explain your findings in a clear and concise way.
            
            Tool Usage Guide:
            1. Data Info Tool (data_info):
               - Use for initial dataset understanding
               - Get dataset purpose and structure
               - Identify data types and quality issues
               - Detect temporal columns automatically
               Example queries: "Tell me about this dataset", "What kind of data do we have?"
            
            2. Data Cleaning Tool (data_cleaning):
               - Handle missing values using strategies: mean, median, mode, or custom value
               - Remove duplicate rows
               - Standardize column names to snake_case
               - Convert column types (int, float, str, datetime)
               - Handle outliers by clipping or removing
               Example queries: "Clean missing values", "Remove duplicates", "Convert date column"
            
            3. EDA Tool (exploratory_data_analysis):
               - Generate basic statistics (count, mean, std, quartiles)
               - Analyze correlations between numeric columns
               - Create visualizations based on data type:
                 * Numeric: histograms
                 * Categorical: bar charts
                 * Temporal: line plots
               - Provide AI-powered insights about patterns and implications
               Example queries: "Show basic statistics", "Analyze correlations", "Plot distributions"
            
            When analyzing data:
            1. Start with data_info to understand the dataset
            2. Address any data quality issues using data_cleaning
            3. Perform analysis with exploratory_data_analysis
            4. Explain findings in business-friendly terms
            5. Make actionable recommendations
            
            If you're unsure about an operation or user intention, ask for confirmation.
            When encountering errors, explain what went wrong and suggest alternatives."""
        )
        
        # Create a prompt template that includes the system message, chat history, user input, and agent scratchpad
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Initialize memory with correct input/output keys
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )

        # Create an agent using the OpenAI functions agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=prompt,
            tools=self.tools
        )

        # Create an agent executor with proper configuration
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

    def set_data(self, df: pd.DataFrame):
        """Set the dataframe for analysis"""
        self.df = df

    def _detect_visualization_intent(self, query: str) -> Optional[Dict[str, Any]]:
        """Detect if the query is asking for visualization and extract relevant details
        
        Args:
            query: User's query string
        
        Returns:
            Dictionary with visualization parameters if visualization intent detected,
            None otherwise
        """
        # Common patterns for visualization requests
        time_patterns = [
            r'over time', r'trend', r'timeline', r'time series',
            r'historical', r'evolution', r'changes'
        ]
        plot_patterns = [
            r'plot', r'graph', r'chart', r'visualize', r'show',
            r'display', r'draw', r'create'
        ]
        
        # Check if query contains visualization intent
        has_plot_intent = any(re.search(pattern, query.lower()) for pattern in plot_patterns)
        has_time_intent = any(re.search(pattern, query.lower()) for pattern in time_patterns)
        
        if not has_plot_intent:
            return None
            
        # Use LLM to extract visualization parameters
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data visualization expert. Extract visualization parameters from the query.
            Return a JSON with these fields:
            - time_column: Column to use for x-axis if time series (null if not applicable)
            - value_column: Column to plot on y-axis
            - plot_type: Type of plot to create (line, bar, histogram, scatter)
            - plot_columns: List of columns to include in the plot
            
            Available columns: {columns}"""),
            ("user", "{query}")
        ])
        
        response = self.llm.invoke(
            extract_prompt.format_messages(
                columns=list(self.df.columns),
                query=query
            )
        )
        
        try:
            import json
            params = json.loads(response.content)
            
            # Add default plot type for time series
            if has_time_intent and params.get('time_column'):
                params['plot_type'] = params.get('plot_type', 'line')
            
            return params
            
        except Exception as e:
            if self.monitoring:
                self.monitoring.log_error(
                    "visualization_parsing",
                    e,
                    "parse_error",
                    {"query": query, "response": response.content}
                )
            return None
    
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
            
            # Check for visualization intent
            viz_params = self._detect_visualization_intent(query)
            if viz_params:
                try:
                    # Call EDATool with visualization parameters
                    tool = EDATool()
                    result = tool._run(
                        data=data_dict,
                        llm=self.llm,
                        time_column=viz_params.get('time_column'),
                        value_column=viz_params.get('value_column'),
                        plot_type=viz_params.get('plot_type'),
                        plot_columns=viz_params.get('plot_columns')
                    )
                    return {"success": True, "result": result}
                except Exception as e:
                    if self.monitoring:
                        self.monitoring.log_error(
                            run_id=run_context.get('run_id'),
                            error=e,
                            error_type="visualization_error",
                            context={
                                "query": query,
                                "viz_params": viz_params
                            }
                        )
                    raise
            
            # For non-visualization queries, use the agent executor
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

    def handle_error(self, error: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
        """Handle errors with retry logic"""
        if retry_count >= self.max_retries:
            return {
                "success": False,
                "error": "Maximum retry attempts reached.",
                "suggestion": "Please try a different approach or rephrase your request."
            }

        try:
            # Try with a more explicit error handling approach
            modified_query = f"Error occurred: {error.get('error', '')}. Trying again with: {error.get('original_query', '')}"
            return self.analyze(modified_query)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Please try a different approach."
            }

    def get_suggested_actions(self) -> List[str]:
        """Get suggested actions based on the current state of the data"""
        suggestions = []
        if self.df is not None:
            # Add data-specific suggestions
            if self.df.isnull().any().any():
                suggestions.append("Clean missing values")
            if len(self.df.select_dtypes(include=['number']).columns) > 0:
                suggestions.append("Analyze numerical distributions")
            if len(self.df.select_dtypes(include=['object']).columns) > 0:
                suggestions.append("Analyze categorical variables")
            suggestions.append("Show basic data statistics")
            suggestions.append("Generate data summary")
        return suggestions 