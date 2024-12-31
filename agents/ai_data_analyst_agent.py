from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool, Tool
from langchain_core.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.memory import ConversationBufferMemory
from typing import List, Union, Dict, Any, Optional
import pandas as pd
from tools.data_analysis_tools import DataInfoTool, DataCleaningTool, EDATool

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
        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()
        self.df = None

    def _setup_tools(self) -> List[BaseTool]:
        """Setup tools for the agent"""
        return [
            DataInfoTool(),
            DataCleaningTool(),
            EDATool()
        ]

    def _setup_agent(self) -> AgentExecutor:
        """Setup the agent with tools and memory"""
        system_message = SystemMessage(
            content="""You are an AI Data Analyst that helps users understand and analyze their data.
            You can perform various tasks like data exploration, cleaning, and basic analysis.
            Always explain your findings in a clear and concise way.
            If you're unsure about an operation, ask the user for confirmation.
            When you encounter an error, explain what went wrong and suggest alternatives.
            
            When analyzing data:
            1. Start by understanding the data structure
            2. Check for data quality issues
            3. Suggest appropriate analysis methods
            4. Explain your findings in simple terms
            5. Make recommendations based on the analysis"""
        )
        
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = create_openai_functions_agent(
            llm=self.llm,
            prompt=prompt,
            tools=self.tools
        )

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=self.max_retries,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )

    def set_data(self, df: pd.DataFrame):
        """Set the dataframe for analysis"""
        self.df = df

    def analyze(self, query: str) -> Dict[str, Any]:
        """Process user query and return analysis results"""
        if self.df is None:
            return {"error": "No data has been loaded. Please upload a dataset first."}

        try:
            result = self.agent_executor.invoke(
                {"input": query, "df": self.df}
            )
            return {"success": True, "result": result["output"]}
        except Exception as e:
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