from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.tools import BaseTool
from typing import List, Union, Dict, Any
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
            When you encounter an error, explain what went wrong and suggest alternatives."""
        )
        
        prompt = ChatPromptTemplate.from_messages([
            system_message,
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        return AgentExecutor.from_agent_and_tools(
            agent=self._create_agent(prompt),
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            max_iterations=self.max_retries,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )

    def _create_agent(self, prompt):
        """Create the agent with the specified prompt"""
        # Implementation details for agent creation
        # This would include the logic for parsing tool inputs and outputs
        pass

    def set_data(self, df: pd.DataFrame):
        """Set the dataframe for analysis"""
        self.df = df

    def analyze(self, query: str) -> Dict[str, Any]:
        """Process user query and return analysis results"""
        if self.df is None:
            return {"error": "No data has been loaded. Please upload a dataset first."}

        try:
            result = self.agent_executor.run(
                input=query,
                df=self.df
            )
            return {"success": True, "result": result}
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

        # Implement specific error handling logic here
        # For example, adjusting parameters or trying alternative methods
        return self.analyze(error.get("original_query", ""))

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
        return suggestions 