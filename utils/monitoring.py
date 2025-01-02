from langsmith import Client
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from typing import Optional, Dict, Any
import os
from datetime import datetime

class MonitoringConfig:
    """Configuration for LangSmith monitoring"""
    
    def __init__(self, project_name: str = "ai_data_analyst"):
        """Initialize monitoring configuration
        
        Args:
            project_name: Name of the project in LangSmith
        """
        # Initialize LangSmith client
        self.client = Client()
        self.project_name = f"{project_name}_{os.getenv('ENVIRONMENT', 'dev')}"
        
        # Create tracer for the project
        self.tracer = LangChainTracer(
            project_name=self.project_name,
            client=self.client
        )
        
        # Initialize callback manager
        self.callback_manager = CallbackManager([self.tracer])
    
    def create_run_tags(self, 
                       operation_type: str,
                       data_size: int,
                       additional_tags: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Create standardized tags for runs
        
        Args:
            operation_type: Type of analysis being performed
            data_size: Size of the dataset being analyzed
            additional_tags: Additional custom tags to include
        
        Returns:
            Dictionary of tags
        """
        tags = {
            "operation": operation_type,
            "data_size": str(data_size),
            "timestamp": datetime.utcnow().isoformat(),
            "environment": os.getenv("ENVIRONMENT", "dev")
        }
        
        if additional_tags:
            tags.update(additional_tags)
        
        return tags
    
    def log_feedback(self,
                    run_id: str,
                    score: float,
                    feedback_type: str,
                    comment: Optional[str] = None) -> None:
        """Log user feedback for a run
        
        Args:
            run_id: ID of the run to log feedback for
            score: Feedback score (0-1)
            feedback_type: Type of feedback (e.g., "accuracy", "quality")
            comment: Optional comment with the feedback
        """
        self.client.create_feedback(
            run_id,
            feedback_type,
            score=score,
            comment=comment
        )
    
    def log_error(self,
                 run_id: str,
                 error: Exception,
                 error_type: str,
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Log error information
        
        Args:
            run_id: ID of the run where error occurred
            error: The exception that was raised
            error_type: Category of the error
            context: Additional context about the error
        """
        error_data = {
            "error_type": error_type,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if context:
            error_data.update(context)
        
        self.client.update_run(
            run_id,
            error=error_data,
            outputs={"error_details": error_data}
        )
    
    def create_run_name(self, operation: str, dataset_name: str) -> str:
        """Create standardized run names
        
        Args:
            operation: Type of operation being performed
            dataset_name: Name of the dataset being analyzed
        
        Returns:
            Formatted run name
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{operation}_{dataset_name}_{timestamp}"

# Example usage in your agent:
"""
from utils.monitoring import MonitoringConfig

class DataAnalystAgent:
    def __init__(self, openai_api_key: str):
        self.monitoring = MonitoringConfig()
        self.llm = ChatOpenAI(
            callbacks=self.monitoring.callback_manager,
            temperature=0,
            openai_api_key=openai_api_key
        )
        
    def analyze(self, query: str) -> Dict[str, Any]:
        try:
            # Create run tags
            tags = self.monitoring.create_run_tags(
                operation_type="data_analysis",
                data_size=len(self.df),
                additional_tags={"query_type": "user_question"}
            )
            
            # Create run name
            run_name = self.monitoring.create_run_name(
                operation="analyze",
                dataset_name="user_data"
            )
            
            # Run analysis with monitoring
            with self.monitoring.tracer.start_trace(
                run_name=run_name,
                tags=tags
            ) as run:
                result = self._perform_analysis(query)
                
                # Log feedback if available
                if "user_rating" in result:
                    self.monitoring.log_feedback(
                        run_id=run.run_id,
                        score=result["user_rating"],
                        feedback_type="user_satisfaction",
                        comment=result.get("user_comment")
                    )
                
                return result
                
        except Exception as e:
            # Log error if something goes wrong
            self.monitoring.log_error(
                run_id=run.run_id,
                error=e,
                error_type="analysis_error",
                context={"query": query}
            )
            raise
""" 