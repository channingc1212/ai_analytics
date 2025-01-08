from langsmith import Client
from langchain.callbacks.tracers.langchain import LangChainTracer
from langchain.callbacks.manager import CallbackManager
from typing import Optional, Dict, Any
import os
from datetime import datetime
import uuid

class MonitoringConfig:
    """Configuration for LangSmith monitoring"""
    
    def __init__(self, project_name: str = "ai_data_analyst"):
        """Initialize monitoring configuration
        
        Args:
            project_name: Name of the project in LangSmith
        """
        # Initialize LangSmith client
        self.client = Client()
        
        # Use environment variable for project name if set, otherwise use default
        self.project_name = os.getenv("LANGCHAIN_PROJECT", project_name)
        
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
        """Create standardized tags for runs"""
        tags = {
            "operation": operation_type,
            "data_size": str(data_size),
            "timestamp": datetime.utcnow().isoformat(),
            "environment": os.getenv("ENVIRONMENT", "dev")  # Environment as a tag, not part of project name
        }
        
        if additional_tags:
            tags.update(additional_tags)
        
        return tags
    
    def log_feedback(self,
                    run_id: Optional[str],
                    score: float,
                    feedback_type: str = "user_feedback",
                    comment: Optional[str] = None) -> None:
        """Log user feedback for a specific run
        
        Args:
            run_id: The ID of the run to attach feedback to
            score: Feedback score (1.0 for positive, 0.0 for negative)
            feedback_type: Type of feedback (default: user_feedback)
            comment: Optional comment explaining the feedback
        """
        if not run_id:
            run_id = str(uuid.uuid4())
            
        try:
            self.client.create_feedback(
                run_id=run_id,
                key=feedback_type,
                score=score,
                comment=comment,
                value={"score": score, "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            print(f"Error logging feedback: {str(e)}")
    
    def log_error(self,
                 run_id: Optional[str] = None,
                 error: Optional[Exception] = None,
                 error_type: str = "unknown",
                 context: Optional[Dict[str, Any]] = None) -> None:
        """Log error information for a specific run"""
        if not run_id:
            run_id = str(uuid.uuid4())
            
        error_data = {
            "error_type": error_type,
            "error_message": str(error) if error else "Unknown error",
            "timestamp": datetime.now().isoformat()
        }
        if context:
            error_data.update(context)
            
        try:
            self.client.update_run(
                run_id=run_id,
                error=error_data,
                end_time=datetime.now()
            )
        except Exception as e:
            print(f"Error logging error: {str(e)}")
    
    def create_run_name(self, operation: str, dataset_name: str) -> str:
        """Create standardized run names"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{operation}_{dataset_name}_{timestamp}"
    
    def start_run(self, operation: str, dataset_name: str = None, tags: Dict[str, str] = None) -> Dict[str, Any]:
        """Start a new run and return run context"""
        run_id = str(uuid.uuid4())
        
        # Create run metadata
        metadata = {
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name
        }
        if tags:
            metadata.update(tags)
            
        try:
            self.client.create_run(
                project_name=self.project_name,
                run_id=run_id,
                name=f"{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                extra=metadata
            )
        except Exception as e:
            print(f"Error creating run: {str(e)}")
            
        return {"run_id": run_id}

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