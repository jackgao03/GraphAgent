from pydantic import BaseModel
from typing import Dict, Any

class AgentAnalysisResponse(BaseModel):
    industry_name: str
    model_name: str
    llm_analysis: Dict[str, Any]