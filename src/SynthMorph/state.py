from pydantic import BaseModel
from typing import Annotated, Optional, Dict, Any, List, TypedDict
from langgraph.graph.message import add_messages

class ElasticMatrix(BaseModel):

    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C22: Optional[float] = None
    C23: Optional[float] = None
    C33: Optional[float] = None

    def to_matrix(self) -> List[List[float]]:
        return [
            [self.C11 or 0.0, self.C12 or 0.0, self.C13 or 0.0],
            [self.C12 or 0.0, self.C22 or 0.0, self.C23 or 0.0],
            [self.C13 or 0.0, self.C23 or 0.0, self.C33 or 0.0],
        ]


class SPAgentState(TypedDict, total=False):
    # WORKDIR: Annotated[str, lambda old, new: new if new is not None else old]
    WORKDIR: str 
    abq_script_path: str 
    messages: Annotated[List[Any], add_messages]
    CurrentTask: str

    C11: Optional[float]
    C12: Optional[float]
    C13: Optional[float]
    C22: Optional[float]
    C23: Optional[float]
    C33: Optional[float]

    elastic_matrix: Optional[List[List[float]]]
    PredictedStructureImage : Optional[str]  
    user_input_image : Optional[str] 
    PredictedElasticPerformance : Optional[Dict[str, float]]  
    analyze_C_result : Optional[Dict[str, Any]]  