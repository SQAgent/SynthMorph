from pydantic import BaseModel
from typing import Annotated, Optional, Dict, Any, List, TypedDict
from langgraph.graph.message import add_messages

class ElasticMatrix(BaseModel):
    """定义弹性刚度矩阵的数据结构"""

    C11: Optional[float] = None
    C12: Optional[float] = None
    C13: Optional[float] = None
    C22: Optional[float] = None
    C23: Optional[float] = None
    C33: Optional[float] = None

    def to_matrix(self) -> List[List[float]]:
        """将弹性矩阵转换为二维列表形式"""
        return [
            [self.C11 or 0.0, self.C12 or 0.0, self.C13 or 0.0],
            [self.C12 or 0.0, self.C22 or 0.0, self.C23 or 0.0],
            [self.C13 or 0.0, self.C23 or 0.0, self.C33 or 0.0],
        ]


class SPAgentState(TypedDict, total=False):
    """
    LangGraph 使用的 State 结构。
    - messages: 对话历史（System / Human / AI / Tool）
    - C11~C33: 六个独立的刚度矩阵元素
    - elastic_matrix: 最终 3x3 刚度矩阵（全部元素收集完才写入）
    """
    # WORKDIR: Annotated[str, lambda old, new: new if new is not None else old] = "/home/shangqing/sqdata/project/sqagents/log/2D"
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
    PredictedStructureImage : Optional[str]  # 用于存储预测结构的图片
    
    user_input_image : Optional[str]  # 用于存储用户上传的图片
    PredictedElasticPerformance : Optional[Dict[str, float]]  # 用于存储计算得到的弹性刚度矩阵


    analyze_C_result : Optional[Dict[str, Any]]  # 用于存储 analyze_C 的计算结果