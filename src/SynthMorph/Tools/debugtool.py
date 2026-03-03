import json
import time
from typing import Annotated, Optional, Dict, Any, List, TypedDict
from pydantic import BaseModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

def save_state_to_json(state, filename="state_dump.json"):
    """
    将state对象保存为json文件，默认文件名为state_dump.json。
    支持消息对象和BaseModel的序列化。
    """
    def serialize(obj):
        if isinstance(obj, (HumanMessage, AIMessage, SystemMessage, ToolMessage)):
            return {"type": obj.__class__.__name__, "content": obj.content}
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        # if isinstance(obj, HunyuanConfig):
        #     # 假设 HunyuanConfig 有一个 to_dict 方法
        #     return obj.to_dict() if hasattr(obj, 'to_dict') else obj.__dict__
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return obj
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    state_to_save = dict(state)
    state_to_save["time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    with open(filename, "w") as f:
        json.dump(state_to_save, f, indent=2, default=serialize, ensure_ascii=False)

    
def save_graph_image(graph,filename="graph.png"):
    # 将StateGraph对象保存为PNG图像文件
    from IPython.display import Image, display
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_bytes)
        print(f"graph image saved to {filename}")
    except Exception as e:
        print(f"无法生成或保存图像: {e}")
        pass