from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from SynthMorph.state import SPAgentState
from SynthMorph.nodes import *

def last_ai_message_has_tool_calls(state: SPAgentState) -> bool:
    messages = state.get("messages", [])
    if not messages:
        return False
    last = messages[-1]
    return isinstance(last, AIMessage) and bool(getattr(last, "tool_calls", []))

def route_after_process(state: SPAgentState) -> str:
    return "NODE_predict_c_from_image" if state.get("user_input_image") else "model"

def route_after_model(state: SPAgentState) -> str:
    return "tools" if last_ai_message_has_tool_calls(state) else "check_matrix"

def route_after_check(state: SPAgentState) -> str:
    if state.get("elastic_matrix") is not None:
        return "predict_image_from_c"
    return "continue"

def build_elastic_matrix_graph():
    builder = StateGraph(SPAgentState)
    builder.add_node("NODE_Preprocessing", NODE_Preprocessing)
    builder.add_node("NODE_predict_c_from_image", NODE_predict_c_from_image)
    builder.add_node("NODE_analyze_C", NODE_analyze_C)
    builder.add_node("model", model_node)
    builder.add_node("tools", ToolNode(TOOLS))
    builder.add_node("check_matrix", check_matrix_completion)
    builder.add_node("predict_image_from_c", NODE_predict_image_from_c)
    builder.add_node("NODE_show_figure", NODE_show_figure)
    builder.add_node("NODE_Structure_Create", NODE_Structure_Create)
    builder.add_node("NODE_FEM_calc", NODE_FEM_calc)
    builder.add_node("debug", debug_NODE)

    builder.add_edge(START, "NODE_Preprocessing")
    builder.add_conditional_edges(
        "NODE_Preprocessing", 
        route_after_process, 
        {
        "NODE_predict_c_from_image": "NODE_predict_c_from_image",
        "model": "model"
        },
    )
    builder.add_edge("NODE_predict_c_from_image", "NODE_analyze_C")

    builder.add_conditional_edges(
        "model",
        route_after_model,
        {
            "tools": "tools",
            "check_matrix": "check_matrix",
        },
    )

    builder.add_edge("tools", "check_matrix")

    builder.add_conditional_edges(
        "check_matrix",
        route_after_check,
        {
            "predict_image_from_c": "predict_image_from_c",
            "continue": "model",
            "end": "debug",
        },
    )
    builder.add_edge("predict_image_from_c", "NODE_analyze_C")
    builder.add_edge("NODE_analyze_C", "NODE_show_figure")
    builder.add_edge("NODE_show_figure", "NODE_Structure_Create")
    builder.add_edge("NODE_Structure_Create", "NODE_FEM_calc")
    builder.add_edge("NODE_FEM_calc", "debug")
    builder.add_edge("debug", END)

    return builder.compile()



# agent = build_elastic_matrix_graph()

# C11=1 C12=2 C22=C11 C33=8 C13=0 C23=0
# C11= 0.07593492 C12=-0.04600782 C22=0.08051132 C33=0.00788642 C13=0.00441639 C23=-0.00441008

# C11=0.133934076 C12=0.100481599 C13=-0.062865005 C22=0.125312033 C23=-0.05942723 C33=0.038711063
# C11=0.231262883	C12=0.068509024	C13=-0.081485633 c22=0.095230301 c23=-0.024744723 c33=0.035968979