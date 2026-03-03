import os
import uuid
import base64
import subprocess
from typing import Annotated, Dict, Any, List

from langchain.tools import tool, InjectedToolCallId
from langchain_core.messages.tool import tool_call
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.types import Command, interrupt
from langgraph.prebuilt.interrupt import HumanInterrupt, HumanInterruptConfig

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
print(os.environ.get("VLM_MODEL"))
llm = ChatOpenAI(
    model=os.environ.get("VLM_MODEL"), 
    api_key=os.environ.get("VLM_API_KEY"),
    base_url=os.environ.get("VLM_URL"),
    timeout=None,
    temperature=0.7,
    max_retries=5,
    # max_completion_tokens = 2048,
    )

from SynthMorph.Tools.debugtool import save_state_to_json
from SynthMorph.Tools.ResponseProcess.Qwen import QwenProcess
from SynthMorph.state import SPAgentState, ElasticMatrix
from SynthMorph.Tools.Difussion import predict
from SynthMorph.Tools.ImgProcess import from_image_to_contour
from SynthMorph.Tools.img2gif import images_to_gif

def NODE_Preprocessing(state: SPAgentState) -> Dict[str, Any]:  
    """Preprocess user input to ensure image content can be extracted correctly."""  
    messages = state["messages"]  
    human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]  
      
    updates = {}  
    updates["WORKDIR"] = "./log/2D"
    os.makedirs(updates["WORKDIR"], exist_ok=True)
    updates["abq_script_path"] = "./bin/CAE_FE/run_abq.sh"
    if human_messages:  
        last_human_msg = human_messages[-1]  
        for bulk in last_human_msg.content:  
            if bulk["type"] == "image":  
                base64_str = bulk["data"]  
                image_bytes = base64.b64decode(base64_str)  
                image_path = os.path.join("./log/2D", "predictimage.png")  
                with open(image_path, "wb") as f:  
                    f.write(image_bytes)  
                updates["user_input_image"] = image_path  
      
    save_state_to_json(state, os.path.join(updates.get("WORKDIR"),"debug_preprocessing_node_state.json"))
    print("[DEBUG] NODE_Preprocessing user_input_image:", state.get("user_input_image")) 
    return updates

def NODE_predict_c_from_image(state:SPAgentState) -> SPAgentState:
    """If image_path exists, predict the elastic stiffness matrix from the image."""
    save_state_to_json(state,os.path.join(state.get("WORKDIR", "."),"debug_predict_c_from_image.json"))
    image_path = state.get("user_input_image")
    if image_path :
        state["CurrentTask"] = "predict_c_from_image"
        output_path, C_end = predict(Picture_path = image_path, C=None,output_dir=state["WORKDIR"])
        state["PredictedElasticPerformance"] = C_end
        # state["user_input_image"] = None
    else:
        state["CurrentTask"] = "predict_image_from_c"

    return state

from SynthMorph.Tools.C_cloud_pi import plot_2d_properties

def NODE_analyze_C(state:SPAgentState) -> SPAgentState:
    """Analyze the elastic stiffness matrix predicted from the image and add results to the conversation."""
    save_state_to_json(state,os.path.join(state.get("WORKDIR"),"debug_analyze_C_node_state.json"))
    if state.get("CurrentTask") == "predict_c_from_image":
        C_matrix = state.get("PredictedElasticPerformance")
    elif state.get("CurrentTask") == "predict_image_from_c":
        C_matrix = state.get("elastic_matrix")
    else:
        return state
    print("[DEBUG] C_matrix:", C_matrix)
    workdir = state.get("WORKDIR")
    save_path = os.path.join(workdir, "C_properties_polar_plot.png")
    result, save_path  = plot_2d_properties(C_matrix,save_path)
    prompt = (
        "You are a 2D material elastic stiffness matrix analysis assistant. A 2D material has a 3×3 symmetric matrix. The user computed the elastic stiffness matrix as below. Please analyze the mechanical properties of this matrix.\n"
        "The meanings of each returned value are: E1: Young's modulus along principal direction 1 (usually the x-axis), reflecting tensile stiffness in that direction. E2: Young's modulus along principal direction 2 (usually the y-axis), reflecting tensile stiffness in that direction. nu12: Poisson's ratio of principal direction 2 when principal direction 1 is stretched (i.e., y contraction when x is stretched). nu21: Poisson's ratio of principal direction 1 when principal direction 2 is stretched (i.e., x contraction when y is stretched). G12: Shear modulus on the principal plane, reflecting resistance to shear deformation. E_range: the minimum and maximum of Young's modulus over all directions in polar coordinates ([min, max]), reflecting anisotropy. nu_range: the minimum and maximum of Poisson's ratio over all directions in polar coordinates. G_range: the minimum and maximum of shear modulus over all directions in polar coordinates. E_mean: the mean Young's modulus over all directions in polar coordinates. nu_mean: the mean Poisson's ratio over all directions in polar coordinates. G_mean: the mean shear modulus over all directions in polar coordinates. polar_plot: the file path of the polar plot image, showing the directional variation of Young's modulus, Poisson's ratio, and shear modulus."
        f"Please analyze the following results.\n{result}"
        f"You should first tell the user that you have predicted the performance from the image, and the elastic stiffness matrix is {C_matrix}, then analyze these data."
    )
    if C_matrix:
        response = llm.invoke([HumanMessage(content=prompt)])
        with open(save_path, "rb") as f:
            fig_base64 = base64.b64encode(f.read()).decode("utf-8")
        think_part, tool_call_content, response_part = QwenProcess(response.content)
        state["messages"].append(
            AIMessage(content_blocks=[
                    {"type": "text", "text": response_part},  
                    # {
                    #     "type": "image",
                    #     "source_type": "base64",
                    #     "mime_type": "image/png",
                    #     "data": fig_base64,
                    #     "metadata": {"name": "polar_plot"}
                    # }
                ])
            )
        # state["PredictedElasticPerformance"] = None 
    return state
###########################################################################################################################################################
@tool
def update_matrix_elements(
    elements: Dict[str, float],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Update any number of elements in a 2D elastic stiffness matrix (no more than 6).

    Args:
        elements: {C11, C12, C13, C22, C23, C33} -> GPa
    """
    print(f"[DEBUG] Received elements for update: {elements}")

    valid_elements = {"C11", "C12", "C13", "C22", "C23", "C33"}
    invalid_keys = set(elements.keys()) - valid_elements
    if invalid_keys:
        raise ValueError(f"Invalid matrix element names: {', '.join(invalid_keys)}")
    update_data = {key: value for key, value in elements.items() if key in valid_elements}

    if not update_data:
        raise ValueError("No valid matrix elements provided for update.")

    return Command(
        update={
            **update_data,
            "messages": [
                ToolMessage(
                    content="Updated matrix elements: "
                    + ", ".join([f"{k}={v} GPa" for k, v in update_data.items()]),
                    tool_call_id=tool_call_id,
                )
            ],
        }
    )

TOOLS = [ update_matrix_elements] 

def model_node(state: SPAgentState) -> SPAgentState:
    """
    Node that calls the LLM.
    - Automatically prepends a SystemMessage
    - Uses bind_tools to allow tool calls
    """
    save_state_to_json(state,os.path.join(state.get("WORKDIR", "."),"debug_model_node_state.json"))
    print("[DEBUG] model_node")
    print("[DEBUG] model_node WORKDIR:", state.get("WORKDIR"))
    messages: List[Any] = state.get("messages", [])

    llm_messages = []  
    for msg in messages:  
        if hasattr(msg, 'content') and isinstance(msg.content, list):    
            filtered_content = []  
            for content_item in msg.content:  
                if isinstance(content_item, dict) and content_item.get("type") == "image":  
                    continue  
                elif isinstance(content_item, dict) and content_item.get("type") == "image_url":  
                    continue  
                else:  
                    filtered_content.append(content_item)  
            if filtered_content:  
                new_msg = msg.__class__(content=filtered_content)  

                if hasattr(msg, 'id'):  
                    new_msg.id = msg.id  
                if hasattr(msg, 'name'):  
                    new_msg.name = msg.name  
                llm_messages.append(new_msg)  
        else:  
 
            llm_messages.append(msg)

    SYSTEM_PROMPT = SystemMessage(
        content=(
            "You are a materials science assistant responsible for collecting the 6 independent elements of a 2D metamaterial 3×3 elastic stiffness matrix.\n\n"  # Must respond in English
            "Workflow:\n"
            "1. If the user provides all matrix elements at once (C11, C12, C13, C22, C23, C33),"
            "   call the `update_matrix_elements` tool to update the state and tell the user what matrix you recorded.\n"
            "2. If the matrix elements are incomplete, determine which elements are missing and summarize the current progress."
            "3. After the user replies, continue parsing the input and use `update_matrix_elements` to write the new values into the state.\n "
            "4. Repeat the above steps until all 6 elements are collected.\n"
            "5. After all elements are collected, you can summarize the final stiffness matrix for the user.\n\n"
            "Note: Units are GPa."
        )
    )
 
    has_system = any(isinstance(m, SystemMessage) for m in llm_messages)  
    if not has_system:  
        llm_messages = [SYSTEM_PROMPT] + llm_messages  
      
    print(f"Origin message: {len(messages)}, Filtered message: {len(llm_messages)}")  
    response = llm.bind_tools(TOOLS).invoke(llm_messages)
    think_part, tool_call_content, response_part = QwenProcess(response.content)
    print("[DEBUG] LLM think_part:", think_part)
    print("[DEBUG] LLM response_part:", response_part)
    if not tool_call_content:
        state["messages"].append(AIMessage(content=response_part))
    else:
        tool_call_object = tool_call(
            name=tool_call_content["name"],
            args=tool_call_content["arguments"],
            id=str(uuid.uuid4()) 
        )
        state["messages"].append(AIMessage(content = response_part, tool_calls=[tool_call_object]))
    return state


def NODE_predict_image_from_c(state: SPAgentState) -> Dict[str, Any]:
    """
    Node that generates a predicted structure image from the elastic stiffness matrix.
    """
    save_state_to_json(state,os.path.join(state.get("WORKDIR", "."),"debug_predict_image_from_c.json"))
    state["CurrentTask"] = "predict_image_from_c"
    C_matrix = state.get("elastic_matrix")
    print(C_matrix)
    if C_matrix is None:
        raise ValueError("The elastic stiffness matrix is incomplete and cannot be used to generate a predicted structure image.")

    try:
        output_path, C_end = predict(Picture_path = None, C=C_matrix,output_dir=state["WORKDIR"])
        image_path = os.path.join(state.get("WORKDIR"), "generate.png")
        state["PredictedStructureImage"] = image_path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Generated image path is invalid or the file does not exist: {image_path}")
        return state

    except Exception as e:
        print(f"[ERROR] Error while generating the predicted structure image: {e}")
        raise RuntimeError("Failed to generate the predicted structure image. Please check the input data and model configuration.") from e
    

def check_matrix_completion(state: SPAgentState) -> Command:  
    """  
    Check whether the stiffness matrix is complete after each tool execution.  
    - If all 6 elements exist, write to elastic_matrix  
    - If information is incomplete, call interrupt to ask and wait for human input.  
    """  
    matrix = ElasticMatrix(  
        C11=state.get("C11"),  
        C12=state.get("C12"),  
        C13=state.get("C13"),  
        C22=state.get("C22"),  
        C23=state.get("C23"),  
        C33=state.get("C33"),  
    )  
    missing_elements = []  
    for field_name in ["C11", "C12", "C13", "C22", "C23", "C33"]:  
        if getattr(matrix, field_name) is None:  
            missing_elements.append(field_name)  

    if missing_elements:  
        question = f"""
        📋 Elastic stiffness matrix collection progress

        Currently recorded elements:
        {chr(10).join(
            f"{field}={getattr(matrix, field)} GPa"
            for field in ["C11", "C12", "C13", "C22", "C23", "C33"]
            if getattr(matrix, field) is not None
        )}

        Missing elements:

        Please provide the values for the missing elements {missing_elements} (unit: GPa, e.g., C11=120).
        """
  
        print(f"[CHECK] Matrix not complete, missing: {missing_elements}")  
          
        interrupt({  
            "action_request": {  
                "action": "Provide stiffness matrix elements",  
                "args": {
                    "question": question,
                }  
            },  
        }) 
        return Command(update={})  
    else:  
        print("[CHECK] Matrix complete, writing elastic_matrix.")  
        return Command(update={"elastic_matrix": matrix.to_matrix()})

def NODE_show_figure(state: SPAgentState) -> SPAgentState:
    "Node that displays the predicted structure."
    print("[DEBUG] NODE_show_figure")
    if state.get("CurrentTask") == "predict_c_from_image":
        image = state.get("user_input_image")
        gif_path = None
        # state["user_input_image"] = None

    elif state.get("CurrentTask") == "predict_image_from_c":
        image = state.get("PredictedStructureImage")
        gif_path = "./log/2D/generation_process/generation_process.gif"
    else:
        return state
    content_blocks=[{"type": "text", "text": f"Here is the image of the structure after optimization:"}]

    if gif_path:
        with open(gif_path, "rb") as f:
            gif_base64 = base64.b64encode(f.read()).decode("utf-8")
        content_blocks.append(
            {
                "type": "image",
                "source_type": "base64",
                "mime_type": "image/gif",
                "data": gif_base64,
                "metadata": {
                    "name": "optimized_structure"
                }
            }   
        )

    if image:
        with open(image, "rb") as f:
            fig_base64 = base64.b64encode(f.read()).decode("utf-8")
        content_blocks.append(
            {
                "type": "image",
                "source_type": "base64",
                "mime_type": "image/jpeg",
                "data": fig_base64,
                "metadata": {
                    "name": "optimized_structure"
                }
            }  
        )
    propertiesPNG = os.path.join(state.get("WORKDIR"), "C_properties_polar_plot.png")
    if os.path.exists(propertiesPNG):
        with open(propertiesPNG, "rb") as f:
            prop_base64 = base64.b64encode(f.read()).decode("utf-8")
        content_blocks.append(
            {
                "type": "image",
                "source_type": "base64",
                "mime_type": "image/png",
                "data": prop_base64,
                "metadata": {
                    "name": "C_properties_polar_plot"
                }
            }  
        )
    state["messages"].append(AIMessage(content_blocks=content_blocks))

    return state

def NODE_Structure_Create(state:SPAgentState)-> SPAgentState:
    """Pre-processing step for finite element calculation."""
    review_request = HumanInterrupt(
        action_request={
            "action": "CAE Review",
            "args": {
                "question": "Do you want to create a CAE model?",
            }
        },
        config=HumanInterruptConfig(
            allow_ignore=True,  
            allow_accept=True, 
            allow_edit=False,
            allow_respond=False, 
        ),
        description="CAE model create",
    )

    resp = interrupt([review_request])[0]
    if resp["type"] == "ignore":
        state["messages"].append(AIMessage(content="Skip the finite element simulation."))
        return state

    if state.get("CurrentTask") == "predict_c_from_image":
        image_path = state.get("user_input_image")
    elif state.get("CurrentTask") == "predict_image_from_c":
        image_path = state.get("PredictedStructureImage")
    else:
        image_path = None
    calc_path = os.path.join(state.get("WORKDIR"),"CAE_FE")
    os.makedirs(calc_path, exist_ok=True)
    if not image_path or not os.path.exists(image_path):
            state["messages"].append(AIMessage(content="Structure image not found. Finite element simulation cannot proceed."))
            return state

    from_image_to_contour(image_path, calc_path)
    contour_path = os.path.join(calc_path, "test.txt")
    contour_path = os.path.abspath(contour_path)
    if not os.path.exists(contour_path):
        state["messages"].append(AIMessage(content="Failed to generate contour file. Finite element simulation cannot proceed."))
        return state
    
    Abq_Create_model_path = os.path.abspath("./bin/CAE_FE/Abq_Create_model.py")
    cmd = ["/home/shangqing/ABAQUS22/Commands/abq2022", "cae", f"noGUI={Abq_Create_model_path}","--", "./", contour_path]
    print("FEM CMD:", cmd)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=calc_path)
        try:
            with open(os.path.join(calc_path, "abq.log"), "w") as logf:
                logf.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
        except Exception as logerr:
            print("[WARN] error write abq.log:", logerr)
        content_blocks = [{"type": "text", "text": "The structure in the figure has been reconstructed as a 3D model."}]
        fig_path = os.path.join(calc_path, f"model.png")
        with open(fig_path, "rb") as f:
            fig_base64 = base64.b64encode(f.read()).decode("utf-8")
        content_blocks.append({
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": "image/png",
                    "data": fig_base64,
                    "metadata": {"name": f"Structure"}
                })
        state["messages"].append(AIMessage(content_blocks=content_blocks))
    except Exception as e:
        print("error", e)
        state["messages"].append(AIMessage(content=f"Error during model creation: {e}"))
    return state

def NODE_FEM_calc(state: SPAgentState) -> SPAgentState:
    """Finite Element Method calculation node, supports Human in the loop review before performing FEM calculation."""
    print("[DEBUG] NODE_FEM_calc")
    # Human in the loop review
    review_request = HumanInterrupt(
        action_request={
            "action": "FEM review",
            "args": {
                "question": "Do you want to perform finite element simulation for this structure?",
                # "structure_info": state.get("PredictedStructureImage")
            }
        },
        config=HumanInterruptConfig(
            allow_accept=True,    # Show "Accept" button
            allow_ignore=True,    # Show "Ignore" button
            allow_edit=False,     # No edit
            allow_respond=False   # No text input
        ),
        description="Please choose whether to perform finite element simulation.",
        # id="FEM calculation review"
    )

    resp = interrupt([review_request])[0]
    print(resp["type"])
    if resp["type"] == "ignore":
        state["messages"].append(AIMessage(content="Skip the finite element simulation."))
        return state


    if state.get("CurrentTask") == "predict_c_from_image":
        image_path = state.get("user_input_image")
    elif state.get("CurrentTask") == "predict_image_from_c":
        image_path = state.get("PredictedStructureImage")
    else:
        image_path = None

    calc_path = os.path.join(state.get("WORKDIR"),"CAE_FE")
    os.makedirs(calc_path, exist_ok=True)
    script_path = state.get("abq_script_path", "./bin/CAE_FE/run_abq.sh")
    if not image_path or not os.path.exists(image_path):
        state["messages"].append(AIMessage(content="Structure image not found. Finite element simulation cannot proceed."))
        return state
    # 2. FEM calculation NEW
    from_image_to_contour(image_path, calc_path)
    contour_path = os.path.join(calc_path, "test.txt")
    contour_path = os.path.abspath(contour_path)
    if not os.path.exists(contour_path):
        state["messages"].append(AIMessage(content="Failed to generate contour file. Finite element simulation cannot proceed."))
        return state
    
    AbqScript_path = os.path.abspath("./bin/CAE_FE/AbqScript.py")
    cmd = ["/home/shangqing/ABAQUS22/Commands/abq2022", "cae", f"noGUI={AbqScript_path}","--", contour_path]
    print("FEM CMD:", cmd)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=calc_path)
        try:
            with open(os.path.join(calc_path, "abq.log"), "w") as logf:
                logf.write("STDOUT:\n" + result.stdout + "\n\nSTDERR:\n" + result.stderr)
        except Exception as logerr:
            print("[WARN] error write abq.log:", logerr)
        content_blocks = [{"type": "text", "text": "The structure in the figure has been reconstructed as a 3D model and the compression-torsion process has been simulated:"}]
        image_files = [os.path.join(calc_path, f"frame_{i}.png") for i in range(10)] 

        gif_path = os.path.join(calc_path, "animation.gif")
        images_to_gif(
            image_files=image_files,
            output_gif=gif_path,
            duration=300,  
            loop=0
        )
        with open(gif_path, "rb") as f:
            gif_base64 = base64.b64encode(f.read()).decode("utf-8")
        content_blocks.append({
            "type": "image",
            "source_type": "base64",
            "mime_type": "image/gif",
            "data": gif_base64,
            "metadata": {"name": "output.gif"}
        })
        state["messages"].append(AIMessage(content_blocks=content_blocks))
        content_blocks = []
        for fig_path in image_files:
            if os.path.exists(fig_path):
                with open(fig_path, "rb") as f:
                    fig_base64 = base64.b64encode(f.read()).decode("utf-8")
                content_blocks.append({
                    "type": "image",
                    "source_type": "base64",
                    "mime_type": "image/png",
                    "data": fig_base64,
                    "metadata": {"name": f"fem_result_{image_files.index(fig_path)}"}
                })
        state["messages"].append(AIMessage(content_blocks=content_blocks))
    except Exception as e:
        print("error", e)
        state["messages"].append(AIMessage(content=f"Error during finite element simulation: {e}"))

    return state

def debug_NODE(state: SPAgentState) -> Dict[str, Any]:
    """Debug node that saves state to a JSON file."""
    if state.get("CurrentTask") == "predict_c_from_image":
        state["user_input_image"] = None  
    elif state.get("CurrentTask") == "predict_image_from_c":
        pass
    else:
        pass
    save_state_to_json(state,os.path.join(state.get("WORKDIR", "."),"DEBUG.json"))
    return state