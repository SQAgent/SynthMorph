import re
import json



def QwenProcess(llm_response: str):
    """
    Preprocessing scheme for Qwen model: remove thinking process and extract tool call content.
    llm_response: str Qwen model text response
    Content before </think> tag is the thinking process.
    Content inside <tool_call></tool_call> tags is the tool call content.
    The rest is the reply content.

    return:
    think_part: str Thinking process
    tool_call_content: dict or None Tool call content, None if not present
    response_part: str Reply content
    """
    match = re.match(r'(.*?)</think>', llm_response, re.DOTALL)
    if match:
        think_part = match.group(1).strip()
        others_part = llm_response[match.end():].strip()
    else:
        think_part = ""
        others_part = llm_response.strip()
    
    match_tool = re.search(r'<tool_call>(.*?)</tool_call>', others_part, re.DOTALL)
    if match_tool:
        tool_call_content = match_tool.group(1).strip()  # Only extract content inside the tag
        try:
            tool_call_content = json.loads(tool_call_content)  # Try to parse content as JSON
        except json.JSONDecodeError:
            tool_call_content = None  # If parsing fails, return None
        before_tool = others_part[:match_tool.start()].strip()
        after_tool = others_part[match_tool.end():].strip()
        response_part = "\n".join(filter(None, [before_tool, after_tool]))
    else:
        tool_call_content = None
        response_part = others_part.strip()
    return think_part, tool_call_content, response_part


example = """After passing all elements with update_matrix_elements, feedback is given to the user that collection is complete and the matrix is displayed. </think>
<tool_call> {"name": "update_matrix_elements", "arguments": {"elements": {"C11": 1, "C12": 2, "C13": 0, "C22": 1, "C23": 0, "C33": 8}}} </tool_call>
The complete 3×3 elastic stiffness matrix (unit: GPa) has been recorded for you:
C = [ [1, 2, 0], [2, 1, 0], [0, 0, 8] ]
All 6 independent elements have been fully collected, matrix verification passed.
"""
if __name__ == "__main__":
    think_part, tool_call_content, response_part = QwenProcess(example)
    print("Thinking process:", think_part)
    print("Tool call:", tool_call_content)
    print(type(tool_call_content))
    print("Reply content:", response_part)

