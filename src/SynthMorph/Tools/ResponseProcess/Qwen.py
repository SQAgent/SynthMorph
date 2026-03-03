import re
import json



def QwenProcess(llm_response: str):
    """
    Qwen 模型的预处理方案，删除思考过程并提取工具调用内容
    llm_response: str Qwen 模型的文本响应
    </think> 标签前面是思考过程
    <tool_call></tool_call> 标签内是工具调用内容
    其余部分是回复内容

    return:
    think_part: str 思考过程
    tool_call_content: dict or None 工具调用内容，若无则为 None
    response_part: str 回复内容
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
        tool_call_content = match_tool.group(1).strip()  # 仅提取标签内的内容
        try:
            tool_call_content = json.loads(tool_call_content)  # 尝试将内容解析为 JSON
        except json.JSONDecodeError:
            tool_call_content = None  # 如果解析失败，返回 None
        before_tool = others_part[:match_tool.start()].strip()
        after_tool = others_part[match_tool.end():].strip()
        response_part = "\n".join(filter(None, [before_tool, after_tool]))
    else:
        tool_call_content = None
        response_part = others_part.strip()
    return think_part, tool_call_content, response_part


example = """用update_matrix_elements传递所有元素后，再向用户反馈收集完成，并展示矩阵。 </think>
<tool_call> {"name": "update_matrix_elements", "arguments": {"elements": {"C11": 1, "C12": 2, "C13": 0, "C22": 1, "C23": 0, "C33": 8}}} </tool_call>
已为您记录完整的3×3弹性刚度矩阵（单位：GPa）：
C = [ [1, 2, 0], [2, 1, 0], [0, 0, 8] ]
所有6个独立元素已完整收集，矩阵验证通过。
"""
if __name__ == "__main__":
    think_part, tool_call_content, response_part = QwenProcess(example)
    print("思考过程：", think_part)
    print("工具调用：", tool_call_content)
    print(type(tool_call_content))
    print("回复内容：", response_part)

