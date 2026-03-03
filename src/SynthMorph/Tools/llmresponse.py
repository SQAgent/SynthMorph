import re
import json
from typing import Annotated, Optional, Dict, Any, List, TypedDict


def get_structure_response(llm, message, max_retries = 5) -> dict:
    """直接解析LLM的回答，返回一个字典"""
    retries = 0 # 初始化重试次数
    while retries < max_retries:
        try:
            response = llm.invoke([message])
            print(f"[INFO] VLM 原始回答: {response.content}")
            try:
                match = re.search(r'```json\s*([\s\S]*?)\s*```', response.content, re.DOTALL)
                if match:
                    json_content = match.group(1)
                else:
                    json_content = response.content.strip()
                structure_response = json.loads(json_content)
                break
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON 解析失败: {e}")
                raise e  # 抛出异常以触发重试逻辑
        except Exception as e:
            retries += 1
            print(f"[ERROR] 第 {retries} 次解析失败：{e}")
            if retries >= max_retries:
                print("[ERROR] 达到最大重试次数，返回默认值。")
                structure_response = {}
                break
    return structure_response
