import subprocess
import os

def run_abq_script(image_path, calc_path, script_path=None):
    """
    执行 run_abq.sh 脚本
    :param image_path: 原始图片路径（可为相对或绝对路径）
    :param calc_path: 计算目录路径（可为相对或绝对路径）
    :param script_path: run_abq.sh 路径，默认与本py文件同目录
    :return: (returncode, stdout, stderr)
    """
    if script_path is None:
        script_path = os.path.join(os.path.dirname(__file__), 'run_abq.sh')
    # 确保脚本有执行权限
    if not os.access(script_path, os.X_OK):
        os.chmod(script_path, 0o755)
    cmd = [script_path, image_path, calc_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

# 示例用法：
# code, out, err = run_abq_script('test.png', './test')
# print('返回码:', code)
# print('输出:', out)
# print('错误:', err)
