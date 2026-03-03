#!/bin/bash

env_name="freecad"

# 获取原始图片路径和计算路径参数
if [ $# -lt 2 ]; then
    echo "用法: $0 <image_path> <calc_path>"
    exit 1
fi
# 将图片路径转为绝对路径
image_path="$1"
if [[ "$image_path" != /* ]]; then
    image_path="$(pwd)/$image_path"
fi
calc_path="$2"
if [[ "$calc_path" != /* ]]; then
    calc_path="$(pwd)/$calc_path"
fi
# 保证calc_path以/结尾
case "$calc_path" in
    */) ;; # 已有/
    *) calc_path="${calc_path}/" ;;
esac
# 检查并创建计算目录
if [ ! -d "$calc_path" ]; then
    mkdir -p "$calc_path"
fi
# 查找conda路径
if command -v conda >/dev/null 2>&1; then
    CONDA_PATH=$(command -v conda)
else
    for p in "$HOME/anaconda3/bin/conda" "$HOME/miniconda3/bin/conda" "/opt/anaconda3/bin/conda" "/opt/miniconda3/bin/conda"; do
        if [ -f "$p" ]; then
            CONDA_PATH="$p"
            break
        fi
    done
fi

if [ -z "$CONDA_PATH" ]; then
    echo "未找到conda路径！"
    exit 1
fi

echo "conda路径: $CONDA_PATH"

# 激活base环境
CONDA_SH="$(dirname "$CONDA_PATH")/../etc/profile.d/conda.sh"
if [ -f "$CONDA_SH" ]; then
    source "$CONDA_SH"
    conda activate $env_name
    echo "已激活 $env_name 环境，当前python版本："
    python --version
else
    echo "未找到conda.sh，无法激活环境。"
    exit 2
fi

script1="/home/shangqing/sqdata/m3agent/bin/CAE_FE/origin/LLM_CAE_FE_1.py"
script2="/home/shangqing/sqdata/m3agent/bin/CAE_FE/origin/LLM_CAE_FE_2.py"
# script3="/home/shangqing/sqdata/m3agent/bin/CAE_FE/origin/LLM_CAE_FE_3.py"

# 进入计算路径
if [ ! -d "$calc_path" ]; then
    mkdir -p "$calc_path"
fi
cd "$calc_path"
pwd

# 执行脚本
python "$script1" "$image_path" "$calc_path"
/home/shangqing/ABAQUS22/Commands/abq2022 cae noGUI="$script2" -- "./" "${calc_path}test.txt"
# python "$script3" --image_path="$calc_path" --output_gif=output.gif



# bash run_abq.sh test.png ./test