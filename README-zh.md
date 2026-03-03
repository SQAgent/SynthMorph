# SynthMorph
> 论文 "Semantic Structure-Property Alignment via Cross-Modal Contrastive Learning in Mechanical Metamaterials" 的智能体部分开源代码
> 基于langchain开发，
> 有限元计算部分使用abaqus2022版本（不包括在当前项目，请自行安装）
> 前端配合 https://github.com/SQAgent/agent-chat-ui 使用， Langchain 原生 `agent-chat-ui` 项目不支持AI显示图片和3D结构。

# run
```bash
cd Path/To/Your/SynthMorph
pip install -e . "langgraph-cli[inmem]"
langgraph dev
```
# update to GitHub
```shell
git add .
git status 
git commit -m "update"
git push -u origin main 
```