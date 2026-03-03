# SynthMorph

Open-source agent code for the paper:
**"Semantic Structure-Property Alignment via Cross-Modal Contrastive Learning in Mechanical Metamaterials"**.

## Overview

- Built with LangChain/LangGraph.
- The finite element analysis (FEA) workflow uses **Abaqus 2022**.
- Abaqus is **not included** in this repository; please install it separately.
- For the frontend, use: https://github.com/SQAgent/agent-chat-ui
- The original LangChain `agent-chat-ui` does not support AI display for images and 3D structures.

## Run

```bash
cd Path/To/Your/SynthMorph
pip install -e . "langgraph-cli[inmem]"
langgraph dev
```

## Update to GitHub

```bash
git add .
git status
git commit -m "update"
git push -u SynthMorph main
```
