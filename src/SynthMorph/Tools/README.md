# Tools

该目录下存储所有Agent可以共同使用的工具。
避免重复造轮子

## debugtool
调试工具。推荐在每个节点最开始调用该函数。避免出现问题

## llmresponse
用于从llm中得到json格式的文件。重试次数默认五次，可自定义