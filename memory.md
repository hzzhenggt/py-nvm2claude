# 项目记忆

## 项目概述
- NVIDIA NIM API → Anthropic Claude Messages API 代理服务器
- 单文件 Python 项目: `pynvmproxy.py` (Flask + requests)
- 提供 Web UI 配置界面（单页内嵌HTML）
- 目标: 兼容 Claude Code CLI

## 技术要点
- Python 3.12.3
- 上游: NVIDIA NIM `/chat/completions`
- 下游: Anthropic `/v1/messages` 格式
- 支持 SSE 流式响应、thinking/reasoning 转换
- 模型别名映射 (Claude模型名 → NIM模型名)
- **v2 新增**: tools/tool_choice 支持、流式 tool_use、SSE异常恢复、socket超时

## 关键文件
- `pynvmproxy.py`: 主服务代码 (~1031行)
- `start.bat`: Windows Terminal 启动脚本
- `.python-version`: Python版本锁定
