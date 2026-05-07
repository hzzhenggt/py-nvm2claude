# 反馈记录

## 版本变更记录

| 状态 | 类型 | 编号 | 描述 | 优先级 | 日期 |
|------|------|------|------|--------|------|
| ✅ | feature | 001 | 初始化项目文档 | low | 2026-05-06 |
| ✅ | fix | 002 | 修复SSE流式4个严重缺陷+补充tools/tool_choice支持 | high | 2026-05-06 |
| ✅ | fix | 003 | 修复break只退出内层循环+tool_use启动时序+新增waitress生产模式 | high | 2026-05-06 |

## 详细记录

### fix-003: 修复流式循环退出+tool_use时序+生产级服务器

**问题描述：**
`[DONE]`/`finish` 的 break 只退出内层 while 循环，外层 for 继续空读网络。tool_use content_block_start 在 name 未知时提前发送。缺少生产级 WSGI 服务器。

**修复内容：**

1. **break 退出外层循环** — 加 `done` 标志，同时退出 for 和 while
2. **tool_use content_block_start 延迟** — 等待 `td["name"]` 就绪后才发送，符合 Anthropic 协议
3. **默认 Waitress 生产服务器** — 替换 Flask dev server，支持 `--debug=true` 切回老方案
4. **支持 --host / --port 命令行参数** — `python pynvmproxy.py --host=127.0.0.1 --port=9090`
5. **依赖: waitress** — 新增 `pip install waitress`

**影响范围：** `pynvmproxy.py` - `__main__`, `stream_convert`

### fix-002: 修复SSE流式严重缺陷 + API兼容性补全

**问题描述：**
Claude Code 在分析问题时有概率卡住不继续。根因是SSE流式转换链路存在4个关键缺陷，加上tools/tool_choice完全不支持。

**修复内容：**

1. **`generate()` 异常处理** — catch Exception 发送 SSE error + message_stop
2. **`iter_lines` → `iter_content` + socket超时** — 30秒无数据断开，手动行缓冲
3. **流式 stop_reason 映射** — stop→end_turn, length→max_tokens, tool_calls→tool_use
4. **tools/tool_choice 请求转换** — Anthropic→OpenAI function calling
5. **非流式 tool_calls 响应** — OpenAI tool_calls→Anthropic tool_use blocks
6. **流式 tool_use 输出** — input_json_delta 支持，多工具并行流式
7. **thinking adaptive 类型** — 支持 enabled + adaptive
8. **top_k 透传**
