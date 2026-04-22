#!/usr/bin/env python3
"""
NVIDIA NIM -> Anthropic Claude API 代理服务器 v2
专门针对 Claude Code CLI 优化兼容性

修复:
  - ANTHROPIC_BASE_URL 路径双重拼接问题（/v1/messages/v1/messages）
  - anthropic-version / anthropic-beta 头透传
  - Claude Code 特有的 extended thinking 格式
  - 完整请求路径调试日志

用法:
  pip install flask requests
  export NVIDIA_API_KEY="nvapi-xxx"
  python nim_to_claude_proxy.py

Claude Code 配置 (~/.claude/settings.json):
  "env": {
    "ANTHROPIC_BASE_URL": "http://你的IP:8899",   <- 注意：不要加 /v1/messages
    "ANTHROPIC_AUTH_TOKEN": "sk-proxy-change-me"
  }
"""

import json, time, uuid, logging, threading, os, sys, re
from datetime import datetime
from collections import deque

from flask import Flask, request, Response, jsonify

CONFIG = {
    "nvidia_base_url": "https://integrate.api.nvidia.com/v1",
    "nvidia_api_key": os.getenv("NVIDIA_API_KEY", ""),
    "proxy_api_key": os.getenv("PROXY_API_KEY", "sk-proxy-change-me"),
    "default_model": "z-ai/glm-5.1",
    "server_port": int(os.getenv("SERVER_PORT", "8899")),
    "enable_thinking": True,
    "max_tokens": 16384,
    "temperature": 1.0,
    "top_p": 1.0,
    "model_aliases": {
        "claude-sonnet-4-20250514": "z-ai/glm-5.1",
        "claude-opus-4-20250514": "z-ai/glm-5.1",
        "claude-3-5-sonnet-20241022": "z-ai/glm-5.1",
        "claude-3-5-haiku-20241022": "z-ai/glm-5.1",
        "claude-3-opus-20240229": "z-ai/glm-5.1",
        "claude-3-haiku-20240307": "z-ai/glm-5.1",
    },
}
CONFIG_LOCK = threading.Lock()

_USE_COLOR = sys.stderr.isatty() and os.getenv("NO_COLOR") is None
_C = lambda c: f"\033[{c}m" if _USE_COLOR else ""
_RST = _C(0)

logging.basicConfig(
    level=logging.INFO,
    format=f"{_C(90)}%(asctime)s{_RST} | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nim-proxy")
logging.getLogger("werkzeug").setLevel(logging.WARNING)

USAGE_LOG = deque(maxlen=500)
USAGE_LOCK = threading.Lock()
TOTAL_STATS = {"input": 0, "output": 0, "thinking": 0, "requests": 0}


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff"
              or "\u3000" <= c <= "\u303f"
              or "\uff00" <= c <= "\uffef")
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


def log_usage(model, input_tokens, output_tokens, request_id, thinking_tokens=0):
    entry = {
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rid": request_id,
        "model": model,
        "in": input_tokens,
        "out": output_tokens,
        "think": thinking_tokens,
        "total": input_tokens + output_tokens + thinking_tokens,
    }
    with USAGE_LOCK:
        USAGE_LOG.append(entry)
        TOTAL_STATS["input"] += input_tokens
        TOTAL_STATS["output"] += output_tokens
        TOTAL_STATS["thinking"] += thinking_tokens
        TOTAL_STATS["requests"] += 1
    logger.info(
        f"{_C(36)}Token{_RST} | {request_id[:16]}... | {_C(33)}{model}{_RST} | "
        f"in={_C(32)}{input_tokens}{_RST} out={_C(32)}{output_tokens}{_RST} "
        f"think={_C(90)}{thinking_tokens}{_RST} total={entry['total']}"
    )


def resolve_model(model_name: str) -> str:
    if not model_name:
        with CONFIG_LOCK:
            return CONFIG["default_model"]
    with CONFIG_LOCK:
        aliases = CONFIG.get("model_aliases", {})
        default = CONFIG["default_model"]
    return aliases.get(model_name, model_name if "/" in model_name else default)


def anthropic_to_openai(body: dict) -> dict:
    messages = []
    system = body.get("system")
    if system:
        if isinstance(system, list):
            system = "\n".join(
                b.get("text", "") for b in system if b.get("type") == "text"
            )
        messages.append({"role": "system", "content": system})

    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "thinking":
                        pass
                    elif btype == "tool_use":
                        parts.append(f"[tool_use: {block.get('name', '')}({json.dumps(block.get('input', {}), ensure_ascii=False)})]")
                    elif btype == "tool_result":
                        content_val = block.get("content", "")
                        if isinstance(content_val, list):
                            content_val = "\n".join(
                                b.get("text", "") for b in content_val
                                if isinstance(b, dict) and b.get("type") == "text"
                            )
                        parts.append(f"[tool_result: {content_val}]")
                    elif btype == "image":
                        parts.append("[image]")
                    else:
                        parts.append(block.get("text", str(block)))
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(p for p in parts if p)
        messages.append({"role": role, "content": content or ""})

    raw_model = body.get("model", "")
    model = resolve_model(raw_model)
    if model != raw_model and raw_model:
        logger.info(f"{_C(90)}   Model map: {raw_model} -> {model}{_RST}")

    with CONFIG_LOCK:
        cfg = dict(CONFIG)

    result = {
        "model": model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", cfg["max_tokens"]),
        "temperature": body.get("temperature", cfg["temperature"]),
        "top_p": body.get("top_p", cfg["top_p"]),
        "stream": body.get("stream", False),
    }

    thinking = body.get("thinking", {})
    enable_think = False
    if isinstance(thinking, dict):
        enable_think = thinking.get("type") == "enabled"
    if not enable_think:
        enable_think = cfg.get("enable_thinking", False)

    if enable_think:
        result["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "clear_thinking": False,
            }
        }
        budget = thinking.get("budget_tokens") if isinstance(thinking, dict) else None
        if budget and isinstance(budget, int):
            result["max_tokens"] = max(result["max_tokens"], budget)

    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]

    return result


def openai_to_anthropic(oai: dict, model: str, rid: str, input_est: int) -> dict:
    choice = oai.get("choices", [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "")

    blocks = []
    think_tok = 0
    if reasoning:
        blocks.append({
            "type": "thinking",
            "thinking": reasoning,
            "signature": f"EpR_{uuid.uuid4().hex[:40]}sig"
        })
        think_tok = estimate_tokens(reasoning)
    blocks.append({"type": "text", "text": text})

    usage = oai.get("usage", {})
    in_tok = usage.get("prompt_tokens") or input_est
    out_tok = usage.get("completion_tokens") or estimate_tokens(text)

    log_usage(model, in_tok, out_tok, rid, think_tok)

    stop_map = {"stop": "end_turn", "length": "max_tokens", "content_filter": "end_turn"}
    finish = choice.get("finish_reason", "stop")

    return {
        "id": rid,
        "type": "message",
        "role": "assistant",
        "content": blocks,
        "model": model,
        "stop_reason": stop_map.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok + think_tok,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }


def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_convert(resp, model, rid, input_est):
    output_text = ""
    thinking_text = ""
    content_started = False
    thinking_open = False
    block_idx = 0
    think_idx = 0
    text_idx = 0
    thinking_sig = f"EpR_{uuid.uuid4().hex[:40]}sig"

    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": rid, "type": "message", "role": "assistant",
            "content": [], "model": model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {
                "input_tokens": input_est, "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
    })
    yield _sse("ping", {"type": "ping"})

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line or not raw_line.startswith("data: "):
            continue
        payload = raw_line[6:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = chunk.get("choices")
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        finish = choices[0].get("finish_reason")

        reasoning = delta.get("reasoning_content")
        if reasoning:
            if not thinking_open:
                thinking_open = True
                think_idx = block_idx
                block_idx += 1
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": think_idx,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    },
                })
            thinking_text += reasoning
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": think_idx,
                "delta": {"type": "thinking_delta", "thinking": reasoning},
            })

        text = delta.get("content")
        if text:
            if thinking_open:
                yield _sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": think_idx,
                    "delta": {"type": "signature_delta", "signature": thinking_sig},
                })
                yield _sse("content_block_stop", {
                    "type": "content_block_stop", "index": think_idx,
                })
                thinking_open = False
            if not content_started:
                content_started = True
                text_idx = block_idx
                block_idx += 1
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": text_idx,
                    "content_block": {"type": "text", "text": ""},
                })
            output_text += text
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": text_idx,
                "delta": {"type": "text_delta", "text": text},
            })

        if finish:
            break

    if thinking_open:
        yield _sse("content_block_delta", {
            "type": "content_block_delta",
            "index": think_idx,
            "delta": {"type": "signature_delta", "signature": thinking_sig},
        })
        yield _sse("content_block_stop", {
            "type": "content_block_stop", "index": think_idx,
        })
    if content_started:
        yield _sse("content_block_stop", {
            "type": "content_block_stop", "index": text_idx,
        })
    else:
        yield _sse("content_block_start", {
            "type": "content_block_start",
            "index": block_idx,
            "content_block": {"type": "text", "text": ""},
        })
        yield _sse("content_block_stop", {
            "type": "content_block_stop", "index": block_idx,
        })

    out_tok = estimate_tokens(output_text)
    think_tok = estimate_tokens(thinking_text)
    log_usage(model, input_est, out_tok, rid, think_tok)

    yield _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": out_tok + think_tok},
    })
    yield _sse("message_stop", {"type": "message_stop"})


app = Flask(__name__)
app.logger.setLevel(logging.WARNING)


def _auth_check():
    with CONFIG_LOCK:
        expected = CONFIG["proxy_api_key"]
    if not expected:
        return True
    key = (request.headers.get("x-api-key")
           or request.headers.get("Authorization", "").removeprefix("Bearer ").strip())
    return key == expected


@app.before_request
def log_request_path():
    if request.path.startswith("/api/") or request.path == "/":
        return
    logger.info(
        f"{_C(90)}<- {request.method} {request.path} | "
        f"x-api-key={'Y' if request.headers.get('x-api-key') else 'N'} | "
        f"anthropic-version={request.headers.get('anthropic-version', '-')} | "
        f"anthropic-beta={request.headers.get('anthropic-beta', '-')}{_RST}"
    )


def handle_messages():
    if not _auth_check():
        return jsonify({"type": "error", "error": {"type": "authentication_error",
                        "message": "Invalid API key"}}), 401

    body = request.get_json(force=True)
    rid = f"msg_{uuid.uuid4().hex[:24]}"
    raw_model = body.get("model", "")
    model = resolve_model(raw_model)
    is_stream = body.get("stream", False)

    msg_count = len(body.get("messages", []))
    system_len = 0
    sys_val = body.get("system")
    if sys_val:
        system_len = len(json.dumps(sys_val, ensure_ascii=False)) if isinstance(sys_val, list) else len(str(sys_val))
    thinking_cfg = body.get("thinking", {})

    logger.info(
        f"{_C(35)}-> REQ{_RST} | {rid[:16]}... | "
        f"model={_C(33)}{raw_model}{_RST}->{_C(32)}{model}{_RST} | "
        f"stream={is_stream} | msgs={msg_count} | "
        f"sys={system_len}c | thinking={thinking_cfg}"
    )

    oai_body = anthropic_to_openai(body)
    input_est = estimate_tokens(json.dumps(oai_body.get("messages", []), ensure_ascii=False))

    with CONFIG_LOCK:
        base_url = CONFIG["nvidia_base_url"].rstrip("/")
        api_key = CONFIG["nvidia_api_key"]

    if not api_key:
        logger.error(f"{_C(31)}X NVIDIA API Key not set!{_RST}")
        return jsonify({"type": "error", "error": {"type": "authentication_error",
                        "message": "NVIDIA API Key not configured. Set via Web UI or NVIDIA_API_KEY env var."
                        }}), 500

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    extra = oai_body.pop("extra_body", None)
    if extra:
        oai_body.update(extra)

    import requests as http_req

    target_url = f"{base_url}/chat/completions"
    logger.info(f"{_C(90)}   Upstream: POST {target_url} | body keys={list(oai_body.keys())}{_RST}")

    try:
        upstream = http_req.post(
            target_url,
            headers=headers,
            json=oai_body,
            stream=is_stream,
            timeout=300,
        )
    except http_req.exceptions.Timeout:
        logger.error(f"{_C(31)}X Upstream timeout (300s){_RST}")
        return jsonify({"type": "error", "error": {"type": "api_error",
                        "message": "Upstream request timed out (300s)"}}), 504
    except Exception as e:
        logger.error(f"{_C(31)}X Upstream connection failed: {e}{_RST}")
        return jsonify({"type": "error", "error": {"type": "api_error",
                        "message": str(e)}}), 502

    if upstream.status_code != 200:
        err_text = upstream.text[:500]
        logger.error(f"{_C(31)}X Upstream {upstream.status_code}: {err_text}{_RST}")
        return jsonify({"type": "error", "error": {"type": "api_error",
                        "message": f"Upstream {upstream.status_code}: {err_text}"
                        }}), upstream.status_code

    logger.info(f"{_C(32)}OK Upstream 200{_RST} | stream={is_stream}")

    anthropic_headers = {
        "x-request-id": rid,
        "anthropic-version": "2023-06-01",
    }

    if is_stream:
        def generate():
            try:
                yield from stream_convert(upstream, model, rid, input_est)
            finally:
                upstream.close()
        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                **anthropic_headers,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        oai_resp = upstream.json()
    except Exception:
        return jsonify({"type": "error", "error": {"type": "api_error",
                        "message": "Invalid JSON from upstream"}}), 502

    result = openai_to_anthropic(oai_resp, model, rid, input_est)
    resp = jsonify(result)
    for k, v in anthropic_headers.items():
        resp.headers[k] = v
    return resp


@app.route("/v1/messages", methods=["POST"])
def messages_endpoint():
    return handle_messages()


@app.route("/v1/messages/v1/messages", methods=["POST"])
def messages_double_path():
    logger.warning(
        f"{_C(33)}WARNING: Double path /v1/messages/v1/messages detected!{_RST}\n"
        f"{_C(33)}  Fix ANTHROPIC_BASE_URL to: http://YOUR_IP:8899 (no /v1/messages){_RST}"
    )
    return handle_messages()


@app.route("/messages", methods=["POST"])
def messages_no_prefix():
    logger.info(f"{_C(90)}   Path /messages (no /v1 prefix), handling normally{_RST}")
    return handle_messages()


@app.route("/v1/models", methods=["GET"])
def list_models():
    models = [
        "z-ai/glm-5.1",
        "deepseek-ai/deepseek-r1",
        "meta/llama-4-maverick-17b-128e-instruct",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "qwen/qwen2.5-72b-instruct",
    ]
    return jsonify({
        "data": [{"id": m, "object": "model", "created": 0, "owned_by": "nvidia-nim"}
                 for m in models],
        "object": "list",
    })


@app.route("/v1/messages/count_tokens", methods=["POST"])
@app.route("/v1/messages/v1/messages/count_tokens", methods=["POST"])
def count_tokens_endpoint():
    body = request.get_json(force=True)
    messages = body.get("messages", [])
    system = body.get("system", "")
    est = estimate_tokens(json.dumps(messages, ensure_ascii=False))
    if system:
        est += estimate_tokens(json.dumps(system, ensure_ascii=False) if isinstance(system, list) else system)
    return jsonify({"input_tokens": est})


@app.route("/api/config", methods=["GET"])
def get_config():
    with CONFIG_LOCK:
        safe = {k: v for k, v in CONFIG.items() if k != "model_aliases"}
        safe["model_aliases"] = dict(CONFIG.get("model_aliases", {}))
        if safe.get("nvidia_api_key"):
            k = safe["nvidia_api_key"]
            safe["nvidia_api_key"] = k[:8] + "..." + k[-4:] if len(k) > 12 else "***"
    return jsonify(safe)


@app.route("/api/config", methods=["POST"])
def set_config():
    data = request.get_json(force=True)
    with CONFIG_LOCK:
        for key in ["nvidia_base_url", "nvidia_api_key", "proxy_api_key",
                     "default_model", "enable_thinking"]:
            if key in data and data[key] != "":
                CONFIG[key] = data[key]
        if "max_tokens" in data:
            CONFIG["max_tokens"] = int(data["max_tokens"])
        if "temperature" in data:
            CONFIG["temperature"] = float(data["temperature"])
        if "top_p" in data:
            CONFIG["top_p"] = float(data["top_p"])
        if "model_aliases" in data and isinstance(data["model_aliases"], dict):
            CONFIG["model_aliases"].update(data["model_aliases"])
    logger.info(f"{_C(32)}Config updated{_RST}")
    return jsonify({"status": "ok"})


@app.route("/api/usage", methods=["GET"])
def get_usage():
    with USAGE_LOCK:
        return jsonify({
            "total": dict(TOTAL_STATS),
            "recent": list(USAGE_LOG)[-50:],
        })


WEB_PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIM to Claude Proxy</title>
<style>
:root{--bg:#0f172a;--card:#1e293b;--accent:#38bdf8;--accent2:#818cf8;
      --text:#e2e8f0;--muted:#94a3b8;--border:#334155;--green:#22c55e;--red:#ef4444;--warn:#f59e0b}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);
     min-height:100vh;padding:20px}
.container{max-width:960px;margin:0 auto}
h1{font-size:1.5rem;margin-bottom:4px;background:linear-gradient(135deg,var(--accent),var(--accent2));
   -webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:var(--muted);font-size:.85rem;margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:24px;margin-bottom:20px}
.card h2{font-size:1.1rem;margin-bottom:16px;color:var(--accent)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.field{margin-bottom:14px}
.field label{display:block;font-size:.78rem;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px}
.field input,.field select,.field textarea{width:100%;padding:10px 12px;background:#0f172a;
     border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:.9rem;outline:none;transition:border .2s;font-family:inherit}
.field input:focus,.field textarea:focus{border-color:var(--accent)}
.field textarea{min-height:80px;resize:vertical;font-family:'Cascadia Code','Fira Code',monospace;font-size:.8rem}
.check-row{display:flex;align-items:center;gap:8px;margin:8px 0}
.check-row input{accent-color:var(--accent)}
button{padding:10px 24px;border:none;border-radius:8px;font-size:.9rem;font-weight:600;cursor:pointer;transition:all .2s}
.btn-primary{background:var(--accent);color:#0f172a}
.btn-primary:hover{opacity:.85}
.toast{position:fixed;top:20px;right:20px;padding:12px 20px;border-radius:8px;
       background:var(--green);color:#fff;font-weight:600;opacity:0;transition:opacity .3s;z-index:99}
.toast.show{opacity:1}
table{width:100%;border-collapse:collapse;font-size:.82rem}
th{text-align:left;color:var(--muted);border-bottom:1px solid var(--border);padding:8px 6px;font-weight:500}
td{padding:6px;border-bottom:1px solid #1e293b}
.mono{font-family:'Cascadia Code','Fira Code',monospace;font-size:.8rem}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
.stat-box{text-align:center;padding:16px;background:#0f172a;border-radius:8px;border:1px solid var(--border)}
.stat-box .num{font-size:1.6rem;font-weight:700;color:var(--accent)}
.stat-box .lbl{font-size:.7rem;color:var(--muted);text-transform:uppercase;margin-top:4px}
.endpoint{background:#0f172a;border:1px solid var(--border);border-radius:8px;padding:12px 16px;
          font-family:monospace;font-size:.82rem;margin:8px 0;color:var(--accent);word-break:break-all}
.help{background:#1e293b;border-left:3px solid var(--accent2);padding:12px 16px;margin:12px 0;
      border-radius:0 8px 8px 0;font-size:.84rem;line-height:1.6}
.help code{background:#0f172a;padding:2px 6px;border-radius:4px;font-size:.8rem;color:var(--accent)}
.warn-box{background:#f59e0b15;border:1px solid #f59e0b44;border-radius:8px;padding:12px 16px;
          margin:8px 0;font-size:.84rem;color:var(--warn)}
.tag{display:inline-block;background:var(--accent2);color:#fff;font-size:.7rem;padding:2px 8px;
     border-radius:99px;margin-left:6px;font-weight:600;vertical-align:middle}
.alias-row{display:flex;gap:8px;margin-bottom:6px;align-items:center}
.alias-row input{flex:1}
.alias-row .arr{color:var(--accent);font-weight:bold;flex:0 0 auto}
.alias-row .del{background:var(--red);color:#fff;padding:4px 10px;border-radius:6px;font-size:.8rem;cursor:pointer;border:none}
</style>
</head><body>
<div class="container">
  <h1>NIM to Claude API Proxy <span class="tag">v2</span></h1>
  <p class="subtitle">NVIDIA NIM to Anthropic Claude Messages API Proxy - Claude Code Optimized</p>

  <div class="card">
    <h2>Claude Code Setup</h2>
    <div class="warn-box">
      WARNING: ANTHROPIC_BASE_URL should end at port only, do NOT add /v1/messages
    </div>
    <div class="help">
      In <code>~/.claude/settings.json</code>:
      <pre style="margin-top:8px;color:var(--text);white-space:pre-wrap">{
  "env": {
    "ANTHROPIC_BASE_URL": "<span id="epBase" style="color:var(--accent)"></span>",
    "ANTHROPIC_AUTH_TOKEN": "<span id="epKey2" style="color:var(--accent2)"></span>",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "<span id="epMdl" style="color:var(--accent2)"></span>",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "<span id="epMdl2" style="color:var(--accent2)"></span>",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "<span id="epMdl3" style="color:var(--accent2)"></span>"
  }
}</pre>
    </div>
    <div style="margin-top:10px">
      <span style="color:var(--muted);font-size:.8rem">Messages Endpoint:</span>
      <div class="endpoint" id="epUrl"></div>
    </div>
  </div>

  <div class="card">
    <h2>Server Config</h2>
    <form id="configForm">
      <div class="field">
        <label>NVIDIA API Base URL</label>
        <input name="nvidia_base_url" id="f_base" placeholder="https://integrate.api.nvidia.com/v1">
      </div>
      <div class="field">
        <label>NVIDIA API Key</label>
        <input name="nvidia_api_key" id="f_key" type="password" placeholder="nvapi-xxx">
      </div>
      <div class="row">
        <div class="field">
          <label>Proxy API Key (for client auth)</label>
          <input name="proxy_api_key" id="f_proxy" placeholder="sk-proxy-change-me">
        </div>
        <div class="field">
          <label>Default Model</label>
          <input name="default_model" id="f_model" placeholder="z-ai/glm-5.1">
        </div>
      </div>
      <div class="row">
        <div class="field"><label>Max Tokens</label>
          <input name="max_tokens" id="f_maxtok" type="number" value="16384"></div>
        <div class="field"><label>Temperature</label>
          <input name="temperature" id="f_temp" type="number" step="0.1" value="1.0"></div>
      </div>
      <div class="row">
        <div class="field"><label>Top P</label>
          <input name="top_p" id="f_topp" type="number" step="0.1" value="1.0"></div>
        <div class="field">
          <label>&nbsp;</label>
          <div class="check-row">
            <input type="checkbox" id="f_think" name="enable_thinking" checked>
            <label for="f_think" style="font-size:.9rem;color:var(--text)">Enable Thinking (default)</label>
          </div>
        </div>
      </div>
      <button type="submit" class="btn-primary">Save Config</button>
    </form>
  </div>

  <div class="card">
    <h2>Model Aliases</h2>
    <p style="color:var(--muted);font-size:.84rem;margin-bottom:12px">
      Claude Code sends Claude model names (e.g. claude-sonnet-4-20250514).
      Map them to NIM models here. Names containing "/" are treated as NIM-native and passed through.
    </p>
    <div id="aliasContainer"></div>
    <button onclick="addAliasRow()" class="btn-primary" style="margin-top:8px;padding:6px 16px;font-size:.82rem">
      + Add Alias</button>
    <button onclick="saveAliases()" class="btn-primary" style="margin-top:8px;margin-left:8px;padding:6px 16px;font-size:.82rem">
      Save Aliases</button>
  </div>

  <div class="card">
    <h2>Token Usage (Estimated)</h2>
    <div class="stat-grid">
      <div class="stat-box"><div class="num" id="sReq">0</div><div class="lbl">Requests</div></div>
      <div class="stat-box"><div class="num" id="sIn">0</div><div class="lbl">Input Tokens</div></div>
      <div class="stat-box"><div class="num" id="sOut">0</div><div class="lbl">Output Tokens</div></div>
      <div class="stat-box"><div class="num" id="sThink">0</div><div class="lbl">Thinking Tokens</div></div>
    </div>
    <table>
      <thead><tr><th>Time</th><th>Request ID</th><th>Model</th><th>In</th><th>Out</th><th>Think</th><th>Total</th></tr></thead>
      <tbody id="usageBody"></tbody>
    </table>
    <div style="text-align:right;margin-top:8px">
      <button onclick="loadUsage()" class="btn-primary" style="padding:6px 16px;font-size:.8rem">Refresh</button>
    </div>
  </div>
</div>

<div class="toast" id="toast">Config saved</div>

<script>
const $=s=>document.querySelector(s);
let currentAliases={};

function toast(msg,ok=true){
  const t=$('#toast');t.textContent=msg;
  t.style.background=ok?'var(--green)':'var(--red)';
  t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2000);
}

function addAliasRow(from='',to=''){
  const c=$('#aliasContainer');
  const d=document.createElement('div');d.className='alias-row';
  d.innerHTML=`<input class="af" placeholder="claude-sonnet-4-20250514" value="${from}" style="font-size:.82rem">
    <span class="arr">&rarr;</span>
    <input class="at" placeholder="z-ai/glm-5.1" value="${to}" style="font-size:.82rem">
    <button class="del" onclick="this.parentElement.remove()">X</button>`;
  c.appendChild(d);
}

function collectAliases(){
  const rows=document.querySelectorAll('.alias-row');
  const aliases={};
  rows.forEach(r=>{
    const f=r.querySelector('.af').value.trim();
    const t=r.querySelector('.at').value.trim();
    if(f&&t)aliases[f]=t;
  });
  return aliases;
}

async function saveAliases(){
  const a=collectAliases();
  const r=await fetch('/api/config',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({model_aliases:a})});
  toast(r.ok?'Aliases saved':'Save failed',r.ok);
}

async function loadConfig(){
  const r=await fetch('/api/config');const c=await r.json();
  $('#f_base').value=c.nvidia_base_url||'';
  $('#f_key').placeholder=c.nvidia_api_key||'Not set';
  $('#f_proxy').value=c.proxy_api_key||'';
  $('#f_model').value=c.default_model||'';
  $('#f_maxtok').value=c.max_tokens||16384;
  $('#f_temp').value=c.temperature||1;
  $('#f_topp').value=c.top_p||1;
  $('#f_think').checked=!!c.enable_thinking;
  const base=location.origin;
  $('#epBase').textContent=base;
  $('#epUrl').textContent=base+'/v1/messages';
  $('#epKey2').textContent=c.proxy_api_key||'(none)';
  const mdl=c.default_model||'z-ai/glm-5.1';
  $('#epMdl').textContent=mdl;
  $('#epMdl2').textContent=mdl;
  $('#epMdl3').textContent=mdl;
  currentAliases=c.model_aliases||{};
  const ac=$('#aliasContainer');ac.innerHTML='';
  Object.entries(currentAliases).forEach(([f,t])=>addAliasRow(f,t));
}

$('#configForm').addEventListener('submit',async e=>{
  e.preventDefault();
  const d=Object.fromEntries(new FormData(e.target));
  d.enable_thinking=$('#f_think').checked;
  if(!d.nvidia_api_key)delete d.nvidia_api_key;
  const r=await fetch('/api/config',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  toast(r.ok?'Config saved':'Save failed',r.ok);if(r.ok)loadConfig();
});

async function loadUsage(){
  const r=await fetch('/api/usage');const u=await r.json();
  $('#sReq').textContent=u.total.requests.toLocaleString();
  $('#sIn').textContent=u.total.input.toLocaleString();
  $('#sOut').textContent=u.total.output.toLocaleString();
  $('#sThink').textContent=u.total.thinking.toLocaleString();
  const rows=u.recent.reverse().map(e=>
    `<tr><td class="mono">${e.ts}</td><td class="mono">${e.rid.slice(0,12)}...</td>
     <td>${e.model}</td><td>${e.in}</td><td>${e.out}</td><td>${e.think}</td>
     <td style="font-weight:600">${e.total}</td></tr>`).join('');
  $('#usageBody').innerHTML=rows||
    '<tr><td colspan="7" style="text-align:center;color:var(--muted)">No data yet</td></tr>';
}

loadConfig();loadUsage();setInterval(loadUsage,15000);
</script>
</body></html>'''


@app.route("/", methods=["GET"])
def web_ui():
    return Response(WEB_PAGE, mimetype="text/html")


@app.route("/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def catch_all(path):
    logger.warning(
        f"{_C(33)}404 Unknown path: {request.method} /{path}{_RST}"
    )
    return jsonify({"type": "error", "error": {"type": "not_found_error",
                    "message": f"Unknown endpoint: /{path}. Use /v1/messages for API calls."
                    }}), 404


if __name__ == "__main__":
    port = CONFIG["server_port"]
    has_key = bool(CONFIG["nvidia_api_key"])

    print(f"""
{_C(36)}======================================================
     NIM -> Claude API Proxy Server v2
======================================================
  Anthropic endpoint:  http://0.0.0.0:{port}/v1/messages
  Web UI:              http://0.0.0.0:{port}/
  NVIDIA API Key:      {'SET' if has_key else 'NOT SET (configure via Web UI)'}
  Proxy API Key:       {CONFIG['proxy_api_key'][:20]}
======================================================{_RST}
""")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
