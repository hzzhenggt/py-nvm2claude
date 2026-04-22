#!/usr/bin/env python3
"""
NVIDIA NIM → Anthropic Claude API 代理服务器
将 NVIDIA NIM (OpenAI 兼容) API 转换为 Anthropic Claude Messages API 格式
支持流式/非流式、thinking 透传、token 估算日志、Web 配置界面

用法:
  pip install flask requests
  export NVIDIA_API_KEY="nvapi-xxx"
  python nim_to_claude_proxy.py
"""

import json, time, uuid, logging, threading, os, sys
from datetime import datetime
from collections import deque

from flask import Flask, request, Response, jsonify

# ═══════════════════════════════════════════════════════════════════════
#  全局配置（可通过 Web UI 动态修改）
# ═══════════════════════════════════════════════════════════════════════

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
}
CONFIG_LOCK = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════
#  日志
# ═══════════════════════════════════════════════════════════════════════

_USE_COLOR = sys.stderr.isatty() and os.getenv("NO_COLOR") is None
_C = lambda c: f"\033[{c}m" if _USE_COLOR else ""
_RST = _C(0)

logging.basicConfig(
    level=logging.INFO,
    format=f"{_C(90)}%(asctime)s{_RST} │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nim-proxy")

# ═══════════════════════════════════════════════════════════════════════
#  Token 使用量追踪（虚假估算 + 日志）
# ═══════════════════════════════════════════════════════════════════════

USAGE_LOG = deque(maxlen=500)
USAGE_LOCK = threading.Lock()
TOTAL_STATS = {"input": 0, "output": 0, "thinking": 0, "requests": 0}


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数: 中文 ~1.5 字符/token, 英文 ~4 字符/token"""
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
        f"{_C(36)}📊 Token{_RST} │ {request_id[:12]}… │ {_C(33)}{model}{_RST} │ "
        f"in={_C(32)}{input_tokens}{_RST} out={_C(32)}{output_tokens}{_RST} "
        f"think={_C(90)}{thinking_tokens}{_RST} Σ={entry['total']}"
    )


# ═══════════════════════════════════════════════════════════════════════
#  格式转换: Anthropic → OpenAI
# ═══════════════════════════════════════════════════════════════════════

def anthropic_to_openai(body: dict) -> dict:
    """将 Anthropic Messages API 请求体转为 OpenAI Chat Completion 格式"""
    messages = []

    # system prompt
    system = body.get("system")
    if system:
        if isinstance(system, list):
            system = "\n".join(
                b.get("text", "") for b in system if b.get("type") == "text"
            )
        messages.append({"role": "system", "content": system})

    # 对话消息
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "thinking":
                        pass  # 跳过历史 thinking 块
                    elif block.get("type") == "image":
                        parts.append("[image]")
                elif isinstance(block, str):
                    parts.append(block)
            content = "\n".join(parts)
        messages.append({"role": role, "content": content})

    with CONFIG_LOCK:
        cfg = dict(CONFIG)

    model = body.get("model") or cfg["default_model"]

    result = {
        "model": model,
        "messages": messages,
        "max_tokens": body.get("max_tokens", cfg["max_tokens"]),
        "temperature": body.get("temperature", cfg["temperature"]),
        "top_p": body.get("top_p", cfg["top_p"]),
        "stream": body.get("stream", False),
    }

    # Thinking 支持
    thinking = body.get("thinking", {})
    enable_think = (thinking.get("type") == "enabled") or cfg.get("enable_thinking")
    if enable_think:
        result["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": True,
                "clear_thinking": False,
            }
        }

    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]

    return result


# ═══════════════════════════════════════════════════════════════════════
#  格式转换: OpenAI 响应 → Anthropic 响应（非流式）
# ═══════════════════════════════════════════════════════════════════════

def openai_to_anthropic(oai: dict, model: str, rid: str, input_est: int) -> dict:
    choice = oai.get("choices", [{}])[0]
    msg = choice.get("message", {})
    text = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "")

    blocks = []
    think_tok = 0
    if reasoning:
        blocks.append({"type": "thinking", "thinking": reasoning})
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
        },
    }


# ═══════════════════════════════════════════════════════════════════════
#  流式转换: OpenAI SSE → Anthropic SSE
# ═══════════════════════════════════════════════════════════════════════

def _sse(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_convert(resp, model, rid, input_est):
    """生成器: 读取 OpenAI SSE 流，转换为 Anthropic 格式 yield"""

    output_text = ""
    thinking_text = ""
    content_started = False
    thinking_open = False
    block_idx = 0
    think_idx = 0
    text_idx = 0

    # ── message_start ──
    yield _sse("message_start", {
        "type": "message_start",
        "message": {
            "id": rid, "type": "message", "role": "assistant",
            "content": [], "model": model,
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": input_est, "output_tokens": 0},
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

        # ── reasoning / thinking ──
        reasoning = delta.get("reasoning_content")
        if reasoning:
            if not thinking_open:
                thinking_open = True
                think_idx = block_idx
                block_idx += 1
                yield _sse("content_block_start", {
                    "type": "content_block_start",
                    "index": think_idx,
                    "content_block": {"type": "thinking", "thinking": ""},
                })
            thinking_text += reasoning
            yield _sse("content_block_delta", {
                "type": "content_block_delta",
                "index": think_idx,
                "delta": {"type": "thinking_delta", "thinking": reasoning},
            })

        # ── text content ──
        text = delta.get("content")
        if text:
            # 关闭 thinking 块（如果打开）
            if thinking_open:
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

    # ── 关闭剩余块 ──
    if thinking_open:
        yield _sse("content_block_stop", {
            "type": "content_block_stop", "index": think_idx,
        })
    if content_started:
        yield _sse("content_block_stop", {
            "type": "content_block_stop", "index": text_idx,
        })
    else:
        # 至少保证一个 text block
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


# ═══════════════════════════════════════════════════════════════════════
#  Flask App & Routes
# ═══════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.logger.setLevel(logging.WARNING)


def _auth_check():
    """验证请求中的 API key"""
    with CONFIG_LOCK:
        expected = CONFIG["proxy_api_key"]
    if not expected:
        return True
    key = (request.headers.get("x-api-key")
           or request.headers.get("Authorization", "").removeprefix("Bearer ").strip())
    return key == expected


# ────────────────── /v1/messages（核心代理端点）──────────────────

@app.route("/v1/messages", methods=["POST"])
def messages_proxy():
    if not _auth_check():
        return jsonify({"type": "error", "error": {"type": "authentication_error",
                        "message": "Invalid API key"}}), 401

    body = request.get_json(force=True)
    rid = f"msg_{uuid.uuid4().hex[:24]}"
    model = body.get("model") or CONFIG["default_model"]
    is_stream = body.get("stream", False)

    logger.info(
        f"{_C(35)}➜ REQ{_RST} │ {rid[:12]}… │ model={_C(33)}{model}{_RST} │ "
        f"stream={is_stream} │ msgs={len(body.get('messages', []))}"
    )

    # 转换格式
    oai_body = anthropic_to_openai(body)
    input_est = estimate_tokens(json.dumps(oai_body.get("messages", [])))

    # 准备 NVIDIA 请求
    with CONFIG_LOCK:
        base_url = CONFIG["nvidia_base_url"].rstrip("/")
        api_key = CONFIG["nvidia_api_key"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    # 从 oai_body 中分离 extra_body（requests 库不支持，需合并到顶层）
    extra = oai_body.pop("extra_body", None)
    if extra:
        oai_body.update(extra)

    import requests as http_req

    try:
        upstream = http_req.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=oai_body,
            stream=is_stream,
            timeout=300,
        )
    except Exception as e:
        logger.error(f"上游请求失败: {e}")
        return jsonify({"type": "error", "error": {"type": "api_error",
                        "message": str(e)}}), 502

    if upstream.status_code != 200:
        err_text = upstream.text[:500]
        logger.error(f"上游错误 {upstream.status_code}: {err_text}")
        return jsonify({"type": "error", "error": {"type": "api_error",
                        "message": f"Upstream {upstream.status_code}: {err_text}"
                        }}), upstream.status_code

    # ── 流式 ──
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
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── 非流式 ──
    try:
        oai_resp = upstream.json()
    except Exception:
        return jsonify({"type": "error", "error": {"type": "api_error",
                        "message": "Invalid JSON from upstream"}}), 502

    result = openai_to_anthropic(oai_resp, model, rid, input_est)
    return jsonify(result)


# ────────────────── 兼容端点 ──────────────────

@app.route("/v1/models", methods=["GET"])
def list_models():
    """返回可用模型列表（兼容某些客户端）"""
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


# ────────────────── 配置接口 ──────────────────

@app.route("/api/config", methods=["GET"])
def get_config():
    with CONFIG_LOCK:
        safe = {k: v for k, v in CONFIG.items()}
        if safe.get("nvidia_api_key"):
            k = safe["nvidia_api_key"]
            safe["nvidia_api_key"] = k[:8] + "…" + k[-4:] if len(k) > 12 else "***"
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
    logger.info(f"{_C(32)}⚙ Config updated{_RST}")
    return jsonify({"status": "ok"})


# ────────────────── 使用量统计 ──────────────────

@app.route("/api/usage", methods=["GET"])
def get_usage():
    with USAGE_LOCK:
        return jsonify({
            "total": dict(TOTAL_STATS),
            "recent": list(USAGE_LOG)[-50:],
        })


# ────────────────── Web 管理界面 ──────────────────

WEB_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIM → Claude 代理配置</title>
<style>
  :root{--bg:#0f172a;--card:#1e293b;--accent:#38bdf8;--accent2:#818cf8;
        --text:#e2e8f0;--muted:#94a3b8;--border:#334155;--green:#22c55e;--red:#ef4444}
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);
       min-height:100vh;padding:20px}
  .container{max-width:900px;margin:0 auto}
  h1{font-size:1.5rem;margin-bottom:4px;background:linear-gradient(135deg,var(--accent),var(--accent2));
     -webkit-background-clip:text;-webkit-text-fill-color:transparent}
  .subtitle{color:var(--muted);font-size:.85rem;margin-bottom:24px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:24px;margin-bottom:20px}
  .card h2{font-size:1.1rem;margin-bottom:16px;color:var(--accent)}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .field{margin-bottom:14px}
  .field label{display:block;font-size:.8rem;color:var(--muted);margin-bottom:4px;
               text-transform:uppercase;letter-spacing:.5px}
  .field input,.field select{width:100%;padding:10px 12px;background:#0f172a;
       border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:.9rem;
       outline:none;transition:border .2s}
  .field input:focus{border-color:var(--accent)}
  .check-row{display:flex;align-items:center;gap:8px;margin:8px 0}
  .check-row input{accent-color:var(--accent)}
  button{padding:10px 24px;border:none;border-radius:8px;font-size:.9rem;font-weight:600;
         cursor:pointer;transition:all .2s}
  .btn-primary{background:var(--accent);color:#0f172a}
  .btn-primary:hover{opacity:.85}
  .toast{position:fixed;top:20px;right:20px;padding:12px 20px;border-radius:8px;
         background:var(--green);color:#fff;font-weight:600;opacity:0;transition:opacity .3s;z-index:99}
  .toast.show{opacity:1}
  table{width:100%;border-collapse:collapse;font-size:.82rem}
  th{text-align:left;color:var(--muted);border-bottom:1px solid var(--border);padding:8px 6px;
     font-weight:500}
  td{padding:6px;border-bottom:1px solid #1e293b}
  .mono{font-family:'Cascadia Code','Fira Code',monospace;font-size:.8rem}
  .stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}
  .stat-box{text-align:center;padding:16px;background:#0f172a;border-radius:8px;
            border:1px solid var(--border)}
  .stat-box .num{font-size:1.6rem;font-weight:700;color:var(--accent)}
  .stat-box .lbl{font-size:.7rem;color:var(--muted);text-transform:uppercase;margin-top:4px}
  .endpoint{background:#0f172a;border:1px solid var(--border);border-radius:8px;padding:12px 16px;
            font-family:monospace;font-size:.82rem;margin:8px 0;color:var(--accent);word-break:break-all}
</style>
</head><body>
<div class="container">
  <h1>⚡ NIM → Claude API Proxy</h1>
  <p class="subtitle">NVIDIA NIM (OpenAI 兼容) → Anthropic Claude Messages API 转换代理</p>

  <div class="card">
    <h2>🔗 接入端点</h2>
    <p style="color:var(--muted);font-size:.85rem;margin-bottom:8px">
      在 Claude 客户端中使用以下配置：</p>
    <div class="endpoint" id="epUrl"></div>
    <div class="row" style="margin-top:8px">
      <div><span style="color:var(--muted);font-size:.8rem">API Key:</span>
           <code class="mono" id="epKey" style="color:var(--accent2)"></code></div>
      <div><span style="color:var(--muted);font-size:.8rem">默认模型:</span>
           <code class="mono" id="epModel" style="color:var(--accent2)"></code></div>
    </div>
  </div>

  <div class="card">
    <h2>⚙️ 服务配置</h2>
    <form id="configForm">
      <div class="field">
        <label>NVIDIA API Base URL</label>
        <input name="nvidia_base_url" id="f_base"
               placeholder="https://integrate.api.nvidia.com/v1">
      </div>
      <div class="field">
        <label>NVIDIA API Key</label>
        <input name="nvidia_api_key" id="f_key" type="password" placeholder="nvapi-xxx">
      </div>
      <div class="row">
        <div class="field">
          <label>代理 API Key（客户端认证用）</label>
          <input name="proxy_api_key" id="f_proxy" placeholder="sk-proxy-change-me">
        </div>
        <div class="field">
          <label>默认模型</label>
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
            <label for="f_think" style="font-size:.9rem;color:var(--text)">启用 Thinking</label>
          </div>
        </div>
      </div>
      <button type="submit" class="btn-primary">💾 保存配置</button>
    </form>
  </div>

  <div class="card">
    <h2>📊 Token 使用量</h2>
    <div class="stat-grid">
      <div class="stat-box"><div class="num" id="sReq">0</div><div class="lbl">请求数</div></div>
      <div class="stat-box"><div class="num" id="sIn">0</div><div class="lbl">Input Tokens</div></div>
      <div class="stat-box"><div class="num" id="sOut">0</div><div class="lbl">Output Tokens</div></div>
      <div class="stat-box"><div class="num" id="sThink">0</div><div class="lbl">Thinking Tokens</div></div>
    </div>
    <table>
      <thead><tr><th>时间</th><th>Request ID</th><th>Model</th><th>In</th><th>Out</th>
                 <th>Think</th><th>Total</th></tr></thead>
      <tbody id="usageBody"></tbody>
    </table>
    <div style="text-align:right;margin-top:8px">
      <button onclick="loadUsage()" class="btn-primary"
              style="padding:6px 16px;font-size:.8rem">🔄 刷新</button>
    </div>
  </div>
</div>

<div class="toast" id="toast">✅ 配置已保存</div>

<script>
const $ = s => document.querySelector(s);
function toast(msg,ok=true){
  const t=$('#toast');t.textContent=msg;
  t.style.background=ok?'var(--green)':'var(--red)';
  t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2000);
}
async function loadConfig(){
  const r=await fetch('/api/config');const c=await r.json();
  $('#f_base').value=c.nvidia_base_url||'';
  $('#f_key').placeholder=c.nvidia_api_key||'未设置';
  $('#f_proxy').value=c.proxy_api_key||'';
  $('#f_model').value=c.default_model||'';
  $('#f_maxtok').value=c.max_tokens||16384;
  $('#f_temp').value=c.temperature||1;
  $('#f_topp').value=c.top_p||1;
  $('#f_think').checked=!!c.enable_thinking;
  $('#epUrl').textContent=location.origin+'/v1/messages';
  $('#epKey').textContent=c.proxy_api_key||'(无)';
  $('#epModel').textContent=c.default_model||'';
}
$('#configForm').addEventListener('submit',async e=>{
  e.preventDefault();
  const d=Object.fromEntries(new FormData(e.target));
  d.enable_thinking=$('#f_think').checked;
  if(!d.nvidia_api_key)delete d.nvidia_api_key;
  const r=await fetch('/api/config',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(d)});
  toast(r.ok?'✅ 配置已保存':'❌ 保存失败',r.ok);if(r.ok)loadConfig();
});
async function loadUsage(){
  const r=await fetch('/api/usage');const u=await r.json();
  $('#sReq').textContent=u.total.requests.toLocaleString();
  $('#sIn').textContent=u.total.input.toLocaleString();
  $('#sOut').textContent=u.total.output.toLocaleString();
  $('#sThink').textContent=u.total.thinking.toLocaleString();
  const rows=u.recent.reverse().map(e=>
    `<tr><td class="mono">${e.ts}</td><td class="mono">${e.rid.slice(0,12)}…</td>
     <td>${e.model}</td><td>${e.in}</td><td>${e.out}</td><td>${e.think}</td>
     <td style="font-weight:600">${e.total}</td></tr>`).join('');
  $('#usageBody').innerHTML=rows||
    '<tr><td colspan="7" style="text-align:center;color:var(--muted)">暂无数据</td></tr>';
}
loadConfig();loadUsage();setInterval(loadUsage,15000);
</script>
</body></html>"""


@app.route("/", methods=["GET"])
def web_ui():
    return Response(WEB_PAGE, mimetype="text/html")


# ═══════════════════════════════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = CONFIG["server_port"]
    has_key = bool(CONFIG["nvidia_api_key"])

    print(f"""
{_C(36)}╔══════════════════════════════════════════════════════════════╗
║          ⚡  NIM → Claude API Proxy Server  ⚡               ║
╠══════════════════════════════════════════════════════════════╣
║  Anthropic 端点:  http://0.0.0.0:{port}/v1/messages              ║
║  Web 管理界面:    http://0.0.0.0:{port}/                          ║
║  NVIDIA API Key:  {'✅ 已配置' if has_key else '❌ 未设置（请通过 Web UI 配置）'}  ║
╚══════════════════════════════════════════════════════════════╝{_RST}
""")

    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
