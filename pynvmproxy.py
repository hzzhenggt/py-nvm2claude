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

import json, time, uuid, logging, threading, os, sys, re, socket, hmac
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
    "rate_limit": int(os.getenv("RATE_LIMIT", "40")),
    "stream_idle_timeout": int(os.getenv("STREAM_IDLE_TIMEOUT", "600")),
    "upstream_read_timeout": int(os.getenv("UPSTREAM_READ_TIMEOUT", "600")),
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
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

_MODEL_CACHE = {"models": [], "ts": 0}
_MODEL_CACHE_TTL = 3600

_FALLBACK_MODELS = [
    "z-ai/glm-5.1",
    "deepseek-ai/deepseek-r1",
    "meta/llama-4-maverick-17b-128e-instruct",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
    "qwen/qwen2.5-72b-instruct",
]


def _fetch_nvidia_models():
    now = time.time()
    if _MODEL_CACHE["models"] and (now - _MODEL_CACHE["ts"]) < _MODEL_CACHE_TTL:
        return _MODEL_CACHE["models"]

    with CONFIG_LOCK:
        api_key = CONFIG["nvidia_api_key"]
        base_url = CONFIG["nvidia_base_url"].rstrip("/")

    if not api_key:
        logger.warning("No API key, using fallback models")
        return _FALLBACK_MODELS

    try:
        import requests as _r
        resp = _r.get(f"{base_url}/models", headers={
            "Authorization": f"Bearer {api_key}",
        }, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = []
            for m in data.get("data", []):
                mid = m.get("id", "")
                if mid:
                    models.append(mid)
            if models:
                _MODEL_CACHE["models"] = models
                _MODEL_CACHE["ts"] = now
                logger.info(f"{_C(32)}Fetched {len(models)} models from NVIDIA{_RST}")
                return models
    except Exception as e:
        logger.warning(f"{_C(33)}Failed to fetch models from NVIDIA: {e}{_RST}")

    if _MODEL_CACHE["models"]:
        return _MODEL_CACHE["models"]
    return _FALLBACK_MODELS


RATE_WINDOW = deque()
RATE_LOCK = threading.Lock()


def _check_rate_limit():
    with CONFIG_LOCK:
        limit = CONFIG.get("rate_limit", 40)
    now = time.time()
    with RATE_LOCK:
        window = 60
        while RATE_WINDOW and RATE_WINDOW[0] < now - window:
            RATE_WINDOW.popleft()
        current = len(RATE_WINDOW)
        allowed = current < limit
        if allowed:
            RATE_WINDOW.append(now)
        return allowed, current, limit


def _get_qps():
    now = time.time()
    with RATE_LOCK:
        window = 60
        while RATE_WINDOW and RATE_WINDOW[0] < now - window:
            RATE_WINDOW.popleft()
        return len(RATE_WINDOW)

    try:
        import requests as _r
        resp = _r.get(f"{base_url}/models", headers={
            "Authorization": f"Bearer {api_key}",
        }, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            models = []
            for m in data.get("data", []):
                mid = m.get("id", "")
                if mid:
                    models.append(mid)
            if models:
                _MODEL_CACHE["models"] = models
                _MODEL_CACHE["ts"] = now
                logger.info(f"{_C(32)}Fetched {len(models)} models from NVIDIA{_RST}")
                return models
    except Exception as e:
        logger.warning(f"{_C(33)}Failed to fetch models: {e}{_RST}")

    if _MODEL_CACHE["models"]:
        return _MODEL_CACHE["models"]
    return _FALLBACK_MODELS


def _save_config():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(CONFIG, f, ensure_ascii=False, indent=2)
        logger.info(f"{_C(32)}Config saved to {CONFIG_FILE}{_RST}")
    except Exception as e:
        logger.warning(f"{_C(33)}Failed to save config: {e}{_RST}")


def _load_config():
    if not os.path.exists(CONFIG_FILE):
        return
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        with CONFIG_LOCK:
            for k, v in saved.items():
                if k in CONFIG and v:
                    CONFIG[k] = v
        logger.info(f"{_C(32)}Config loaded from {CONFIG_FILE}{_RST}")
    except Exception as e:
        logger.warning(f"{_C(33)}Failed to load config: {e}{_RST}")

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
TRACE_LOG = deque(maxlen=200)
TRACE_LOCK = threading.Lock()
TOTAL_STATS = {"input": 0, "output": 0, "thinking": 0, "requests": 0}
CIRCUIT_STATE = {}
CIRCUIT_LOCK = threading.Lock()
CIRCUIT_THRESHOLD = 3
CIRCUIT_COOLDOWN = 30


def _update_circuit(model, success):
    with CIRCUIT_LOCK:
        cb = CIRCUIT_STATE.get(model, {"failures": 0, "open_until": 0})
        if success:
            cb["failures"] = 0
        else:
            cb["failures"] += 1
            if cb["failures"] >= CIRCUIT_THRESHOLD:
                cb["open_until"] = time.time() + CIRCUIT_COOLDOWN
        CIRCUIT_STATE[model] = cb


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff"
              or "\u3000" <= c <= "\u303f"
              or "\uff00" <= c <= "\uffef")
    other = len(text) - cjk
    return max(1, int(cjk / 1.5 + other / 4))


def log_usage(model, input_tokens, output_tokens, request_id, thinking_tokens=0, elapsed=None):
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
    elapsed_str = f" {elapsed:.0f}s" if elapsed else ""
    logger.info(
        f"{_C(36)}Token{_RST} | {request_id[:16]}... | {_C(33)}{model}{_RST} | "
        f"in={_C(32)}{input_tokens}{_RST} out={_C(32)}{output_tokens}{_RST} "
        f"think={_C(90)}{thinking_tokens}{_RST} total={entry['total']}{elapsed_str}"
    )


def _trace(rid, phase, data):
    with TRACE_LOCK:
        for t in TRACE_LOG:
            if t["rid"] == rid:
                t["steps"].append({"ts": datetime.now().strftime("%H:%M:%S.%f")[:-3], "phase": phase, "data": data})
                return
        TRACE_LOG.append({"rid": rid, "ts": datetime.now().strftime("%H:%M:%S"), "steps": [
            {"ts": datetime.now().strftime("%H:%M:%S.%f")[:-3], "phase": phase, "data": data}
        ]})


def _trim_body(oai_body, max_bytes):
    msgs = oai_body.get("messages", [])
    while len(json.dumps(oai_body, ensure_ascii=False)) > max_bytes and len(msgs) > 2:
        if msgs[0]["role"] == "system":
            msgs.pop(0)
        elif len(msgs) >= 4:
            del msgs[1:3]
        else:
            break
    return oai_body


def _sanitize_log(s, max_len=512):
    s = re.sub(r'"data:([A-Za-z0-9+/=]{100,})"', '"data:<redacted base64>"', str(s))
    if len(s) > max_len:
        s = s[:max_len] + f"...<truncated {len(s)}>"
    return s


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
                        prefix = "[tool_result"
                        if block.get("is_error"):
                            prefix += " ERROR"
                        parts.append(f"{prefix}: {content_val}]")
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

    if "top_k" in body:
        result["top_k"] = body["top_k"]

    tools = body.get("tools")
    if tools:
        oai_tools = []
        for t in tools:
            if isinstance(t, dict):
                oai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                    }
                })
        if oai_tools:
            result["tools"] = oai_tools

    tool_choice = body.get("tool_choice")
    if tool_choice:
        tc_type = tool_choice.get("type", "") if isinstance(tool_choice, dict) else ""
        if tc_type == "any":
            result["tool_choice"] = "required"
        elif tc_type == "auto":
            result["tool_choice"] = "auto"
        elif tc_type == "none":
            result["tool_choice"] = "none"
        elif tc_type == "tool":
            name = tool_choice.get("name", "")
            if name:
                result["tool_choice"] = {"type": "function", "function": {"name": name}}

    thinking = body.get("thinking", {})
    enable_think = False
    if isinstance(thinking, dict):
        ttype = thinking.get("type")
        enable_think = ttype in ("enabled", "adaptive")
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
    tool_calls = msg.get("tool_calls") or []

    blocks = []
    think_tok = 0
    if reasoning:
        blocks.append({
            "type": "thinking",
            "thinking": reasoning,
            "signature": f"EpR_{uuid.uuid4().hex[:40]}sig"
        })
        think_tok = estimate_tokens(reasoning)

    for tc in tool_calls:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        arguments = fn.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                inp = json.loads(arguments)
            except json.JSONDecodeError:
                inp = {}
        else:
            inp = arguments
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": name,
            "input": inp,
        })

    if text or not blocks:
        blocks.append({"type": "text", "text": text})

    usage = oai.get("usage", {})
    in_tok = usage.get("prompt_tokens") or input_est
    out_tok = usage.get("completion_tokens") or estimate_tokens(text)

    log_usage(model, in_tok, out_tok, rid, think_tok)

    stop_map = {"stop": "end_turn", "length": "max_tokens", "content_filter": "end_turn",
                "tool_calls": "tool_use"}
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


def _safe_json_padding(args_str):
    """Return padding to append to args_str so the result parses as JSON,
    or None if no padding is needed or padding is hopeless."""
    trimmed = args_str.strip()
    if trimmed == "":
        return "{}"
    try:
        json.loads(trimmed)
        return None  # already valid
    except json.JSONDecodeError:
        pass
    for cand in ('"}', '}', '"}}', '}}', '"]}', ']}',
                 'null}', 'null}}', 'null]}', '0}', '0}}', '""}', '""}}'):
        try:
            json.loads(trimmed + cand)
            return cand
        except json.JSONDecodeError:
            continue
    return None  # give up -- let caller only emit content_block_stop


def stream_convert(resp, model, rid, input_est):
    start_ts = time.time()
    output_text = ""
    thinking_text = ""
    content_started = False
    thinking_open = False
    block_idx = 0
    think_idx = 0
    text_idx = 0
    thinking_sig = f"EpR_{uuid.uuid4().hex[:40]}sig"
    finish_reason = "stop"
    stream_stop_map = {"stop": "end_turn", "length": "max_tokens", "content_filter": "end_turn",
                       "tool_calls": "tool_use"}
    tool_data = {}

    # --- unified SSE closure (defined here so it can close over all state) ---
    _finalized = [False]  # closure-cell idempotency flag

    def _stop_reason_for(hint, fr, trunc):
        # design §stop_reason 语义映射表
        if hint == "idle_timeout":
            return "max_tokens"
        if hint == "conn_error":
            return "end_turn"
        if hint == "unexpected":
            return "end_turn"
        if hint == "eof":
            return stream_stop_map.get(fr, "end_turn")
        # "done"
        if trunc:
            return "max_tokens"
        return stream_stop_map.get(fr, "end_turn")

    def _finalize_stream(reason_hint, error_payload=None):
        if _finalized[0]:
            return
        _finalized[0] = True

        nonlocal thinking_open

        # Step 1: close thinking (signature_delta + content_block_stop)
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

        # Step 2: close tool_use blocks (pad partial_json if needed)
        for td in tool_data.values():
            if td.get("started"):
                padding = _safe_json_padding(td["arguments"])
                if padding is not None:
                    yield _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": td["block_idx"],
                        "delta": {"type": "input_json_delta", "partial_json": padding},
                    })
                yield _sse("content_block_stop", {
                    "type": "content_block_stop", "index": td["block_idx"],
                })

        # Step 3: close text block, or emit empty-text placeholder when truly empty
        if content_started:
            yield _sse("content_block_stop", {
                "type": "content_block_stop", "index": text_idx,
            })
        elif not tool_data and reason_hint == "done":
            yield _sse("content_block_start", {
                "type": "content_block_start",
                "index": block_idx,
                "content_block": {"type": "text", "text": ""},
            })
            yield _sse("content_block_stop", {
                "type": "content_block_stop", "index": block_idx,
            })

        # Step 4: token accounting + usage log
        out_tok = estimate_tokens(output_text)
        think_tok = estimate_tokens(thinking_text)
        log_usage(model, input_est, out_tok, rid, think_tok, time.time() - start_ts)
        elapsed = time.time() - start_ts
        _trace(rid, "stream_done", {"elapsed": f"{elapsed:.1f}s", "in_tok": input_est, "out_tok": out_tok, "think_tok": think_tok, "truncated": truncated})

        # Step 5: message_delta with semantic stop_reason
        stop_reason = _stop_reason_for(reason_hint, finish_reason, truncated)
        yield _sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": out_tok + think_tok},
        })

        # Step 6: optional error event (unexpected path only) — emit BEFORE message_stop
        # so that message_stop remains the genuine terminal SSE event.
        if reason_hint == "unexpected" and error_payload is not None:
            yield _sse("error", error_payload)

        # Step 7: sole terminal message_stop
        yield _sse("message_stop", {"type": "message_stop"})

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

    # Load configurable idle timeout once (avoid lock contention per read)
    with CONFIG_LOCK:
        stream_idle_timeout_val = CONFIG.get("stream_idle_timeout", 600)

    try:
        sock = None
        for chain in [
            lambda r: r.raw._fp.fp.raw._sock,
            lambda r: r.raw._fp.fp._sock,
            lambda r: r.raw._fp.fp,
        ]:
            try:
                sock = chain(resp)
                if hasattr(sock, "settimeout"):
                    sock.settimeout(15)
                    break
            except Exception:
                continue
        if sock is None:
            logger.warning(f"{_C(33)}Could not set socket timeout on upstream connection{_RST}")
    except Exception:
        pass

        buffer = ""
        done = False
        truncated = False
        last_data = time.time()
        while not done:
            try:
                chunk = resp.raw.read(8192, decode_content=True)
            except GeneratorExit:
                raise
            except socket.timeout:
                if time.time() - last_data > stream_idle_timeout_val:
                    logger.warning(f"{_C(33)}Stream silent for {stream_idle_timeout_val}s, terminating{_RST}")
                    yield from _finalize_stream("idle_timeout")
                    return
                yield _sse("ping", {"type": "ping"})
                continue
            except (ConnectionError, OSError) as e:
                logger.warning(f"{_C(33)}Stream read error: {e}{_RST}")
                yield from _finalize_stream("conn_error")
                return
            except Exception as e:
                err_str = str(e).lower()
                if "timed out" in err_str or "timeout" in err_str:
                    if time.time() - last_data > stream_idle_timeout_val:
                        logger.warning(f"{_C(33)}Stream silent for {stream_idle_timeout_val}s, terminating{_RST}")
                        yield from _finalize_stream("idle_timeout")
                        return
                    yield _sse("ping", {"type": "ping"})
                    continue
                raise

            if not chunk:
                break

            last_data = time.time()
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="replace")
            buffer += chunk

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    finish_reason = finish_reason or "stop"
                    done = True
                    break
                try:
                    ck = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                choices = ck.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                finish = choices[0].get("finish_reason")
                if finish:
                    finish_reason = finish

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

                tool_calls = delta.get("tool_calls")
                if tool_calls:
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
                    for tc in tool_calls:
                        tc_idx = tc.get("index", 0)
                        if tc_idx not in tool_data:
                            tool_block = block_idx
                            block_idx += 1
                            tool_data[tc_idx] = {
                                "block_idx": tool_block,
                                "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:24]}",
                                "name": "",
                                "arguments": "",
                                "started": False,
                            }
                        td = tool_data[tc_idx]
                        if tc.get("id"):
                            td["id"] = tc["id"]
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            td["name"] = fn["name"]
                        args = fn.get("arguments", "")
                        if args:
                            td["arguments"] += args
                        if not td["started"] and td["name"]:
                            td["started"] = True
                            yield _sse("content_block_start", {
                                "type": "content_block_start",
                                "index": td["block_idx"],
                                "content_block": {
                                    "type": "tool_use",
                                    "id": td["id"],
                                    "name": td["name"],
                                    "input": {},
                                },
                            })
                        if args:
                            yield _sse("content_block_delta", {
                                "type": "content_block_delta",
                                "index": td["block_idx"],
                                "delta": {"type": "input_json_delta", "partial_json": args},
                            })

                if finish:
                    done = True
                    break

        # Reached here via `done=True` (finish_reason or [DONE]) OR `not chunk` (EOF)
        if done:
            yield from _finalize_stream("done")
        else:
            yield from _finalize_stream("eof")
    except GeneratorExit:
        raise
    except Exception as e:
        logger.error(f"{_C(31)}X stream_convert unexpected error: {e}{_RST}")
        yield from _finalize_stream("unexpected", error_payload={
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        })
        return


app = Flask(__name__)
app.logger.setLevel(logging.WARNING)


def _auth_check():
    with CONFIG_LOCK:
        expected = CONFIG["proxy_api_key"]
    if not expected:
        return True
    key = (request.headers.get("x-api-key")
           or request.headers.get("Authorization", "").removeprefix("Bearer ").strip())
    return hmac.compare_digest(key.encode(), expected.encode())


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

    allowed, current, limit = _check_rate_limit()
    if not allowed:
        logger.warning(f"{_C(33)}Rate limit: {current}/{limit} per minute{_RST}")
        return jsonify({"type": "error", "error": {"type": "rate_limit_error",
                        "message": f"Rate limit exceeded: {current}/{limit} requests per minute. Retry later."
                        }}), 429

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

    if "/" in model and _MODEL_CACHE["models"]:
        if model not in _MODEL_CACHE["models"]:
            logger.warning(f"{_C(33)}Unknown model: {model}{_RST}")
            return jsonify({"type": "error", "error": {"type": "invalid_request_error",
                            "message": f"Unknown model: {model}. Check available models at /api/models"
                            }}), 400

    oai_body = anthropic_to_openai(body)
    input_est = estimate_tokens(json.dumps(oai_body.get("messages", []), ensure_ascii=False))
    body_size = len(json.dumps(oai_body, ensure_ascii=False))

    logger.info(
        f"{_C(35)}-> REQ{_RST} | {rid[:16]}... | "
        f"model={_C(33)}{raw_model}{_RST}->{_C(32)}{model}{_RST} | "
        f"stream={is_stream} | msgs={msg_count} | "
        f"sys={system_len}c | thinking={thinking_cfg} | {body_size//1024}KB"
    )
    _trace(rid, "request", {"model": model, "stream": is_stream, "msgs": msg_count, "sys_chars": system_len, "thinking": str(thinking_cfg)})

    if body_size > 200 * 1024:
        oai_body = _trim_body(oai_body, 200 * 1024)
        new_size = len(json.dumps(oai_body, ensure_ascii=False))
        logger.warning(f"{_C(33)}Body trimmed: {body_size//1024}KB -> {new_size//1024}KB{_RST}")

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

    with CIRCUIT_LOCK:
        cb = CIRCUIT_STATE.get(model, {"failures": 0, "open_until": 0})
        now_ts = time.time()
        if cb["failures"] >= CIRCUIT_THRESHOLD and now_ts < cb["open_until"]:
            retry_after = int(cb["open_until"] - now_ts)
            logger.warning(f"{_C(33)}Circuit OPEN for {model} | retry-after={retry_after}s{_RST}")
            _trace(rid, "circuit_open", {"model": model, "retry_after": retry_after})
            resp = jsonify({"type": "error", "error": {"type": "overloaded_error",
                            "message": f"Upstream temporarily unavailable. Retry after {retry_after}s."}})
            resp.headers["Retry-After"] = str(retry_after)
            resp.status_code = 503
            return resp

    import requests as http_req

    target_url = f"{base_url}/chat/completions"
    _trace(rid, "upstream", {"url": target_url})
    logger.info(f"{_C(90)}   Upstream: POST {target_url} | body keys={list(oai_body.keys())}{_RST}")

    with CONFIG_LOCK:
        read_timeout = CONFIG.get("upstream_read_timeout", 600)

    RETRYABLE_STATUS = {502, 503, 504}
    MAX_RETRIES = 3
    upstream = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            upstream = http_req.post(
                target_url,
                headers=headers,
                json=oai_body,
                stream=is_stream,
                timeout=(15, read_timeout),
            )
            if upstream.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                upstream.close()
                delay = 2 ** attempt
                err_text = _sanitize_log(upstream.text, 200)
                logger.warning(
                    f"{_C(33)}Retry {attempt + 1}/{MAX_RETRIES}: upstream {upstream.status_code}"
                    f" | delay={delay}s | {err_text}{_RST}"
                )
                time.sleep(delay)
                continue
            break
        except http_req.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                delay = 2 ** attempt
                logger.warning(f"{_C(33)}Retry {attempt + 1}/{MAX_RETRIES}: timeout | delay={delay}s{_RST}")
                time.sleep(delay)
                continue
            logger.error(f"{_C(31)}X Upstream timeout after {MAX_RETRIES + 1} attempts{_RST}")
            return jsonify({"type": "error", "error": {"type": "api_error",
                            "message": "Upstream request timed out"}}), 504
        except Exception as e:
            if attempt < MAX_RETRIES:
                delay = 2 ** attempt
                logger.warning(f"{_C(33)}Retry {attempt + 1}/{MAX_RETRIES}: {e} | delay={delay}s{_RST}")
                time.sleep(delay)
                continue
            logger.error(f"{_C(31)}X Upstream connection failed after {MAX_RETRIES + 1} attempts: {e}{_RST}")
            _update_circuit(model, False)
            return jsonify({"type": "error", "error": {"type": "api_error",
                            "message": str(e)}}), 502

    if upstream is None or upstream.status_code != 200:
        err_text = _sanitize_log(upstream.text, 500) if upstream else "no response"
        status = upstream.status_code if upstream else 502
        logger.error(f"{_C(31)}X Upstream {status} after {MAX_RETRIES + 1} attempts: {err_text}{_RST}")
        _trace(rid, "upstream_error", {"status": status, "error": err_text[:200]})
        _update_circuit(model, False)
        return jsonify({"type": "error", "error": {"type": "api_error",
                        "message": f"Upstream {status}: {err_text}"
                        }}), status

    _update_circuit(model, True)

    logger.info(f"{_C(32)}OK Upstream 200{_RST} | stream={is_stream}")
    _trace(rid, "upstream_ok", {"status": 200, "stream": is_stream})

    anthropic_headers = {
        "x-request-id": rid,
        "anthropic-version": "2023-06-01",
    }

    if is_stream:
        def generate():
            try:
                yield from stream_convert(upstream, model, rid, input_est)
            except GeneratorExit:
                pass
            except Exception as e:
                # stream_convert 已通过 _finalize_stream("unexpected") 下发合规闭合序列
                # 这里只记录日志，不再补发 error / message_stop，避免双重收尾
                logger.error(f"{_C(31)}X Stream outer error (already finalized): {e}{_RST}")
            finally:
                upstream.close()
        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                **anthropic_headers,
                "Cache-Control": "no-cache",
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


def _openai_passthrough():
    if not _auth_check():
        return jsonify({"type": "error", "error": {"type": "authentication_error",
                        "message": "Invalid API key"}}), 401

    allowed, current, limit = _check_rate_limit()
    if not allowed:
        logger.warning(f"{_C(33)}Rate limit: {current}/{limit} per minute{_RST}")
        return jsonify({"error": {"message": f"Rate limit exceeded: {current}/{limit} per minute", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}), 429

    body = request.get_json(force=True)
    model = body.get("model", "")
    is_stream = body.get("stream", False)
    rid = f"chat_{uuid.uuid4().hex[:24]}"

    if "/" in model and _MODEL_CACHE["models"]:
        if model not in _MODEL_CACHE["models"]:
            logger.warning(f"{_C(33)}Unknown model: {model}{_RST}")
            return jsonify({"error": {"message": f"Unknown model: {model}", "type": "invalid_request_error", "code": "model_not_found"}}), 400

    logger.info(
        f"{_C(35)}-> OAI{_RST} | {rid[:16]}... | "
        f"model={_C(32)}{model}{_RST} | stream={is_stream} | msgs={len(body.get('messages', []))}"
    )

    with CONFIG_LOCK:
        base_url = CONFIG["nvidia_base_url"].rstrip("/")
        api_key = CONFIG["nvidia_api_key"]

    if not api_key:
        return jsonify({"error": {"message": "API key not configured", "type": "server_error"}}), 500

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    target_url = f"{base_url}/chat/completions"

    with CIRCUIT_LOCK:
        cb = CIRCUIT_STATE.get(model, {"failures": 0, "open_until": 0})
        now_ts = time.time()
        if cb["failures"] >= CIRCUIT_THRESHOLD and now_ts < cb["open_until"]:
            retry_after = int(cb["open_until"] - now_ts)
            logger.warning(f"{_C(33)}Circuit OPEN for {model} | retry-after={retry_after}s{_RST}")
            resp = jsonify({"error": {"message": f"Upstream temporarily unavailable. Retry after {retry_after}s.", "type": "overloaded_error"}})
            resp.headers["Retry-After"] = str(retry_after)
            resp.status_code = 503
            return resp

    import requests as http_req
    RETRYABLE = {502, 503, 504}
    MAX_RETRIES = 3
    upstream = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            upstream = http_req.post(target_url, headers=headers, json=body, stream=is_stream, timeout=(15, 120))
            if upstream.status_code in RETRYABLE and attempt < MAX_RETRIES:
                upstream.close()
                time.sleep(2 ** attempt)
                continue
            break
        except (http_req.exceptions.Timeout, Exception):
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            return jsonify({"error": {"message": "Upstream unavailable", "type": "server_error"}}), 502

    if upstream is None or upstream.status_code != 200:
        status = upstream.status_code if upstream else 502
        try:
            err = upstream.json()
        except Exception:
            err = {"message": upstream.text[:500] if upstream else "no response"}
        logger.error(f"{_C(31)}X OAI upstream {status}{_RST}")
        _update_circuit(model, False)
        return jsonify(err), status

    _update_circuit(model, True)
    logger.info(f"{_C(32)}OK OAI upstream 200{_RST} | stream={is_stream}")

    if is_stream:
        def stream_passthrough():
            try:
                for chunk in upstream.iter_content(chunk_size=8192, decode_unicode=True):
                    if chunk:
                        yield chunk
            finally:
                upstream.close()
        return Response(stream_passthrough(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                 "x-request-id": rid})

    try:
        result = upstream.json()
    except Exception:
        return jsonify({"error": {"message": "Invalid JSON from upstream", "type": "server_error"}}), 502
    resp = jsonify(result)
    resp.headers["x-request-id"] = rid
    return resp


@app.route("/v1/chat/completions", methods=["POST"])
def oai_completions():
    return _openai_passthrough()


@app.route("/v1/models", methods=["GET"])
def list_models():
    models = _fetch_nvidia_models()
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
        if "rate_limit" in data:
            CONFIG["rate_limit"] = int(data["rate_limit"])
        if "model_aliases" in data and isinstance(data["model_aliases"], dict):
            CONFIG["model_aliases"].update(data["model_aliases"])
    logger.info(f"{_C(32)}Config updated{_RST}")
    _save_config()
    return jsonify({"status": "ok"})


@app.route("/api/usage", methods=["GET"])
def get_usage():
    with USAGE_LOCK:
        qps = _get_qps()
        with CONFIG_LOCK:
            max_qps = CONFIG.get("rate_limit", 40)
        return jsonify({
            "total": dict(TOTAL_STATS),
            "recent": list(USAGE_LOG)[-50:],
            "qps": qps,
            "max_qps": max_qps,
        })


@app.route("/health", methods=["GET"])
def health():
    with CONFIG_LOCK:
        api_key_set = bool(CONFIG["nvidia_api_key"])
    return jsonify({
        "status": "ok",
        "api_key_configured": api_key_set,
    })


@app.route("/api/trace", methods=["GET"])
def list_traces():
    with TRACE_LOCK:
        return jsonify([{"rid": t["rid"], "ts": t["ts"], "steps": len(t["steps"])} for t in TRACE_LOG])


@app.route("/api/trace/<rid>", methods=["GET"])
def get_trace(rid):
    with TRACE_LOCK:
        for t in TRACE_LOG:
            if t["rid"] == rid or t["rid"].startswith(rid):
                return jsonify(t)
    return jsonify({"error": "not found"}), 404


@app.route("/api/models", methods=["GET"])
def api_models():
    models = _fetch_nvidia_models()
    q = request.args.get("q", "").lower()
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 200))
    result = [m for m in models if q in m.lower()] if q else list(models)
    total = len(result)
    start = (page - 1) * per_page
    result = result[start:start + per_page]
    return jsonify({"models": result, "total": total, "cached": bool(_MODEL_CACHE["models"])})


WEB_PAGE = r'''<!DOCTYPE html>
<html lang="zh-CN" data-theme="dark"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NIM to Claude Proxy</title>
<style>
html{transition:background .3s,color .3s}
html[data-theme="dark"]{
  --bg:#0b0f19;--surface:#131a2b;--card:#182035;--accent:#38bdf8;--accent2:#818cf8;
  --text:#e4e8f0;--muted:#7b89a0;--border:#1e2d45;--green:#22c55e;--red:#ef4444;
  --warn:#f59e0b;--orange:#fb923c;--input-bg:#0b0f19;--hover:rgba(255,255,255,.03)
}
html[data-theme="light"]{
  --bg:#f5f6fa;--surface:#fff;--card:#fff;--accent:#0284c7;--accent2:#6366f1;
  --text:#1e293b;--muted:#64748b;--border:#e2e8f0;--green:#16a34a;--red:#dc2626;
  --warn:#d97706;--orange:#ea580c;--input-bg:#f8fafc;--hover:rgba(0,0,0,.02)
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Inter','Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
.container{max-width:1060px;margin:0 auto;padding:24px 20px 40px}

header{display:flex;align-items:center;justify-content:space-between;margin-bottom:28px;flex-wrap:wrap;gap:12px}
header h1{font-size:1.4rem;font-weight:700;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
header .status{display:flex;align-items:center;gap:6px;font-size:.78rem;color:var(--muted)}
header .dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0}
header .dot.on{background:var(--green);box-shadow:0 0 6px var(--green)}
header .dot.off{background:var(--red);box-shadow:0 0 6px var(--red)}

.theme-btn{background:var(--surface);border:1px solid var(--border);color:var(--text);width:34px;height:34px;border-radius:8px;cursor:pointer;font-size:1rem;display:flex;align-items:center;justify-content:center;transition:all .2s;flex-shrink:0}
.theme-btn:hover{border-color:var(--accent);background:var(--hover)}

.stats-row{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:24px}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px 16px;text-align:center;transition:border-color .2s,transform .15s}
.stat-card:hover{border-color:var(--accent);transform:translateY(-1px)}
.stat-card .val{font-size:1.8rem;font-weight:800;line-height:1.2;letter-spacing:-.02em}
.stat-card .val.accent{color:var(--accent)}
.stat-card .val.green{color:var(--green)}
.stat-card .val.warn{color:var(--orange)}
.stat-card .val.red{color:var(--red)}
.stat-card .lbl{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.8px;margin-top:6px}

.qps-row{margin-bottom:24px;display:flex;align-items:center;gap:12px;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:14px 18px}
.qps-row .qps-label{font-size:.8rem;color:var(--muted);white-space:nowrap;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
.qps-bar-wrap{flex:1;height:10px;background:var(--bg);border-radius:99px;overflow:hidden}
.qps-bar{height:100%;border-radius:99px;transition:width .6s ease,background .4s}
.qps-bar.safe{background:var(--green)}
.qps-bar.warn{background:var(--orange)}
.qps-bar.danger{background:var(--red)}
.qps-row .qps-val{font-size:.85rem;font-weight:700;min-width:60px;text-align:right;font-variant-numeric:tabular-nums}

.panel{background:var(--card);border:1px solid var(--border);border-radius:12px;margin-bottom:16px;overflow:hidden}
.panel-header{padding:14px 18px;cursor:pointer;display:flex;align-items:center;justify-content:space-between;user-select:none;transition:background .15s}
.panel-header:hover{background:var(--hover)}
.panel-header h2{font-size:.95rem;font-weight:600;color:var(--text)}
.panel-header .arrow{font-size:.7rem;color:var(--muted);transition:transform .25s}
.panel.open .panel-header .arrow{transform:rotate(90deg)}
.panel-body{display:none;padding:0 18px 18px}
.panel.open .panel-body{display:block}

.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.field{margin-bottom:12px}
.field label{display:block;font-size:.72rem;color:var(--muted);margin-bottom:4px;text-transform:uppercase;letter-spacing:.4px;font-weight:600}
.field input,.field select{width:100%;padding:9px 12px;background:var(--input-bg);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:.85rem;outline:none;transition:border .2s}
.field input:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(56,189,248,.1)}
.check-row{display:flex;align-items:center;gap:8px;margin:8px 0}
.check-row input{accent-color:var(--accent);width:16px;height:16px}
button{padding:9px 20px;border:none;border-radius:8px;font-size:.82rem;font-weight:600;cursor:pointer;transition:all .15s;font-family:inherit}
.btn-primary{background:var(--accent);color:#0b0f19}
.btn-primary:hover{background:#7dd3fc;transform:translateY(-1px)}
.btn-sm{padding:5px 12px;font-size:.75rem}
.btn-danger{background:var(--red);color:#fff}
.btn-danger:hover{background:#f87171}

.endpoint{background:var(--input-bg);border:1px solid var(--border);border-radius:8px;padding:10px 14px;font-family:'Cascadia Code','Fira Code',monospace;font-size:.78rem;color:var(--accent);word-break:break-all}

.help{background:var(--surface);border-left:3px solid var(--accent2);padding:12px 16px;margin:12px 0;border-radius:0 8px 8px 0;font-size:.82rem;line-height:1.7}
.help code{background:var(--input-bg);padding:2px 6px;border-radius:4px;font-size:.78rem;color:var(--accent)}
.warn-box{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-radius:8px;padding:10px 14px;margin:8px 0;font-size:.8rem;color:var(--warn)}

table{width:100%;border-collapse:collapse;font-size:.8rem}
th{text-align:left;color:var(--muted);border-bottom:1px solid var(--border);padding:8px 6px;font-weight:600;font-size:.72rem;text-transform:uppercase;letter-spacing:.4px}
td{padding:7px 6px;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums}
tr:hover td{background:var(--hover)}
.mono{font-family:'Cascadia Code','Fira Code',monospace;font-size:.75rem}

.alias-row{display:flex;gap:8px;margin-bottom:6px;align-items:center}
.alias-row input{flex:1}
.alias-row .arr{color:var(--accent);font-weight:bold;flex:0 0 auto;font-size:.9rem}
.alias-row .del{background:var(--red);color:#fff;padding:4px 10px;border-radius:6px;font-size:.75rem;cursor:pointer;border:none}

.toast{position:fixed;top:20px;right:20px;padding:12px 20px;border-radius:8px;font-weight:600;font-size:.85rem;opacity:0;transform:translateY(-10px);transition:all .3s;z-index:99;pointer-events:none}
.toast.show{opacity:1;transform:translateY(0)}
.toast.ok{background:var(--green);color:#fff}
.toast.err{background:var(--red);color:#fff}

.refresh-badge{font-size:.68rem;color:var(--muted);margin-left:8px;font-weight:400}

.empty-state{text-align:center;padding:32px 16px;color:var(--muted);font-size:.85rem}

.search-box{width:100%;padding:9px 12px;background:var(--input-bg);border:1px solid var(--border);border-radius:8px;color:var(--text);font-size:.85rem;outline:none;margin-bottom:12px;transition:border .2s}
.search-box:focus{border-color:var(--accent);box-shadow:0 0 0 3px rgba(56,189,248,.1)}

.model-count{font-size:.75rem;color:var(--muted);margin-bottom:8px}

.model-row{cursor:pointer;transition:background .15s}
.model-row.copied{animation:flash .6s ease}

@keyframes flash{
  0%,100%{background:transparent}
  50%{background:rgba(34,197,94,.12)}
}

@media(max-width:768px){
  .stats-row{grid-template-columns:repeat(3,1fr)}
  .row{grid-template-columns:1fr}
  header{flex-direction:column;align-items:flex-start}
}
@media(max-width:480px){
  .stats-row{grid-template-columns:repeat(2,1fr)}
}
</style>
</head><body>
<div class="container">
<header>
  <h1>NIM to Claude Proxy</h1>
  <div style="display:flex;align-items:center;gap:12px">
    <div class="status">
      <span class="dot" id="keyDot"></span>
      <span id="keyStatus">Checking...</span>
      <span class="refresh-badge">auto-refresh 15s</span>
    </div>
    <button class="theme-btn" id="themeBtn" onclick="toggleTheme()" title="Toggle theme">&#9728;&#65039;</button>
  </div>
</header>

<div class="stats-row">
  <div class="stat-card"><div class="val accent" id="sReq">0</div><div class="lbl">Requests</div></div>
  <div class="stat-card"><div class="val accent" id="sIn">0</div><div class="lbl">Input Tokens</div></div>
  <div class="stat-card"><div class="val green"  id="sOut">0</div><div class="lbl">Output Tokens</div></div>
  <div class="stat-card"><div class="val green"  id="sThink">0</div><div class="lbl">Thinking</div></div>
  <div class="stat-card"><div class="val"       id="sQps">0</div><div class="lbl">QPM</div></div>
</div>

<div class="qps-row">
  <span class="qps-label">Rate Limit</span>
  <div class="qps-bar-wrap"><div class="qps-bar safe" id="qpsBar" style="width:0%"></div></div>
  <span class="qps-val" id="qpsVal">0 / 40</span>
</div>

<div class="panel open" id="panelUsage">
  <div class="panel-header" onclick="togglePanel('panelUsage')">
    <h2>Recent Activity</h2><span class="arrow">&#9654;</span>
  </div>
  <div class="panel-body">
    <table>
      <thead><tr><th>Time</th><th>Request ID</th><th>Model</th><th>In</th><th>Out</th><th>Think</th><th>Total</th></tr></thead>
      <tbody id="usageBody"></tbody>
    </table>
    <div style="text-align:right;margin-top:10px">
      <button onclick="loadUsage()" class="btn-primary btn-sm">Refresh</button>
    </div>
  </div>
</div>

<div class="panel" id="panelModels">
  <div class="panel-header" onclick="togglePanel('panelModels')">
    <h2>Available Models (from NVIDIA)</h2><span class="arrow">&#9654;</span>
  </div>
  <div class="panel-body">
    <input class="search-box" id="modelSearch" type="text" placeholder="Search models..." oninput="filterModels()">
    <div class="model-count" id="modelCount"></div>
    <div style="max-height:400px;overflow-y:auto">
      <table>
        <thead><tr><th>Model ID</th><th>Provider</th></tr></thead>
        <tbody id="modelsBody"></tbody>
      </table>
    </div>
  </div>
</div>

<div class="panel" id="panelSetup">
  <div class="panel-header" onclick="togglePanel('panelSetup')">
    <h2>Claude Code Setup</h2><span class="arrow">&#9654;</span>
  </div>
  <div class="panel-body">
    <div class="warn-box">WARNING: ANTHROPIC_BASE_URL must end at port only &mdash; do <b>NOT</b> append /v1/messages</div>
    <div class="help">
      <div style="font-weight:600;margin-bottom:6px;font-size:.8rem">~/.claude/settings.json</div>
      <pre style="color:var(--text);white-space:pre-wrap;font-family:'Cascadia Code',monospace;font-size:.78rem;line-height:1.8">{
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
      <span style="color:var(--muted);font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.4px">Anthropic Endpoint (Claude Code)</span>
      <div class="endpoint" id="epUrl"></div>
    </div>
    <div style="margin-top:10px">
      <span style="color:var(--muted);font-size:.75rem;font-weight:600;text-transform:uppercase;letter-spacing:.4px">OpenAI Endpoint</span>
      <div class="endpoint" id="epOaiUrl"></div>
    </div>
  </div>
</div>

<div class="panel" id="panelConfig">
  <div class="panel-header" onclick="togglePanel('panelConfig')">
    <h2>Server Config</h2><span class="arrow">&#9654;</span>
  </div>
  <div class="panel-body">
    <form id="configForm">
      <div class="field"><label>NVIDIA API Base URL</label>
        <input name="nvidia_base_url" id="f_base" placeholder="https://integrate.api.nvidia.com/v1"></div>
      <div class="field"><label>NVIDIA API Key</label>
        <input name="nvidia_api_key" id="f_key" type="password" placeholder="nvapi-xxx"></div>
      <div class="row">
        <div class="field"><label>Proxy API Key</label>
          <input name="proxy_api_key" id="f_proxy" placeholder="sk-proxy-change-me"></div>
        <div class="field"><label>Default Model</label>
          <input name="default_model" id="f_model" placeholder="z-ai/glm-5.1"></div>
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
        <div class="field"><label>Rate Limit (req/min)</label>
          <input name="rate_limit" id="f_ratelimit" type="number" value="40"></div>
      </div>
      <div class="check-row">
        <input type="checkbox" id="f_think" name="enable_thinking" checked>
        <label for="f_think" style="font-size:.85rem;color:var(--text);cursor:pointer">Enable Thinking (default)</label>
      </div>
      <button type="submit" class="btn-primary">Save Config</button>
    </form>
  </div>
</div>

<div class="panel" id="panelAliases">
  <div class="panel-header" onclick="togglePanel('panelAliases')">
    <h2>Model Aliases</h2><span class="arrow">&#9654;</span>
  </div>
  <div class="panel-body">
    <p style="color:var(--muted);font-size:.8rem;margin-bottom:12px">
      Map Claude model names (e.g. claude-sonnet-4-20250514) to NIM models.
      Names containing "/" are treated as NIM-native and passed through.
    </p>
    <div id="aliasContainer"></div>
    <button onclick="addAliasRow()" class="btn-primary btn-sm" style="margin-top:8px">+ Add Alias</button>
    <button onclick="saveAliases()" class="btn-primary btn-sm" style="margin-top:8px;margin-left:6px">Save</button>
  </div>
</div>
</div>

<div class="toast" id="toast"></div>

<script>
const $=s=>document.querySelector(s);
let currentAliases={};
let allModels=[];

(function(){
  const saved=localStorage.getItem('theme')||'dark';
  document.documentElement.setAttribute('data-theme',saved);
  updateThemeIcon();
})();

function toggleTheme(){
  const html=document.documentElement;
  const cur=html.getAttribute('data-theme');
  const next=cur==='dark'?'light':'dark';
  html.setAttribute('data-theme',next);
  localStorage.setItem('theme',next);
  updateThemeIcon();
}

function updateThemeIcon(){
  const btn=document.getElementById('themeBtn');
  const theme=document.documentElement.getAttribute('data-theme')||'dark';
  btn.innerHTML=theme==='dark'?'&#9728;&#65039;':'&#127769;';
}

function togglePanel(id){
  const panel=document.getElementById(id);
  panel.classList.toggle('open');
  if(id==='panelModels'&&panel.classList.contains('open')){loadModels();}
}

function toast(msg,ok=true){
  const t=$('#toast');t.textContent=msg;
  t.className='toast '+(ok?'ok':'err')+' show';
  setTimeout(()=>t.classList.remove('show'),2000);
}

function addAliasRow(from='',to=''){
  const c=$('#aliasContainer');
  const d=document.createElement('div');d.className='alias-row';
  d.innerHTML=`<input class="af" placeholder="claude-sonnet-4-20250514" value="${from}" style="font-size:.8rem">
    <span class="arr">&rarr;</span>
    <input class="at" placeholder="z-ai/glm-5.1" value="${to}" style="font-size:.8rem">
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
  $('#f_ratelimit').value=c.rate_limit||40;
  $('#f_think').checked=!!c.enable_thinking;
  const base=location.origin;
  $('#epBase').textContent=base;
  $('#epUrl').textContent=base+'/v1/messages';
  $('#epOaiUrl').textContent=base+'/v1/chat/completions';
  $('#epKey2').textContent=c.proxy_api_key||'(none)';
  const mdl=c.default_model||'z-ai/glm-5.1';
  $('#epMdl').textContent=mdl;
  $('#epMdl2').textContent=mdl;
  $('#epMdl3').textContent=mdl;
  const keySet=c.nvidia_api_key&&c.nvidia_api_key.length>4;
  $('#keyDot').className='dot '+(keySet?'on':'off');
  $('#keyStatus').textContent=keySet?'API Key configured':'API Key not set';
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
  $('#sQps').textContent=u.qps;
  const pct=u.max_qps>0?Math.min(100,(u.qps/u.max_qps)*100):0;
  const bar=$('#qpsBar');bar.style.width=pct+'%';
  bar.className='qps-bar '+(pct>=90?'danger':pct>=70?'warn':'safe');
  $('#qpsVal').textContent=u.qps+' / '+u.max_qps;
  const rows=u.recent.reverse().map(e=>
    `<tr><td class="mono">${e.ts}</td><td class="mono">${e.rid.slice(0,12)}...</td>
     <td>${e.model}</td><td>${e.in}</td><td>${e.out}</td><td>${e.think}</td>
     <td style="font-weight:600">${e.total}</td></tr>`).join('');
  $('#usageBody').innerHTML=rows||
    '<tr><td colspan="7"><div class="empty-state">No requests yet</div></td></tr>';
}

async function loadModels(){
  try{
    const r=await fetch('/api/models');const d=await r.json();
    allModels=d.models||[];
    renderModels(allModels,d.total||allModels.length);
  }catch(e){
    $('#modelsBody').innerHTML='<tr><td colspan="2"><div class="empty-state">Failed to load models</div></td></tr>';
    $('#modelCount').textContent='';
  }
}

function filterModels(){
  const q=document.getElementById('modelSearch').value.toLowerCase();
  const filtered=q?allModels.filter(m=>m.toLowerCase().includes(q)):allModels;
  renderModels(filtered,allModels.length);
}

function renderModels(models,total){
  document.getElementById('modelCount').textContent=models.length+' / '+total+' models';
  const rows=models.map(m=>{
    const prov=m.split('/')[0];
    return '<tr class="model-row" onclick="copyModel(\''+m.replace(/'/g,"\\'")+'\',this)" title="Click to copy"><td class="mono">'+m+'</td><td style="color:var(--muted)">'+prov+'</td></tr>';
  }).join('');
  document.getElementById('modelsBody').innerHTML=rows||
    '<tr><td colspan="2"><div class="empty-state">No models found</div></td></tr>';
}

function copyModel(id,row){
  navigator.clipboard.writeText(id).then(()=>{
    row.classList.add('copied');
    setTimeout(()=>row.classList.remove('copied'),600);
    toast('Copied: '+id);
  }).catch(()=>toast('Copy failed',false));
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
    _load_config()

    debug_mode = False
    host = "0.0.0.0"
    port = CONFIG["server_port"]
    threads = 8

    for arg in sys.argv[1:]:
        if arg.startswith("--debug="):
            debug_mode = arg.split("=", 1)[1].lower() == "true"
        elif arg.startswith("--host="):
            host = arg.split("=", 1)[1]
        elif arg.startswith("--port="):
            port = int(arg.split("=", 1)[1])
            CONFIG["server_port"] = port
        elif arg.startswith("--threads="):
            threads = int(arg.split("=", 1)[1])

    has_key = bool(CONFIG["nvidia_api_key"])
    server_type = "Flask (debug)" if debug_mode else f"Waitress threads={threads}"

    print(f"""
{_C(36)}======================================================
     NIM -> Claude API Proxy Server v2
======================================================
  Server:              {server_type}
  Anthropic endpoint:  http://{host}:{port}/v1/messages
  Web UI:              http://{host}:{port}/
  Health:              http://{host}:{port}/health
  NVIDIA API Key:      {'SET' if has_key else 'NOT SET (configure via Web UI)'}
  Proxy API Key:       {CONFIG['proxy_api_key'][:20]}
======================================================{_RST}
""")

    if has_key:
        _fetch_nvidia_models()

    if debug_mode:
        app.run(host=host, port=port, debug=False, threaded=True)
    else:
        from waitress import serve
        serve(app, host=host, port=port, threads=threads)

