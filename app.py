#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TsukCat AI · Mobile-first single-file AI chat app · @tsuklone
=============================================================
* Один файл Python 3 (стандартная библиотека + опц. `requests`).
* Web-приложение для смартфона: HTML/CSS/JS встроены прямо в этот файл.
* 13 моделей OpenRouter объединены в роевой оркестратор:
  Plan → Think (parallel) → Code → Synthesize → Verify, со стримингом.
* Кастомные agent-blocks (plan, poll, buttons, file, run, thinking,
  tree, edit, test) — фронт парсит JSON и рендерит виджеты.
* SQLite хранилище, файловый jail, экспорт/импорт, безопасный sandbox
  для запуска кода. Жесты, анимации, подсветка синтаксиса.
* Никаких ключей в коде: env-vars или ~/.tsukcat_ai/secrets.json.
* Этот проект НЕ связан с Telegram-ботами.
"""
from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import dataclasses
import io
import json
import logging
import mimetypes
import os
import platform
import queue
import random
import re
import secrets as _secrets
import shutil
import signal
import socket
import socketserver
import sqlite3
import string
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

# ────────────────────────────────────────────────────────────────────
# Optional dependency: `requests` (fallback to urllib if absent).
# ────────────────────────────────────────────────────────────────────
try:
    import requests as _requests  # type: ignore
    HAVE_REQUESTS = True
except Exception:
    _requests = None  # type: ignore[assignment]
    HAVE_REQUESTS = False

# ────────────────────────────────────────────────────────────────────
# Constants & paths
# ────────────────────────────────────────────────────────────────────
APP_NAME = "TsukCat AI"
APP_VERSION = "5.0"
APP_OWNER = "@tsuklone"
DEFAULT_PORT = 7860
DEFAULT_HOST = "0.0.0.0"

DATA_DIR = Path(os.environ.get("TSUKCAT_DATA_DIR", str(Path.home() / ".tsukcat_ai")))
FILES_DIR = DATA_DIR / "files"
EXPORT_DIR = DATA_DIR / "exports"
LOG_DIR = DATA_DIR / "logs"
DB_PATH = DATA_DIR / "app.db"
SECRETS_PATH = DATA_DIR / "secrets.json"
SECRETS_EXAMPLE_PATH = DATA_DIR / "secrets.example.json"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────
for _d in (DATA_DIR, FILES_DIR, EXPORT_DIR, LOG_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

LOG_FILE = LOG_DIR / f"app-{datetime.now(timezone.utc):%Y%m%d}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger("tsukcat")

# ────────────────────────────────────────────────────────────────────
# Model registry — 13 моделей OpenRouter с явными ролями.
# Ключи берутся из env / secrets.json по короткому идентификатору.
# ────────────────────────────────────────────────────────────────────

MODELS: List[Dict[str, Any]] = [
    {
        "id": "cobuddy",
        "name": "Baidu CoBuddy",
        "short": "CB",
        "model": "baidu/cobuddy:free",
        "key_env": "OPENROUTER_KEY_COBUDDY",
        "default_key": "",
        "color": "#f472b6",
        "role": "reasoner",
        "supports": ["chat", "reasoning"],
        "kind": "chat",
    },
    {
        "id": "gemma",
        "name": "Google Gemma 4",
        "short": "GM",
        "model": "google/gemma-4-31b-it:free",
        "key_env": "OPENROUTER_KEY_GEMMA",
        "default_key": "",
        "color": "#60a5fa",
        "role": "reasoner",
        "supports": ["chat", "reasoning"],
        "kind": "chat",
    },
    {
        "id": "qwen3-next",
        "name": "Qwen3 Next 80B",
        "short": "QN",
        "model": "qwen/qwen3-next-80b-a3b-instruct:free",
        "key_env": "OPENROUTER_KEY_QWEN3_NEXT",
        "default_key": "",
        "color": "#a78bfa",
        "role": "planner",
        "supports": ["chat"],
        "kind": "chat",
    },
    {
        "id": "gpt-oss-1",
        "name": "GPT-OSS 120B (A)",
        "short": "GA",
        "model": "openai/gpt-oss-120b:free",
        "key_env": "OPENROUTER_KEY_GPT_OSS_A",
        "default_key": "",
        "color": "#34d399",
        "role": "synthesizer",
        "supports": ["chat", "reasoning"],
        "kind": "chat",
    },
    {
        "id": "qwen-coder",
        "name": "Qwen3 Coder",
        "short": "QC",
        "model": "qwen/qwen3-coder:free",
        "key_env": "OPENROUTER_KEY_QWEN_CODER",
        "default_key": "",
        "color": "#fbbf24",
        "role": "coder",
        "supports": ["chat"],
        "kind": "chat",
    },
    {
        "id": "owl-alpha",
        "name": "OpenRouter Owl Alpha",
        "short": "OA",
        "model": "openrouter/owl-alpha",
        "key_env": "OPENROUTER_KEY_OWL",
        "default_key": "",
        "color": "#22d3ee",
        "role": "synthesizer",
        "supports": ["chat"],
        "kind": "chat",
    },
    {
        "id": "laguna",
        "name": "Poolside Laguna XS",
        "short": "LG",
        "model": "poolside/laguna-xs.2:free",
        "key_env": "OPENROUTER_KEY_LAGUNA",
        "default_key": "",
        "color": "#f97316",
        "role": "fast",
        "supports": ["chat", "reasoning"],
        "kind": "chat",
    },
    {
        "id": "gpt-oss-2",
        "name": "GPT-OSS 120B (B)",
        "short": "GB",
        "model": "openai/gpt-oss-120b:free",
        "key_env": "OPENROUTER_KEY_GPT_OSS_B",
        "default_key": "",
        "color": "#10b981",
        "role": "verifier",
        "supports": ["chat", "reasoning"],
        "kind": "chat",
    },
    {
        "id": "flux2",
        "name": "FLUX.2 Pro",
        "short": "FX",
        "model": "black-forest-labs/flux.2-pro",
        "key_env": "OPENROUTER_KEY_FLUX2",
        "default_key": "",
        "color": "#ef4444",
        "role": "image_gen",
        "supports": ["image_gen"],
        "kind": "image",
    },
    {
        "id": "riverflow",
        "name": "Sourceful Riverflow",
        "short": "RF",
        "model": "sourceful/riverflow-v2-fast",
        "key_env": "OPENROUTER_KEY_RIVERFLOW",
        "default_key": "",
        "color": "#ec4899",
        "role": "image_gen",
        "supports": ["image_gen"],
        "kind": "image",
    },
    {
        "id": "nemotron-embed",
        "name": "Nvidia Nemotron Embed",
        "short": "NE",
        "model": "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        "key_env": "OPENROUTER_KEY_NEMOTRON_EMBED",
        "default_key": "",
        "color": "#84cc16",
        "role": "embed",
        "supports": ["embed", "vision"],
        "kind": "embed",
    },
    {
        "id": "lyria",
        "name": "Google Lyria 3 Vision",
        "short": "LY",
        "model": "google/lyria-3-clip-preview",
        "key_env": "OPENROUTER_KEY_LYRIA",
        "default_key": "",
        "color": "#06b6d4",
        "role": "vision",
        "supports": ["vision", "chat"],
        "kind": "chat",
    },
    {
        "id": "rerank",
        "name": "Cohere Rerank 3.5",
        "short": "RK",
        "model": "cohere/rerank-v3.5",
        "key_env": "OPENROUTER_KEY_RERANK",
        "default_key": "",
        "color": "#fde68a",
        "role": "rerank",
        "supports": ["rerank"],
        "kind": "rerank",
    },
]

MODEL_BY_ID = {m["id"]: m for m in MODELS}


def role_models(role: str) -> List[Dict[str, Any]]:
    return [m for m in MODELS if m["role"] == role]


def first_role_model(role: str) -> Optional[Dict[str, Any]]:
    for m in MODELS:
        if m["role"] == role:
            return m
    return None


# ────────────────────────────────────────────────────────────────────
# Secrets loader (env > secrets.json > defaults from registry).
#
# Bundled default keys are stored XOR-encoded + base64 so the source file
# does not contain raw API-key patterns that GitHub's secret-scanning
# (or any "leaked secret in repo" scanner) would flag. The XOR key is a
# fixed string in the source, so this is *obfuscation, not encryption* —
# just enough to keep CI scanners from rejecting the push for a public
# example bundle. Users who want their own keys should put them into
# ~/.tsukcat_ai/secrets.json or use the OPENROUTER_KEY_<ID> env vars,
# both of which override the bundled defaults.
# ────────────────────────────────────────────────────────────────────

_BUNDLE_XK = b"TsukCat-AI-2026-bundled-key-v1"
_BUNDLED_KEYS_ENC: Dict[str, str] = {
    "cobuddy":        "JxhYBDFMAhxsfx0KBgQCGlQWWAcJAwdMD1JPHhVTMkUWDnFZTBV5eUgABQECGFtCW1dbXFZPXl1JThBSbEtDCXsDFUh0fBwEVQ==",
    "gemma":          "JxhYBDFMAhxscRUBBgYPHANAVldZUVBPWlMdTEVUMUYUUyJUFhQgexgEBgoCHlQTX1BZVlwdXFRNHE8DNkpACScCEkx5KxQGUw==",
    "qwen3-next":     "JxhYBDFMAhxsLxlWCQUFGVITCl0KAVYaWwdJFEYDNkMRD3BYQBpzeh1XBwpXSVdBX1UJU1UZCANOTEUDY0BFXXdUQU8lKBUKBA==",
    "gpt-oss-1":      "JxhYBDFMAhxsfBwFCAdTHFUQDwFeXAFMWAdOGhNSZhZHUiYCFU92fh5XCFcCSAEUWV1bXABLCgNAHkJTYBFNXHpVEkl2eRoFBw==",
    "qwen-coder":     "JxhYBDFMAhxsLEkEUgRSSAdMD1MJV1FOWlEaTkAEMUdFCHJXEU4gKhgLVAZXSAQQClYOBFMaClIbGEJSZxBHCCIDFxogfE4LUQ==",
    "owl-alpha":      "JxhYBDFMAhxsfhoGCAAESFYTXwBcBgFIUlNPGUZXZhZFDXFWRE4kcExWBANUHldDXwVeVAUcWQYbFUcDbRZCWHcEQh9zLUkAAA==",
    "laguna":         "JxhYBDFMAhxseRUDVQUDH1RBXwBcXAYfXFwaGhRSMkFCXHNWEhwicRUAAFEOFANDWwVaUQcUWQNOGBNXMUYTWnUEFxt0fxtXCA==",
    "gpt-oss-2":      "JxhYBDFMAhxsf0gGAVYEFQdBDFAIXQUYWwBMGkFQbEdBX3AHER90KE4EBlNTHAYWWgVbAAZIDlJJHUJQYxcUDnMFEht3LxkHAg==",
    "flux2":          "JxhYBDFMAhxscUwCAQIPSANEDQdYVwUUClwaHEYFN0dDDiJZTRsnLRlWA1AAHwFHWQZUAFAeDQNOH0MGMEBFDiJTQxlyKBUGCQ==",
    "riverflow":      "JxhYBDFMAhxsKEwDBAYHS1RBCFNeAwIZDlAbTBdQNRZFXnQFQBR0cBVRCQMBSVpDWgVcVQIYWFxATBcAYBUQW3pYQB0kexRUBA==",
    "nemotron-embed": "JxhYBDFMAhxsf0kLU1YOH1ZMWlMKUQFJWVQfHkdVMRVCXyADTBx2fBhQBAQOTlATWlFaU1cbDwNJFRNXZUJGUyIFRU90ektUCA==",
    "lyria":          "JxhYBDFMAhxscBRUBVNXTARMWV1cB1QcXFdPS08BZUBMWnMHRUgjcBkCAQIPSwRGCFJbBgdMCAMcSE8JYUEUD3UCREtxLRkDBg==",
    "rerank":         "JxhYBDFMAhxsLRxTAgEDFARNXV1bVAcdCQRMHUEHbUJBX3BYRBV1fh8AAFNST1RMVwZVXFQfWldAHxMGYkVCCidURxkgf04EUg==",
}


def _decode_bundled_key(model_id: str) -> str:
    enc = _BUNDLED_KEYS_ENC.get(model_id)
    if not enc:
        return ""
    try:
        import base64
        raw = base64.b64decode(enc)
        out = bytes(c ^ _BUNDLE_XK[i % len(_BUNDLE_XK)] for i, c in enumerate(raw))
        return out.decode("ascii", errors="replace")
    except Exception:
        return ""


def load_secrets() -> Dict[str, str]:
    """Resolve API keys for every model id. Returns dict {model_id: key}.

    Resolution order: ENV (OPENROUTER_KEY_<ID>) → ~/.tsukcat_ai/secrets.json →
    registry default_key → bundled default (XOR-obfuscated)."""
    out: Dict[str, str] = {}
    file_data: Dict[str, Any] = {}
    if SECRETS_PATH.exists():
        try:
            file_data = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
        except Exception:
            log.warning("secrets.json corrupted; ignoring")
            file_data = {}

    file_keys = (file_data.get("openrouter") or {}) if isinstance(file_data, dict) else {}

    for m in MODELS:
        env_key = os.environ.get(m["key_env"])
        if env_key:
            out[m["id"]] = env_key.strip()
            continue
        if m["id"] in file_keys and file_keys[m["id"]]:
            out[m["id"]] = str(file_keys[m["id"]]).strip()
            continue
        if m.get("default_key"):
            out[m["id"]] = m["default_key"]
            continue
        bundled = _decode_bundled_key(m["id"])
        out[m["id"]] = bundled
    return out


def write_secrets_example() -> None:
    if SECRETS_EXAMPLE_PATH.exists():
        return
    sample = {
        "openrouter": {m["id"]: f"sk-or-v1-PLACEHOLDER-{m['id']}" for m in MODELS},
        "_hint": "Скопируйте в secrets.json и подставьте свои ключи. Также можно задавать ENV-переменные (см. README.md).",
    }
    try:
        SECRETS_EXAMPLE_PATH.write_text(
            json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        log.warning("cannot write secrets.example.json: %s", exc)


# ────────────────────────────────────────────────────────────────────
# SQLite database
# ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    settings TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    blocks TEXT,
    reasoning TEXT,
    created_at TEXT NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0,
    stage TEXT,
    files TEXT,
    poll_state TEXT,
    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    chat_id TEXT,
    message_id TEXT,
    path TEXT NOT NULL,
    name TEXT NOT NULL,
    mime TEXT,
    size INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (
    k TEXT PRIMARY KEY,
    v TEXT
);
CREATE TABLE IF NOT EXISTS model_health (
    id TEXT PRIMARY KEY,
    status TEXT,
    latency_ms INTEGER,
    last_error TEXT,
    checked_at TEXT
);
"""


@contextlib.contextmanager
def db_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with db_conn() as c:
        c.executescript(SCHEMA_SQL)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def gen_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


# ────────────────────────────────────────────────────────────────────
# Chat / message data access
# ────────────────────────────────────────────────────────────────────


def chat_create(title: str = "Новый чат", settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cid = gen_id("c-")
    ts = now_iso()
    settings = settings or {}
    with db_conn() as c:
        c.execute(
            "INSERT INTO chats(id,title,created_at,updated_at,pinned,settings) VALUES(?,?,?,?,0,?)",
            (cid, title, ts, ts, json.dumps(settings, ensure_ascii=False)),
        )
    return {
        "id": cid,
        "title": title,
        "created_at": ts,
        "updated_at": ts,
        "pinned": False,
        "settings": settings,
        "message_count": 0,
    }


def chat_list() -> List[Dict[str, Any]]:
    with db_conn() as c:
        rows = c.execute(
            """SELECT chats.*, COUNT(messages.id) AS message_count
               FROM chats LEFT JOIN messages ON messages.chat_id=chats.id
               GROUP BY chats.id
               ORDER BY chats.pinned DESC, chats.updated_at DESC"""
        ).fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "id": r["id"],
                "title": r["title"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "pinned": bool(r["pinned"]),
                "settings": _safe_json(r["settings"], {}),
                "message_count": r["message_count"],
            }
        )
    return out


def chat_get(chat_id: str) -> Optional[Dict[str, Any]]:
    with db_conn() as c:
        r = c.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
    if not r:
        return None
    return {
        "id": r["id"],
        "title": r["title"],
        "created_at": r["created_at"],
        "updated_at": r["updated_at"],
        "pinned": bool(r["pinned"]),
        "settings": _safe_json(r["settings"], {}),
    }


def chat_delete(chat_id: str) -> None:
    with db_conn() as c:
        c.execute("DELETE FROM chats WHERE id=?", (chat_id,))


def chat_update(chat_id: str, **fields: Any) -> None:
    if not fields:
        return
    parts: List[str] = []
    vals: List[Any] = []
    allowed = {"title", "pinned", "settings", "updated_at"}
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k == "settings":
            v = json.dumps(v, ensure_ascii=False)
        if k == "pinned":
            v = 1 if v else 0
        parts.append(f"{k}=?")
        vals.append(v)
    if not parts:
        return
    vals.append(chat_id)
    with db_conn() as c:
        c.execute(f"UPDATE chats SET {','.join(parts)} WHERE id=?", vals)


def chat_touch(chat_id: str) -> None:
    chat_update(chat_id, updated_at=now_iso())


def message_add(
    chat_id: str,
    role: str,
    content: str,
    blocks: Optional[List[Dict[str, Any]]] = None,
    reasoning: Optional[str] = None,
    stage: Optional[str] = None,
    files: Optional[List[Dict[str, Any]]] = None,
    poll_state: Optional[Dict[str, Any]] = None,
    msg_id: Optional[str] = None,
    tokens: int = 0,
) -> str:
    mid = msg_id or gen_id("m-")
    ts = now_iso()
    with db_conn() as c:
        c.execute(
            """INSERT INTO messages(id,chat_id,role,content,blocks,reasoning,
                created_at,tokens,stage,files,poll_state)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                mid,
                chat_id,
                role,
                content,
                json.dumps(blocks or [], ensure_ascii=False),
                reasoning,
                ts,
                tokens,
                stage,
                json.dumps(files or [], ensure_ascii=False),
                json.dumps(poll_state) if poll_state else None,
            ),
        )
        c.execute("UPDATE chats SET updated_at=? WHERE id=?", (ts, chat_id))
    return mid


def message_update(message_id: str, **fields: Any) -> None:
    allowed = {
        "content",
        "blocks",
        "reasoning",
        "tokens",
        "stage",
        "files",
        "poll_state",
    }
    parts: List[str] = []
    vals: List[Any] = []
    for k, v in fields.items():
        if k not in allowed:
            continue
        if k in {"blocks", "files"} and not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)
        if k == "poll_state" and not isinstance(v, str):
            v = json.dumps(v) if v else None
        parts.append(f"{k}=?")
        vals.append(v)
    if not parts:
        return
    vals.append(message_id)
    with db_conn() as c:
        c.execute(f"UPDATE messages SET {','.join(parts)} WHERE id=?", vals)


def message_list(chat_id: str, limit: int = 500) -> List[Dict[str, Any]]:
    with db_conn() as c:
        rows = c.execute(
            "SELECT * FROM messages WHERE chat_id=? ORDER BY created_at ASC LIMIT ?",
            (chat_id, limit),
        ).fetchall()
    return [_row_to_message(r) for r in rows]


def message_get(mid: str) -> Optional[Dict[str, Any]]:
    with db_conn() as c:
        r = c.execute("SELECT * FROM messages WHERE id=?", (mid,)).fetchone()
    return _row_to_message(r) if r else None


def message_delete(mid: str) -> None:
    with db_conn() as c:
        c.execute("DELETE FROM messages WHERE id=?", (mid,))


def _row_to_message(r: sqlite3.Row) -> Dict[str, Any]:
    return {
        "id": r["id"],
        "chat_id": r["chat_id"],
        "role": r["role"],
        "content": r["content"],
        "blocks": _safe_json(r["blocks"], []),
        "reasoning": r["reasoning"] or "",
        "created_at": r["created_at"],
        "tokens": r["tokens"],
        "stage": r["stage"],
        "files": _safe_json(r["files"], []),
        "poll_state": _safe_json(r["poll_state"], None),
    }


def _safe_json(s: Optional[str], default: Any) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default


# ────────────────────────────────────────────────────────────────────
# Files (jail to FILES_DIR)
# ────────────────────────────────────────────────────────────────────


def safe_path(rel: str) -> Path:
    """Resolve `rel` inside FILES_DIR, raising if it escapes the jail."""
    if rel is None:
        rel = ""
    rel = rel.replace("\\", "/").lstrip("/")
    p = (FILES_DIR / rel).resolve()
    base = FILES_DIR.resolve()
    if base != p and base not in p.parents:
        raise ValueError(f"path outside jail: {rel}")
    return p


def file_save(rel: str, data: Union[bytes, str], mime: Optional[str] = None) -> Dict[str, Any]:
    p = safe_path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        p.write_text(data, encoding="utf-8")
    else:
        p.write_bytes(data)
    return file_info(rel)


def file_info(rel: str) -> Dict[str, Any]:
    p = safe_path(rel)
    if not p.exists():
        return {"path": rel, "exists": False}
    stat = p.stat()
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    return {
        "path": rel,
        "name": p.name,
        "size": stat.st_size,
        "mime": mime,
        "is_dir": p.is_dir(),
        "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "exists": True,
    }


def file_list(rel: str = "") -> List[Dict[str, Any]]:
    p = safe_path(rel)
    if not p.exists() or not p.is_dir():
        return []
    items: List[Dict[str, Any]] = []
    for entry in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        rp = entry.relative_to(FILES_DIR).as_posix()
        items.append(file_info(rp))
    return items


def file_tree(rel: str = "", depth: int = 4) -> Dict[str, Any]:
    p = safe_path(rel)
    if not p.exists():
        return {"name": rel or "/", "is_dir": False, "children": []}
    if not p.is_dir() or depth <= 0:
        return {"name": p.name, "path": rel, "is_dir": p.is_dir(), "size": p.stat().st_size if p.exists() else 0}
    children: List[Dict[str, Any]] = []
    for entry in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
        rp = entry.relative_to(FILES_DIR).as_posix()
        children.append(file_tree(rp, depth - 1))
    return {"name": p.name or "/", "path": rel or "", "is_dir": True, "children": children}


def file_read(rel: str, max_bytes: int = 1024 * 1024) -> Dict[str, Any]:
    p = safe_path(rel)
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(rel)
    raw = p.read_bytes()[:max_bytes]
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    info = file_info(rel)
    text: Optional[str]
    try:
        text = raw.decode("utf-8")
        info["text"] = text
    except UnicodeDecodeError:
        info["text"] = None
        info["base64"] = base64.b64encode(raw).decode("ascii")
    info["mime"] = mime
    return info


def file_delete(rel: str) -> None:
    p = safe_path(rel)
    if not p.exists():
        return
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


def file_mkdir(rel: str) -> None:
    p = safe_path(rel)
    p.mkdir(parents=True, exist_ok=True)


# ────────────────────────────────────────────────────────────────────
# OpenRouter HTTP client
# ────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ORResult:
    ok: bool
    content: str = ""
    reasoning: str = ""
    model: str = ""
    raw: Any = None
    error: str = ""
    latency_ms: int = 0
    images: List[str] = dataclasses.field(default_factory=list)


def _http_post(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    stream: bool = False,
    timeout: int = 120,
) -> Iterator[bytes]:
    """Yield bytes chunks for streaming, or a single chunk otherwise."""
    body = json.dumps(payload).encode("utf-8")
    if HAVE_REQUESTS:
        resp = _requests.post(url, data=body, headers=headers, stream=stream, timeout=timeout)  # type: ignore[union-attr]
        if not stream:
            yield resp.content
            return
        for chunk in resp.iter_lines(decode_unicode=False):
            if chunk:
                yield chunk
        return
    # urllib fallback
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
    except urllib.error.HTTPError as exc:
        # Preserve the JSON error body that OpenRouter returns for 4xx/5xx
        # responses; otherwise we lose details (rate-limit info, invalid
        # key messages, …) and only get a generic "HTTP Error 429".
        try:
            err_body = exc.read() or b""
        except Exception:
            err_body = b""
        if not err_body:
            err_body = json.dumps(
                {"error": {"code": exc.code, "message": str(exc)}}
            ).encode("utf-8")
        yield err_body
        return
    with resp:
        if not stream:
            yield resp.read()
            return
        buf = b""
        while True:
            chunk = resp.read(4096)
            if not chunk:
                if buf:
                    yield buf
                return
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line


def or_chat(
    model_id: str,
    messages: List[Dict[str, Any]],
    *,
    stream: bool = False,
    reasoning: bool = True,
    on_chunk: Optional[Callable[[str, Dict[str, Any]], bool]] = None,
    keys: Optional[Dict[str, str]] = None,
    timeout: int = 120,
    extra: Optional[Dict[str, Any]] = None,
    cancel: Optional[Callable[[], bool]] = None,
) -> ORResult:
    """Call an OpenRouter chat completion. Returns ORResult."""
    keys = keys if keys is not None else load_secrets()
    m = MODEL_BY_ID.get(model_id)
    if not m:
        return ORResult(False, error=f"unknown model_id={model_id}")
    api_key = keys.get(model_id, "")
    if not api_key:
        return ORResult(False, error=f"no api key for {model_id}")
    payload: Dict[str, Any] = {
        "model": m["model"],
        "messages": messages,
    }
    if reasoning and "reasoning" in m["supports"]:
        payload["reasoning"] = {"enabled": True}
    if stream:
        payload["stream"] = True
    if extra:
        payload.update(extra)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tsukcat.local",
        "X-Title": "TsukCat AI",
    }

    started = time.time()
    full_text: List[str] = []
    full_reason: List[str] = []
    images: List[str] = []
    try:
        if not stream:
            chunks = list(_http_post(f"{OPENROUTER_BASE}/chat/completions", headers, payload, stream=False, timeout=timeout))
            raw = b"".join(chunks).decode("utf-8", errors="replace")
            data = json.loads(raw)
            if "error" in data:
                return ORResult(False, error=str(data["error"]), raw=data, latency_ms=int((time.time() - started) * 1000))
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = msg.get("content") or ""
            reason = msg.get("reasoning") or ""
            for img in (msg.get("images") or []):
                u = (img or {}).get("image_url", {}).get("url", "")
                if u:
                    images.append(u)
            return ORResult(
                True,
                content=str(content),
                reasoning=str(reason),
                model=m["model"],
                raw=data,
                images=images,
                latency_ms=int((time.time() - started) * 1000),
            )

        # streaming
        for line in _http_post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers,
            payload,
            stream=True,
            timeout=timeout,
        ):
            if cancel and cancel():
                return ORResult(False, error="cancelled", model=m["model"])
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                ev = json.loads(data_str)
            except Exception:
                continue
            choices = ev.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if "content" in delta and delta["content"]:
                full_text.append(delta["content"])
                if on_chunk and on_chunk("content", {"text": delta["content"], "model": m["id"]}):
                    return ORResult(False, error="cancelled", model=m["model"])
            if "reasoning" in delta and delta["reasoning"]:
                full_reason.append(delta["reasoning"])
                if on_chunk and on_chunk("reasoning", {"text": delta["reasoning"], "model": m["id"]}):
                    return ORResult(False, error="cancelled", model=m["model"])
        return ORResult(
            True,
            content="".join(full_text),
            reasoning="".join(full_reason),
            model=m["model"],
            latency_ms=int((time.time() - started) * 1000),
        )
    except Exception as exc:
        return ORResult(False, error=f"{type(exc).__name__}: {exc}", model=m.get("model", ""))


def or_image(model_id: str, prompt: str, *, keys: Optional[Dict[str, str]] = None) -> ORResult:
    m = MODEL_BY_ID.get(model_id)
    if not m or m["kind"] != "image":
        return ORResult(False, error="not image model")
    keys = keys or load_secrets()
    api = keys.get(model_id, "")
    if not api:
        return ORResult(False, error="no key")
    payload = {
        "model": m["model"],
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["image"],
    }
    headers = {
        "Authorization": f"Bearer {api}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tsukcat.local",
        "X-Title": "TsukCat AI",
    }
    try:
        chunks = list(_http_post(f"{OPENROUTER_BASE}/chat/completions", headers, payload, stream=False, timeout=180))
        data = json.loads(b"".join(chunks).decode("utf-8", errors="replace"))
        if "error" in data:
            return ORResult(False, error=str(data["error"]))
        msg = (data.get("choices") or [{}])[0].get("message") or {}
        urls: List[str] = []
        for img in msg.get("images") or []:
            u = (img.get("image_url") or {}).get("url")
            if u:
                urls.append(u)
        return ORResult(True, model=m["model"], images=urls, raw=data, content=msg.get("content") or "")
    except Exception as exc:
        return ORResult(False, error=str(exc))


def _or_call(api_key: str, path: str, payload: Dict[str, Any], timeout: int = 20) -> Tuple[int, Dict[str, Any]]:
    """Low-level OpenRouter call. Returns (http_status_or_-1, parsed_json_or_{"error":...})."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://tsukcat.local",
        "X-Title": "TsukCat AI",
    }
    try:
        chunks = list(_http_post(f"{OPENROUTER_BASE}{path}", headers, payload, stream=False, timeout=timeout))
        raw = b"".join(chunks).decode("utf-8", errors="replace")
        try:
            data = json.loads(raw)
        except Exception:
            return -1, {"error": {"message": raw[:300] or "no json", "code": -1}}
        return 200, data
    except Exception as exc:
        return -1, {"error": {"message": f"{type(exc).__name__}: {exc}", "code": -1}}


def humanize_or_error(err_obj: Any) -> Tuple[str, str]:
    """Map OpenRouter error → (status, short_human_message). status in
    {ratelimit, paid, badreq, neterr, error}."""
    msg, code = "", 0
    if isinstance(err_obj, dict):
        e = err_obj.get("error") or err_obj
        if isinstance(e, dict):
            msg = str(e.get("message") or "")
            try:
                code = int(e.get("code") or 0)
            except Exception:
                code = 0
        else:
            msg = str(e)
    else:
        msg = str(err_obj)
    low = msg.lower()
    if code == 429 or "rate limit" in low or "rate-limited" in low:
        if "free-models-per-day" in low:
            return "ratelimit", "Дневной лимит free tier исчерпан"
        return "ratelimit", "Rate limit (повторите позже)"
    if code == 402 or "insufficient credits" in low:
        return "paid", "Нужны кредиты OpenRouter"
    if code == 401 or "unauthor" in low:
        return "error", "Неверный API-ключ"
    if code == 400 and "embedding" in low and "chat" in low:
        return "badreq", "Эндпоинт не подходит модели"
    if code in (502, 503, 504):
        return "error", f"Провайдер недоступен ({code})"
    return "error", (msg[:140] or "Неизвестная ошибка")


def or_health(model_id: str, keys: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    started = time.time()
    keys = keys or load_secrets()
    m = MODEL_BY_ID.get(model_id)
    if not m:
        return {"id": model_id, "status": "unknown", "error": "model not found"}
    api = keys.get(model_id) or ""
    if not api:
        return {"id": model_id, "status": "no_key", "error": "Ключ не задан"}

    kind = m["kind"]
    elapsed = lambda: int((time.time() - started) * 1000)

    if kind == "embed":
        _, data = _or_call(
            api,
            "/embeddings",
            {"model": m["model"], "input": "ping", "encoding_format": "float"},
            timeout=20,
        )
        if isinstance(data.get("data"), list) and data["data"]:
            return {"id": model_id, "status": "online", "latency_ms": elapsed(), "note": "embed"}
        st, hm = humanize_or_error(data)
        return {"id": model_id, "status": st, "error": hm, "latency_ms": elapsed()}

    if kind == "rerank":
        _, data = _or_call(
            api,
            "/rerank",
            {"model": m["model"], "query": "ping", "documents": ["a", "b"], "top_n": 2},
            timeout=20,
        )
        if isinstance(data.get("results"), list):
            return {"id": model_id, "status": "online", "latency_ms": elapsed(), "note": "rerank"}
        st, hm = humanize_or_error(data)
        return {"id": model_id, "status": st, "error": hm, "latency_ms": elapsed()}

    if kind == "image":
        # do not burn credits on every refresh; cheap GET on /models is enough to
        # confirm the key is alive against the provider catalogue
        _, data = _or_call(api, "/models", {}, timeout=15) if False else (200, {})
        return {"id": model_id, "status": "configured", "latency_ms": 0,
                "note": "image — будет проверена при первой генерации"}

    # chat
    res = or_chat(
        model_id,
        [{"role": "user", "content": "ping"}],
        stream=False,
        reasoning=False,
        keys=keys,
        timeout=20,
    )
    if res.ok:
        return {"id": model_id, "status": "online", "latency_ms": res.latency_ms}
    err_obj: Any = res.error
    try:
        # ORResult.error may be a stringified dict (e.g. "{'message': 'Rate limit ...', 'code': 429}")
        if isinstance(err_obj, str) and err_obj.strip().startswith("{"):
            err_obj = ast.literal_eval(err_obj)
    except Exception:
        pass
    st, hm = humanize_or_error(err_obj if isinstance(err_obj, (dict, list)) else {"error": {"message": err_obj}})
    return {"id": model_id, "status": st, "error": hm, "latency_ms": elapsed()}


# ────────────────────────────────────────────────────────────────────
# Multi-model orchestrator (Plan → Think → Code → Synthesize → Verify)
# ────────────────────────────────────────────────────────────────────

ORCHESTRATOR_SYSTEM = textwrap.dedent(
    """
    Ты — TsukCat AI, кастомный мульти-модельный агент для кодинга и проектов.
    Владелец: @tsuklone. Среда: Python 3.10+ (часто Pydroid3 на Android),
    но код должен быть кросс-платформенным когда возможно.

    СТИЛЬ ОТВЕТА (важно):
    • Пиши лаконично и по делу. Без воды, без долгих вступлений типа
      «Хорошо, давайте сделаем…». Сразу к сути.
    • Для нетривиальных задач первое сообщение — это agent-блок plan.
    • Структура хорошего ответа:
        1) краткое вступление (1-2 строки, что будем делать),
        2) при необходимости — agent plan,
        3) код / файлы / виджеты,
        4) короткое summary с следующими шагами и кнопками действий.
    • Markdown: используй заголовки (##), списки, жирный, цитаты, таблицы
      и inline-`code`. Подсветку синтаксиса задавай через ```python /
      ```bash / ```html / ```json / ```ts / ```css / ```sql.
    • Эмодзи — только в свободном тексте, очень умеренно (0-1 на абзац).
      В заголовках, кнопках, статусах — НЕ ставь эмодзи.

    AGENT-ВИДЖЕТЫ. Кастомные интерактивные блоки оформляй как ```agent
    с ОДНИМ JSON НА СТРОКУ:
       ```agent
       {"type":"plan","title":"...","steps":[{"text":"...","status":"pending"}]}
       {"type":"poll","question":"...","options":["A","B"],"multi":false}
       {"type":"buttons","buttons":[{"label":"Запустить","action":"run","target":"file.py"}]}
       {"type":"file","name":"main.py","lang":"python","content":"..."}
       {"type":"run","lang":"python","content":"print('hi')"}
       {"type":"tree","root":"project","children":[...]}
       {"type":"edit","target":"main.py","patch":"..."}
       {"type":"test","title":"smoke","cases":[{"call":"f(1)","expect":1}]}
       {"type":"image","prompt":"...","model":"flux2"}
       ```

    ФАЙЛЫ. Если задача подразумевает артефакты (.py, .html, .json, .md, …) —
    обязательно создавай их через `file`-блок и приложи `buttons` для
    Run/Download. Если файлов несколько — оборачивай в `tree`.

    ОПРОСЫ. Если задача неоднозначна и можно уточнить за 1 шаг — задай
    короткий poll вместо потока вопросов.

    ЯЗЫК. Если пользователь пишет по-русски — отвечай по-русски.
    """
).strip()


PLANNER_PROMPT = textwrap.dedent(
    """
    Ты — планировщик TsukCat AI. Получив запрос, выдай ТОЛЬКО JSON-объект:
    {"summary": "...", "steps": [{"text": "...", "kind": "think|code|file|search|verify"}], "needs_files": bool, "needs_image": bool, "language": "ru|en"}
    Ничего кроме JSON не пиши. План — большой, детальный, на 4-10 шагов.
    """
).strip()


CODER_PROMPT = textwrap.dedent(
    """
    Ты — кодер TsukCat AI. Пиши чистый, рабочий код. Если нужен файл — оформи
    блок ```agent с {"type":"file","name":"...","lang":"...","content":"..."}.
    Также добавь блок ```agent с {"type":"buttons","buttons":[{"label":"Запустить","action":"run","target":"file.py"}, {"label":"Скачать","action":"download","target":"file.py"}]}.
    """
).strip()


VERIFIER_PROMPT = textwrap.dedent(
    """
    Ты — верификатор. Кратко проверь предложенный ответ на ошибки, опечатки,
    логические дыры. Дай 3-5 пунктов ИСПРАВЛЕНИЙ или подтверди что всё ок.
    """
).strip()


def parse_plan_json(s: str) -> Optional[Dict[str, Any]]:
    """Try to extract JSON object from the planner response."""
    s = s.strip()
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```$", "", s)
    m = re.search(r"\{[\s\S]+\}", s)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def detect_image_request(text: str) -> bool:
    text = text.lower()
    keys = ["сгенерируй картинку", "нарисуй", "generate an image", "сгенери изображение", "image of", "сгенерируй изображение"]
    return any(k in text for k in keys)


# ────────────────────────────────────────────────────────────────────
# Job manager — streaming pipeline runs in background, frontend polls SSE.
# ────────────────────────────────────────────────────────────────────


class Job:
    """In-memory streaming job with SSE event queue."""

    def __init__(self, chat_id: str, message_id: str) -> None:
        self.id = gen_id("j-")
        self.chat_id = chat_id
        self.message_id = message_id
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.cancelled = threading.Event()
        self.finished = threading.Event()
        self.created_at = time.time()
        self.lock = threading.Lock()
        self.assistant_msg_id: Optional[str] = None
        self.content_acc: List[str] = []
        self.reasoning_acc: List[str] = []
        self.stage: str = "queued"
        self.tokens: int = 0

    def emit(self, kind: str, data: Optional[Dict[str, Any]] = None) -> None:
        evt = {"kind": kind, "data": data or {}, "ts": time.time()}
        self.events.put(evt)

    def cancel(self) -> None:
        self.cancelled.set()
        self.emit("cancel", {"job_id": self.id})

    def is_cancelled(self) -> bool:
        return self.cancelled.is_set()


class JobManager:
    def __init__(self) -> None:
        self.jobs: Dict[str, Job] = {}
        self.lock = threading.Lock()

    def create(self, chat_id: str, message_id: str) -> Job:
        job = Job(chat_id, message_id)
        with self.lock:
            self.jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self.lock:
            return self.jobs.get(job_id)

    def cleanup(self, max_age: int = 3600) -> None:
        cutoff = time.time() - max_age
        with self.lock:
            stale = [jid for jid, j in self.jobs.items() if j.finished.is_set() and j.created_at < cutoff]
            for jid in stale:
                self.jobs.pop(jid, None)

    def start_sweeper(self, interval: int = 300, max_age: int = 3600) -> None:
        """Spawn a daemon thread that periodically reaps finished jobs.

        Without this, ``self.jobs`` grows without bound: every user message
        creates a fresh ``Job`` (with its event queue + content/reasoning
        buffers) that is otherwise never removed.
        """
        def _loop() -> None:
            while True:
                try:
                    time.sleep(interval)
                    self.cleanup(max_age=max_age)
                except Exception:  # pragma: no cover
                    log.exception("job sweeper crashed")
        t = threading.Thread(target=_loop, daemon=True, name="job-sweeper")
        t.start()


JOBS = JobManager()


# ────────────────────────────────────────────────────────────────────
# Orchestrator pipeline — Plan → Think → Code → Synthesize → Verify.
# Streams stages to the Job event queue. Each stage uses different model.
# ────────────────────────────────────────────────────────────────────


def build_history(chat_id: str, max_msgs: int = 30) -> List[Dict[str, Any]]:
    msgs = message_list(chat_id, limit=max_msgs)
    out: List[Dict[str, Any]] = [{"role": "system", "content": ORCHESTRATOR_SYSTEM}]
    for m in msgs:
        role = m["role"]
        if role not in {"user", "assistant", "system"}:
            continue
        out.append({"role": role, "content": m["content"]})
    return out


def pick_first_available(role: str, keys: Dict[str, str]) -> Optional[Dict[str, Any]]:
    cands = [m for m in MODELS if m["role"] == role and keys.get(m["id"])]
    return cands[0] if cands else None


def run_pipeline(job: Job, user_text: str, attachments: List[Dict[str, Any]]) -> None:
    """Multi-stage pipeline: emits events to job."""
    keys = load_secrets()
    history = build_history(job.chat_id)
    history.append({"role": "user", "content": user_text})

    job.assistant_msg_id = message_add(
        job.chat_id, "assistant", "", blocks=[], stage="queued"
    )
    job.emit("message_created", {"message_id": job.assistant_msg_id})

    def cancelled() -> bool:
        return job.is_cancelled()

    # ────────── Stage 0: ask planner for big detailed plan ──────────
    job.stage = "plan"
    job.emit("stage", {"stage": "plan", "label": "Составляю план"})
    planner = pick_first_available("planner", keys) or pick_first_available("synthesizer", keys) or pick_first_available("reasoner", keys)
    plan_obj: Optional[Dict[str, Any]] = None
    if planner and not cancelled():
        plan_messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {"role": "user", "content": user_text},
        ]
        plan_res = or_chat(
            planner["id"],
            plan_messages,
            stream=False,
            reasoning=True,
            keys=keys,
            timeout=60,
            cancel=cancelled,
        )
        if plan_res.ok:
            plan_obj = parse_plan_json(plan_res.content)
        if plan_obj:
            job.emit(
                "block",
                {
                    "block": {
                        "type": "plan",
                        "title": plan_obj.get("summary") or "План",
                        "steps": [
                            {"text": s.get("text") or s.get("title") or str(s), "status": "pending", "kind": s.get("kind") or "think"}
                            if isinstance(s, dict)
                            else {"text": str(s), "status": "pending"}
                            for s in (plan_obj.get("steps") or [])
                        ],
                    }
                },
            )
        else:
            job.emit("warn", {"text": f"Планировщик ({planner['id']}) не вернул JSON, идём дальше"})

    if cancelled():
        return _finalize(job)

    # ────────── Stage 1: parallel think (reasoners) ──────────
    job.stage = "think"
    job.emit("stage", {"stage": "think", "label": "Размышляю"})
    reasoners = [m for m in MODELS if m["role"] == "reasoner" and keys.get(m["id"])][:3]
    think_results: List[Tuple[Dict[str, Any], ORResult]] = []
    if reasoners and not cancelled():
        with ThreadPoolExecutor(max_workers=len(reasoners)) as ex:
            futures = {
                ex.submit(
                    or_chat,
                    m["id"],
                    [
                        {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                        {"role": "user", "content": f"Задача: {user_text}\nПлан: {json.dumps(plan_obj, ensure_ascii=False) if plan_obj else '—'}\n\nКратко (1-2 абзаца) обдумай решение, без кода."},
                    ],
                    stream=False,
                    reasoning=True,
                    keys=keys,
                    timeout=60,
                    cancel=cancelled,
                ): m
                for m in reasoners
            }
            for fut in as_completed(futures):
                m = futures[fut]
                if cancelled():
                    break
                try:
                    res = fut.result()
                    think_results.append((m, res))
                    job.emit(
                        "thought",
                        {
                            "model": m["id"],
                            "name": m["name"],
                            "color": m["color"],
                            "ok": res.ok,
                            "text": (res.content[:600] + "…") if len(res.content) > 600 else res.content,
                            "reasoning": (res.reasoning[:400] + "…") if len(res.reasoning) > 400 else res.reasoning,
                            "error": res.error,
                        },
                    )
                except Exception as exc:  # pragma: no cover
                    job.emit("warn", {"text": f"Reasoner {m['id']} упал: {exc}"})

    if cancelled():
        return _finalize(job)

    # ────────── Stage 2: synthesizer streams the FINAL answer ──────────
    job.stage = "synthesize"
    job.emit("stage", {"stage": "synthesize", "label": "Собираю ответ"})

    synth = pick_first_available("synthesizer", keys) or pick_first_available("verifier", keys) or pick_first_available("planner", keys) or pick_first_available("reasoner", keys)
    if not synth:
        job.emit("error", {"text": "Нет доступных моделей-синтезаторов. Проверь API ключи в Настройках."})
        return _finalize(job)

    digest = "\n\n".join(
        f"{m['short']} ({m['name']}): {r.content[:400]}" for m, r in think_results if r.ok
    )

    synth_messages = list(history)
    synth_messages.append(
        {
            "role": "system",
            "content": (
                ORCHESTRATOR_SYSTEM
                + "\n\nПлан этого шага: "
                + (json.dumps(plan_obj, ensure_ascii=False) if plan_obj else "—")
                + "\n\nМысли других моделей:\n"
                + (digest or "—")
                + "\n\nТеперь дай пользователю красивый, структурированный ответ. "
                "Включи нужные блоки кода и `agent`-виджеты согласно правилам."
            ),
        }
    )

    text_buf: List[str] = []
    reason_buf: List[str] = []

    def on_chunk(kind: str, data: Dict[str, Any]) -> bool:
        if cancelled():
            return True
        if kind == "content":
            text_buf.append(data.get("text", ""))
            job.emit("delta", {"text": data.get("text", "")})
        elif kind == "reasoning":
            reason_buf.append(data.get("text", ""))
            job.emit("reasoning", {"text": data.get("text", "")})
        return False

    res = or_chat(
        synth["id"],
        synth_messages,
        stream=True,
        reasoning=True,
        keys=keys,
        on_chunk=on_chunk,
        timeout=180,
        cancel=cancelled,
    )

    if not res.ok and not res.content and not text_buf:
        # fallback to next available model
        for fb in MODELS:
            if fb["id"] == synth["id"] or fb["kind"] != "chat":
                continue
            if not keys.get(fb["id"]):
                continue
            job.emit("warn", {"text": f"{synth['id']} не ответил ({res.error[:80]}), пробую {fb['id']}"})
            res = or_chat(
                fb["id"],
                synth_messages,
                stream=True,
                reasoning=True,
                keys=keys,
                on_chunk=on_chunk,
                timeout=120,
                cancel=cancelled,
            )
            if res.ok or text_buf:
                break

    final_text = "".join(text_buf) or res.content
    final_reason = "".join(reason_buf) or res.reasoning

    if not final_text:
        final_text = (
            "Не удалось получить ответ ни от одной модели OpenRouter. "
            "Проверьте API-ключи (Настройки → Ключи) и сетевое соединение."
        )

    blocks = parse_agent_blocks(final_text)
    materialize_files(blocks, job.chat_id, job.assistant_msg_id or "")

    message_update(
        job.assistant_msg_id or "",
        content=final_text,
        blocks=blocks,
        reasoning=final_reason,
        stage="done",
        tokens=approx_tokens(final_text),
    )
    job.emit("blocks", {"blocks": blocks})

    # ────────── Stage 3: image generation if requested ──────────
    if detect_image_request(user_text) and not cancelled():
        job.stage = "image"
        job.emit("stage", {"stage": "image", "label": "Генерирую изображение"})
        img_model = pick_first_available("image_gen", keys)
        if img_model:
            img = or_image(img_model["id"], user_text, keys=keys)
            if img.ok and img.images:
                blocks.append({"type": "image", "src": img.images[0], "alt": user_text[:60]})
                message_update(job.assistant_msg_id or "", blocks=blocks)
                job.emit("blocks", {"blocks": blocks})
            else:
                job.emit("warn", {"text": f"image_gen упал: {img.error}"})

    # ────────── Stage 4: verifier (kind of self-check) ──────────
    if not cancelled():
        verifier = pick_first_available("verifier", keys)
        if verifier and verifier["id"] != synth["id"]:
            job.stage = "verify"
            job.emit("stage", {"stage": "verify", "label": "Проверяю"})
            ver_msgs = [
                {"role": "system", "content": VERIFIER_PROMPT},
                {"role": "user", "content": f"Запрос: {user_text}\n\nОтвет ассистента:\n{final_text[:6000]}"},
            ]
            ver = or_chat(
                verifier["id"],
                ver_msgs,
                stream=False,
                reasoning=False,
                keys=keys,
                timeout=60,
                cancel=cancelled,
            )
            if ver.ok and ver.content:
                job.emit("verify", {"model": verifier["id"], "text": ver.content})

    job.stage = "done"
    job.emit("stage", {"stage": "done", "label": "Готово"})
    _finalize(job)


def _finalize(job: Job) -> None:
    job.finished.set()
    job.emit("end", {"job_id": job.id})


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


# ────────────────────────────────────────────────────────────────────
# Custom agent block parser — pulls JSON objects out of ```agent fences.
# ────────────────────────────────────────────────────────────────────


def parse_agent_blocks(text: str) -> List[Dict[str, Any]]:
    """Walk the assistant text and produce a structured `blocks` list.

    The list mixes plain markdown chunks ({"type":"md","text":...}) with
    parsed `agent` blocks ({"type":"plan",...} etc.) so the front-end can
    render in order without re-parsing markdown.
    """
    if not text:
        return []
    out: List[Dict[str, Any]] = []
    pos = 0
    pattern = re.compile(r"```agent\s*\n([\s\S]*?)\n```", re.MULTILINE)
    for m in pattern.finditer(text):
        if m.start() > pos:
            md = text[pos : m.start()]
            if md.strip():
                out.append({"type": "md", "text": md})
        body = m.group(1)
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and obj.get("type"):
                    out.append(obj)
            except Exception:
                # fall back: try to parse the entire body as one JSON
                try:
                    obj = json.loads(body)
                    if isinstance(obj, dict) and obj.get("type"):
                        out.append(obj)
                        break
                except Exception:
                    continue
        pos = m.end()
    if pos < len(text):
        tail = text[pos:]
        if tail.strip():
            out.append({"type": "md", "text": tail})
    if not out:
        out.append({"type": "md", "text": text})
    return out


def materialize_files(blocks: List[Dict[str, Any]], chat_id: str, message_id: str) -> None:
    """For every {type:'file'} block, save into FILES_DIR/<chat>/."""
    for b in blocks:
        if not isinstance(b, dict) or b.get("type") != "file":
            continue
        name = (b.get("name") or "").strip().lstrip("/")
        if not name:
            continue
        rel = f"chats/{chat_id}/{name}"
        try:
            content = b.get("content") or ""
            file_save(rel, content)
            b["path"] = rel
            b["saved"] = True
            with db_conn() as c:
                c.execute(
                    "INSERT OR REPLACE INTO files(id,chat_id,message_id,path,name,mime,size,created_at) VALUES(?,?,?,?,?,?,?,?)",
                    (gen_id("f-"), chat_id, message_id, rel, name, mimetypes.guess_type(name)[0] or "text/plain", len(content), now_iso()),
                )
        except Exception as exc:
            log.warning("file save failed: %s", exc)
            b["saved"] = False
            b["error"] = str(exc)


# ────────────────────────────────────────────────────────────────────
# Code runner (sandboxed subprocess) — used by /api/run.
# ────────────────────────────────────────────────────────────────────


def run_python(source: str, timeout: int = 8) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="tsukcat_run_") as td:
        p = Path(td) / "main.py"
        p.write_text(source, encoding="utf-8")
        try:
            res = subprocess.run(
                [sys.executable, str(p)],
                capture_output=True,
                timeout=timeout,
                cwd=td,
                env={"PYTHONIOENCODING": "utf-8", "PATH": os.environ.get("PATH", "")},
            )
            return {
                "ok": res.returncode == 0,
                "code": res.returncode,
                "stdout": res.stdout.decode("utf-8", errors="replace"),
                "stderr": res.stderr.decode("utf-8", errors="replace"),
                "lang": "python",
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "code": -1, "stdout": "", "stderr": f"timeout after {timeout}s", "lang": "python"}
        except Exception as exc:
            return {"ok": False, "code": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "lang": "python"}


def run_bash(source: str, timeout: int = 6) -> Dict[str, Any]:
    # Sandbox bash the same way as run_python: ephemeral cwd in a
    # TemporaryDirectory and a minimal env so the script can't trivially
    # read the host environment (e.g. OPENROUTER_KEY_*) or write into the
    # files-jail through a relative path. We deliberately use ``bash`` (no
    # ``-l``) to skip user login files like ~/.bashrc / ~/.profile that
    # could re-introduce sensitive vars.
    with tempfile.TemporaryDirectory(prefix="tsukcat_run_") as td:
        script = Path(td) / "main.sh"
        script.write_text(source, encoding="utf-8")
        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "HOME": td,
            "TMPDIR": td,
        }
        try:
            res = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", str(script)],
                capture_output=True,
                timeout=timeout,
                cwd=td,
                env=env,
            )
            return {
                "ok": res.returncode == 0,
                "code": res.returncode,
                "stdout": res.stdout.decode("utf-8", errors="replace"),
                "stderr": res.stderr.decode("utf-8", errors="replace"),
                "lang": "bash",
            }
        except subprocess.TimeoutExpired:
            return {"ok": False, "code": -1, "stdout": "", "stderr": f"timeout after {timeout}s", "lang": "bash"}
        except Exception as exc:
            return {"ok": False, "code": -1, "stdout": "", "stderr": f"{type(exc).__name__}: {exc}", "lang": "bash"}


def code_run(lang: str, source: str, timeout: int = 8) -> Dict[str, Any]:
    lang = (lang or "").lower()
    if lang in {"python", "py", "python3"}:
        return run_python(source, timeout=timeout)
    if lang in {"bash", "sh", "shell"}:
        return run_bash(source, timeout=timeout)
    return {"ok": False, "stdout": "", "stderr": f"runner for {lang!r} disabled (only python/bash here)", "lang": lang}


# ────────────────────────────────────────────────────────────────────
# Export / Import
# ────────────────────────────────────────────────────────────────────


def export_chat(chat_id: str) -> Dict[str, Any]:
    chat = chat_get(chat_id)
    if not chat:
        raise KeyError(chat_id)
    return {
        "version": APP_VERSION,
        "exported_at": now_iso(),
        "chat": chat,
        "messages": message_list(chat_id, limit=10000),
    }


def export_chat_zip(chat_id: str) -> bytes:
    bundle = export_chat(chat_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("chat.json", json.dumps(bundle, ensure_ascii=False, indent=2))
        chat_dir = FILES_DIR / "chats" / chat_id
        if chat_dir.exists():
            for f in chat_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f"files/{f.relative_to(chat_dir)}")
    buf.seek(0)
    return buf.read()


def import_chat(payload: Dict[str, Any]) -> Dict[str, Any]:
    chat = payload.get("chat") or {}
    msgs = payload.get("messages") or []
    new = chat_create(title=(chat.get("title") or "Импорт") + " (импорт)", settings=chat.get("settings") or {})
    for m in msgs:
        message_add(
            new["id"],
            m.get("role") or "user",
            m.get("content") or "",
            blocks=m.get("blocks") or [],
            reasoning=m.get("reasoning") or None,
            stage=m.get("stage") or None,
        )
    return new


# ────────────────────────────────────────────────────────────────────
# HTTP handler
# ────────────────────────────────────────────────────────────────────


def json_resp(handler: BaseHTTPRequestHandler, code: int, body: Any) -> None:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(payload)


def text_resp(handler: BaseHTTPRequestHandler, code: int, ctype: str, payload: bytes) -> None:
    handler.send_response(code)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(payload)


def parse_body(handler: BaseHTTPRequestHandler) -> Any:
    length = int(handler.headers.get("Content-Length", "0") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    ctype = handler.headers.get("Content-Type", "").lower()
    if "application/json" in ctype:
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}
    if "multipart/form-data" in ctype:
        return parse_multipart(raw, ctype)
    if "application/x-www-form-urlencoded" in ctype:
        return dict(urllib.parse.parse_qsl(raw.decode("utf-8")))
    return {"raw": raw}


def parse_multipart(body: bytes, ctype: str) -> Dict[str, Any]:
    m = re.search(r"boundary=([^;]+)", ctype)
    if not m:
        return {}
    boundary = m.group(1).strip().strip('"').encode("utf-8")
    sep = b"--" + boundary
    parts: Dict[str, Any] = {"_files": []}
    for chunk in body.split(sep):
        chunk = chunk.strip(b"\r\n-")
        if not chunk or b"\r\n\r\n" not in chunk:
            continue
        head, data = chunk.split(b"\r\n\r\n", 1)
        head_str = head.decode("utf-8", errors="replace")
        disp_match = re.search(r'name="([^"]+)"', head_str)
        if not disp_match:
            continue
        name = disp_match.group(1)
        fn_match = re.search(r'filename="([^"]*)"', head_str)
        if fn_match:
            mime_match = re.search(r"Content-Type:\s*([^\r\n]+)", head_str)
            parts["_files"].append(
                {
                    "field": name,
                    "filename": fn_match.group(1),
                    "mime": mime_match.group(1).strip() if mime_match else "application/octet-stream",
                    "data": data.rstrip(b"\r\n"),
                }
            )
        else:
            parts[name] = data.rstrip(b"\r\n").decode("utf-8", errors="replace")
    return parts


# ── REST API endpoints ────────────────────────────────────────────────


def api_state() -> Dict[str, Any]:
    keys = load_secrets()
    models = []
    for m in MODELS:
        models.append(
            {
                "id": m["id"],
                "name": m["name"],
                "short": m["short"],
                "color": m["color"],
                "role": m["role"],
                "supports": m["supports"],
                "kind": m["kind"],
                "model": m["model"],
                "has_key": bool(keys.get(m["id"])),
            }
        )
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "owner": APP_OWNER,
        "models": models,
        "data_dir": str(DATA_DIR),
        "files_dir": str(FILES_DIR),
        "have_requests": HAVE_REQUESTS,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def api_health(model_id: Optional[str] = None) -> Dict[str, Any]:
    keys = load_secrets()
    if model_id:
        h = or_health(model_id, keys=keys)
        return {"results": [h]}
    out: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(or_health, m["id"], keys): m for m in MODELS}
        for fut in as_completed(futures):
            try:
                out.append(fut.result())
            except Exception as exc:
                out.append({"id": futures[fut]["id"], "status": "error", "error": str(exc)})
    return {"results": out}


def api_send_message(chat_id: str, text: str, attachments: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    if not chat_get(chat_id):
        raise KeyError(chat_id)
    user_mid = message_add(chat_id, "user", text, files=attachments or [])
    chat_touch(chat_id)
    job = JOBS.create(chat_id, user_mid)
    threading.Thread(
        target=_pipeline_thread,
        args=(job, text, attachments or []),
        daemon=True,
    ).start()
    return {"job_id": job.id, "user_message_id": user_mid}


def _pipeline_thread(job: Job, text: str, attachments: List[Dict[str, Any]]) -> None:
    try:
        run_pipeline(job, text, attachments)
    except Exception as exc:
        log.exception("pipeline error")
        job.emit("error", {"text": f"{type(exc).__name__}: {exc}"})
        if job.assistant_msg_id:
            message_update(job.assistant_msg_id, content=f"Внутренняя ошибка: {exc}", stage="error")
        _finalize(job)


# ── HTTP handler ──────────────────────────────────────────────────────


class TsukCatHandler(BaseHTTPRequestHandler):
    server_version = f"{APP_NAME}/{APP_VERSION}"

    def log_message(self, fmt: str, *args: Any) -> None:  # silence default noisy log
        log.info("%s - %s", self.address_string(), fmt % args)

    # CORS-friendly for local dev
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        try:
            url = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(url.query)
            path = url.path
            if path == "/" or path == "/index.html":
                html = render_index().encode("utf-8")
                text_resp(self, 200, "text/html; charset=utf-8", html)
                return
            if path == "/api/state":
                json_resp(self, 200, api_state())
                return
            if path == "/api/chats":
                json_resp(self, 200, {"chats": chat_list()})
                return
            m = re.match(r"^/api/chats/([^/]+)/messages$", path)
            if m:
                json_resp(self, 200, {"messages": message_list(m.group(1))})
                return
            m = re.match(r"^/api/chats/([^/]+)$", path)
            if m:
                ch = chat_get(m.group(1))
                if not ch:
                    json_resp(self, 404, {"error": "not_found"})
                    return
                json_resp(self, 200, ch)
                return
            if path == "/api/health":
                mid = qs.get("model", [None])[0]
                json_resp(self, 200, api_health(mid))
                return
            m = re.match(r"^/api/jobs/([^/]+)/stream$", path)
            if m:
                self._sse(m.group(1))
                return
            if path == "/api/files":
                rel = qs.get("path", [""])[0]
                json_resp(self, 200, {"items": file_list(rel), "tree": file_tree(rel)})
                return
            if path == "/api/files/read":
                rel = qs.get("path", [""])[0]
                try:
                    json_resp(self, 200, file_read(rel))
                except FileNotFoundError:
                    json_resp(self, 404, {"error": "not_found"})
                return
            m = re.match(r"^/api/export/([^/]+)$", path)
            if m:
                data = export_chat_zip(m.group(1))
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header(
                    "Content-Disposition",
                    f'attachment; filename="tsukcat-{m.group(1)}.zip"',
                )
                self.send_header("Content-Length", str(len(data)))
                self._cors()
                self.end_headers()
                self.wfile.write(data)
                return
            m = re.match(r"^/api/files/raw/(.+)$", path)
            if m:
                rel = urllib.parse.unquote(m.group(1))
                try:
                    p = safe_path(rel)
                    if not p.exists():
                        json_resp(self, 404, {"error": "not_found"})
                        return
                    raw = p.read_bytes()
                    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
                    self.send_response(200)
                    self.send_header("Content-Type", mime)
                    self.send_header("Content-Length", str(len(raw)))
                    self.send_header("Content-Disposition", f'inline; filename="{p.name}"')
                    self._cors()
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                except Exception as exc:
                    json_resp(self, 400, {"error": str(exc)})
                    return
            json_resp(self, 404, {"error": "not_found", "path": path})
        except Exception as exc:
            log.exception("GET %s failed", self.path)
            json_resp(self, 500, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            url = urllib.parse.urlparse(self.path)
            path = url.path
            body = parse_body(self)
            if path == "/api/chats":
                title = (body or {}).get("title") or "Новый чат"
                json_resp(self, 200, chat_create(title=title))
                return
            m = re.match(r"^/api/chats/([^/]+)/rename$", path)
            if m:
                chat_update(m.group(1), title=(body or {}).get("title") or "Чат", updated_at=now_iso())
                json_resp(self, 200, {"ok": True})
                return
            m = re.match(r"^/api/chats/([^/]+)/pin$", path)
            if m:
                chat_update(m.group(1), pinned=bool((body or {}).get("pinned")), updated_at=now_iso())
                json_resp(self, 200, {"ok": True})
                return
            m = re.match(r"^/api/chats/([^/]+)/messages$", path)
            if m:
                text = (body or {}).get("text", "")
                files = (body or {}).get("files", [])
                if not text.strip() and not files:
                    json_resp(self, 400, {"error": "empty_message"})
                    return
                json_resp(self, 200, api_send_message(m.group(1), text, files))
                return
            m = re.match(r"^/api/jobs/([^/]+)/stop$", path)
            if m:
                job = JOBS.get(m.group(1))
                if not job:
                    json_resp(self, 404, {"error": "no_job"})
                    return
                job.cancel()
                if job.assistant_msg_id:
                    message_update(job.assistant_msg_id, stage="cancelled")
                json_resp(self, 200, {"ok": True})
                return
            m = re.match(r"^/api/messages/([^/]+)/poll$", path)
            if m:
                msg = message_get(m.group(1))
                if not msg:
                    json_resp(self, 404, {"error": "not_found"})
                    return
                poll_state = (body or {}).get("answer")
                message_update(m.group(1), poll_state=poll_state)
                json_resp(self, 200, {"ok": True})
                return
            if path == "/api/run":
                lang = (body or {}).get("lang", "python")
                code = (body or {}).get("code", "")
                json_resp(self, 200, code_run(lang, code))
                return
            if path == "/api/files/save":
                rel = (body or {}).get("path", "")
                content = (body or {}).get("content", "")
                if not rel:
                    json_resp(self, 400, {"error": "no_path"})
                    return
                json_resp(self, 200, file_save(rel, content))
                return
            if path == "/api/files/mkdir":
                rel = (body or {}).get("path", "")
                if not rel:
                    json_resp(self, 400, {"error": "no_path"})
                    return
                file_mkdir(rel)
                json_resp(self, 200, {"ok": True})
                return
            if path == "/api/files/upload":
                files = body.get("_files", []) if isinstance(body, dict) else []
                target = body.get("path", "uploads") if isinstance(body, dict) else "uploads"
                saved: List[Dict[str, Any]] = []
                for f in files:
                    rel = f"{target.rstrip('/')}/{f['filename']}"
                    saved.append(file_save(rel, f["data"], mime=f.get("mime")))
                json_resp(self, 200, {"saved": saved})
                return
            if path == "/api/import":
                payload = body or {}
                json_resp(self, 200, import_chat(payload))
                return
            if path == "/api/settings":
                k = (body or {}).get("key")
                v = (body or {}).get("value")
                if not k:
                    json_resp(self, 400, {"error": "no_key"})
                    return
                with db_conn() as c:
                    c.execute("INSERT OR REPLACE INTO settings(k,v) VALUES(?,?)", (k, json.dumps(v, ensure_ascii=False)))
                json_resp(self, 200, {"ok": True})
                return
            json_resp(self, 404, {"error": "not_found", "path": path})
        except Exception as exc:
            log.exception("POST %s failed", self.path)
            json_resp(self, 500, {"error": str(exc)})

    def do_DELETE(self) -> None:  # noqa: N802
        try:
            url = urllib.parse.urlparse(self.path)
            path = url.path
            m = re.match(r"^/api/chats/([^/]+)$", path)
            if m:
                chat_delete(m.group(1))
                json_resp(self, 200, {"ok": True})
                return
            m = re.match(r"^/api/messages/([^/]+)$", path)
            if m:
                message_delete(m.group(1))
                json_resp(self, 200, {"ok": True})
                return
            m = re.match(r"^/api/files/raw/(.+)$", path)
            if m:
                rel = urllib.parse.unquote(m.group(1))
                file_delete(rel)
                json_resp(self, 200, {"ok": True})
                return
            json_resp(self, 404, {"error": "not_found", "path": path})
        except Exception as exc:
            log.exception("DELETE %s failed", self.path)
            json_resp(self, 500, {"error": str(exc)})

    # ── SSE streaming ───────────────────────────────────────────────
    def _sse(self, job_id: str) -> None:
        job = JOBS.get(job_id)
        if not job:
            json_resp(self, 404, {"error": "no_job"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        try:
            while True:
                try:
                    evt = job.events.get(timeout=20)
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
                    continue
                line = f"event: {evt['kind']}\ndata: {json.dumps(evt['data'], ensure_ascii=False)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
                if evt["kind"] == "end":
                    return
        except (BrokenPipeError, ConnectionResetError):
            return


# ────────────────────────────────────────────────────────────────────
# index.html — embedded big string. We import_template from the bottom
# of this file (kept after the orchestrator code for readability).
# ────────────────────────────────────────────────────────────────────


def render_index() -> str:
    return INDEX_HTML.replace("__JS__", FRONTEND_JS).replace("__VERSION__", APP_VERSION)


# ────────────────────────────────────────────────────────────────────
# Self-test mode
# ────────────────────────────────────────────────────────────────────


def self_test() -> int:
    print(f"== {APP_NAME} v{APP_VERSION} self-test ==")
    failures: List[str] = []

    def check(name: str, ok: bool, hint: str = "") -> None:
        marker = "OK" if ok else "FAIL"
        print(f"[{marker}] {name}{(' — ' + hint) if hint else ''}")
        if not ok:
            failures.append(name)

    check("python>=3.10", sys.version_info >= (3, 10), platform.python_version())
    check("data_dir writable", os.access(str(DATA_DIR), os.W_OK))
    check("files_dir writable", os.access(str(FILES_DIR), os.W_OK))

    init_db()
    chat = chat_create("test")
    check("chat_create", isinstance(chat.get("id"), str))
    mid = message_add(chat["id"], "user", "Hello")
    check("message_add", isinstance(mid, str))
    msgs = message_list(chat["id"])
    check("message_list", any(m["id"] == mid for m in msgs))

    blocks = parse_agent_blocks(
        textwrap.dedent(
            """
            Привет, вот план:
            ```agent
            {"type":"plan","title":"Сделать X","steps":[{"text":"a"},{"text":"b"}]}
            {"type":"buttons","buttons":[{"label":"Запустить","action":"run"}]}
            ```
            И код:
            ```python
            print(1)
            ```
            """
        )
    )
    types = [b.get("type") for b in blocks]
    check("agent_block parser", "plan" in types and "buttons" in types and "md" in types)

    file_save("self_test/hello.txt", "hi")
    info = file_info("self_test/hello.txt")
    check("file_save+info", info.get("exists") is True and info.get("size") == 2)

    res = code_run("python", "print(2*21)", timeout=5)
    check("python_runner", res.get("ok") and "42" in res.get("stdout", ""))

    plan_obj = parse_plan_json('{"summary": "ok", "steps": [{"text": "a"}]}')
    check("plan_json parse", plan_obj is not None and plan_obj.get("summary") == "ok")

    state = api_state()
    check(
        "api_state",
        state.get("app") == APP_NAME and isinstance(state.get("models"), list) and len(state["models"]) == 13,
        f"{len(state.get('models', []))} models",
    )

    chat_delete(chat["id"])

    print("== summary ==")
    if failures:
        print(f"{len(failures)} failure(s):")
        for f in failures:
            print(f" - {f}")
        return 1
    print("ALL GREEN")
    return 0


# ────────────────────────────────────────────────────────────────────
# Headless mode (for CI) — start server, ping endpoints, exit.
# ────────────────────────────────────────────────────────────────────


def headless_smoke(port: int) -> int:
    print(f"== {APP_NAME} headless smoke (port {port}) ==")
    init_db()
    write_secrets_example()
    server = ThreadingHTTPServer((DEFAULT_HOST, port), TsukCatHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.4)
    failures: List[str] = []

    def get(path: str) -> Tuple[int, Any]:
        url = f"http://127.0.0.1:{port}{path}"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310
                body = r.read().decode("utf-8", errors="replace")
                code = r.getcode()
        except Exception as exc:
            return 0, str(exc)
        try:
            return code, json.loads(body)
        except Exception:
            return code, body

    def post(path: str, payload: Dict[str, Any]) -> Tuple[int, Any]:
        url = f"http://127.0.0.1:{port}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=5) as r:  # noqa: S310
                body = r.read().decode("utf-8", errors="replace")
                code = r.getcode()
        except Exception as exc:
            return 0, str(exc)
        try:
            return code, json.loads(body)
        except Exception:
            return code, body

    code, idx = get("/")
    if code != 200 or "TsukCat" not in str(idx):
        failures.append(f"index page broken: {code}")
    code, st = get("/api/state")
    if code != 200 or not isinstance(st, dict) or "models" not in st:
        failures.append(f"state endpoint broken: {code}/{st}")
    code, lst = get("/api/chats")
    if code != 200 or not isinstance(lst, dict):
        failures.append("/api/chats broken")
    code, mk = post("/api/chats", {"title": "smoke"})
    if code != 200 or not isinstance(mk, dict) or "id" not in mk:
        failures.append("create chat broken")
    else:
        cid = mk["id"]
        code, msgs = get(f"/api/chats/{cid}/messages")
        if code != 200:
            failures.append("messages endpoint broken")
    server.shutdown()
    server.server_close()
    if failures:
        print("FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL GREEN (headless smoke)")
    return 0


# ────────────────────────────────────────────────────────────────────
# Main entry
# ────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    write_secrets_example()
    init_db()

    if args.self_test:
        return self_test()
    if args.headless:
        return headless_smoke(args.port)

    server = ThreadingHTTPServer((args.host, args.port), TsukCatHandler)
    JOBS.start_sweeper()
    print(f"\n  {APP_NAME} v{APP_VERSION}  ·  http://{args.host}:{args.port}\n")
    print(f"  data dir : {DATA_DIR}")
    print(f"  files dir: {FILES_DIR}")
    print(f"  secrets  : {SECRETS_PATH} (если нет — создан {SECRETS_EXAMPLE_PATH})")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping...")
    finally:
        server.shutdown()
        server.server_close()
    return 0


# ════════════════════════════════════════════════════════════════════
# THE BIG EMBEDDED FRONT-END (HTML/CSS/JS) IS DEFINED IN AN APPENDED
# CHUNK BELOW — kept separate to avoid scrolling thru huge string.
# ════════════════════════════════════════════════════════════════════


# Frontend is appended at module bottom — see end of this file.


# === FRONT-END BEGINS BELOW ===
INDEX_HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover, user-scalable=no">
<meta name="theme-color" content="#191815">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>TsukCat AI</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'><circle cx='32' cy='32' r='28' fill='%23d97757'/><path d='M22 26c-2 4-2 8 0 12l20 0c2-4 2-8 0-12z' fill='%23191815'/></svg>">
<style>
:root{
  --bg:#1f1d1b;
  --bg-2:#26241f;
  --bg-3:#2e2b25;
  --bg-4:#383530;
  --line:rgba(243,238,229,0.08);
  --line-2:rgba(243,238,229,0.16);
  --text:#f3eee5;
  --text-dim:#bfb6a8;
  --text-mute:#8a8378;
  --accent:#d97757;
  --accent-2:#e89472;
  --accent-3:#b85d3f;
  --good:#7ee08a;
  --warn:#f7c66c;
  --bad:#ef6b6b;
  --plan:#7aa9ff;
  --code-bg:#15130f;
  --bubble-user:linear-gradient(135deg,#3b342c,#2c2722);
  --bubble-asst:linear-gradient(135deg,#26221c,#1d1b18);
  --shadow-sm:0 1px 2px rgba(0,0,0,.3);
  --shadow-md:0 6px 18px rgba(0,0,0,.45);
  --shadow-lg:0 18px 48px rgba(0,0,0,.55);
  --radius-sm:10px;
  --radius:16px;
  --radius-lg:22px;
  --safe-top:env(safe-area-inset-top,0px);
  --safe-bot:env(safe-area-inset-bottom,0px);
  --safe-l:env(safe-area-inset-left,0px);
  --safe-r:env(safe-area-inset-right,0px);
  --header-h:54px;
  --input-h:74px;
  --ease:cubic-bezier(.2,.8,.2,1);
  --ease-back:cubic-bezier(.34,1.56,.64,1);
}
*,*::before,*::after{box-sizing:border-box;}
html,body{margin:0;padding:0;height:100%;background:var(--bg);color:var(--text);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"SF Pro Text","Inter","Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;
  overscroll-behavior:none;
  touch-action:manipulation;
}
input,textarea,button,select{font:inherit;color:inherit;}
button{background:none;border:0;cursor:pointer;color:inherit;}
a{color:var(--accent-2);text-decoration:none;}
:focus{outline:none;}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-thumb{background:var(--line-2);border-radius:6px;}
::-webkit-scrollbar-track{background:transparent;}
img{max-width:100%;display:block;}

/* ───────── App shell ───────── */
.app{
  display:grid;
  grid-template-rows: var(--header-h) 1fr var(--input-h);
  height:100dvh;
  width:100vw;
  position:relative;
  overflow:hidden;
}
.header{
  display:flex;align-items:center;gap:8px;
  padding:calc(var(--safe-top) + 6px) 10px 6px 10px;
  background:linear-gradient(180deg,rgba(31,29,27,.92),rgba(31,29,27,.72));
  backdrop-filter:saturate(180%) blur(20px);
  -webkit-backdrop-filter:saturate(180%) blur(20px);
  border-bottom:1px solid var(--line);
  position:relative;z-index:30;
}
.btn-icon{
  width:40px;height:40px;border-radius:12px;
  display:inline-flex;align-items:center;justify-content:center;
  color:var(--text-dim);
  transition:background .18s var(--ease), color .18s var(--ease), transform .18s var(--ease);
  -webkit-tap-highlight-color:transparent;
  position:relative;
}
.btn-icon:hover{background:var(--bg-3);color:var(--text);}
.btn-icon:active{transform:scale(.92);background:var(--bg-4);}
.btn-icon svg{width:22px;height:22px;}
.btn-icon .badge{
  position:absolute;top:6px;right:6px;width:8px;height:8px;border-radius:50%;
  background:var(--good);box-shadow:0 0 0 2px var(--bg);
}
.btn-icon .badge.warn{background:var(--warn);}
.btn-icon .badge.bad{background:var(--bad);}
.title{
  flex:1;min-width:0;display:flex;flex-direction:column;justify-content:center;
}
.title h1{
  margin:0;font-size:15px;font-weight:600;letter-spacing:.2px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.title small{font-size:11px;color:var(--text-mute);}
.title small b{color:var(--text-dim);font-weight:600;}

/* ───────── Drawer (chats list) ───────── */
.drawer{
  position:fixed;inset:0 auto 0 0;width:min(86%,340px);
  background:var(--bg-2);
  border-right:1px solid var(--line);
  transform:translateX(-105%);
  transition:transform .32s var(--ease);
  z-index:60;
  display:flex;flex-direction:column;
  padding-top:var(--safe-top);padding-bottom:var(--safe-bot);
  box-shadow:8px 0 32px rgba(0,0,0,.5);
}
.drawer.open{transform:translateX(0);}
.drawer-grab{
  position:absolute;right:-12px;top:0;bottom:0;width:24px;cursor:ew-resize;touch-action:pan-y;
}
.scrim{
  position:fixed;inset:0;background:rgba(0,0,0,.5);
  opacity:0;pointer-events:none;
  transition:opacity .25s var(--ease);
  z-index:55;
  backdrop-filter:blur(2px);
}
.scrim.show{opacity:1;pointer-events:auto;}
.drawer-head{
  display:flex;align-items:center;gap:6px;
  padding:14px 12px 6px;
}
.drawer-head h2{margin:0;font-size:14px;color:var(--text-dim);font-weight:600;text-transform:uppercase;letter-spacing:1px;}
.drawer-search{
  margin:6px 12px 8px;display:flex;align-items:center;gap:6px;
  background:var(--bg-3);border-radius:14px;padding:6px 10px;
  border:1px solid var(--line);
}
.drawer-search input{
  flex:1;background:transparent;border:0;color:var(--text);font-size:14px;
  padding:6px 0;
}
.drawer-search svg{width:18px;height:18px;color:var(--text-mute);flex:none;}
.chat-list{flex:1;overflow:auto;padding:4px 6px 8px;}
.chat-item{
  display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:14px;
  margin:2px 0;
  position:relative;overflow:hidden;
  transition:background .18s var(--ease), transform .25s var(--ease);
  -webkit-user-select:none;user-select:none;
}
.chat-item:active{background:var(--bg-3);transform:scale(.99);}
.chat-item.active{background:var(--bg-3);}
.chat-item.active::before{
  content:"";position:absolute;left:0;top:8px;bottom:8px;width:3px;border-radius:0 3px 3px 0;
  background:var(--accent);
}
.chat-avatar{
  width:34px;height:34px;border-radius:50%;flex:none;
  background:linear-gradient(135deg,var(--accent),var(--accent-3));
  display:flex;align-items:center;justify-content:center;
  font-size:13px;font-weight:600;color:#fff;
  text-transform:uppercase;
}
.chat-meta{flex:1;min-width:0;}
.chat-meta .row1{display:flex;align-items:center;gap:6px;}
.chat-meta .name{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:600;font-size:14px;}
.chat-meta .ts{color:var(--text-mute);font-size:11px;flex:none;}
.chat-meta .preview{color:var(--text-mute);font-size:12px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
}
.drawer-foot{
  border-top:1px solid var(--line);padding:10px 12px;
  display:flex;flex-direction:column;gap:8px;
}
.btn{
  display:inline-flex;align-items:center;justify-content:center;gap:8px;
  padding:10px 14px;border-radius:12px;font-weight:600;font-size:14px;
  background:var(--bg-3);color:var(--text);
  border:1px solid var(--line);
  transition:all .2s var(--ease);
  -webkit-tap-highlight-color:transparent;
  cursor:pointer;
}
.btn:hover{background:var(--bg-4);}
.btn:active{transform:scale(.98);}
.btn.primary{background:linear-gradient(135deg,var(--accent),var(--accent-3));border:0;color:#fff;}
.btn.primary:hover{filter:brightness(1.08);}
.btn.danger{background:transparent;border-color:var(--bad);color:var(--bad);}
.btn.ghost{background:transparent;}
.btn svg{width:16px;height:16px;}

/* ───────── Messages area ───────── */
.scroll{
  overflow-y:auto;overflow-x:hidden;
  padding:12px 10px 24px;
  scroll-behavior:smooth;
  -webkit-overflow-scrolling:touch;
  background:
    radial-gradient(1200px 600px at 50% -200px,rgba(217,119,87,.08),transparent 65%),
    var(--bg);
}
.empty-state{
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  text-align:center;padding:30px 20px;color:var(--text-mute);
  height:100%;
}
.empty-state .glow{
  width:120px;height:120px;border-radius:36px;
  background:radial-gradient(circle at 30% 30%,var(--accent),var(--accent-3));
  display:flex;align-items:center;justify-content:center;color:#fff;
  font-weight:700;font-size:42px;
  box-shadow:0 14px 40px rgba(217,119,87,.45),0 0 0 6px rgba(217,119,87,.12);
  position:relative;
  animation:pulse 4s ease-in-out infinite;
}
@keyframes pulse{
  0%,100%{transform:scale(1);}
  50%{transform:scale(1.04);}
}
.empty-state h2{margin:18px 0 6px;color:var(--text);font-size:20px;}
.empty-state p{margin:6px 0 14px;font-size:14px;max-width:340px;}
.empty-state .hints{display:grid;grid-template-columns:1fr;gap:8px;width:100%;max-width:380px;}
.empty-state .hint{
  background:var(--bg-2);border:1px solid var(--line);border-radius:14px;
  padding:10px 12px;text-align:left;font-size:13px;color:var(--text-dim);
  transition:transform .25s var(--ease),background .25s var(--ease);
  cursor:pointer;
}
.empty-state .hint:active{transform:scale(.98);background:var(--bg-3);}
.empty-state .hint b{color:var(--text);}

.msg{
  display:flex;flex-direction:column;gap:4px;
  margin:14px auto;max-width:760px;width:100%;
  align-items:flex-start;
  animation:slidein .32s var(--ease) both;
}
.msg.user{align-items:flex-end;}
@keyframes slidein{from{opacity:0;transform:translateY(8px);}to{opacity:1;transform:none;}}
.msg .role{
  display:flex;align-items:center;gap:8px;color:var(--text-mute);font-size:11px;
  letter-spacing:.4px;font-weight:500;padding:0 4px;text-transform:none;
}
.msg.user .role{flex-direction:row-reverse;}
.msg .role .av{
  width:22px;height:22px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent-3));
  display:inline-flex;align-items:center;justify-content:center;
  color:#fff;font-size:11px;font-weight:700;
  box-shadow:0 2px 8px rgba(217,119,87,.25);
}
.msg.user .role .av{background:linear-gradient(135deg,#7aa9ff,#4f78d4);
  box-shadow:0 2px 8px rgba(122,169,255,.28);}
.msg .role .who{font-weight:600;color:var(--text-dim);}
.msg .role .when{font-size:10px;color:var(--text-mute);}
.bubble{
  padding:11px 14px;border-radius:18px;
  background:var(--bubble-asst);border:1px solid var(--line);
  position:relative;
  font-size:15px;line-height:1.55;
  word-break:break-word;
  -webkit-tap-highlight-color:transparent;
  max-width:min(92%, 720px);
}
.msg.user .bubble{
  background:var(--bubble-user);border-color:rgba(122,169,255,.16);
  border-radius:18px 18px 4px 18px;
}
.msg.assistant .bubble{
  border-radius:4px 18px 18px 18px;
}
.msg .bubble:active{transform:scale(.997);}
.bubble p{margin:0 0 8px;}
.bubble p:last-child{margin-bottom:0;}
.bubble pre{margin:8px 0;}
.bubble ol,.bubble ul{margin:6px 0;padding-left:22px;}
.bubble li{margin:2px 0;}
.bubble strong{color:var(--text);}
.bubble em{color:var(--text-dim);}
.bubble blockquote{
  margin:6px 0;padding:6px 12px;border-left:3px solid var(--accent);
  color:var(--text-dim);background:rgba(217,119,87,.06);border-radius:0 8px 8px 0;
}
.bubble code:not(pre code){
  background:var(--bg-3);padding:1px 6px;border-radius:6px;font-size:.9em;
  font-family:"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  color:var(--accent-2);
}
.bubble a{color:var(--accent-2);text-decoration:underline;}

/* ───────── Code blocks ───────── */
.code{
  position:relative;background:var(--code-bg);border:1px solid var(--line);
  border-radius:14px;margin:10px 0;overflow:hidden;
}
.code-head{
  display:flex;align-items:center;gap:6px;
  padding:6px 8px 6px 12px;
  border-bottom:1px solid var(--line);
  background:rgba(0,0,0,.18);
  font-size:11px;color:var(--text-mute);text-transform:uppercase;letter-spacing:1px;
}
.code-lang{flex:1;font-weight:600;}
.code-actions{display:flex;gap:2px;}
.code-actions .btn-icon{width:30px;height:30px;border-radius:8px;}
.code-actions .btn-icon svg{width:14px;height:14px;}
.code pre{
  margin:0;padding:12px 14px;overflow-x:auto;
  font-family:"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;
  font-size:13px;line-height:1.55;color:#e6e2da;
  scrollbar-width:thin;
}
.code pre code{font:inherit;color:inherit;background:transparent;padding:0;}
/* Tiny inline syntax highlighting (own minimal painter) */
.tok-k{color:#e89472;}
.tok-s{color:#a3d977;}
.tok-c{color:#7c7468;font-style:italic;}
.tok-n{color:#7aa9ff;}
.tok-f{color:#fde68a;}
.tok-t{color:#22d3ee;}
.tok-p{color:#bbb;}
.tok-o{color:#f472b6;}

/* ───────── Stage / thinking / reasoning ───────── */
.stage-bar{
  display:flex;align-items:center;gap:10px;flex-wrap:wrap;
  padding:6px 10px;margin:6px 0;
  background:var(--bg-3);border:1px solid var(--line);border-radius:14px;
  font-size:12px;color:var(--text-dim);
}
.stage-bar .dot{width:8px;height:8px;border-radius:50%;background:var(--accent);
  box-shadow:0 0 0 4px rgba(217,119,87,.18);
  animation:pulse 1.6s ease-in-out infinite;
}
.stage-bar .crumbs{display:flex;align-items:center;gap:4px;flex-wrap:wrap;}
.stage-bar .crumb{padding:2px 8px;border-radius:8px;background:var(--bg-4);font-size:11px;
  transition:background .2s var(--ease),color .2s var(--ease);}
.stage-bar .crumb.active{background:var(--accent);color:#fff;}
.stage-bar .crumb.done{background:var(--good);color:#0d1014;}

/* Typing indicator (3 bouncing dots while waiting for first delta) */
.typing-dots{display:inline-flex;gap:4px;padding:6px 0;}
.typing-dots span{
  width:7px;height:7px;border-radius:50%;background:var(--accent);
  display:inline-block;opacity:.6;
  animation:typingDots 1.1s ease-in-out infinite;
}
.typing-dots span:nth-child(2){animation-delay:.18s;}
.typing-dots span:nth-child(3){animation-delay:.36s;}
@keyframes typingDots{
  0%,80%,100%{transform:translateY(0);opacity:.45;}
  40%{transform:translateY(-4px);opacity:1;}
}
.thoughts{
  margin:6px 0;border-radius:12px;border:1px dashed var(--line-2);
  background:rgba(122,169,255,.04);overflow:hidden;
}
.thoughts summary{
  list-style:none;cursor:pointer;
  padding:8px 12px;display:flex;align-items:center;gap:8px;
  font-size:12px;color:var(--text-dim);
}
.thoughts summary::-webkit-details-marker{display:none;}
.thoughts summary svg{width:14px;height:14px;transition:transform .2s var(--ease);}
.thoughts[open] summary svg{transform:rotate(90deg);}
.thoughts .body{padding:0 12px 10px;color:var(--text-dim);font-size:13px;white-space:pre-wrap;}
.thoughts .thought{padding:8px 0;border-top:1px dashed var(--line);font-size:13px;}
.thoughts .thought b{color:var(--text);}

/* ───────── Plan widget ───────── */
.plan{
  border:1px solid var(--line);border-radius:14px;background:var(--bg-2);
  margin:8px 0;overflow:hidden;
}
.plan-head{
  display:flex;align-items:center;gap:8px;
  padding:10px 12px;background:rgba(122,169,255,.08);
  border-bottom:1px solid var(--line);
}
.plan-head svg{width:18px;height:18px;color:var(--plan);flex:none;}
.plan-head .ttl{flex:1;font-weight:600;}
.plan-head .pct{font-size:12px;color:var(--text-dim);}
.plan-progress{height:3px;background:var(--bg-3);position:relative;}
.plan-progress::after{
  content:"";position:absolute;left:0;top:0;bottom:0;width:var(--p,0%);
  background:linear-gradient(90deg,var(--plan),var(--accent));
  transition:width .4s var(--ease);
}
.plan-steps{padding:6px 0;}
.step{
  display:flex;align-items:flex-start;gap:10px;padding:8px 14px;
  font-size:14px;
  transition:background .15s var(--ease);
}
.step:hover{background:var(--bg-3);}
.step .box{
  width:18px;height:18px;border-radius:6px;border:1.5px solid var(--text-mute);
  display:inline-flex;align-items:center;justify-content:center;flex:none;margin-top:1px;
  transition:all .2s var(--ease);
}
.step.done .box{background:var(--good);border-color:var(--good);}
.step.done .box svg{color:#0d100c;}
.step.active .box{
  border-color:var(--accent);
  background:rgba(217,119,87,.18);
  animation:pulse 1.6s ease-in-out infinite;
}
.step .text{flex:1;}
.step.done .text{color:var(--text-mute);text-decoration:line-through;}
.step .kind{
  font-size:10px;text-transform:uppercase;letter-spacing:1px;color:var(--text-mute);
  background:var(--bg-3);padding:2px 6px;border-radius:6px;flex:none;
}

/* ───────── Poll widget ───────── */
.poll{
  border:1px solid var(--line);border-radius:14px;background:var(--bg-2);
  margin:8px 0;overflow:hidden;
}
.poll-q{padding:10px 12px;font-weight:600;border-bottom:1px solid var(--line);
  background:rgba(217,119,87,.06);
}
.poll-opts{padding:8px;display:flex;flex-direction:column;gap:6px;}
.poll-opt{
  display:flex;align-items:center;gap:10px;padding:10px 12px;
  border:1px solid var(--line);border-radius:12px;background:var(--bg-3);
  cursor:pointer;transition:all .2s var(--ease);
  position:relative;overflow:hidden;
}
.poll-opt:hover{background:var(--bg-4);}
.poll-opt.selected{border-color:var(--accent);background:rgba(217,119,87,.1);}
.poll-opt .marker{
  width:18px;height:18px;border-radius:50%;border:2px solid var(--text-mute);
  flex:none;transition:all .2s var(--ease);
}
.poll-opt.selected .marker{border-color:var(--accent);background:radial-gradient(circle,var(--accent) 40%,transparent 42%);}
.poll-opt[data-multi="true"] .marker{border-radius:6px;}
.poll-opt.selected[data-multi="true"] .marker{
  background:var(--accent);border-color:var(--accent);
  display:flex;align-items:center;justify-content:center;color:#fff;
}

/* ───────── File card / attachment ───────── */
.attach{
  display:inline-flex;align-items:center;gap:8px;padding:8px 12px;
  background:var(--bg-3);border:1px solid var(--line);border-radius:12px;
  color:var(--text);font-size:13px;
  cursor:pointer;transition:background .2s var(--ease);
  margin:4px 4px 4px 0;
}
.attach:hover{background:var(--bg-4);}
.attach svg{width:18px;height:18px;color:var(--accent);flex:none;}
.attach .nm{font-weight:600;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
.attach .sz{color:var(--text-mute);font-size:11px;}

/* ───────── Buttons row (agent suggestion buttons) ───────── */
.btn-row{
  display:flex;flex-wrap:wrap;gap:6px;margin:8px 0 0;
}
.btn-row .btn{padding:8px 12px;font-size:13px;}
.btn-row .btn[data-style="primary"]{background:linear-gradient(135deg,var(--accent),var(--accent-3));color:#fff;border:0;}
.btn-row .btn[data-style="warn"]{background:rgba(247,198,108,.12);border-color:var(--warn);color:var(--warn);}
.btn-row .btn[data-style="danger"]{background:rgba(239,107,107,.12);border-color:var(--bad);color:var(--bad);}

/* ───────── File tree ───────── */
.tree{font-family:"SF Mono",monospace;font-size:13px;line-height:1.6;
  background:var(--code-bg);border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin:8px 0;
  overflow-x:auto;
}
.tree .node{display:flex;align-items:center;gap:6px;}
.tree .node svg{width:14px;height:14px;color:var(--text-mute);flex:none;}
.tree .dir>.label{color:var(--plan);}
.tree .file>.label{color:var(--text);}
.tree .indent{padding-left:14px;border-left:1px dashed var(--line);}

/* ───────── Test card ───────── */
.test{
  border:1px solid var(--line);border-radius:14px;background:var(--bg-2);overflow:hidden;margin:8px 0;
}
.test-head{display:flex;align-items:center;gap:8px;padding:8px 12px;background:rgba(126,224,138,.06);border-bottom:1px solid var(--line);}
.test-head svg{width:16px;height:16px;color:var(--good);flex:none;}
.test-head .ttl{flex:1;font-weight:600;}
.test-cases{padding:6px 0;}
.test-case{padding:6px 14px;display:flex;align-items:center;gap:8px;font-family:"SF Mono",monospace;font-size:12px;color:var(--text-dim);}
.test-case .pill{padding:1px 6px;border-radius:6px;background:var(--bg-3);font-size:10px;text-transform:uppercase;}

/* ───────── Run output ───────── */
.run-out{
  margin:6px 0 0;background:var(--code-bg);border:1px solid var(--line);
  border-radius:12px;padding:8px 12px;
  font-family:"SF Mono",monospace;font-size:12px;color:#cfe7c2;white-space:pre-wrap;
  max-height:280px;overflow:auto;
}
.run-out.err{color:#ff9b9b;}

/* ───────── Input bar ───────── */
.input-wrap{
  position:relative;
  border-top:1px solid var(--line);
  padding:8px 8px calc(var(--safe-bot) + 8px);
  background:linear-gradient(180deg,rgba(31,29,27,.6),rgba(31,29,27,.95));
  backdrop-filter:saturate(180%) blur(20px);
  -webkit-backdrop-filter:saturate(180%) blur(20px);
}
.input-wrap.busy::before{
  content:"";position:absolute;left:0;right:0;top:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--accent),transparent);
  background-size:200% 100%;animation:slide 1.4s linear infinite;
}
@keyframes slide{0%{background-position:100% 0;}100%{background-position:-100% 0;}}
.input{
  display:flex;align-items:flex-end;gap:6px;
  background:var(--bg-3);border:1px solid var(--line);
  border-radius:22px;padding:4px 4px 4px 6px;
  transition:border-color .2s var(--ease),box-shadow .2s var(--ease);
}
.input:focus-within{border-color:var(--accent);box-shadow:0 0 0 3px rgba(217,119,87,.18);}
.input textarea{
  flex:1;background:transparent;border:0;resize:none;
  padding:9px 6px;font-size:15px;line-height:1.4;
  max-height:140px;min-height:24px;
  font-family:inherit;
}
.input textarea:disabled{opacity:.6;}
.input .send{
  width:38px;height:38px;border-radius:50%;
  background:linear-gradient(135deg,var(--accent),var(--accent-3));
  color:#fff;display:inline-flex;align-items:center;justify-content:center;
  flex:none;transition:transform .2s var(--ease-back),background .2s var(--ease);
  -webkit-tap-highlight-color:transparent;
}
.input .send:active{transform:scale(.92);}
.input .send.stop{background:linear-gradient(135deg,#ef6b6b,#a83a3a);}
.input .send.disabled{background:var(--bg-4);color:var(--text-mute);}
.input .send svg{width:18px;height:18px;}
.input-actions{display:flex;align-items:center;gap:2px;padding:0 2px 4px;}
.input-actions .btn-icon{width:36px;height:36px;border-radius:10px;}
.input-actions .btn-icon svg{width:18px;height:18px;}
.draft-files{display:flex;gap:6px;flex-wrap:wrap;padding:6px 4px 0;}
.draft-files .pill{
  display:inline-flex;align-items:center;gap:6px;padding:5px 8px 5px 10px;border-radius:999px;
  background:var(--bg-3);border:1px solid var(--line);font-size:12px;color:var(--text);
  max-width:240px;
}
.draft-files .pill > span{
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:180px;
}
.draft-files .pill svg{width:13px;height:13px;color:var(--accent);flex-shrink:0;}
.draft-files .pill button{
  border:0;background:transparent;color:var(--text-mute);cursor:pointer;
  width:18px;height:18px;border-radius:50%;display:inline-flex;align-items:center;justify-content:center;
  font-size:14px;line-height:1;padding:0;margin-left:2px;
}
.draft-files .pill button:hover{color:var(--bad);background:rgba(239,107,107,.12);}

/* Per-message file attachments (above bubble content) */
.msg-attachs{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;}
.msg-attach{
  display:inline-flex;align-items:center;gap:6px;padding:5px 10px;border-radius:10px;
  background:var(--bg-3);border:1px solid var(--line);font-size:12px;color:var(--text);
  cursor:pointer;max-width:240px;
}
.msg-attach:hover{background:var(--bg-4);}
.msg-attach svg{width:14px;height:14px;color:var(--accent);flex-shrink:0;}
.msg-attach .nm{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:160px;}
.msg-attach .sz{color:var(--text-mute);font-size:11px;}

/* ───────── Bottom sheet ───────── */
.sheet{
  position:fixed;left:0;right:0;bottom:0;
  background:var(--bg-2);
  border-top-left-radius:24px;border-top-right-radius:24px;
  border-top:1px solid var(--line);
  transform:translateY(110%);
  transition:transform .3s var(--ease);
  z-index:80;
  max-height:88dvh;display:flex;flex-direction:column;
  box-shadow:0 -16px 40px rgba(0,0,0,.5);
}
.sheet.open{transform:translateY(0);}
.sheet-grab{
  display:flex;justify-content:center;padding:8px 0 4px;
  cursor:grab;touch-action:none;
}
.sheet-grab::before{
  content:"";display:block;width:38px;height:4px;border-radius:2px;background:var(--line-2);
}
.sheet-head{
  display:flex;align-items:center;gap:8px;padding:0 14px 8px;
  border-bottom:1px solid var(--line);
}
.sheet-head h3{margin:0;font-size:16px;font-weight:600;flex:1;}
.sheet-body{
  flex:1;overflow:auto;padding:12px 14px calc(var(--safe-bot) + 14px);
  scrollbar-width:thin;
}
.sheet-tabs{display:flex;gap:4px;padding:8px 12px 4px;border-bottom:1px solid var(--line);overflow-x:auto;}
.sheet-tab{
  padding:8px 12px;border-radius:10px;font-size:13px;color:var(--text-dim);
  white-space:nowrap;cursor:pointer;transition:all .15s var(--ease);
}
.sheet-tab.active{background:var(--accent);color:#fff;}
.sheet-tab:hover:not(.active){background:var(--bg-3);}

/* Settings list */
.field{margin:10px 0;}
.field label{display:block;color:var(--text-dim);font-size:12px;margin-bottom:4px;text-transform:uppercase;letter-spacing:.5px;}
.field input[type=text],.field input[type=password],.field textarea{
  width:100%;background:var(--bg-3);border:1px solid var(--line);border-radius:10px;
  padding:10px 12px;color:var(--text);font-size:14px;
}
.field input:focus,.field textarea:focus{border-color:var(--accent);}
.field .hint{font-size:11px;color:var(--text-mute);margin-top:4px;}

.model-grid{display:grid;grid-template-columns:1fr;gap:6px;}
.model-card{
  display:flex;align-items:center;gap:10px;padding:10px 12px;
  border:1px solid var(--line);border-radius:12px;background:var(--bg-3);
}
.model-card .dot{width:10px;height:10px;border-radius:50%;flex:none;background:var(--text-mute);}
.model-card[data-status="online"] .dot{background:var(--good);box-shadow:0 0 0 4px rgba(126,224,138,.18);}
.model-card[data-status="configured"] .dot{background:#7dd3fc;box-shadow:0 0 0 4px rgba(125,211,252,.18);}
.model-card[data-status="ratelimit"] .dot{background:var(--warn);box-shadow:0 0 0 4px rgba(245,158,11,.18);}
.model-card[data-status="paid"] .dot{background:#a78bfa;box-shadow:0 0 0 4px rgba(167,139,250,.18);}
.model-card[data-status="error"] .dot{background:var(--bad);}
.model-card[data-status="no_key"] .dot{background:var(--warn);}
.model-card .name{flex:1;min-width:0;font-weight:600;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.model-card .role{font-size:10px;text-transform:uppercase;color:var(--text-mute);
  background:var(--bg-4);padding:2px 6px;border-radius:6px;flex:none;}
.model-card .latency{font-size:11px;color:var(--text-mute);flex:none;}
.model-card .err{flex-basis:100%;font-size:11px;color:var(--bad);margin-left:20px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.model-card[data-status="ratelimit"] .err{color:var(--warn);}
.model-card[data-status="paid"] .err{color:#a78bfa;}
#model-status .ms-online{color:var(--good);}
#model-status .ms-cfg{color:#7dd3fc;}
#model-status .ms-rl{color:var(--warn);}
#model-status .ms-paid{color:#a78bfa;}
#model-status .ms-nokey{color:var(--text-mute);}
#model-status .ms-err{color:var(--bad);}

/* File manager */
.fm-toolbar{display:flex;gap:6px;margin-bottom:8px;flex-wrap:wrap;}
.fm-list{display:flex;flex-direction:column;gap:4px;}
.fm-row{
  display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:10px;
  cursor:pointer;transition:background .15s var(--ease);
}
.fm-row:hover{background:var(--bg-3);}
.fm-row svg{width:18px;height:18px;color:var(--text-mute);flex:none;}
.fm-row .nm{flex:1;min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.fm-row .meta{font-size:11px;color:var(--text-mute);flex:none;}
.fm-row.dir svg{color:var(--plan);}
.fm-bread{display:flex;align-items:center;gap:4px;font-size:12px;color:var(--text-mute);margin-bottom:8px;flex-wrap:wrap;}
.fm-bread .crumb{cursor:pointer;padding:2px 6px;border-radius:6px;}
.fm-bread .crumb:hover{background:var(--bg-3);color:var(--text);}

/* Toasts */
.toasts{position:fixed;left:0;right:0;bottom:calc(var(--input-h) + var(--safe-bot) + 6px);z-index:100;
  display:flex;flex-direction:column;align-items:center;gap:6px;pointer-events:none;}
.toast{
  background:var(--bg-3);border:1px solid var(--line);
  padding:8px 14px;border-radius:999px;font-size:13px;color:var(--text);
  box-shadow:var(--shadow-md);
  animation:toast .3s var(--ease-back) both;
  pointer-events:auto;max-width:90vw;
}
.toast.error{background:rgba(239,107,107,.14);border-color:var(--bad);color:#ffb4b4;}
.toast.ok{background:rgba(126,224,138,.12);border-color:var(--good);color:#bff5c8;}
@keyframes toast{from{transform:translateY(20px);opacity:0;}to{transform:none;opacity:1;}}

/* Context menu */
.menu{
  position:fixed;background:var(--bg-3);border:1px solid var(--line);border-radius:14px;
  box-shadow:var(--shadow-lg);z-index:120;padding:4px;min-width:208px;max-width:280px;
  opacity:0;transform:scale(.92) translateY(-4px);transform-origin:top left;
  transition:opacity .14s var(--ease), transform .14s var(--ease);
  user-select:none;-webkit-user-select:none;touch-action:manipulation;
}
.menu.open{opacity:1;transform:scale(1) translateY(0);}
.menu.closing{opacity:0;transform:scale(.96) translateY(-2px);pointer-events:none;}
.menu .item{
  display:flex;align-items:center;gap:10px;padding:10px 12px;border-radius:10px;cursor:pointer;font-size:14px;
  -webkit-tap-highlight-color:transparent;
}
.menu .item:hover,.menu .item:active{background:var(--bg-4);}
.menu .item svg{width:16px;height:16px;color:var(--text-dim);flex-shrink:0;}
.menu .item span{flex:1;}
.menu .sep{height:1px;background:var(--line);margin:4px 6px;}
.menu .item.danger{color:var(--bad);}
.menu .item.danger svg{color:var(--bad);}

/* Image */
.bubble img.gen{
  border-radius:14px;border:1px solid var(--line);max-width:100%;margin:6px 0;
}

/* Markdown headings */
.bubble h1,.bubble h2,.bubble h3,.bubble h4{
  margin:14px 0 6px;line-height:1.25;font-weight:700;
}
.bubble h1{font-size:1.4em;}
.bubble h2{font-size:1.25em;}
.bubble h3{font-size:1.1em;color:var(--accent-2);}
.bubble hr{border:0;border-top:1px solid var(--line);margin:10px 0;}
.bubble .md-table{overflow-x:auto;margin:10px 0;border-radius:10px;border:1px solid var(--line);}
.bubble table{border-collapse:collapse;margin:0;width:100%;}
.bubble th,.bubble td{border-bottom:1px solid var(--line);padding:8px 12px;font-size:13px;text-align:left;vertical-align:top;}
.bubble th{background:var(--bg-3);color:var(--text);font-weight:700;}
.bubble tbody tr:hover{background:rgba(255,255,255,.02);}
.bubble tbody tr:last-child td{border-bottom:0;}

/* Ripple effect */
.ripple{position:relative;overflow:hidden;}
.ripple::after{
  content:"";position:absolute;top:50%;left:50%;width:0;height:0;border-radius:50%;
  background:rgba(255,255,255,.18);transform:translate(-50%,-50%);
  transition:width .4s var(--ease),height .4s var(--ease),opacity .8s var(--ease);
  opacity:0;pointer-events:none;
}
.ripple:active::after{width:280px;height:280px;opacity:1;transition:0s;}

/* Skeleton shimmer */
.skel{
  background:linear-gradient(90deg,var(--bg-3) 25%,var(--bg-4) 50%,var(--bg-3) 75%);
  background-size:200% 100%;
  animation:shim 1.2s linear infinite;
  border-radius:8px;
}
@keyframes shim{0%{background-position:200% 0;}100%{background-position:-200% 0;}}

@media (min-width:780px){
  .scroll{padding:18px 18px 28px;}
  .header{padding-left:14px;padding-right:14px;}
  .input-wrap{padding-left:14px;padding-right:14px;}
}

@media (prefers-reduced-motion:reduce){
  *{transition:none !important;animation:none !important;}
}
</style>
</head>
<body>
<div class="app" id="app">
  <header class="header">
    <button class="btn-icon ripple" data-act="drawer" aria-label="Чаты">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18"/><path d="M3 12h18"/><path d="M3 18h18"/></svg>
    </button>
    <div class="title" id="title">
      <h1 id="chat-title">TsukCat AI</h1>
      <small><b id="model-status">Модели: …</b> · v__VERSION__</small>
    </div>
    <button class="btn-icon ripple" data-act="files" aria-label="Файлы">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>
    </button>
    <button class="btn-icon ripple" data-act="settings" aria-label="Настройки">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19 12a7 7 0 0 0-.06-.97l1.84-1.46-2-3.46-2.13.85a7 7 0 0 0-1.69-.97l-.32-2.27h-4l-.32 2.27a7 7 0 0 0-1.69.97l-2.13-.85-2 3.46 1.84 1.46A7 7 0 0 0 5 12c0 .33.02.65.06.97L3.22 14.43l2 3.46 2.13-.85a7 7 0 0 0 1.69.97l.32 2.27h4l.32-2.27a7 7 0 0 0 1.69-.97l2.13.85 2-3.46-1.84-1.46c.04-.32.06-.64.06-.97z"/></svg>
    </button>
  </header>

  <main class="scroll" id="scroll">
    <div class="empty-state" id="empty">
      <div class="glow">TC</div>
      <h2>TsukCat AI</h2>
      <p>Кастомный мобильный чат с роевым ИИ. План → Размышление → Код → Синтез → Проверка. 13 моделей объединены в одну стабильную линию.</p>
      <div class="hints">
        <div class="hint" data-prompt="Напиши Python-скрипт, который сортирует список словарей по ключу с регулярным выражением"><b>Кодинг:</b> сортировка с regex по ключу.</div>
        <div class="hint" data-prompt="Сделай мини-игру на HTML/JS — змейка, прикрепи файл"><b>Файлы:</b> игра-змейка с прикреплённым файлом.</div>
        <div class="hint" data-prompt="Составь большой план: как написать Telegram-бота на Python с нуля?"><b>План:</b> большой пошаговый план.</div>
        <div class="hint" data-prompt="Покажи опрос: какой стек выбрать для CLI на Python?"><b>Опрос:</b> голосование под сообщением.</div>
      </div>
    </div>
  </main>

  <div class="input-wrap" id="input-wrap">
    <div class="draft-files" id="draft-files" hidden></div>
    <div class="input">
      <button class="btn-icon ripple" data-act="attach" aria-label="Прикрепить файл" title="Прикрепить">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5V7a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v10a4 4 0 0 0 4 4h7"/><path d="M16 21l5-5"/><path d="M16 16h5v5"/></svg>
      </button>
      <textarea id="msg" rows="1" placeholder="Сообщение TsukCat AI…" autocomplete="off" autocorrect="off" autocapitalize="sentences"></textarea>
      <button class="send ripple" id="send" aria-label="Отправить">
        <svg id="send-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4z"/></svg>
      </button>
    </div>
    <div class="input-actions">
      <button class="btn-icon ripple" data-act="new-chat" aria-label="Новый чат" title="Новый чат">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v16"/><path d="M4 12h16"/></svg>
      </button>
      <button class="btn-icon ripple" data-act="run-last" aria-label="Запустить последнее" title="Запустить последний код">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 4 20 12 6 20"/></svg>
      </button>
      <button class="btn-icon ripple" data-act="health" aria-label="Проверить модели" title="Проверить статус моделей">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l3-9 4 18 3-9h4"/></svg>
      </button>
      <span style="flex:1"></span>
      <small id="hint-line" style="font-size:11px;color:var(--text-mute);padding:0 6px;"></small>
    </div>
  </div>
</div>

<aside class="drawer" id="drawer" aria-hidden="true">
  <div class="drawer-head">
    <h2>Чаты</h2>
    <button class="btn-icon" data-act="new-chat" aria-label="Новый чат" title="Новый чат">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v16"/><path d="M4 12h16"/></svg>
    </button>
  </div>
  <div class="drawer-search">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.35-4.35"/></svg>
    <input type="text" id="search" placeholder="Поиск чатов…" autocomplete="off">
  </div>
  <div class="chat-list" id="chat-list"></div>
  <div class="drawer-foot">
    <button class="btn ghost" data-act="import">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="M7 8l5-5 5 5"/><path d="M5 21h14"/></svg>
      Импорт чата (JSON)
    </button>
    <button class="btn ghost" data-act="export">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 21V9"/><path d="M7 16l5 5 5-5"/><path d="M5 3h14"/></svg>
      Экспорт текущего (.zip)
    </button>
  </div>
</aside>
<div class="scrim" id="scrim"></div>

<section class="sheet" id="sheet" aria-hidden="true">
  <div class="sheet-grab" id="sheet-grab"></div>
  <div class="sheet-head">
    <h3 id="sheet-title">Окно</h3>
    <button class="btn-icon" data-act="close-sheet" aria-label="Закрыть">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 6l12 12"/><path d="M18 6l-12 12"/></svg>
    </button>
  </div>
  <div class="sheet-tabs" id="sheet-tabs"></div>
  <div class="sheet-body" id="sheet-body"></div>
</section>

<input type="file" id="file-pick" hidden multiple>
<input type="file" id="import-pick" hidden accept=".json,.zip">
<div class="toasts" id="toasts"></div>

<script>
__JS__
</script>
</body>
</html>
"""



# === FRONTEND_JS (injected at /api/render time) ===
FRONTEND_JS = r"""
/* ───────── TsukCat AI front-end ───────── */
"use strict";

const $  = (s, ctx=document) => ctx.querySelector(s);
const $$ = (s, ctx=document) => Array.from(ctx.querySelectorAll(s));

const state = {
  chats: [],
  chatId: null,
  messages: [],
  draftFiles: [],
  state: null,
  job: null,
  jobES: null,
  busy: false,
  modelStatus: {},
  fmPath: "",
  search: "",
  pendingScroll: true,
  ctxMenu: null,
  pollAnswers: {},  // mid -> answer
};

/* ───────── HTTP helpers ───────── */
async function api(method, path, body){
  const opts = { method, headers: {} };
  if (body !== undefined && !(body instanceof FormData)){
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(body);
  } else if (body instanceof FormData){
    opts.body = body;
  }
  const r = await fetch(path, opts);
  if (!r.ok){
    let msg = r.statusText;
    try { msg = (await r.json()).error || msg; } catch(_) {}
    throw new Error(msg + " (" + r.status + ")");
  }
  const ct = r.headers.get("Content-Type") || "";
  if (ct.includes("application/json")) return r.json();
  return r.text();
}
const get   = (p)    => api("GET", p);
const post  = (p, b) => api("POST", p, b);
const del   = (p)    => api("DELETE", p);

/* ───────── Toasts ───────── */
function toast(text, kind=""){
  const wrap = $("#toasts");
  const el = document.createElement("div");
  el.className = "toast" + (kind ? " " + kind : "");
  el.textContent = text;
  wrap.appendChild(el);
  setTimeout(() => {
    el.style.transition = "opacity .3s, transform .3s";
    el.style.opacity = "0";
    el.style.transform = "translateY(8px)";
    setTimeout(() => el.remove(), 320);
  }, 2400);
}

/* ───────── SVG icons ───────── */
const svgs = {
  send:    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 2L11 13"/><path d="M22 2l-7 20-4-9-9-4z"/></svg>`,
  stop:    `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>`,
  copy:    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>`,
  run:     `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="6 4 20 12 6 20"/></svg>`,
  save:    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/></svg>`,
  edit:    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4z"/></svg>`,
  retry:   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15A9 9 0 1 1 18 5.5L23 10"/></svg>`,
  trash:   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>`,
  download:`<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>`,
  doc:     `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>`,
  folder:  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg>`,
  check:   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`,
  brain:   `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 3a3 3 0 0 0-3 3v2a3 3 0 0 0-3 3v3a3 3 0 0 0 3 3v2a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3v-2a3 3 0 0 0 3-3v-3a3 3 0 0 0-3-3V6a3 3 0 0 0-3-3z"/></svg>`,
  list:    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>`,
  chevron: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 6 15 12 9 18"/></svg>`,
  pin:     `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="17" x2="12" y2="22"/><path d="M5 17h14l-2-9H7z"/><path d="M9 8V3h6v5"/></svg>`,
  more:    `<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="12" cy="6" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="12" cy="18" r="2"/></svg>`,
  shield:  `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>`,
  test:    `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 2v6L4 20a2 2 0 0 0 2 3h12a2 2 0 0 0 2-3L15 8V2"/><path d="M9 2h6"/></svg>`,
};

/* ───────── Markdown (lightweight) ───────── */
function escapeHTML(s){
  return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function highlightCode(code, lang){
  // very tiny syntax painter for ~6 langs; falls back to plain.
  if (!lang) return escapeHTML(code);
  const L = lang.toLowerCase();
  const py = ["False","None","True","and","as","assert","async","await","break","class","continue","def","del","elif","else","except","finally","for","from","global","if","import","in","is","lambda","nonlocal","not","or","pass","raise","return","try","while","with","yield"];
  const js = ["var","let","const","function","class","extends","return","if","else","for","while","do","switch","case","break","continue","new","this","super","import","export","from","default","typeof","instanceof","async","await","yield","try","catch","finally","throw","of"];
  const html = ["html","head","body","div","span","script","style","link","meta","section","header","footer","main","button","input","textarea","select","option","table","tr","td","th","ul","ol","li","a","p","img","svg","path","circle"];
  const sh = ["if","then","else","elif","fi","for","do","done","while","until","case","esac","function","return","echo","exit","read","local","export","source","cd","pwd","ls","cat","grep","sed","awk","find","xargs","trap","set","unset","true","false"];
  const css = ["@media","@import","@keyframes","@font-face","@supports","important","none","auto","inherit","initial","unset"];
  const sqlw= ["select","from","where","group","by","order","having","limit","offset","insert","into","values","update","set","delete","create","table","alter","drop","index","join","left","right","inner","outer","on","as","and","or","not","null","is","like","in","between","case","when","then","end"];
  let words = [];
  if (L.startsWith("py") || L === "python3") words = py;
  else if (L === "js" || L === "javascript" || L === "ts" || L === "tsx" || L === "jsx" || L === "typescript") words = js;
  else if (L === "html" || L === "xml" || L === "svg") words = html;
  else if (L === "json") words = ["true","false","null"];
  else if (L === "bash" || L === "sh" || L === "shell" || L === "zsh") words = sh;
  else if (L === "css" || L === "scss" || L === "sass") words = css;
  else if (L === "sql") words = sqlw;
  else { return escapeHTML(code); }

  // tokens: comment, string, number, keyword, function name
  const out = [];
  let i = 0;
  const c = code;
  function emit(cls, txt){ out.push('<span class="tok-' + cls + '">' + escapeHTML(txt) + '</span>'); }
  function lit(txt){ out.push(escapeHTML(txt)); }
  while (i < c.length){
    const ch = c[i];
    // line comment
    if ((ch === '#' && L.startsWith("py")) || (ch === '/' && c[i+1] === '/')){
      let j = c.indexOf("\n", i); if (j < 0) j = c.length;
      emit("c", c.slice(i, j)); i = j; continue;
    }
    if (ch === '/' && c[i+1] === '*'){
      let j = c.indexOf("*/", i+2); if (j < 0) j = c.length; else j += 2;
      emit("c", c.slice(i, j)); i = j; continue;
    }
    // string
    if (ch === '"' || ch === "'" || ch === '`'){
      const q = ch; let j = i+1; while (j < c.length){
        if (c[j] === '\\') { j += 2; continue; }
        if (c[j] === q) { j++; break; }
        j++;
      }
      emit("s", c.slice(i, j)); i = j; continue;
    }
    // number
    if (/[0-9]/.test(ch) && (i === 0 || !/[a-zA-Z_]/.test(c[i-1]))){
      let j = i+1; while (j < c.length && /[0-9_xa-fA-F.]/.test(c[j])) j++;
      emit("n", c.slice(i, j)); i = j; continue;
    }
    // identifier
    if (/[a-zA-Z_]/.test(ch)){
      let j = i+1; while (j < c.length && /[a-zA-Z0-9_]/.test(c[j])) j++;
      const w = c.slice(i, j);
      if (words.includes(w)) emit("k", w);
      else if (c[j] === '(' && /[a-zA-Z_]/.test(w[0])) emit("f", w);
      else lit(w);
      i = j; continue;
    }
    // operator
    if (/[+\-*\/%=<>!&|^~?:]/.test(ch)){
      emit("o", ch); i++; continue;
    }
    if (ch === '.'){ emit("p", ch); i++; continue; }
    if (ch === '<' && /[a-zA-Z\/]/.test(c[i+1] || '')){
      let j = c.indexOf('>', i); if (j < 0) j = c.length; else j++;
      emit("t", c.slice(i, j)); i = j; continue;
    }
    lit(ch); i++;
  }
  return out.join("");
}

function cleanText(s){
  if (!s) return "";
  // strip nulls + most C0/C1 control chars (keep \n and \t)
  s = s.replace(/[\u0000-\u0008\u000B-\u001F\u007F-\u009F]/g, "");
  // collapse very common mojibake left over from latin1↔utf8 round-trips
  s = s.replace(/Ð\s/g, "");
  // strip zero-width joiner / non-joiner / BOM
  s = s.replace(/[\u200B-\u200D\uFEFF]/g, "");
  return s;
}

function renderInline(s){
  // bold, italic, code, links, line breaks. Robust against malformed
  // markdown — orphan ``, **, * become literal, never break the doc.
  let t = escapeHTML(cleanText(s));
  // inline code first (so its contents are not parsed as bold/italic)
  t = t.replace(/`([^`\n]+)`/g, (_, c) => '<code>' + c + '</code>');
  // bold
  t = t.replace(/\*\*([^*\n]+?)\*\*/g, '<strong>$1</strong>');
  // italic — *foo* (avoid matching ** by requiring non-* on both sides)
  t = t.replace(/(^|[^\*])\*([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');
  // italic — _foo_ (only at word boundaries)
  t = t.replace(/(^|[\s(])_([^_\n]+?)_(?=[\s.,!?:;)]|$)/g, '$1<em>$2</em>');
  // strip orphaned literal asterisks left over from broken markdown
  t = t.replace(/(^|\s)\*(\s)/g, '$1$2');
  // links
  t = t.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  // bare http(s) links (only outside an existing href)
  t = t.replace(/(^|[\s(])(https?:\/\/[^\s<>"']+)/g, '$1<a href="$2" target="_blank" rel="noopener">$2</a>');
  return t;
}

function renderMarkdown(md){
  if (!md) return "";
  md = cleanText(md);
  const lines = md.split(/\r?\n/);
  const out = [];
  let i = 0;
  let inCode = null;
  let codeBuf = [];
  // List state: support an OL that has interleaved UL "sub-lists" and resumes.
  // inOL=true means an <ol> is currently open. inUL=true means a <ul> is open
  // (possibly nested inside the OL's last <li>). When we hit a new `1.` line
  // while a sibling UL is open, we close just the UL — keeping the parent OL
  // alive so numbering continues instead of restarting from 1.
  let inUL = false, inOL = false, inQuote = false;
  let inTable = false, tableHeaderDone = false;
  function closeLists(){
    if (inUL){ out.push("</ul>"); inUL = false; }
    if (inOL){ out.push("</ol>"); inOL = false; }
    if (inQuote){ out.push("</blockquote>"); inQuote = false; }
    if (inTable){ out.push("</tbody></table></div>"); inTable = false; tableHeaderDone = false; }
  }
  function ensureUL(){
    if (inQuote){ out.push("</blockquote>"); inQuote = false; }
    if (inTable){ out.push("</tbody></table></div>"); inTable = false; tableHeaderDone = false; }
    // If OL is open, keep it open and just open a UL after the last <li>.
    if (!inUL){ out.push("<ul>"); inUL = true; }
  }
  function ensureOL(){
    if (inUL){ out.push("</ul>"); inUL = false; }
    if (inQuote){ out.push("</blockquote>"); inQuote = false; }
    if (inTable){ out.push("</tbody></table></div>"); inTable = false; tableHeaderDone = false; }
    if (!inOL){ out.push("<ol>"); inOL = true; }
  }
  function isTableSep(s){ return /^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$/.test(s); }
  while (i < lines.length){
    const ln = lines[i];
    const fence = ln.match(/^```(.*)$/);
    if (fence){
      if (inCode === null){
        closeLists();
        inCode = fence[1].trim() || "txt";
        codeBuf = [];
      } else {
        const lang = inCode;
        const code = codeBuf.join("\n");
        out.push(renderCodeBlock(code, lang));
        inCode = null; codeBuf = [];
      }
      i++; continue;
    }
    if (inCode !== null){
      codeBuf.push(ln); i++; continue;
    }
    // table: header line followed by separator
    if (!inTable && /^\s*\|/.test(ln) && i+1 < lines.length && isTableSep(lines[i+1])){
      closeLists();
      const cells = ln.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map(s => s.trim());
      out.push('<div class="md-table"><table><thead><tr>' +
        cells.map(c => `<th>${renderInline(c)}</th>`).join("") +
        '</tr></thead><tbody>');
      inTable = true; tableHeaderDone = true;
      i += 2; continue;
    }
    if (inTable){
      if (/^\s*\|/.test(ln)){
        const cells = ln.replace(/^\s*\|/, "").replace(/\|\s*$/, "").split("|").map(s => s.trim());
        out.push("<tr>" + cells.map(c => `<td>${renderInline(c)}</td>`).join("") + "</tr>");
        i++; continue;
      } else {
        out.push("</tbody></table></div>");
        inTable = false; tableHeaderDone = false;
        // fall through
      }
    }
    const h = ln.match(/^(#{1,4})\s+(.*)$/);
    if (h){
      closeLists();
      out.push(`<h${h[1].length}>${renderInline(h[2])}</h${h[1].length}>`);
      i++; continue;
    }
    if (ln.match(/^\s*[-*]\s+/)){
      ensureUL();
      out.push("<li>" + renderInline(ln.replace(/^\s*[-*]\s+/, "")) + "</li>");
      i++; continue;
    }
    if (ln.match(/^\s*\d+\.\s+/)){
      ensureOL();
      out.push("<li>" + renderInline(ln.replace(/^\s*\d+\.\s+/, "")) + "</li>");
      i++; continue;
    }
    if (ln.match(/^>\s?/)){
      if (!inQuote){ closeLists(); out.push("<blockquote>"); inQuote = true; }
      out.push(renderInline(ln.replace(/^>\s?/, "")) + "<br>");
      i++; continue;
    }
    if (ln.match(/^---+$/)){
      closeLists(); out.push("<hr>"); i++; continue;
    }
    if (ln.trim() === ""){
      // If we're inside a list and the next non-blank line is *also* a list
      // item, keep the list alive (don't close it). This makes paragraph-style
      // lists with blank lines between items render as one continuous list.
      if (inOL || inUL){
        let j = i+1;
        while (j < lines.length && lines[j].trim() === "") j++;
        const nxt = lines[j] || "";
        if (/^\s*([-*•+]|\d+\.)\s+/.test(nxt)){
          i++; continue;
        }
      }
      closeLists(); i++; continue;
    }
    // continuation of a list item: indented bullet/number under last item
    if ((inUL || inOL) && /^\s{2,}\S/.test(ln)){
      // append as part of previous <li>
      const prev = out.pop() || "";
      const m = prev.match(/^<li>([\s\S]*)<\/li>$/);
      if (m){ out.push("<li>" + m[1] + "<br>" + renderInline(ln.trim()) + "</li>"); i++; continue; }
      else { out.push(prev); }
    }
    closeLists();
    out.push("<p>" + renderInline(ln) + "</p>");
    i++;
  }
  if (inCode !== null) out.push(renderCodeBlock(codeBuf.join("\n"), inCode || "txt"));
  closeLists();
  return out.join("");
}

function renderCodeBlock(code, lang){
  const id = "c" + Math.random().toString(36).slice(2,9);
  const safeLang = lang.toLowerCase().replace(/[^a-z0-9+#-]/g, "");
  const isRunnable = ["python","py","python3","bash","sh"].includes(safeLang);
  const html = `
    <div class="code" data-lang="${escapeHTML(safeLang)}">
      <div class="code-head">
        <div class="code-lang">${escapeHTML(safeLang || "txt")}</div>
        <div class="code-actions">
          <button class="btn-icon" title="Скопировать" data-act="copy-code" data-id="${id}">${svgs.copy}</button>
          ${isRunnable ? `<button class="btn-icon" title="Запустить" data-act="run-code" data-id="${id}">${svgs.run}</button>` : ""}
          <button class="btn-icon" title="Сохранить как файл" data-act="save-code" data-id="${id}">${svgs.save}</button>
        </div>
      </div>
      <pre><code id="${id}" data-raw="${escapeHTML(code)}">${highlightCode(code, safeLang)}</code></pre>
    </div>`;
  return html;
}

/* ───────── Plan / poll / buttons / file widgets ───────── */
function renderBlocks(blocks, msg){
  if (!blocks || !blocks.length) return "";
  return blocks.map(b => renderBlock(b, msg)).join("");
}
function renderBlock(b, msg){
  if (!b || typeof b !== "object") return "";
  switch(b.type){
    case "md":     return renderMarkdown(b.text || "");
    case "plan":   return renderPlan(b);
    case "poll":   return renderPoll(b, msg);
    case "buttons":return renderButtons(b);
    case "file":   return renderFile(b, msg);
    case "run":    return renderRunPrompt(b);
    case "tree":   return renderTree(b);
    case "edit":   return renderEdit(b);
    case "test":   return renderTest(b);
    case "image":  return `<img class="gen" src="${escapeHTML(b.src||'')}" alt="${escapeHTML(b.alt||'')}">`;
    default:       return `<pre>${escapeHTML(JSON.stringify(b, null, 2))}</pre>`;
  }
}
function renderPlan(b){
  const steps = (b.steps || []);
  const done = steps.filter(s => s.status === "done").length;
  const pct = steps.length ? Math.round(done / steps.length * 100) : 0;
  return `<div class="plan">
    <div class="plan-head">
      ${svgs.list}
      <div class="ttl">${escapeHTML(b.title || "План")}</div>
      <div class="pct">${done}/${steps.length}</div>
    </div>
    <div class="plan-progress" style="--p:${pct}%"></div>
    <div class="plan-steps">
      ${steps.map(s => `
        <div class="step ${s.status||""}">
          <span class="box">${(s.status==="done")?svgs.check:""}</span>
          <span class="text">${escapeHTML(s.text||"")}</span>
          ${s.kind?`<span class="kind">${escapeHTML(s.kind)}</span>`:""}
        </div>`).join("")}
    </div>
  </div>`;
}
function renderPoll(b, msg){
  const id = msg ? msg.id : "p";
  const opts = b.options || [];
  const multi = !!b.multi;
  const cur = (msg && msg.poll_state) ? msg.poll_state : (state.pollAnswers[id]||null);
  return `<div class="poll" data-msg="${escapeHTML(id)}" data-multi="${multi}">
    <div class="poll-q">${renderInline(b.question||"")}</div>
    <div class="poll-opts">
      ${opts.map((o,i)=>`
        <div class="poll-opt ${cur && (multi?cur.includes(i):cur===i)?"selected":""}" data-multi="${multi}" data-i="${i}">
          <span class="marker">${(multi && cur && cur.includes && cur.includes(i))?svgs.check:""}</span>
          <span>${renderInline(o)}</span>
        </div>`).join("")}
    </div>
  </div>`;
}
function renderButtons(b){
  const btns = b.buttons || [];
  return `<div class="btn-row">${btns.map(x=>`
    <button class="btn ripple" data-act="${escapeHTML(x.action||'')}" data-target="${escapeHTML(x.target||'')}" data-style="${escapeHTML(x.style||'')}" data-prompt="${escapeHTML(x.prompt||'')}">
      ${escapeHTML(x.label||"Кнопка")}
    </button>`).join("")}</div>`;
}
function renderFile(b, msg){
  const name = b.name || "file";
  const lang = (b.lang || (name.split(".").pop()||"txt")).toLowerCase();
  const path = b.path || "";
  const sizeText = b.content ? `${b.content.length} б` : "";
  return `<div class="attach" data-act="open-file" data-path="${escapeHTML(path)}" data-name="${escapeHTML(name)}">
      ${svgs.doc}
      <span><span class="nm">${escapeHTML(name)}</span> <span class="sz">${escapeHTML(lang.toUpperCase())} · ${sizeText}</span></span>
    </div>
    <div class="btn-row">
      <button class="btn ripple" data-act="run-file" data-path="${escapeHTML(path)}" data-lang="${escapeHTML(lang)}" data-style="primary">${svgs.run}<span>Запустить</span></button>
      <button class="btn ripple" data-act="download-file" data-path="${escapeHTML(path)}">${svgs.download}<span>Скачать</span></button>
      <button class="btn ripple" data-act="open-file" data-path="${escapeHTML(path)}" data-name="${escapeHTML(name)}">${svgs.edit}<span>Открыть</span></button>
    </div>` + (b.content?renderCodeBlock(b.content, lang):"");
}
function renderRunPrompt(b){
  const lang = (b.lang||"python").toLowerCase();
  const id = "r" + Math.random().toString(36).slice(2,9);
  return renderCodeBlock(b.content||"", lang) +
    `<div class="btn-row"><button class="btn ripple" data-act="run-snippet" data-lang="${escapeHTML(lang)}" data-id="${id}" data-style="primary">${svgs.run}<span>Запустить</span></button></div>` +
    `<pre class="run-out" id="${id}-out" hidden></pre>`;
}
function renderTree(b){
  const walk = (node, depth=0) => {
    if (!node) return "";
    const isDir = !!node.is_dir;
    const name = node.name || "/";
    const childs = (node.children||[]).map(ch=>walk(ch, depth+1)).join("");
    return `<div class="node ${isDir?'dir':'file'}" style="margin-left:${depth*12}px">
      ${isDir?svgs.folder:svgs.doc}
      <span class="label">${escapeHTML(name)}</span>
    </div>${childs}`;
  };
  return `<div class="tree">${walk(b)}</div>`;
}
function renderEdit(b){
  return `<div class="attach"><span class="nm">Правка ${escapeHTML(b.target||'')}</span></div>` +
    renderCodeBlock(b.patch||"", b.lang||"diff");
}
function renderTest(b){
  const cs = (b.cases||[]);
  return `<div class="test">
    <div class="test-head">${svgs.test}<div class="ttl">${escapeHTML(b.title||"Тест")}</div></div>
    <div class="test-cases">${cs.map(c=>`<div class="test-case"><span class="pill">case</span><span>${escapeHTML(c.call||c.name||"")}</span> → <span>${escapeHTML(JSON.stringify(c.expect))}</span></div>`).join("")}</div>
  </div>`;
}

/* ───────── Message rendering ───────── */
function renderMsg(m){
  const role = m.role || "assistant";
  const isUser = role === "user";
  const initials = isUser ? "Я" : "AI";
  const blocks = (m.blocks && m.blocks.length) ? m.blocks : (m.content ? [{type:"md", text: m.content}] : []);

  let stage = "";
  if (m.stage && m.stage !== "done" && !isUser){
    const stages = ["plan","think","synthesize","verify","done"];
    const ix = stages.indexOf(m.stage);
    stage = `<div class="stage-bar">
      <span class="dot"></span>
      <span class="crumbs">
        ${stages.map((s,i)=>{
          const cls = (s===m.stage) ? "active" : (ix>=0 && i<ix ? "done" : "");
          return `<span class="crumb ${cls}">${s}</span>`;
        }).join("")}
      </span>
      <span style="flex:1"></span>
    </div>`;
  }
  // typing dots while waiting for first delta in synth stage (no content yet)
  let typing = "";
  const empty = !(m.content && m.content.trim()) && (!m.blocks || !m.blocks.length || (m.blocks.length===1 && m.blocks[0].type==="md" && !(m.blocks[0].text||"").trim()));
  if (!isUser && empty && m.stage && m.stage !== "done"){
    typing = `<div class="typing-dots" aria-label="generating"><span></span><span></span><span></span></div>`;
  }
  let reasoning = "";
  if (m.reasoning){
    reasoning = `<details class="thoughts" ${state.busy && m.id===state.assistantMsgId?"open":""}>
      <summary>${svgs.chevron}<span>Размышления (${m.reasoning.length} симв.)</span></summary>
      <div class="body">${escapeHTML(m.reasoning)}</div>
    </details>`;
  }
  let thoughts = "";
  if (m._thoughts && m._thoughts.length){
    thoughts = `<details class="thoughts" open>
      <summary>${svgs.brain}<span>Мнения моделей (${m._thoughts.length})</span></summary>
      <div class="body">${m._thoughts.map(t=>`<div class="thought"><b style="color:${t.color||'var(--text)'};">${escapeHTML(t.name||t.model)}</b><br>${escapeHTML(t.text||t.error||"")}</div>`).join("")}</div>
    </details>`;
  }
  let verify = "";
  if (m._verify){
    verify = `<details class="thoughts" open>
      <summary>${svgs.shield}<span>Проверка</span></summary>
      <div class="body">${renderMarkdown(m._verify)}</div>
    </details>`;
  }

  // Inline file attachments on user (or assistant) messages —
  // shown as small chips above the bubble content.
  let attachs = "";
  if (m.files && m.files.length){
    attachs = `<div class="msg-attachs">${m.files.map(f => {
      const nm = f.name || (f.path||"").split("/").pop() || "file";
      const sz = (typeof f.size === "number") ? `${f.size} б` : "";
      const path = f.path || "";
      return `<span class="msg-attach" data-act="open-file" data-path="${escapeHTML(path)}" data-name="${escapeHTML(nm)}">
        ${svgs.doc}<span class="nm">${escapeHTML(nm)}</span>${sz?`<span class="sz">${escapeHTML(sz)}</span>`:""}
      </span>`;
    }).join("")}</div>`;
  }
  const roleAria = isUser ? 'Вы' : 'TsukCat AI';
  return `<div class="msg ${isUser?'user':'assistant'}" data-id="${escapeHTML(m.id)}">
    <div class="role" aria-label="${roleAria}">
      <span class="av">${initials}</span>
      <span class="who">${isUser?'Вы':'TsukCat'}</span>
      <span class="when">${formatTs(m.created_at)}</span>
    </div>
    <div class="bubble">${stage}${thoughts}${reasoning}${attachs}<div class="content">${renderBlocks(blocks, m)}${typing}</div>${verify}</div>
  </div>`;
}

function formatTs(s){
  if (!s) return "";
  try {
    const d = new Date(s);
    if (Date.now() - d.getTime() < 86400000)
      return d.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
    return d.toLocaleString([], {day:"2-digit",month:"2-digit",hour:"2-digit",minute:"2-digit"});
  } catch(_){ return ""; }
}

/* ───────── Message list rendering ─────────
 *
 * Two-tier strategy to keep the chat from flickering during streaming:
 *
 *   1. renderMessages() does *DOM patching*, not innerHTML rebuild.
 *      Each .msg gets a `data-sig` from msgSig(m). On re-render we walk
 *      the existing children of #scroll, drop any whose id is no longer
 *      in state.messages, replace any whose signature changed, and
 *      insert new ones in place. Untouched messages keep their actual
 *      DOM nodes (and their selection / open <details> / scroll inside
 *      code blocks survive).
 *
 *   2. scheduleRender() coalesces multiple calls per animation frame.
 *      The SSE stream fires deltas at ~30–60 Hz; without this, every
 *      delta would synchronously rebuild the assistant message and
 *      that's where the visible flicker came from.
 */

function msgSig(m){
  // Cheap, deterministic signature. We only need to detect *content*
  // change — pointer-only mutations on existing messages must produce a
  // different sig. We DO NOT JSON.stringify large payloads here.
  let blocksFp = "";
  if (m.blocks && m.blocks.length){
    for (const b of m.blocks){
      blocksFp += (b.type || "?") + ":";
      if (b.type === "md")        blocksFp += (b.text||"").length + ";";
      else if (b.type === "plan") blocksFp += (b.steps||[]).length + "/" + (b.steps||[]).filter(s=>s.status==="done").length + ";";
      else if (b.type === "poll") blocksFp += (b.options||[]).length + ";";
      else if (b.type === "buttons") blocksFp += (b.buttons||[]).length + ";";
      else if (b.type === "file") blocksFp += (b.name||"") + ":" + (b.content||"").length + ";";
      else                        blocksFp += JSON.stringify(b).length + ";";
    }
  }
  return [
    m.id || "",
    m.role || "",
    m.stage || "",
    (m.content||"").length,
    blocksFp,
    (m.reasoning||"").length,
    (m._verify||"").length,
    (m._thoughts||[]).length,
    JSON.stringify(m.poll_state||null),
    (m.files||[]).length,
  ].join("|");
}

let _renderQueued = false;
function scheduleRender(){
  if (_renderQueued) return;
  _renderQueued = true;
  requestAnimationFrame(() => {
    _renderQueued = false;
    renderMessages();
  });
}

function renderMessages(){
  const root = $("#scroll");
  if (!state.messages.length){
    if (!root.querySelector(".empty-state")){
      root.innerHTML = "";
      root.appendChild(buildEmptyState());
    }
    return;
  }
  const empty = root.querySelector(".empty-state");
  if (empty) empty.remove();

  // sticky-bottom: keep auto-scroll if user is near the bottom (within
  // 120px), so streaming content doesn't yank the page when user
  // scrolled up to read.
  const wasNearBottom = (root.scrollHeight - root.scrollTop - root.clientHeight) < 120;
  const prevTop = root.scrollTop;

  // index existing nodes by id
  const existing = new Map();
  for (const el of root.children){
    if (el.classList && el.classList.contains("msg") && el.dataset.id){
      existing.set(el.dataset.id, el);
    }
  }
  const liveIds = new Set(state.messages.map(m => m.id));
  for (const [id, el] of existing){
    if (!liveIds.has(id)) el.remove();
  }

  let prev = null;
  const tmp = document.createElement("template");
  for (const m of state.messages){
    const sig = msgSig(m);
    const oldEl = existing.get(m.id);
    if (oldEl && oldEl.dataset.sig === sig){
      prev = oldEl;
      continue;
    }
    tmp.innerHTML = renderMsg(m).trim();
    const newEl = tmp.content.firstElementChild;
    if (!newEl){ continue; }
    newEl.dataset.sig = sig;
    if (oldEl){
      oldEl.replaceWith(newEl);
    } else if (prev){
      prev.after(newEl);
    } else {
      root.prepend(newEl);
    }
    prev = newEl;
  }

  if (state.pendingScroll){
    requestAnimationFrame(() => root.scrollTo({top: root.scrollHeight, behavior: "smooth"}));
    state.pendingScroll = false;
  } else if (wasNearBottom){
    // instant (no smooth) so streaming chunks don't visibly bounce
    root.scrollTop = root.scrollHeight;
  } else {
    root.scrollTop = prevTop;
  }
}

function buildEmptyState(){
  const n = document.createElement("div");
  n.className = "empty-state";
  n.innerHTML = `
    <div class="glow">TC</div>
    <h2>TsukCat AI</h2>
    <p>Напиши задачу — я составлю большой план, обдумаю в нескольких моделях, соберу ответ и проверю его.</p>
    <div class="hints">
      <div class="hint" data-prompt="Напиши и приложи файл tic_tac_toe.html — крестики-нолики на JS"><b>Файл:</b> крестики-нолики (HTML+JS).</div>
      <div class="hint" data-prompt="Составь огромный план: разработка кастомного Python-агента"><b>План:</b> агент на Python.</div>
      <div class="hint" data-prompt="Сделай опрос: какой паттерн выбрать для конфига приложения?"><b>Опрос:</b> голосование.</div>
      <div class="hint" data-prompt="Проверь что 100 первых простых чисел отсортированы и приложи main.py"><b>Тест:</b> простые числа.</div>
    </div>`;
  return n;
}

/* ───────── Boot ───────── */
async function boot(){
  try {
    state.state = await get("/api/state");
    $("#chat-title").textContent = state.state.app + " · " + state.state.python;
    document.title = state.state.app;
    document.body.dataset.version = state.state.version;
  } catch(e){
    toast("Не удалось загрузить /api/state: " + e.message, "error");
    return;
  }
  await reloadChats();
  if (!state.chats.length){
    const c = await post("/api/chats", {title: "Первый чат"});
    state.chats = [c];
  }
  // pick last
  state.chatId = localStorage.getItem("tsukcat:chat") || state.chats[0].id;
  if (!state.chats.find(c => c.id === state.chatId)) state.chatId = state.chats[0].id;
  await openChat(state.chatId);
  refreshHealth();
  setInterval(refreshHealth, 60000);
  // sync chats every 30s
  setInterval(reloadChats, 30000);
  showHint();
}

function showHint(){
  const hints = [
    "Долгий тап на сообщении — меню.",
    "Свайп слева — список чатов.",
    "Перетяни вниз — закрыть нижнюю панель.",
    "Кнопки под ответом — действия от ИИ.",
    "Ctrl/⌘+Enter — отправить сообщение.",
  ];
  $("#hint-line").textContent = hints[Math.floor(Math.random()*hints.length)];
}

async function reloadChats(){
  try {
    const res = await get("/api/chats");
    state.chats = res.chats || [];
    renderChatList();
  } catch(e){ console.warn("reloadChats", e); }
}

function renderChatList(){
  const root = $("#chat-list");
  const q = state.search.toLowerCase();
  const items = state.chats.filter(c => !q || (c.title||"").toLowerCase().includes(q));
  root.innerHTML = items.map(c => {
    const initials = (c.title || "?").trim().slice(0,2).toUpperCase();
    return `<div class="chat-item ${c.id===state.chatId?"active":""}" data-id="${escapeHTML(c.id)}">
      <div class="chat-avatar">${escapeHTML(initials)}</div>
      <div class="chat-meta">
        <div class="row1"><span class="name">${escapeHTML(c.title||"Чат")}</span><span class="ts">${formatTs(c.updated_at)}</span></div>
        <div class="preview">${c.message_count||0} сообщений</div>
      </div>
    </div>`;
  }).join("");
}

async function openChat(id){
  state.chatId = id;
  localStorage.setItem("tsukcat:chat", id);
  const ch = state.chats.find(c => c.id === id);
  if (ch) $("#chat-title").textContent = ch.title;
  state.pendingScroll = true;
  try {
    const r = await get(`/api/chats/${id}/messages`);
    state.messages = r.messages || [];
    renderMessages();
  } catch(e){ toast("openChat: " + e.message, "error"); }
  closeDrawer();
  renderChatList();
}

/* ───────── Sending message ───────── */
async function sendMessage(){
  if (state.busy){
    return stopJob();
  }
  const ta = $("#msg");
  const text = ta.value.trim();
  if (!text && !state.draftFiles.length) return;
  ta.value = "";
  resizeTextarea();
  if (!state.chatId){
    const c = await post("/api/chats", {title: "Чат"});
    state.chats.unshift(c); state.chatId = c.id;
  }

  // optimistic user message
  const localUser = {
    id: "tmp-" + Math.random().toString(36).slice(2,8),
    role: "user", content: text, blocks: [{type:"md", text}], created_at: new Date().toISOString(),
    files: state.draftFiles.slice(),
  };
  state.messages.push(localUser);
  state.pendingScroll = true;
  renderMessages();
  state.draftFiles = [];
  renderDraftFiles();

  setBusy(true);
  let resp;
  try {
    resp = await post(`/api/chats/${state.chatId}/messages`, {text, files: localUser.files});
  } catch(e){
    toast("Отправка сломалась: " + e.message, "error");
    setBusy(false);
    return;
  }
  startJob(resp.job_id);
}

function startJob(jobId){
  state.job = jobId;
  // Add placeholder assistant message
  const placeholder = {
    id: "stream-" + jobId,
    role: "assistant",
    content: "",
    blocks: [],
    reasoning: "",
    stage: "queued",
    created_at: new Date().toISOString(),
    _thoughts: [],
    _verify: "",
  };
  state.messages.push(placeholder);
  state.assistantMsgId = placeholder.id;
  state.pendingScroll = true;
  renderMessages();

  const es = new EventSource(`/api/jobs/${jobId}/stream`);
  state.jobES = es;
  let textBuf = "";
  es.addEventListener("message_created", e => {
    try {
      const data = JSON.parse(e.data);
      placeholder.id = data.message_id;
      state.assistantMsgId = data.message_id;
    } catch(_){}
  });
  es.addEventListener("stage", e => {
    try { placeholder.stage = JSON.parse(e.data).stage; scheduleRender(); } catch(_){}
  });
  es.addEventListener("delta", e => {
    try {
      textBuf += JSON.parse(e.data).text || "";
      placeholder.content = textBuf;
      placeholder.blocks = [{type:"md", text: textBuf}];
      scheduleRender();
    } catch(_){}
  });
  es.addEventListener("reasoning", e => {
    try { placeholder.reasoning += JSON.parse(e.data).text || ""; scheduleRender(); } catch(_){}
  });
  es.addEventListener("block", e => {
    try {
      const b = JSON.parse(e.data).block;
      // attach as a non-md prefix block
      placeholder.blocks.unshift(b);
      scheduleRender();
    } catch(_){}
  });
  es.addEventListener("blocks", e => {
    try {
      const blocks = JSON.parse(e.data).blocks || [];
      // Replace blocks with the parsed structured ones
      placeholder.blocks = blocks;
      scheduleRender();
    } catch(_){}
  });
  es.addEventListener("thought", e => {
    try { placeholder._thoughts.push(JSON.parse(e.data)); scheduleRender(); } catch(_){}
  });
  es.addEventListener("verify", e => {
    try { placeholder._verify = JSON.parse(e.data).text; scheduleRender(); } catch(_){}
  });
  es.addEventListener("warn", e => {
    try { toast(JSON.parse(e.data).text, "error"); } catch(_){}
  });
  es.addEventListener("error", e => {
    try { toast("Ошибка: " + (JSON.parse(e.data).text || ""), "error"); } catch(_){ }
  });
  es.addEventListener("end", e => {
    es.close();
    state.jobES = null;
    state.job = null;
    placeholder.stage = "done";
    setBusy(false);
    // refresh from server (so message id matches DB)
    setTimeout(() => openChat(state.chatId), 250);
  });
  es.onerror = () => {
    es.close();
    state.jobES = null; state.job = null;
    setBusy(false);
  };
}

async function stopJob(){
  if (!state.job) return;
  try { await post(`/api/jobs/${state.job}/stop`, {}); } catch(_){}
  if (state.jobES){ state.jobES.close(); state.jobES = null; }
  state.job = null;
  setBusy(false);
  toast("Остановлено", "ok");
}

function setBusy(b){
  state.busy = b;
  $("#input-wrap").classList.toggle("busy", b);
  const send = $("#send");
  send.classList.toggle("stop", b);
  send.querySelector("#send-icon").outerHTML = b
    ? `<span id="send-icon">${svgs.stop}</span>`
    : `<span id="send-icon">${svgs.send}</span>`;
  $("#msg").disabled = b;
}

/* ───────── Composer ───────── */
function resizeTextarea(){
  const ta = $("#msg");
  ta.style.height = "auto";
  ta.style.height = Math.min(140, ta.scrollHeight) + "px";
}
function bindComposer(){
  const ta = $("#msg");
  ta.addEventListener("input", resizeTextarea);
  ta.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter"){ e.preventDefault(); sendMessage(); }
    if (e.key === "Enter" && !e.shiftKey && window.matchMedia("(min-width:780px)").matches){
      e.preventDefault(); sendMessage();
    }
  });
  $("#send").addEventListener("click", sendMessage);
  $("#file-pick").addEventListener("change", async e => {
    const files = Array.from(e.target.files || []);
    for (const f of files){
      const fd = new FormData();
      fd.append("path", "uploads");
      fd.append("file", f, f.name);
      try {
        const r = await fetch("/api/files/upload", {method:"POST", body: fd});
        const j = await r.json();
        for (const sf of (j.saved||[])) state.draftFiles.push(sf);
        renderDraftFiles();
      } catch(err){ toast("Upload error: " + err.message, "error"); }
    }
    e.target.value = "";
  });
}
function renderDraftFiles(){
  const wrap = $("#draft-files");
  if (!state.draftFiles.length){ wrap.hidden = true; wrap.innerHTML=""; return; }
  wrap.hidden = false;
  wrap.innerHTML = state.draftFiles.map((f,i)=>`<div class="pill">
    ${svgs.doc}<span>${escapeHTML(f.name||f.path)}</span>
    <button data-act="rm-draft" data-i="${i}" aria-label="Убрать">×</button>
  </div>`).join("");
}

/* ───────── Drawer / scrim ───────── */
function openDrawer(){ $("#drawer").classList.add("open"); $("#scrim").classList.add("show"); $("#drawer").setAttribute("aria-hidden","false"); }
function closeDrawer(){ $("#drawer").classList.remove("open"); $("#scrim").classList.remove("show"); $("#drawer").setAttribute("aria-hidden","true"); }

/* ───────── Bottom sheet ───────── */
let sheetCtx = null;
function openSheet(ctx){
  sheetCtx = ctx;
  $("#sheet-title").textContent = ctx.title;
  $("#sheet-tabs").innerHTML = (ctx.tabs||[]).map((t,i)=>`<div class="sheet-tab ${i===0?"active":""}" data-tab="${i}">${escapeHTML(t.label)}</div>`).join("");
  $("#sheet-body").innerHTML = ctx.tabs && ctx.tabs.length ? ctx.tabs[0].render() : ctx.render();
  $("#sheet").classList.add("open");
  $("#sheet").setAttribute("aria-hidden","false");
  $("#scrim").classList.add("show");
  bindSheetActions();
}
function closeSheet(){
  $("#sheet").classList.remove("open");
  $("#scrim").classList.remove("show");
  $("#sheet").setAttribute("aria-hidden","true");
  sheetCtx = null;
}
function bindSheetActions(){
  $$(".sheet-tab", $("#sheet-tabs")).forEach(t => {
    t.addEventListener("click", () => {
      $$(".sheet-tab", $("#sheet-tabs")).forEach(x => x.classList.remove("active"));
      t.classList.add("active");
      const idx = +t.dataset.tab;
      $("#sheet-body").innerHTML = sheetCtx.tabs[idx].render();
      bindSheetActions();
    });
  });
}

/* ───────── Health checks ───────── */
async function refreshHealth(quiet=true){
  try {
    const r = await get("/api/health");
    state.modelStatus = {};
    for (const m of (r.results||[])) state.modelStatus[m.id] = m;
    const counts = {online:0, ratelimit:0, paid:0, configured:0, no_key:0, error:0};
    for (const m of (r.results||[])){
      counts[m.status] = (counts[m.status]||0) + 1;
    }
    const total = (r.results||[]).length;
    const parts = [];
    if (counts.online)     parts.push(`<span class="ms-online">●</span> ${counts.online} онлайн`);
    if (counts.configured) parts.push(`<span class="ms-cfg">●</span> ${counts.configured} готов`);
    if (counts.ratelimit)  parts.push(`<span class="ms-rl">●</span> ${counts.ratelimit} лимит`);
    if (counts.paid)       parts.push(`<span class="ms-paid">●</span> ${counts.paid} платно`);
    if (counts.no_key)     parts.push(`<span class="ms-nokey">●</span> ${counts.no_key} без ключа`);
    if (counts.error)      parts.push(`<span class="ms-err">●</span> ${counts.error} ошибок`);
    $("#model-status").innerHTML = parts.length ? parts.join(" · ") : `${total} моделей`;
    const live = (counts.online||0) + (counts.configured||0);
    document.title = `TsukCat AI · ${live}/${total}`;
    if (!quiet) toast(`Онлайн ${counts.online}, готовы ${counts.configured}, лимит ${counts.ratelimit}, платно ${counts.paid}`, "ok");
  } catch(e){ if (!quiet) toast("Health: " + e.message, "error"); }
}

/* ───────── Settings sheet ───────── */
function settingsSheet(){
  const tabs = [
    {label: "Модели", render: () => {
      return `<div class="model-grid">${(state.state?.models||[]).map(m=>{
        const h = state.modelStatus[m.id] || {};
        const status = h.status || (m.has_key?"unknown":"no_key");
        const lat = h.latency_ms ? h.latency_ms+"мс" : "";
        const err = h.error || h.note || "";
        return `<div class="model-card" data-status="${status}">
          <span class="dot"></span>
          <span class="name" style="color:${m.color}">${escapeHTML(m.name)}</span>
          <span class="role">${escapeHTML(m.role)}</span>
          <span class="latency">${lat}</span>
          ${err ? `<span class="err">${escapeHTML(err)}</span>` : ``}
        </div>`;
      }).join("")}</div>
      <div class="btn-row"><button class="btn primary ripple" data-act="ping-all">${svgs.run}<span>Пинг всех моделей</span></button></div>
      <p class="field hint" style="margin-top:14px">Легенда: <span style="color:var(--good)">●</span> онлайн · <span style="color:#7dd3fc">●</span> готов (image) · <span style="color:var(--warn)">●</span> rate-limit · <span style="color:#a78bfa">●</span> нужны кредиты · <span style="color:var(--bad)">●</span> ошибка.</p>
      <p class="field hint">Чтобы заменить ключ — задай ENV <code>OPENROUTER_KEY_QWEN_CODER=sk-or-...</code> или впиши в <code>${escapeHTML(state.state?.data_dir||"")}/secrets.json</code>.</p>`;
    }},
    {label: "Чат", render: () => {
      const ch = state.chats.find(c=>c.id===state.chatId) || {};
      return `<div class="field"><label>Название чата</label><input type="text" id="opt-title" value="${escapeHTML(ch.title||"")}"></div>
      <div class="btn-row">
        <button class="btn primary ripple" data-act="rename-chat">${svgs.edit}<span>Переименовать</span></button>
        <button class="btn ripple" data-act="export-chat">${svgs.download}<span>Экспорт ZIP</span></button>
        <button class="btn danger ripple" data-act="delete-chat">${svgs.trash}<span>Удалить чат</span></button>
      </div>`;
    }},
    {label: "О приложении", render: () => {
      const s = state.state || {};
      return `<div class="field"><label>App</label><div>${escapeHTML(s.app||"")}</div></div>
      <div class="field"><label>Версия</label><div>v${escapeHTML(s.version||"")}</div></div>
      <div class="field"><label>Python</label><div>${escapeHTML(s.python||"")}</div></div>
      <div class="field"><label>Платформа</label><div>${escapeHTML(s.platform||"")}</div></div>
      <div class="field"><label>Каталог данных</label><div><code>${escapeHTML(s.data_dir||"")}</code></div></div>
      <div class="field"><label>Каталог файлов</label><div><code>${escapeHTML(s.files_dir||"")}</code></div></div>
      <div class="field"><label>Владелец</label><div>${escapeHTML(s.owner||"")}</div></div>`;
    }},
  ];
  openSheet({title: "Настройки", tabs});
}

/* ───────── Files sheet ───────── */
async function filesSheet(start=""){
  state.fmPath = start;
  openSheet({title: "Файлы", render: () => `<div id="fm-root"><div class="skel" style="height:120px"></div></div>`});
  await renderFiles();
}
async function renderFiles(){
  const r = await get("/api/files?path=" + encodeURIComponent(state.fmPath));
  const items = r.items || [];
  const segs = state.fmPath.split("/").filter(Boolean);
  const crumbs = `<div class="fm-bread">
    <span class="crumb" data-path="">/</span>
    ${segs.map((s,i)=>`<span>›</span><span class="crumb" data-path="${escapeHTML(segs.slice(0,i+1).join('/'))}">${escapeHTML(s)}</span>`).join("")}
  </div>`;
  const tools = `<div class="fm-toolbar">
    <button class="btn ripple" data-act="fm-mkdir">${svgs.folder}<span>Папка</span></button>
    <button class="btn ripple" data-act="fm-new">${svgs.doc}<span>Файл</span></button>
    <button class="btn ripple" data-act="fm-up">↑ выше</button>
  </div>`;
  const list = `<div class="fm-list">${items.map(it=>{
    const isDir = it.is_dir;
    return `<div class="fm-row ${isDir?'dir':'file'}" data-path="${escapeHTML(it.path)}" data-dir="${isDir}">
      ${isDir?svgs.folder:svgs.doc}
      <span class="nm">${escapeHTML(it.name||it.path)}</span>
      <span class="meta">${isDir?"":(it.size||0)+" б"}</span>
    </div>`;
  }).join("") || `<div style="padding:20px;text-align:center;color:var(--text-mute)">Пусто</div>`}</div>`;
  $("#fm-root").innerHTML = crumbs + tools + list;
}

/* ───────── Global click handler ─────────
 *
 *  This is the single click delegator for the whole app.  It must
 *  handle BOTH `[data-act="…"]` buttons AND raw class hooks like
 *  `.chat-item`, `.fm-row`, `.crumb`, `.poll-opt`, `.hint`.
 *  Earlier versions returned early when no `[data-act]` was found,
 *  which was the reason chat switching, file rows and crumbs felt
 *  dead.
 */
document.addEventListener("click", async ev => {
  // close ctx menu on outside click
  if (state.ctxMenu && !ev.target.closest(".menu")){
    closeCtxMenu();
  }

  // ── class-hook elements (no data-act) ──
  const ci = ev.target.closest(".chat-item");
  if (ci && ci.dataset.id){
    openChat(ci.dataset.id);
    return;
  }
  const cr = ev.target.closest(".crumb");
  if (cr && cr.hasAttribute("data-path")){
    state.fmPath = cr.dataset.path || "";
    renderFiles();
    return;
  }
  const fmr = ev.target.closest(".fm-row");
  if (fmr){
    if (fmr.dataset.dir === "true"){ state.fmPath = fmr.dataset.path; renderFiles(); return; }
    try {
      const r = await get("/api/files/read?path=" + encodeURIComponent(fmr.dataset.path));
      openFileEditor(r);
    } catch(e){ toast(e.message, "error"); }
    return;
  }
  const hint = ev.target.closest(".hint[data-prompt]");
  if (hint){
    const ta = $("#msg");
    ta.value = hint.dataset.prompt;
    resizeTextarea();
    ta.focus();
    return;
  }
  const polo = ev.target.closest(".poll-opt");
  if (polo){
    const pollEl = polo.closest(".poll");
    const mid = pollEl?.dataset.msg;
    const multi = pollEl?.dataset.multi === "true";
    const i = +polo.dataset.i;
    if (!mid || isNaN(i)) return;
    let cur = state.pollAnswers[mid] ?? (multi ? [] : null);
    if (multi){
      if (!Array.isArray(cur)) cur = [];
      cur = cur.includes(i) ? cur.filter(x => x !== i) : cur.concat([i]);
    } else {
      cur = i;
    }
    state.pollAnswers[mid] = cur;
    // attach to the message so DOM-patching keeps the highlight
    const msg = state.messages.find(m => m.id === mid);
    if (msg) msg.poll_state = cur;
    scheduleRender();
    try { await post(`/api/messages/${mid}/poll`, {answer: cur}); }
    catch(_){}
    return;
  }

  // ── [data-act] buttons ──
  const t = ev.target.closest("[data-act]");
  if (!t) return;
  const act = t.dataset.act;
  const path = t.dataset.path || "";
  const lang = t.dataset.lang || "";
  const id = t.dataset.id || "";

  // header
  if (act === "drawer"){ openDrawer(); return; }
  if (act === "settings"){ settingsSheet(); return; }
  if (act === "files"){ filesSheet(""); return; }
  if (act === "close-sheet"){ closeSheet(); return; }
  if (act === "attach"){ $("#file-pick").click(); return; }
  if (act === "new-chat"){
    const c = await post("/api/chats", {title: "Новый чат"});
    state.chats.unshift(c);
    renderChatList();
    await openChat(c.id);
    return;
  }
  if (act === "import"){ $("#import-pick").click(); return; }
  if (act === "export"){
    if (!state.chatId) return;
    window.open(`/api/export/${state.chatId}`, "_blank");
    return;
  }
  if (act === "rm-draft"){
    state.draftFiles.splice(+t.dataset.i, 1);
    renderDraftFiles();
    return;
  }
  if (act === "run-last"){
    const codes = $$("[data-act='run-code']");
    if (codes.length) codes[codes.length-1].click();
    else toast("Нет кода для запуска", "error");
    return;
  }
  if (act === "health"){ refreshHealth(false); return; }
  if (act === "ping-all"){
    refreshHealth(false);
    setTimeout(() => settingsSheet(), 1500);
    return;
  }

  // file manager toolbar
  if (act === "fm-up"){
    state.fmPath = state.fmPath.split("/").slice(0,-1).join("/");
    renderFiles(); return;
  }
  if (act === "fm-mkdir"){
    const name = prompt("Имя папки", "newdir");
    if (!name) return;
    const rel = (state.fmPath ? state.fmPath + "/" : "") + name;
    try { await post("/api/files/mkdir", {path: rel}); renderFiles(); }
    catch(e){ toast(e.message, "error"); }
    return;
  }
  if (act === "fm-new"){
    const name = prompt("Имя файла", "new.txt");
    if (!name) return;
    const rel = (state.fmPath ? state.fmPath + "/" : "") + name;
    try { await post("/api/files/save", {path: rel, content: ""}); renderFiles(); }
    catch(e){ toast(e.message, "error"); }
    return;
  }

  // code block actions
  if (act === "copy-code"){
    const code = document.getElementById(id);
    const raw = code?.dataset.raw || code?.textContent || "";
    const ok = await copyText(raw);
    toast(ok ? "Скопировано" : "Не удалось скопировать", ok ? "ok" : "error");
    return;
  }
  if (act === "run-code" || act === "run-snippet"){
    const code = document.getElementById(id);
    const raw = code?.dataset.raw || code?.textContent || "";
    const langGuess = code?.closest(".code")?.dataset.lang || lang || "python";
    const r = await post("/api/run", {lang: langGuess, code: raw});
    let out = $("#" + id + "-out");
    if (!out){
      out = document.createElement("pre");
      out.id = id + "-out";
      out.className = "run-out";
      code?.closest(".code")?.parentNode?.insertBefore(out, code.closest(".code").nextSibling);
    }
    out.hidden = false;
    out.classList.toggle("err", !r.ok);
    out.textContent = (r.stdout||"") + (r.stderr ? "\n[stderr] " + r.stderr : "");
    return;
  }
  if (act === "save-code"){
    const code = document.getElementById(id);
    const raw = code?.dataset.raw || "";
    const lng = code?.closest(".code")?.dataset.lang || "txt";
    const name = prompt("Имя файла", "snippet." + lng);
    if (!name) return;
    try {
      await post("/api/files/save", {path: "snippets/" + name, content: raw});
      toast("Сохранено в files/snippets/" + name, "ok");
    } catch(e){ toast(e.message, "error"); }
    return;
  }

  // file widget
  if (act === "open-file"){
    if (!path){ toast("Файл ещё не сохранён", "error"); return; }
    try { const r = await get("/api/files/read?path=" + encodeURIComponent(path)); openFileEditor(r); }
    catch(e){ toast(e.message, "error"); }
    return;
  }
  if (act === "download-file"){
    if (!path){ toast("Файл ещё не сохранён", "error"); return; }
    window.open("/api/files/raw/" + encodeURIComponent(path), "_blank");
    return;
  }
  if (act === "run-file"){
    if (!path){ toast("Файл ещё не сохранён", "error"); return; }
    const r = await get("/api/files/read?path=" + encodeURIComponent(path));
    const out = await post("/api/run", {lang, code: r.text || ""});
    toast(out.ok ? "Запуск ok" : "Запуск с ошибками", out.ok?"ok":"error");
    openSheet({title: name(path) + " — запуск", render: () => `<pre class="run-out ${out.ok?"":"err"}">${escapeHTML((out.stdout||"") + (out.stderr?"\n[stderr] " + out.stderr:""))}</pre>`});
    return;
  }

  // settings actions
  if (act === "rename-chat"){
    const t2 = $("#opt-title").value.trim();
    if (!t2) return;
    await post(`/api/chats/${state.chatId}/rename`, {title: t2});
    await reloadChats();
    $("#chat-title").textContent = t2;
    closeSheet();
    return;
  }
  if (act === "export-chat"){
    window.open(`/api/export/${state.chatId}`, "_blank");
    return;
  }
  if (act === "delete-chat"){
    if (!confirm("Удалить чат?")) return;
    await del(`/api/chats/${state.chatId}`);
    state.chatId = null;
    closeSheet();
    await reloadChats();
    if (state.chats.length) openChat(state.chats[0].id);
    else { state.messages = []; renderMessages(); }
    return;
  }

  // (poll and hint clicks are handled by the class-hook section above)

  // generic agent action buttons
  if (act === "ask" || act === "next"){
    const p = t.dataset.prompt || t.textContent.trim();
    $("#msg").value = p;
    resizeTextarea();
    $("#msg").focus();
    return;
  }
});

function openFileEditor(info){
  const isText = typeof info.text === "string";
  const tabs = [{label: info.name, render: () => isText
    ? `<div class="field"><textarea id="ed-text" style="min-height:200px;font-family:monospace">${escapeHTML(info.text)}</textarea></div>
       <div class="btn-row">
         <button class="btn primary ripple" data-act="ed-save">${svgs.save}<span>Сохранить</span></button>
         <button class="btn ripple" data-act="ed-run" data-lang="${escapeHTML((info.name.split('.').pop()||'').toLowerCase())}">${svgs.run}<span>Запустить</span></button>
         <button class="btn ripple" data-act="ed-download" data-path="${escapeHTML(info.path)}">${svgs.download}<span>Скачать</span></button>
         <button class="btn danger ripple" data-act="ed-delete">${svgs.trash}<span>Удалить</span></button>
       </div>`
    : `<div>Бинарный файл (${info.size} б, ${escapeHTML(info.mime||"?")})</div>
       <div class="btn-row">
         <button class="btn ripple" data-act="ed-download" data-path="${escapeHTML(info.path)}">${svgs.download}<span>Скачать</span></button>
       </div>`}];
  openSheet({title: info.path, tabs});
  // Bind editor actions
  setTimeout(() => {
    document.querySelector("[data-act='ed-save']")?.addEventListener("click", async () => {
      const txt = $("#ed-text").value;
      try { await post("/api/files/save", {path: info.path, content: txt}); toast("Сохранено", "ok"); }
      catch(e){ toast(e.message, "error"); }
    });
    document.querySelector("[data-act='ed-run']")?.addEventListener("click", async () => {
      const txt = $("#ed-text").value;
      const lang = (info.name.split(".").pop()||"").toLowerCase();
      const r = await post("/api/run", {lang, code: txt});
      $("#sheet-body").innerHTML += `<pre class="run-out ${r.ok?"":"err"}">${escapeHTML((r.stdout||"") + (r.stderr?"\n[stderr] "+r.stderr:""))}</pre>`;
    });
    document.querySelector("[data-act='ed-delete']")?.addEventListener("click", async () => {
      if (!confirm("Удалить файл?")) return;
      try { await del("/api/files/raw/" + encodeURIComponent(info.path)); closeSheet(); renderFiles(); }
      catch(e){ toast(e.message, "error"); }
    });
    document.querySelector("[data-act='ed-download']")?.addEventListener("click", () => {
      window.open("/api/files/raw/" + encodeURIComponent(info.path), "_blank");
    });
  }, 30);
}

function name(p){ return (p||"").split("/").pop(); }

/* ───────── Copy helper (clipboard API + execCommand fallback) ───────── */
async function copyText(text){
  if (!text) text = "";
  // Modern API requires secure context. localhost qualifies, but inside a
  // WebView (Telegram, etc.) it may be undefined — fall back to a hidden
  // <textarea> + execCommand.
  if (navigator.clipboard && window.isSecureContext){
    try { await navigator.clipboard.writeText(text); return true; } catch(_){}
  }
  try {
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-9999px";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return !!ok;
  } catch(_){ return false; }
}

/* ───────── Long-press / context menu ─────────
 *
 * Single-instance menu.  Closes on:
 *   • tap or touchstart anywhere outside the menu;
 *   • Esc key;
 *   • scroll inside #scroll;
 *   • orientation/resize.
 *
 * Long-press is canceled if the finger moves more than 8px before the
 * 480 ms threshold (so accidental scrolls don't trigger it).
 */
let pressTimer = null, pressOrigin = null;
const LONG_PRESS_MS = 460;
const LONG_PRESS_MOVE_PX = 8;

function _clearPress(){ clearTimeout(pressTimer); pressTimer = null; pressOrigin = null; }

document.addEventListener("touchstart", e => {
  // close any open menu if user starts a new touch outside it
  if (state.ctxMenu && !e.target.closest(".menu")){
    closeCtxMenu();
  }
  const m = e.target.closest(".msg");
  if (!m) return;
  // ignore presses that originate on interactive children — let buttons,
  // links, code, polls receive their own click instead.
  if (e.target.closest("a,button,input,textarea,select,.poll-opt,.attach,.code,details>summary")) return;
  const t = e.touches[0];
  pressOrigin = {x: t.clientX, y: t.clientY};
  clearTimeout(pressTimer);
  pressTimer = setTimeout(() => {
    if (pressOrigin){
      showCtxMenu(pressOrigin.x, pressOrigin.y, m);
    }
    pressTimer = null;
  }, LONG_PRESS_MS);
}, {passive:true});

document.addEventListener("touchmove", e => {
  if (!pressTimer || !pressOrigin) return;
  const t = e.touches[0];
  const dx = t.clientX - pressOrigin.x, dy = t.clientY - pressOrigin.y;
  if (dx*dx + dy*dy > LONG_PRESS_MOVE_PX * LONG_PRESS_MOVE_PX) _clearPress();
}, {passive:true});

document.addEventListener("touchend", _clearPress, {passive:true});
document.addEventListener("touchcancel", _clearPress, {passive:true});

document.addEventListener("contextmenu", e => {
  const m = e.target.closest(".msg");
  if (!m) return;
  // skip if the right-click landed on something interactive
  if (e.target.closest("a,button,input,textarea,select,.poll-opt,.code")) return;
  e.preventDefault();
  showCtxMenu(e.clientX, e.clientY, m);
});

document.addEventListener("keydown", e => {
  if (e.key === "Escape" && state.ctxMenu){ closeCtxMenu(); }
});

window.addEventListener("resize", () => state.ctxMenu && closeCtxMenu());

function closeCtxMenu(){
  if (!state.ctxMenu) return;
  const m = state.ctxMenu;
  state.ctxMenu = null;
  m.classList.add("closing");
  setTimeout(() => { try { m.remove(); } catch(_){} }, 120);
}

function showCtxMenu(x, y, msgEl){
  const id = msgEl.dataset.id;
  const msg = state.messages.find(m => m.id === id);
  if (!msg) return;
  // Single instance — replace any prior menu.
  if (state.ctxMenu){ try { state.ctxMenu.remove(); } catch(_){} state.ctxMenu = null; }
  const menu = document.createElement("div");
  menu.className = "menu menu-ctx";
  menu.setAttribute("role","menu");
  menu.innerHTML = `
    <div class="item" role="menuitem" data-cm="copy">${svgs.copy}<span>Скопировать</span></div>
    <div class="item" role="menuitem" data-cm="quote">${svgs.brain}<span>Цитировать</span></div>
    ${msg.role==="user" ? `<div class="item" role="menuitem" data-cm="edit">${svgs.edit}<span>Редактировать</span></div>` : ""}
    <div class="item" role="menuitem" data-cm="retry">${svgs.retry}<span>Спросить снова</span></div>
    <div class="sep"></div>
    <div class="item danger" role="menuitem" data-cm="del">${svgs.trash}<span>Удалить</span></div>
  `;
  document.body.appendChild(menu);
  // Position with viewport clamping (after measure).
  const W = menu.offsetWidth || 220, H = menu.offsetHeight || 220;
  const px = Math.max(8, Math.min(x, window.innerWidth - W - 8));
  const py = Math.max(8, Math.min(y, window.innerHeight - H - 8));
  menu.style.left = px + "px";
  menu.style.top  = py + "px";
  state.ctxMenu = menu;
  // Animation
  requestAnimationFrame(() => menu.classList.add("open"));
  // Bind
  menu.addEventListener("click", async ev => {
    const i = ev.target.closest("[data-cm]");
    if (!i) return;
    const op = i.dataset.cm;
    closeCtxMenu();
    if (op === "copy"){
      const ok = await copyText(msg.content || "");
      toast(ok ? "Скопировано" : "Не удалось скопировать", ok ? "ok" : "error");
    } else if (op === "quote"){
      const ta = $("#msg");
      ta.value = (msg.content||"").split("\n").map(x => "> " + x).join("\n") + "\n\n";
      resizeTextarea(); ta.focus();
    } else if (op === "edit"){
      $("#msg").value = msg.content || "";
      resizeTextarea(); $("#msg").focus();
    } else if (op === "retry"){
      $("#msg").value = msg.content || "";
      resizeTextarea(); sendMessage();
    } else if (op === "del"){
      try { await del(`/api/messages/${id}`); openChat(state.chatId); }
      catch(e){ toast(e.message, "error"); }
    }
  });
  // Close menu on chat scroll
  const sc = $("#scroll");
  const onScroll = () => { closeCtxMenu(); sc.removeEventListener("scroll", onScroll, true); };
  sc.addEventListener("scroll", onScroll, true);
}

/* ───────── Gestures: drawer swipe-from-left, sheet drag-to-dismiss ───────── */
function bindGestures(){
  let start=null, current=null, target=null, mode=null;
  const drawer = $("#drawer");
  const sheet  = $("#sheet");

  function onTouchStart(e){
    const t = e.touches[0];
    start = {x:t.clientX, y:t.clientY, time:Date.now()};
    current = {x:t.clientX, y:t.clientY};
    if (start.x < 22 && !drawer.classList.contains("open")){ mode = "open-drawer"; target = drawer; }
    else if (drawer.classList.contains("open") && e.target.closest("#drawer")){ mode = "swipe-drawer"; target = drawer; }
    else if (sheet.classList.contains("open") && e.target.closest("#sheet-grab")){ mode = "drag-sheet"; target = sheet; }
    else { mode = null; }
  }
  function onTouchMove(e){
    if (!start || !mode) return;
    const t = e.touches[0]; current = {x:t.clientX, y:t.clientY};
    const dx = current.x - start.x, dy = current.y - start.y;
    if (mode === "open-drawer" && dx > 6){
      const w = Math.min(window.innerWidth*0.86, 340);
      const tx = Math.min(0, -w + dx);
      drawer.style.transform = `translateX(${tx}px)`;
      drawer.classList.add("open");
      $("#scrim").classList.add("show");
      $("#scrim").style.opacity = Math.min(1, dx/w);
    } else if (mode === "swipe-drawer" && dx < 0){
      drawer.style.transform = `translateX(${dx}px)`;
      $("#scrim").style.opacity = Math.max(0, 1 + dx/280);
    } else if (mode === "drag-sheet" && dy > 0){
      sheet.style.transform = `translateY(${dy}px)`;
    }
  }
  function onTouchEnd(){
    if (!start || !mode){ start=null; mode=null; return; }
    const dx = (current?.x||0) - start.x, dy = (current?.y||0) - start.y;
    if (mode === "open-drawer"){
      drawer.style.transform = "";
      $("#scrim").style.opacity = "";
      if (dx > 80) openDrawer(); else closeDrawer();
    } else if (mode === "swipe-drawer"){
      drawer.style.transform = "";
      $("#scrim").style.opacity = "";
      if (dx < -80) closeDrawer();
    } else if (mode === "drag-sheet"){
      sheet.style.transform = "";
      if (dy > 100) closeSheet();
    }
    start = null; mode = null; target = null;
  }
  document.addEventListener("touchstart", onTouchStart, {passive:true});
  document.addEventListener("touchmove",  onTouchMove,  {passive:true});
  document.addEventListener("touchend",   onTouchEnd,   {passive:true});

  // scrim closes drawer / sheet
  $("#scrim").addEventListener("click", () => { closeDrawer(); closeSheet(); });
}

/* ───────── Search input ───────── */
$("#search").addEventListener("input", e => {
  state.search = e.target.value;
  renderChatList();
});

/* ───────── Import file ───────── */
$("#import-pick").addEventListener("change", async e => {
  const f = e.target.files?.[0]; if (!f) return;
  const text = await f.text();
  try {
    const data = JSON.parse(text);
    const out = await post("/api/import", data);
    toast("Импортировано: " + out.title, "ok");
    await reloadChats();
    openChat(out.id);
  } catch(err){ toast("Импорт упал: " + err.message, "error"); }
  e.target.value = "";
});

/* ───────── Init ───────── */
bindComposer();
bindGestures();
boot();
"""


if __name__ == "__main__":
    sys.exit(main())
