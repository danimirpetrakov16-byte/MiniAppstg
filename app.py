#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI Chat + Telegram Bot — single-file Python 3 application.

Кастомный AI-чат с агентом, который полностью управляет Telegram-ботом.
Всё в одном файле: GUI (Tkinter), AI-агент поверх OpenRouter (несколько
моделей с фолбэком), Telegram-бот на python-telegram-bot 20.7, инструменты
(переименование бота, отправка сообщений, опросы, работа с файлами,
запуск кода) и кастомная разметка ответов с подсветкой синтаксиса,
кнопками, опросами и стримингом «печати».

Запуск:
    python3 app.py                  # запустить GUI
    python3 app.py --headless       # импорт-смоук для CI / без X-сервера
    python3 app.py --self-test      # минимальные оффлайн-тесты

Все API-ключи и токен бота заданы как DEFAULTS в этом файле, но любой из
них можно переопределить через переменную окружения с тем же именем
(например ``OPENROUTER_KEY_MINIMAX`` или ``TELEGRAM_BOT_TOKEN``).
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - dependency check
    requests = None  # type: ignore

# ---------------------------------------------------------------------------
# Логирование
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("ai-chat")

APP_NAME = "AI Chat + Telegram Bot Agent"
APP_VERSION = "1.0.0"
DATA_DIR = Path(os.environ.get("AI_CHAT_DATA_DIR", str(Path.home() / ".ai_chat_agent")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHATS_FILE = DATA_DIR / "chats.json"
WORKSPACE_DIR = DATA_DIR / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# OpenRouter и Telegram — конфигурация
# ---------------------------------------------------------------------------
# Все секреты грузятся в порядке приоритета:
#   1) переменные окружения (``OPENROUTER_KEY_<NAME>``, ``TELEGRAM_BOT_TOKEN``)
#   2) локальный файл ``$AI_CHAT_DATA_DIR/secrets.json`` (по умолчанию
#      ``~/.ai_chat_agent/secrets.json``); шаблон лежит рядом с app.py.
#   3) пустая строка — модель/бот будут отключены, пока ключ не задан.
#
# Никакие секреты не зашиты в исходник: при запуске первый раз скрипт сам
# подскажет, как заполнить ``secrets.json`` (он создаст пример, если файла нет).

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

_SECRETS_CACHE: Optional[Dict[str, Any]] = None


def _load_secrets_file() -> Dict[str, Any]:
    global _SECRETS_CACHE
    if _SECRETS_CACHE is not None:
        return _SECRETS_CACHE
    path = DATA_DIR / "secrets.json"
    if path.exists():
        try:
            _SECRETS_CACHE = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Не удалось прочитать %s", path)
            _SECRETS_CACHE = {}
    else:
        _SECRETS_CACHE = {}
    return _SECRETS_CACHE  # type: ignore[return-value]


def _envk(name: str) -> str:
    """Ключ OpenRouter для роли ``name``: env -> secrets.json -> ''."""
    env = os.environ.get(f"OPENROUTER_KEY_{name.upper()}")
    if env:
        return env
    sec = _load_secrets_file().get("openrouter", {})
    return str(sec.get(name, ""))


def _build_models() -> List[Dict[str, str]]:
    spec: List[Dict[str, str]] = [
        {"name": "minimax",                "kind": "chat",   "model": "minimax/minimax-m2:free",                  "label": "MiniMax M2 (free)"},
        {"name": "hy3",                    "kind": "chat",   "model": "tencent/hunyuan-a13b-instruct:free",        "label": "Tencent Hunyuan A13B (free)"},
        {"name": "qwen",                   "kind": "chat",   "model": "qwen/qwen3-30b-a3b:free",                   "label": "Qwen3 30B A3B (free)"},
        {"name": "whisper_large_v3",       "kind": "audio",  "model": "openai/whisper-large-v3",                   "label": "OpenAI Whisper Large V3"},
        {"name": "whisper_large_v3_turbo", "kind": "audio",  "model": "openai/whisper-large-v3-turbo",             "label": "OpenAI Whisper Large V3 Turbo"},
        {"name": "whisper_1",              "kind": "audio",  "model": "openai/whisper-1",                          "label": "OpenAI Whisper 1"},
        {"name": "gemini_image",           "kind": "image",  "model": "google/gemini-2.5-flash-image-preview",     "label": "Google Gemini Flash Image"},
        {"name": "veo",                    "kind": "video",  "model": "google/veo-3-lite",                         "label": "Google Veo 3 Lite"},
        {"name": "rerank",                 "kind": "rerank", "model": "cohere/rerank-3.5",                         "label": "Cohere Rerank"},
        {"name": "embed",                  "kind": "embed",  "model": "thenlper/gte-base",                         "label": "Thenlper GTE-Base"},
        {"name": "flux",                   "kind": "image",  "model": "black-forest-labs/flux-1.1-pro",            "label": "FLUX 1.1 Pro"},
        {"name": "seedream",               "kind": "image",  "model": "bytedance/seedream-3",                      "label": "ByteDance Seedream"},
        {"name": "kokoro",                 "kind": "tts",    "model": "hexgrad/kokoro-82m",                        "label": "Kokoro TTS"},
    ]
    out: List[Dict[str, str]] = []
    for s in spec:
        s = dict(s)
        s["key"] = _envk(s["name"])
        out.append(s)
    return out


MODEL_REGISTRY: List[Dict[str, str]] = _build_models()

# Текстовые модели — порядок попыток (фолбэк по комбинированию).
CHAT_MODELS = [m for m in MODEL_REGISTRY if m["kind"] == "chat" and m.get("key")]
AUDIO_MODELS = [m for m in MODEL_REGISTRY if m["kind"] == "audio" and m.get("key")]
IMAGE_MODELS = [m for m in MODEL_REGISTRY if m["kind"] == "image" and m.get("key")]


# ---------------------------------------------------------------------------
# Telegram-бот — данные владельца и токен
# ---------------------------------------------------------------------------

OWNER_USERNAME = (
    os.environ.get("AI_CHAT_OWNER_USERNAME")
    or _load_secrets_file().get("owner_username")
    or "@tsuklone"
)
DEFAULT_BOT_TOKEN = (
    os.environ.get("TELEGRAM_BOT_TOKEN")
    or str(_load_secrets_file().get("telegram_bot_token") or "")
)
DEFAULT_BOT_USERNAME = (
    os.environ.get("TELEGRAM_BOT_USERNAME")
    or str(_load_secrets_file().get("telegram_bot_username") or "")
)


def ensure_secrets_template() -> Path:
    """Создаёт ``secrets.json.example`` рядом с данными, если его нет."""
    example_path = DATA_DIR / "secrets.json.example"
    if not example_path.exists():
        example_path.write_text(
            json.dumps(
                {
                    "owner_username": "@tsuklone",
                    "telegram_bot_token": "PUT-YOUR-TOKEN-HERE",
                    "telegram_bot_username": "@YourBot",
                    "openrouter": {
                        "minimax": "sk-or-v1-...",
                        "hy3": "sk-or-v1-...",
                        "qwen": "sk-or-v1-...",
                        "whisper_large_v3": "sk-or-v1-...",
                        "whisper_large_v3_turbo": "sk-or-v1-...",
                        "whisper_1": "sk-or-v1-...",
                        "gemini_image": "sk-or-v1-...",
                        "veo": "sk-or-v1-...",
                        "rerank": "sk-or-v1-...",
                        "embed": "sk-or-v1-...",
                        "flux": "sk-or-v1-...",
                        "seedream": "sk-or-v1-...",
                        "kokoro": "sk-or-v1-...",
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return example_path


# ---------------------------------------------------------------------------
# Системный промпт ИИ-агента
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
Ты — встроенный ИИ-агент кастомного приложения «{app_name}».
У тебя есть полный контекст проекта: это один Python-файл (Python 3,
python-telegram-bot 20.7), который запускает GUI-чат и Telegram-бота
в одном процессе. Владелец и единственный администратор бота — {owner}.

Текущий чат:
- Имя: {chat_name}
- Бот: {bot_username} (токен задан в коде)
- Описание: {chat_description}

Твоя задача — отвечать пользователю и при необходимости управлять ботом
и проектом через инструменты. Можно предлагать пользователю варианты
действий (кнопки, опросы), редактировать код проекта, переименовывать
бота, отправлять сообщения, создавать опросы, читать и писать файлы,
запускать код Python в безопасной песочнице.

ФОРМАТ ОТВЕТА — строгий, парсится приложением.

Обычный текст пишется как есть. Поддерживается **жирный** и *курсив*.
Списки начинаются с «- » или «1. ». Подсветка синтаксиса включается
тройными бэктиками с языком, например ```python ... ```.

Чтобы вызвать инструмент или показать пользователю интерактивный
элемент — добавь в ответ блок ``` ``` agent\n{{...}}\n``` ``` (без
пробела после слова agent), где JSON описывает действие. Поддерживаются
типы:

  {{"type": "tool", "name": "<имя>", "args": {{...}}}}
      — вызов инструмента (см. ниже). Результат приложение пришлёт тебе
        отдельным system-сообщением.

  {{"type": "buttons", "title": "Что сделать?",
    "items": [
      {{"label": "Обновить", "style": "primary",
        "tool": {{"name": "rename_bot", "args": {{"name": "Crazy Bot"}}}}}},
      {{"label": "Просто проверить", "style": "ghost", "prompt": "проверь связь"}}
    ]}}
      — кнопки под ответом. style: primary | success | warning | danger | ghost.
        В кнопке либо tool (немедленный вызов), либо prompt (отправит за
        пользователя сообщение).

  {{"type": "poll", "title": "Какой стиль?",
    "options": ["формальный", "дружелюбный", "технический"],
    "multi": false}}
      — опрос. Выбор пользователя возвращается тебе как user-сообщение.

  {{"type": "checklist", "title": "План",
    "items": ["Прочитать код", "Найти баг", "Написать тест"]}}
      — чеклист с галочками.

Доступные инструменты (имя → описание args):

  rename_bot          — args: {{"name": "<новое имя бота>"}}
  set_bot_description — args: {{"description": "<текст>"}}
  set_bot_short_desc  — args: {{"description": "<текст>"}}
  send_telegram       — args: {{"chat": "<@user|id>", "text": "<text>"}}
  send_poll           — args: {{"chat": "<@user|id>", "question": "...",
                                "options": ["a","b","c"]}}
  read_file           — args: {{"path": "..."}}
  write_file          — args: {{"path": "...", "content": "..."}}
  list_dir            — args: {{"path": "."}}
  run_python          — args: {{"code": "..."}}  # короткий скрипт с таймаутом
  shell               — args: {{"cmd": "echo hi"}} # подтверждение пользователя

Думай по шагам, но финальный ответ — короткий и по делу. Если пользователь
просит «сделай то-то с ботом», вызывай инструмент сразу через блок agent
и кратко комментируй. Никогда не выдумывай результат инструмента — жди
system-сообщения с результатом перед тем как отчитаться.

Текущая дата/время: {now}.
"""


# ---------------------------------------------------------------------------
# Структуры данных
# ---------------------------------------------------------------------------


@dataclass
class BotConfig:
    token: str = ""
    username: str = ""
    owner: str = OWNER_USERNAME
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotConfig":
        return cls(
            token=str(data.get("token", "")),
            username=str(data.get("username", "")),
            owner=str(data.get("owner", OWNER_USERNAME)),
            description=str(data.get("description", "")),
        )


@dataclass
class Message:
    role: str  # user | assistant | system | tool
    content: str
    ts: float = field(default_factory=lambda: time.time())
    tokens: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=str(data.get("role", "user")),
            content=str(data.get("content", "")),
            ts=float(data.get("ts", time.time())),
            tokens=int(data.get("tokens", 0)),
            meta=dict(data.get("meta", {})),
        )


@dataclass
class Chat:
    id: str
    name: str
    bot: BotConfig
    messages: List[Message] = field(default_factory=list)
    total_tokens: int = 0
    created: float = field(default_factory=lambda: time.time())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "bot": self.bot.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "total_tokens": self.total_tokens,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chat":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex),
            name=str(data.get("name", "Новый чат")),
            bot=BotConfig.from_dict(data.get("bot", {})),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            total_tokens=int(data.get("total_tokens", 0)),
            created=float(data.get("created", time.time())),
        )


# ---------------------------------------------------------------------------
# Хранилище чатов
# ---------------------------------------------------------------------------


class ChatStore:
    def __init__(self, path: Path = CHATS_FILE) -> None:
        self.path = path
        self.chats: Dict[str, Chat] = {}
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Не получилось прочитать %s", self.path)
            return
        for item in raw.get("chats", []):
            chat = Chat.from_dict(item)
            self.chats[chat.id] = chat

    def save(self) -> None:
        data = {"chats": [c.to_dict() for c in self.chats.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, chat: Chat) -> None:
        self.chats[chat.id] = chat
        self.save()

    def remove(self, chat_id: str) -> None:
        self.chats.pop(chat_id, None)
        self.save()


# ---------------------------------------------------------------------------
# OpenRouter клиент со стримингом
# ---------------------------------------------------------------------------


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    """Простой клиент OpenRouter с поддержкой стриминга и фолбэков."""

    def __init__(self, models: Optional[List[Dict[str, str]]] = None) -> None:
        if requests is None:
            raise OpenRouterError("Не установлен пакет 'requests' (pip install requests)")
        self.models = models if models is not None else CHAT_MODELS

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        stop_event: threading.Event,
        on_delta: Callable[[str], None],
        on_meta: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_tokens: int = 1500,
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.models:
            raise OpenRouterError(
                "Не настроен ни один OpenRouter-ключ. Заполните "
                f"{DATA_DIR / 'secrets.json'} (шаблон: secrets.json.example) "
                "или экспортируйте OPENROUTER_KEY_<имя>."
            )
        last_err: Optional[Exception] = None
        for spec in self.models:
            if stop_event.is_set():
                raise OpenRouterError("Остановлено пользователем")
            try:
                full, meta = self._stream_one(spec, messages, stop_event, on_delta, max_tokens)
                if on_meta:
                    on_meta({"model": spec["model"], "label": spec["label"], **meta})
                return full, meta
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("Модель %s не сработала: %s", spec["name"], exc)
                if stop_event.is_set():
                    break
        raise OpenRouterError(f"Все текстовые модели OpenRouter недоступны: {last_err}")

    def _stream_one(
        self,
        spec: Dict[str, str],
        messages: List[Dict[str, str]],
        stop_event: threading.Event,
        on_delta: Callable[[str], None],
        max_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        url = f"{OPENROUTER_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {spec['key']}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/" + (os.environ.get("AI_CHAT_REFERER", "danimirpetrakov16-byte/MiniAppstg")),
            "X-Title": APP_NAME,
        }
        payload = {
            "model": spec["model"],
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.6,
        }
        full: List[str] = []
        meta: Dict[str, Any] = {}
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as resp:
            if resp.status_code >= 400:
                text = resp.text[:600]
                raise OpenRouterError(f"HTTP {resp.status_code}: {text}")
            for raw in resp.iter_lines(decode_unicode=True):
                if stop_event.is_set():
                    break
                if not raw:
                    continue
                if raw.startswith(":"):
                    continue
                if raw.startswith("data: "):
                    data = raw[6:].strip()
                else:
                    data = raw.strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                meta.update({k: v for k, v in obj.items() if k in {"id", "model", "usage"}})
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                token = delta.get("content")
                if token:
                    full.append(token)
                    on_delta(token)
                finish = choices[0].get("finish_reason")
                if finish:
                    meta["finish_reason"] = finish
        return "".join(full), meta


# ---------------------------------------------------------------------------
# Оценка количества токенов (грубая, без tiktoken)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # ~ 1 токен ≈ 4 символа в среднем (грубо, но работает без зависимостей)
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Парсинг ответа агента: отделяем agent-блоки и подсветку синтаксиса
# ---------------------------------------------------------------------------


CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)


def parse_response(text: str) -> List[Dict[str, Any]]:
    """Разбирает текст ответа на список «частей» для рендера."""

    parts: List[Dict[str, Any]] = []
    last = 0
    for m in CODE_BLOCK_RE.finditer(text):
        if m.start() > last:
            parts.append({"type": "text", "text": text[last:m.start()]})
        lang = (m.group(1) or "").strip().lower()
        body = m.group(2)
        if lang == "agent":
            try:
                obj = json.loads(body.strip())
                parts.append({"type": "agent", "data": obj})
            except json.JSONDecodeError as exc:
                parts.append({"type": "code", "lang": "json", "code": body, "error": f"agent JSON: {exc}"})
        else:
            parts.append({"type": "code", "lang": lang or "text", "code": body})
        last = m.end()
    if last < len(text):
        parts.append({"type": "text", "text": text[last:]})
    return parts


# ---------------------------------------------------------------------------
# Подсветка синтаксиса (без зависимостей)
# ---------------------------------------------------------------------------


PY_KEYWORDS = (
    "False None True and as assert async await break class continue def del elif else except "
    "finally for from global if import in is lambda nonlocal not or pass raise return try while "
    "with yield match case"
).split()
JS_KEYWORDS = (
    "var let const function return if else for while do switch case break continue class new "
    "this typeof instanceof in of try catch finally throw async await import export from default true false null undefined"
).split()


def _word_re(words: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(map(re.escape, words)) + r")\b")


SYNTAX_RULES: Dict[str, List[Tuple[str, re.Pattern[str]]]] = {
    "python": [
        ("comment", re.compile(r"#[^\n]*")),
        ("string", re.compile(r"(?s)\"\"\".*?\"\"\"|'''.*?'''|\"[^\"\n]*\"|'[^'\n]*'")),
        ("decorator", re.compile(r"@\w+")),
        ("number", re.compile(r"\b\d+(?:\.\d+)?\b")),
        ("keyword", _word_re(PY_KEYWORDS)),
        ("def", re.compile(r"\b(?:def|class)\s+(\w+)")),
    ],
    "json": [
        ("string", re.compile(r"\"(?:\\.|[^\"\\])*\"")),
        ("number", re.compile(r"\b-?\d+(?:\.\d+)?\b")),
        ("keyword", re.compile(r"\b(?:true|false|null)\b")),
    ],
    "javascript": [
        ("comment", re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)),
        ("string", re.compile(r"`(?:\\.|[^`\\])*`|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'")),
        ("number", re.compile(r"\b\d+(?:\.\d+)?\b")),
        ("keyword", _word_re(JS_KEYWORDS)),
    ],
    "html": [
        ("string", re.compile(r"\"[^\"]*\"|'[^']*'")),
        ("comment", re.compile(r"<!--.*?-->", re.DOTALL)),
        ("keyword", re.compile(r"</?\w+|>")),
    ],
    "bash": [
        ("comment", re.compile(r"#[^\n]*")),
        ("string", re.compile(r"\"[^\"]*\"|'[^']*'")),
        ("keyword", _word_re(["if", "then", "fi", "for", "while", "do", "done", "case", "esac", "function", "in", "echo", "exit"])),
    ],
}
SYNTAX_RULES["py"] = SYNTAX_RULES["python"]
SYNTAX_RULES["js"] = SYNTAX_RULES["javascript"]
SYNTAX_RULES["sh"] = SYNTAX_RULES["bash"]


# ---------------------------------------------------------------------------
# AI-агент
# ---------------------------------------------------------------------------


class AIAgent:
    """Связывает чат, инструменты и OpenRouter."""

    def __init__(self, app: "App") -> None:
        self.app = app
        self.client = OpenRouterClient(CHAT_MODELS)

    # --- системный промпт ------------------------------------------------
    def system_prompt(self, chat: Chat) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            app_name=APP_NAME,
            owner=chat.bot.owner or OWNER_USERNAME,
            chat_name=chat.name,
            bot_username=chat.bot.username or "(не указан)",
            chat_description=chat.bot.description or "(нет описания)",
            now=datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        )

    # --- сборка messages для API ----------------------------------------
    def build_messages(self, chat: Chat, extra_user: Optional[str] = None) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": self.system_prompt(chat)}]
        for m in chat.messages[-40:]:
            role = m.role
            if role == "tool":
                role = "system"  # OpenRouter ждёт стандартные роли
            msgs.append({"role": role, "content": m.content})
        if extra_user is not None:
            msgs.append({"role": "user", "content": extra_user})
        return msgs

    # --- основной цикл генерации ----------------------------------------
    def generate(
        self,
        chat: Chat,
        on_delta: Callable[[str], None],
        on_meta: Callable[[Dict[str, Any]], None],
        stop_event: threading.Event,
    ) -> Tuple[str, Dict[str, Any]]:
        msgs = self.build_messages(chat)
        return self.client.stream_chat(msgs, stop_event, on_delta, on_meta)


# ---------------------------------------------------------------------------
# Telegram bot — менеджер. Запускает PTB v20 в отдельном потоке/loop.
# ---------------------------------------------------------------------------


class TelegramBotManager:
    """Один менеджер на чат — он же и держит фоновое приложение PTB."""

    def __init__(self, config: BotConfig, on_event: Callable[[str, Dict[str, Any]], None]) -> None:
        self.config = config
        self.on_event = on_event
        self._loop = None
        self._thread: Optional[threading.Thread] = None
        self._app = None  # telegram.ext.Application
        self._started = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()

    # --- управление жизненным циклом ------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if not self.config.token:
            raise RuntimeError("Не задан токен бота")
        self._stop.clear()
        self._started.clear()
        self._thread = threading.Thread(target=self._run, name="ptb-loop", daemon=True)
        self._thread.start()
        if not self._started.wait(timeout=15):
            raise RuntimeError("Бот не успел запуститься за 15 секунд")

    def stop(self) -> None:
        if not self._thread:
            return
        self._stop.set()
        if self._loop and self._app is not None:
            try:
                fut = self._submit(self._shutdown())
                fut.result(timeout=15)
            except Exception:
                log.exception("Ошибка при остановке бота")
        self._thread.join(timeout=5)
        self._thread = None
        self._app = None
        self._loop = None

    # --- вспомогательные методы -----------------------------------------
    def _submit(self, coro):
        if self._loop is None:
            raise RuntimeError("Бот не запущен")
        import asyncio
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run(self) -> None:
        import asyncio
        try:
            from telegram.ext import Application, CommandHandler, MessageHandler, filters
        except Exception:
            log.exception("python-telegram-bot не установлен")
            self.on_event("bot_error", {"error": "python-telegram-bot not installed"})
            self._started.set()
            return

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            app = Application.builder().token(self.config.token).build()
            self._app = app
            app.add_handler(CommandHandler("start", self._cmd_start))
            app.add_handler(CommandHandler("help", self._cmd_help))
            app.add_handler(CommandHandler("whoami", self._cmd_whoami))
            app.add_handler(MessageHandler(filters.ALL, self._on_message))

            async def runner() -> None:
                await app.initialize()
                await app.start()
                me = await app.bot.get_me()
                self.config.username = "@" + (me.username or "")
                self.on_event("bot_started", {"username": self.config.username, "name": me.first_name})
                await app.updater.start_polling(allowed_updates=None, drop_pending_updates=True)
                self._started.set()
                while not self._stop.is_set():
                    await asyncio.sleep(0.5)

            loop.run_until_complete(runner())
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка бота")
            self.on_event("bot_error", {"error": str(exc)})
            self._started.set()
        finally:
            loop.close()

    async def _shutdown(self) -> None:
        if self._app is None:
            return
        try:
            if self._app.updater and self._app.updater.running:
                await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception:
            log.exception("Bot shutdown failed")

    # --- handlers ---------------------------------------------------------
    async def _cmd_start(self, update, context) -> None:
        await update.effective_message.reply_text(
            f"Привет! Я кастомный AI-бот. Владелец: {self.config.owner}.\n"
            "Команды: /help, /whoami"
        )
        self.on_event("incoming", {"text": "/start", "from": update.effective_user.username})

    async def _cmd_help(self, update, context) -> None:
        await update.effective_message.reply_text(
            "Этот бот управляется встроенным ИИ-агентом приложения. "
            "Пиши, что нужно сделать — агент решит."
        )

    async def _cmd_whoami(self, update, context) -> None:
        u = update.effective_user
        await update.effective_message.reply_text(
            f"Ты: @{u.username} ({u.id})\nВладелец бота: {self.config.owner}"
        )

    async def _on_message(self, update, context) -> None:
        msg = update.effective_message
        if not msg or not msg.text:
            return
        self.on_event(
            "incoming",
            {"text": msg.text, "from": "@" + (update.effective_user.username or "?"),
             "chat_id": msg.chat_id},
        )

    # --- API для агента --------------------------------------------------
    def send_message(self, chat: str, text: str) -> Dict[str, Any]:
        async def _do():
            target = self._resolve_chat(chat)
            return await self._app.bot.send_message(chat_id=target, text=text)
        m = self._submit(_do()).result(timeout=20)
        return {"message_id": m.message_id, "chat_id": m.chat_id}

    def send_poll(self, chat: str, question: str, options: List[str]) -> Dict[str, Any]:
        async def _do():
            target = self._resolve_chat(chat)
            return await self._app.bot.send_poll(chat_id=target, question=question, options=options, is_anonymous=False)
        m = self._submit(_do()).result(timeout=20)
        return {"poll_id": m.poll.id, "chat_id": m.chat_id}

    def set_my_name(self, name: str) -> Dict[str, Any]:
        async def _do():
            ok = await self._app.bot.set_my_name(name=name)
            return ok
        ok = self._submit(_do()).result(timeout=15)
        return {"ok": bool(ok), "name": name}

    def set_my_description(self, description: str) -> Dict[str, Any]:
        async def _do():
            return await self._app.bot.set_my_description(description=description)
        ok = self._submit(_do()).result(timeout=15)
        return {"ok": bool(ok)}

    def set_my_short_description(self, description: str) -> Dict[str, Any]:
        async def _do():
            return await self._app.bot.set_my_short_description(short_description=description)
        ok = self._submit(_do()).result(timeout=15)
        return {"ok": bool(ok)}

    def _resolve_chat(self, chat: str) -> Any:
        if not chat:
            raise ValueError("Не указан адресат")
        chat = str(chat).strip()
        if chat.startswith("@"):
            return chat
        try:
            return int(chat)
        except ValueError:
            return chat


# ---------------------------------------------------------------------------
# Инструменты — единый исполнитель
# ---------------------------------------------------------------------------


class ToolError(RuntimeError):
    pass


class ToolRunner:
    def __init__(self, app: "App") -> None:
        self.app = app

    def run(self, chat: Chat, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        log.info("Tool %s args=%s", name, args)
        bot = self.app.get_bot_manager(chat)
        try:
            if name == "rename_bot":
                return bot.set_my_name(str(args.get("name", "")))
            if name == "set_bot_description":
                return bot.set_my_description(str(args.get("description", "")))
            if name == "set_bot_short_desc":
                return bot.set_my_short_description(str(args.get("description", "")))
            if name == "send_telegram":
                return bot.send_message(str(args.get("chat", "")), str(args.get("text", "")))
            if name == "send_poll":
                return bot.send_poll(
                    str(args.get("chat", "")),
                    str(args.get("question", "")),
                    [str(o) for o in (args.get("options") or [])],
                )
            if name == "read_file":
                return self._read_file(str(args.get("path", "")))
            if name == "write_file":
                return self._write_file(str(args.get("path", "")), str(args.get("content", "")))
            if name == "list_dir":
                return self._list_dir(str(args.get("path", ".")))
            if name == "run_python":
                return self._run_python(str(args.get("code", "")))
            if name == "shell":
                return self._shell(str(args.get("cmd", "")))
            raise ToolError(f"Неизвестный инструмент: {name}")
        except Exception as exc:  # noqa: BLE001
            log.exception("Ошибка инструмента %s", name)
            return {"error": str(exc)}

    def _safe_path(self, p: str) -> Path:
        p = (WORKSPACE_DIR / p).resolve() if not Path(p).is_absolute() else Path(p).resolve()
        ws = WORKSPACE_DIR.resolve()
        if not str(p).startswith(str(ws)):
            raise ToolError(f"Путь вне рабочей директории {ws}")
        return p

    def _read_file(self, path: str) -> Dict[str, Any]:
        p = self._safe_path(path)
        if not p.exists():
            return {"error": f"Нет файла {p}"}
        try:
            return {"path": str(p), "content": p.read_text(encoding="utf-8")}
        except UnicodeDecodeError:
            return {"path": str(p), "content_b64": p.read_bytes().hex(), "encoding": "hex"}

    def _write_file(self, path: str, content: str) -> Dict[str, Any]:
        p = self._safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return {"path": str(p), "bytes": len(content.encode("utf-8"))}

    def _list_dir(self, path: str) -> Dict[str, Any]:
        p = self._safe_path(path)
        if not p.exists():
            return {"error": f"Нет директории {p}"}
        items = []
        for child in sorted(p.iterdir()):
            items.append({
                "name": child.name,
                "is_dir": child.is_dir(),
                "size": child.stat().st_size if child.is_file() else None,
            })
        return {"path": str(p), "items": items}

    def _run_python(self, code: str) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(dir=str(WORKSPACE_DIR)) as tmp:
            script = Path(tmp) / "snippet.py"
            script.write_text(code, encoding="utf-8")
            try:
                out = subprocess.run(
                    [sys.executable, str(script)],
                    capture_output=True, text=True, timeout=10,
                    cwd=tmp,
                )
            except subprocess.TimeoutExpired:
                return {"error": "Превышен таймаут (10s)"}
            return {
                "returncode": out.returncode,
                "stdout": out.stdout[-4000:],
                "stderr": out.stderr[-4000:],
            }

    def _shell(self, cmd: str) -> Dict[str, Any]:
        # требует подтверждения пользователя через GUI
        if not self.app.confirm_shell(cmd):
            return {"error": "Пользователь отказал в выполнении shell"}
        try:
            out = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=20)
        except subprocess.TimeoutExpired:
            return {"error": "Превышен таймаут (20s)"}
        return {"returncode": out.returncode, "stdout": out.stdout[-4000:], "stderr": out.stderr[-4000:]}


# ---------------------------------------------------------------------------
# GUI: рендер сообщений, агентских блоков, подсветка синтаксиса
# ---------------------------------------------------------------------------


# Импортируем tkinter лениво — чтобы файл можно было импортировать без X-сервера
def _import_tk():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog, scrolledtext
    return tk, ttk, filedialog, messagebox, simpledialog, scrolledtext


PALETTE = {
    "bg":          "#0f1115",
    "bg_alt":      "#161a22",
    "bg_panel":    "#1e2330",
    "border":      "#2a3040",
    "fg":          "#e7e9ee",
    "fg_dim":      "#9ba3b4",
    "user":        "#7aa2ff",
    "ai":          "#e7e9ee",
    "system":      "#9ba3b4",
    "tool":        "#f4b860",
    "code_bg":     "#0b1020",
    "code_fg":     "#e7e9ee",
    "kw":          "#c084fc",
    "string":      "#f5b864",
    "comment":     "#6b7488",
    "number":      "#69d6cf",
    "decorator":   "#f87171",
    "def":         "#7aa2ff",
    "btn_primary": "#3b82f6",
    "btn_success": "#16a34a",
    "btn_warning": "#d97706",
    "btn_danger":  "#dc2626",
    "btn_ghost":   "#374151",
    "thinking":    "#9ba3b4",
}


class App:
    """Главное окно приложения."""

    def __init__(self) -> None:
        tk, ttk, filedialog, messagebox, simpledialog, scrolledtext = _import_tk()
        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.simpledialog = simpledialog

        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{APP_VERSION}")
        self.root.geometry("1280x820")
        self.root.configure(bg=PALETTE["bg"])
        self._configure_styles()

        self.store = ChatStore()
        self.agent = AIAgent(self)
        self.tools = ToolRunner(self)
        self.bot_managers: Dict[str, TelegramBotManager] = {}

        # стейт активной генерации
        self._gen_thread: Optional[threading.Thread] = None
        self._gen_stop = threading.Event()
        self._ui_queue: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self._is_generating = False
        self._streaming_buffer = ""

        self._build_ui()
        self._refresh_chat_list()
        if self.store.chats:
            first = next(iter(self.store.chats.values()))
            self.select_chat(first.id)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self._drain_ui_queue)

    # ---- Стили ----------------------------------------------------------
    def _configure_styles(self) -> None:
        s = self.ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure("TFrame", background=PALETTE["bg"])
        s.configure("Side.TFrame", background=PALETTE["bg_alt"])
        s.configure("Panel.TFrame", background=PALETTE["bg_panel"])
        s.configure(
            "TLabel", background=PALETTE["bg"], foreground=PALETTE["fg"], font=("Segoe UI", 10)
        )
        s.configure("Side.TLabel", background=PALETTE["bg_alt"], foreground=PALETTE["fg_dim"])
        s.configure("Title.TLabel", background=PALETTE["bg_panel"], foreground=PALETTE["fg"], font=("Segoe UI", 12, "bold"))
        s.configure("Dim.TLabel", background=PALETTE["bg_panel"], foreground=PALETTE["fg_dim"], font=("Segoe UI", 9))
        s.configure(
            "TButton", background=PALETTE["bg_panel"], foreground=PALETTE["fg"], borderwidth=0,
            padding=(10, 6), font=("Segoe UI", 10),
        )
        s.map(
            "TButton",
            background=[("active", PALETTE["border"]), ("disabled", PALETTE["bg_alt"])],
        )
        s.configure(
            "Primary.TButton", background=PALETTE["btn_primary"], foreground="#ffffff",
        )
        s.map("Primary.TButton", background=[("active", "#2563eb")])
        s.configure("Danger.TButton", background=PALETTE["btn_danger"], foreground="#ffffff")
        s.map("Danger.TButton", background=[("active", "#b91c1c")])
        s.configure(
            "Treeview", background=PALETTE["bg_alt"], fieldbackground=PALETTE["bg_alt"],
            foreground=PALETTE["fg"], borderwidth=0, rowheight=28,
        )
        s.map("Treeview", background=[("selected", PALETTE["btn_primary"])])

    # ---- Конструктор UI -------------------------------------------------
    def _build_ui(self) -> None:
        tk = self.tk
        ttk = self.ttk

        root = self.root
        # MENU
        menu = tk.Menu(root)
        filem = tk.Menu(menu, tearoff=0)
        filem.add_command(label="Импорт чатов…", command=self.import_chats)
        filem.add_command(label="Экспорт чатов…", command=self.export_chats)
        filem.add_separator()
        filem.add_command(label="Выход", command=self.on_close)
        menu.add_cascade(label="Файл", menu=filem)
        helpm = tk.Menu(menu, tearoff=0)
        helpm.add_command(label="О программе", command=self._about)
        menu.add_cascade(label="Помощь", menu=helpm)
        root.config(menu=menu)

        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)

        # ---- сайдбар чатов
        sidebar = ttk.Frame(main, style="Side.TFrame", width=280)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        title = ttk.Label(sidebar, text="Чаты", style="Side.TLabel", font=("Segoe UI", 13, "bold"))
        title.pack(anchor="w", padx=14, pady=(12, 4))
        ttk.Label(sidebar, text=f"Владелец: {OWNER_USERNAME}", style="Side.TLabel").pack(anchor="w", padx=14)

        new_btn = ttk.Button(sidebar, text="+ Новый чат", style="Primary.TButton", command=self.new_chat_dialog)
        new_btn.pack(fill="x", padx=10, pady=10)

        self.chat_tree = ttk.Treeview(sidebar, show="tree", selectmode="browse")
        self.chat_tree.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.chat_tree.bind("<<TreeviewSelect>>", self._on_chat_select)
        self.chat_tree.bind("<Button-3>", self._on_chat_right_click)

        del_btn = ttk.Button(sidebar, text="Удалить чат", style="Danger.TButton", command=self.delete_current_chat)
        del_btn.pack(fill="x", padx=10, pady=(0, 10))

        # ---- центральная панель
        center = ttk.Frame(main, style="Panel.TFrame")
        center.pack(side="left", fill="both", expand=True)

        header = ttk.Frame(center, style="Panel.TFrame")
        header.pack(fill="x")
        self.title_lbl = ttk.Label(header, text="Выберите или создайте чат", style="Title.TLabel")
        self.title_lbl.pack(side="left", padx=14, pady=10)
        self.token_lbl = ttk.Label(header, text="0 токенов", style="Dim.TLabel")
        self.token_lbl.pack(side="right", padx=14)
        self.status_lbl = ttk.Label(header, text="готов", style="Dim.TLabel")
        self.status_lbl.pack(side="right", padx=8)

        self.bot_btn = ttk.Button(header, text="Запустить бота", command=self.toggle_bot)
        self.bot_btn.pack(side="right", padx=8, pady=6)

        # ---- область сообщений
        msg_wrap = ttk.Frame(center, style="Panel.TFrame")
        msg_wrap.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.text = tk.Text(
            msg_wrap, wrap="word", bg=PALETTE["bg_panel"], fg=PALETTE["fg"],
            insertbackground=PALETTE["fg"], borderwidth=0, padx=14, pady=14,
            font=("Segoe UI", 11), state="disabled", spacing1=4, spacing3=4,
        )
        self.text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(msg_wrap, command=self.text.yview)
        sb.pack(side="right", fill="y")
        self.text.config(yscrollcommand=sb.set)
        self._setup_text_tags()

        # ---- ввод
        input_box = ttk.Frame(center, style="Panel.TFrame")
        input_box.pack(fill="x", padx=12, pady=(0, 12))
        self.input = tk.Text(
            input_box, height=4, wrap="word", bg=PALETTE["bg_alt"], fg=PALETTE["fg"],
            insertbackground=PALETTE["fg"], borderwidth=0, padx=10, pady=8, font=("Segoe UI", 11),
        )
        self.input.pack(side="left", fill="both", expand=True)
        self.input.bind("<Control-Return>", lambda e: (self.send_message(), "break"))
        self.input.bind("<Return>", self._maybe_send)

        btn_col = ttk.Frame(input_box, style="Panel.TFrame")
        btn_col.pack(side="right", fill="y", padx=(8, 0))
        self.send_btn = ttk.Button(btn_col, text="Отправить", style="Primary.TButton", command=self.send_message)
        self.send_btn.pack(fill="x", pady=(0, 4))
        self.stop_btn = ttk.Button(btn_col, text="Стоп", style="Danger.TButton", command=self.stop_generation, state="disabled")
        self.stop_btn.pack(fill="x")

        self.current_chat: Optional[Chat] = None

    def _setup_text_tags(self) -> None:
        T = self.text
        T.tag_configure("user_label", foreground=PALETTE["user"], font=("Segoe UI", 11, "bold"))
        T.tag_configure("ai_label", foreground=PALETTE["btn_primary"], font=("Segoe UI", 11, "bold"))
        T.tag_configure("system_label", foreground=PALETTE["tool"], font=("Segoe UI", 10, "italic"))
        T.tag_configure("user", foreground=PALETTE["fg"])
        T.tag_configure("ai", foreground=PALETTE["fg"])
        T.tag_configure("system", foreground=PALETTE["fg_dim"], font=("Segoe UI", 10, "italic"))
        T.tag_configure("thinking", foreground=PALETTE["thinking"], font=("Segoe UI", 10, "italic"))
        T.tag_configure("bold", font=("Segoe UI", 11, "bold"))
        T.tag_configure("italic", font=("Segoe UI", 11, "italic"))
        T.tag_configure("code", background=PALETTE["code_bg"], foreground=PALETTE["code_fg"],
                        font=("JetBrains Mono", 10), lmargin1=14, lmargin2=14, rmargin=14, spacing1=4, spacing3=4)
        T.tag_configure("kw", foreground=PALETTE["kw"])
        T.tag_configure("string", foreground=PALETTE["string"])
        T.tag_configure("comment", foreground=PALETTE["comment"], font=("JetBrains Mono", 10, "italic"))
        T.tag_configure("number", foreground=PALETTE["number"])
        T.tag_configure("decorator", foreground=PALETTE["decorator"])
        T.tag_configure("def", foreground=PALETTE["def"], font=("JetBrains Mono", 10, "bold"))
        T.tag_configure("error", foreground=PALETTE["btn_danger"])

    # ---- меню/диалоги ---------------------------------------------------
    def _about(self) -> None:
        self.messagebox.showinfo(
            APP_NAME,
            f"{APP_NAME} v{APP_VERSION}\n"
            f"Однофайловое приложение: AI-чат + Telegram-бот.\n"
            f"Владелец: {OWNER_USERNAME}\n"
            f"Данные: {DATA_DIR}",
        )

    # ---- список чатов ---------------------------------------------------
    def _refresh_chat_list(self) -> None:
        for i in self.chat_tree.get_children():
            self.chat_tree.delete(i)
        for c in self.store.chats.values():
            self.chat_tree.insert("", "end", iid=c.id, text=f"{c.name}  ·  {c.bot.username or 'без бота'}")

    def _on_chat_select(self, _evt) -> None:
        sel = self.chat_tree.selection()
        if sel:
            self.select_chat(sel[0])

    def _on_chat_right_click(self, evt) -> None:
        iid = self.chat_tree.identify_row(evt.y)
        if not iid:
            return
        m = self.tk.Menu(self.root, tearoff=0)
        m.add_command(label="Переименовать", command=lambda: self.rename_chat(iid))
        m.add_command(label="Удалить", command=lambda: self.delete_chat(iid))
        m.tk_popup(evt.x_root, evt.y_root)

    def select_chat(self, chat_id: str) -> None:
        chat = self.store.chats.get(chat_id)
        if not chat:
            return
        self.current_chat = chat
        if not self.chat_tree.selection() or self.chat_tree.selection()[0] != chat_id:
            self.chat_tree.selection_set(chat_id)
        self.title_lbl.config(text=f"{chat.name}    бот: {chat.bot.username or '—'}")
        self.token_lbl.config(text=f"{chat.total_tokens} токенов")
        self.bot_btn.config(text=("Остановить бота" if chat.id in self.bot_managers else "Запустить бота"))
        self._render_chat()

    def new_chat_dialog(self) -> None:
        NewChatDialog(self)

    def add_chat(self, chat: Chat) -> None:
        self.store.add(chat)
        self._refresh_chat_list()
        self.select_chat(chat.id)

    def rename_chat(self, chat_id: str) -> None:
        chat = self.store.chats.get(chat_id)
        if not chat:
            return
        new = self.simpledialog.askstring("Переименовать", "Новое имя чата:", initialvalue=chat.name)
        if new:
            chat.name = new
            self.store.save()
            self._refresh_chat_list()
            if self.current_chat and self.current_chat.id == chat.id:
                self.select_chat(chat.id)

    def delete_current_chat(self) -> None:
        if self.current_chat:
            self.delete_chat(self.current_chat.id)

    def delete_chat(self, chat_id: str) -> None:
        chat = self.store.chats.get(chat_id)
        if not chat:
            return
        if not self.messagebox.askyesno("Удалить", f"Удалить чат «{chat.name}»?"):
            return
        bm = self.bot_managers.pop(chat.id, None)
        if bm:
            try:
                bm.stop()
            except Exception:
                log.exception("stop bot failed")
        self.store.remove(chat.id)
        self._refresh_chat_list()
        self.current_chat = None
        self._clear_text()
        self.title_lbl.config(text="Выберите или создайте чат")

    def import_chats(self) -> None:
        path = self.filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as exc:
            self.messagebox.showerror("Импорт", f"Не удалось прочитать: {exc}")
            return
        for item in data.get("chats", []):
            chat = Chat.from_dict(item)
            chat.id = uuid.uuid4().hex
            self.store.add(chat)
        self._refresh_chat_list()

    def export_chats(self) -> None:
        path = self.filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return
        Path(path).write_text(
            json.dumps({"chats": [c.to_dict() for c in self.store.chats.values()]}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.messagebox.showinfo("Экспорт", "Готово")

    # ---- бот ------------------------------------------------------------
    def get_bot_manager(self, chat: Chat) -> TelegramBotManager:
        bm = self.bot_managers.get(chat.id)
        if bm is None:
            raise ToolError("Бот не запущен. Нажмите «Запустить бота».")
        return bm

    def toggle_bot(self) -> None:
        if not self.current_chat:
            return
        chat = self.current_chat
        if chat.id in self.bot_managers:
            try:
                self.bot_managers.pop(chat.id).stop()
            except Exception:
                log.exception("stop bot failed")
            self.bot_btn.config(text="Запустить бота")
            self._append_system(f"Бот {chat.bot.username or ''} остановлен.")
            return
        if not chat.bot.token:
            self.messagebox.showerror("Бот", "Не задан токен.")
            return
        bm = TelegramBotManager(chat.bot, on_event=lambda kind, payload, c=chat: self._on_bot_event(c, kind, payload))
        try:
            bm.start()
        except Exception as exc:
            self.messagebox.showerror("Бот", f"Не удалось запустить: {exc}")
            return
        self.bot_managers[chat.id] = bm
        chat.bot = bm.config
        self.store.save()
        self.bot_btn.config(text="Остановить бота")
        self._append_system(f"Бот {chat.bot.username} запущен.")
        self.select_chat(chat.id)

    def _on_bot_event(self, chat: Chat, kind: str, payload: Dict[str, Any]) -> None:
        self._ui_queue.put(("bot_event", (chat.id, kind, payload)))

    # ---- рендер чата ----------------------------------------------------
    def _clear_text(self) -> None:
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")

    def _render_chat(self) -> None:
        self._clear_text()
        if not self.current_chat:
            return
        for m in self.current_chat.messages:
            self._render_message(m)
        self.text.see("end")

    def _render_message(self, m: Message) -> None:
        self.text.config(state="normal")
        if m.role == "user":
            self.text.insert("end", "Вы\n", "user_label")
            self._render_text_with_inline_markup(m.content, "user")
        elif m.role == "assistant":
            self.text.insert("end", "Агент\n", "ai_label")
            for part in parse_response(m.content):
                self._render_part(part)
        elif m.role == "tool":
            self.text.insert("end", "Инструмент\n", "system_label")
            self._render_text_with_inline_markup(m.content, "system")
        else:
            self.text.insert("end", "Система\n", "system_label")
            self._render_text_with_inline_markup(m.content, "system")
        self.text.insert("end", "\n\n")
        self.text.config(state="disabled")

    def _render_part(self, part: Dict[str, Any]) -> None:
        if part["type"] == "text":
            self._render_text_with_inline_markup(part["text"], "ai")
        elif part["type"] == "code":
            self._render_code(part.get("code", ""), part.get("lang", "text"))
        elif part["type"] == "agent":
            self._render_agent_block(part.get("data") or {})

    def _render_text_with_inline_markup(self, text: str, base_tag: str) -> None:
        # **bold**, *italic*, списки, заголовки, чеклисты [-]/[x]
        for line in text.splitlines(True):
            stripped = line.lstrip()
            if stripped.startswith("# "):
                self.text.insert("end", line, ("bold", base_tag))
                continue
            if stripped.startswith(("- ", "* ", "• ")):
                self.text.insert("end", "  • " + stripped[2:], base_tag)
                continue
            if re.match(r"^\d+\.\s", stripped):
                self.text.insert("end", "  " + line, base_tag)
                continue
            self._insert_inline(line, base_tag)

    def _insert_inline(self, line: str, base_tag: str) -> None:
        i = 0
        n = len(line)
        while i < n:
            if line.startswith("**", i):
                end = line.find("**", i + 2)
                if end != -1:
                    self.text.insert("end", line[i + 2:end], ("bold", base_tag))
                    i = end + 2
                    continue
            if line[i] == "*":
                end = line.find("*", i + 1)
                if end != -1 and end - i > 1:
                    self.text.insert("end", line[i + 1:end], ("italic", base_tag))
                    i = end + 1
                    continue
            self.text.insert("end", line[i], base_tag)
            i += 1

    def _render_code(self, code: str, lang: str) -> None:
        T = self.text
        start = T.index("end-1c")
        T.insert("end", code if code.endswith("\n") else code + "\n", "code")
        end = T.index("end-1c")
        # подсветка
        rules = SYNTAX_RULES.get(lang.lower())
        if rules:
            for tag_name, regex in rules:
                for m in regex.finditer(code):
                    s = self._offset_to_index(start, m.start())
                    e = self._offset_to_index(start, m.end())
                    T.tag_add(tag_name, s, e)
        # кнопки под кодом
        if lang.lower() in {"python", "py"}:
            self._add_inline_button("Запустить", lambda c=code: self._run_code_button(c), style="Primary.TButton")
        self._add_inline_button("Скопировать", lambda c=code: self._copy_to_clipboard(c))
        self._add_inline_button("Сохранить в файл…", lambda c=code, l=lang: self._save_code_to_file(c, l))
        T.insert("end", "\n")

    def _offset_to_index(self, start_index: str, offset: int) -> str:
        return self.text.index(f"{start_index} + {offset} chars")

    def _add_inline_button(self, label: str, cmd: Callable[[], None], style: str = "TButton") -> None:
        btn = self.ttk.Button(self.text, text=label, style=style, command=cmd)
        self.text.window_create("end", window=btn, padx=4, pady=2)

    def _render_agent_block(self, data: Dict[str, Any]) -> None:
        kind = (data.get("type") or "").lower()
        T = self.text
        if kind == "tool":
            name = data.get("name", "?")
            T.insert("end", f"  ⚙ инструмент: {name}\n", "system")
            return
        if kind == "buttons":
            title = data.get("title")
            if title:
                T.insert("end", f"  {title}\n", ("italic", "ai"))
            for it in data.get("items", []):
                style_map = {
                    "primary": "Primary.TButton",
                    "danger":  "Danger.TButton",
                }
                style = style_map.get(str(it.get("style", "")).lower(), "TButton")
                self._add_inline_button(
                    str(it.get("label", "Кнопка")),
                    (lambda i=it: self._on_agent_button(i)),
                    style=style,
                )
            T.insert("end", "\n")
            return
        if kind == "poll":
            title = data.get("title", "Опрос")
            T.insert("end", f"  📊 {title}\n", ("bold", "ai"))
            multi = bool(data.get("multi"))
            options = data.get("options", [])
            state = {"selected": set()}

            def make_cb(idx, var):
                def cb():
                    if var.get():
                        if not multi:
                            state["selected"].clear()
                        state["selected"].add(idx)
                    else:
                        state["selected"].discard(idx)
                return cb

            for idx, opt in enumerate(options):
                var = self.tk.BooleanVar(value=False)
                cb = self.ttk.Checkbutton(T, text=str(opt), variable=var, command=make_cb(idx, var))
                T.window_create("end", window=cb, padx=8)
                T.insert("end", "\n")
            self._add_inline_button(
                "Отправить выбор",
                lambda s=state, opts=options: self._send_poll_choice(opts, s["selected"]),
                style="Primary.TButton",
            )
            T.insert("end", "\n")
            return
        if kind == "checklist":
            title = data.get("title", "Чеклист")
            T.insert("end", f"  ✅ {title}\n", ("bold", "ai"))
            for it in data.get("items", []):
                var = self.tk.BooleanVar(value=False)
                cb = self.ttk.Checkbutton(T, text=str(it), variable=var)
                T.window_create("end", window=cb, padx=8)
                T.insert("end", "\n")
            T.insert("end", "\n")
            return
        T.insert("end", f"  agent({kind}): {json.dumps(data, ensure_ascii=False)}\n", "system")

    # ---- обработчики кнопок --------------------------------------------
    def _on_agent_button(self, item: Dict[str, Any]) -> None:
        if "tool" in item and isinstance(item["tool"], dict):
            self._dispatch_tool_call(item["tool"])
        elif "prompt" in item:
            self.input.delete("1.0", "end")
            self.input.insert("1.0", str(item["prompt"]))
            self.send_message()

    def _send_poll_choice(self, options: List[Any], selected: set) -> None:
        if not selected:
            self.messagebox.showinfo("Опрос", "Ничего не выбрано")
            return
        chosen = [str(options[i]) for i in sorted(selected) if 0 <= i < len(options)]
        self.input.delete("1.0", "end")
        self.input.insert("1.0", f"[Ответ опроса] {', '.join(chosen)}")
        self.send_message()

    def _run_code_button(self, code: str) -> None:
        res = self.tools._run_python(code)
        self._append_system(f"run_python → {json.dumps(res, ensure_ascii=False)}")

    def _copy_to_clipboard(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_lbl.config(text="скопировано")

    def _save_code_to_file(self, code: str, lang: str) -> None:
        ext = {"python": ".py", "py": ".py", "javascript": ".js", "js": ".js", "json": ".json", "html": ".html", "bash": ".sh", "sh": ".sh"}.get(lang.lower(), ".txt")
        path = self.filedialog.asksaveasfilename(defaultextension=ext)
        if not path:
            return
        Path(path).write_text(code, encoding="utf-8")
        self.status_lbl.config(text=f"сохранено: {Path(path).name}")

    # ---- отправка сообщения --------------------------------------------
    def _maybe_send(self, evt) -> Optional[str]:
        # Enter без shift — отправить, shift+enter — новая строка
        if evt.state & 0x0001:  # Shift
            return None
        self.send_message()
        return "break"

    def send_message(self) -> None:
        if self._is_generating or not self.current_chat:
            return
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        self.input.delete("1.0", "end")
        chat = self.current_chat
        m = Message(role="user", content=text, tokens=estimate_tokens(text))
        chat.messages.append(m)
        chat.total_tokens += m.tokens
        self.store.save()
        self._render_message(m)
        self._start_generation()

    def _start_generation(self) -> None:
        if self._gen_thread and self._gen_thread.is_alive():
            return
        self._is_generating = True
        self.send_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.input.config(state="disabled")
        self._gen_stop = threading.Event()
        self._streaming_buffer = ""
        self._show_thinking()
        chat = self.current_chat
        self._gen_thread = threading.Thread(
            target=self._gen_worker, args=(chat,), daemon=True
        )
        self._gen_thread.start()

    def _gen_worker(self, chat: Chat) -> None:
        try:
            full, meta = self.agent.generate(
                chat,
                on_delta=lambda t: self._ui_queue.put(("delta", t)),
                on_meta=lambda m: self._ui_queue.put(("meta", m)),
                stop_event=self._gen_stop,
            )
            self._ui_queue.put(("done", (full, meta)))
        except Exception as exc:
            log.exception("Ошибка генерации")
            self._ui_queue.put(("error", str(exc)))

    def stop_generation(self) -> None:
        self._gen_stop.set()
        self.status_lbl.config(text="остановка…")

    def _show_thinking(self) -> None:
        self.text.config(state="normal")
        self.text.insert("end", "Агент\n", "ai_label")
        self._think_start = self.text.index("end-1c")
        self.text.insert("end", "печатает…", ("thinking", "stream"))
        self.text.config(state="disabled")
        self.text.see("end")
        self.status_lbl.config(text="печатает…")

    def _drain_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                if kind == "delta":
                    self._on_stream_delta(payload)
                elif kind == "meta":
                    self._on_stream_meta(payload)
                elif kind == "done":
                    self._on_stream_done(*payload)
                elif kind == "error":
                    self._on_stream_error(payload)
                elif kind == "bot_event":
                    self._handle_bot_event(*payload)
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self._drain_ui_queue)

    def _on_stream_delta(self, token: str) -> None:
        self._streaming_buffer += token
        self.text.config(state="normal")
        # удаляем «печатает…» при первом токене
        if self._streaming_buffer == token:
            self.text.delete(self._think_start, "end-1c")
            self.text.insert("end", "")
        self.text.insert("end", token, ("ai", "stream"))
        self.text.see("end")
        self.text.config(state="disabled")

    def _on_stream_meta(self, meta: Dict[str, Any]) -> None:
        label = meta.get("label") or meta.get("model") or ""
        self.status_lbl.config(text=f"модель: {label}")

    def _on_stream_done(self, full: str, meta: Dict[str, Any]) -> None:
        chat = self.current_chat
        if not chat:
            self._reset_generation_ui()
            return
        # снимаем стриминг, перерисовываем сообщение полностью с разметкой
        self.text.config(state="normal")
        # удаляем последний «Агент\n…» который мы стримили
        self._delete_last_assistant_block()
        self.text.config(state="disabled")
        msg = Message(role="assistant", content=full, tokens=estimate_tokens(full), meta=meta)
        chat.messages.append(msg)
        chat.total_tokens += msg.tokens
        self.store.save()
        self._render_message(msg)
        self._reset_generation_ui()
        # автоматически выполняем все agent tool-блоки
        for part in parse_response(full):
            if part["type"] == "agent":
                data = part.get("data") or {}
                if data.get("type") == "tool":
                    self._dispatch_tool_call(data)
        self.token_lbl.config(text=f"{chat.total_tokens} токенов")

    def _delete_last_assistant_block(self) -> None:
        # удаляем последний кусок, начинающийся с "Агент\n"
        idx = self.text.search("Агент\n", "end", backwards=True, regexp=False)
        if idx:
            self.text.delete(idx, "end")

    def _on_stream_error(self, err: str) -> None:
        self._delete_last_assistant_block()
        self._append_system(f"Ошибка: {err}")
        self._reset_generation_ui()

    def _reset_generation_ui(self) -> None:
        self._is_generating = False
        self.send_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.input.config(state="normal")
        self.status_lbl.config(text="готов")

    def _append_system(self, text: str) -> None:
        if not self.current_chat:
            return
        m = Message(role="system", content=text)
        self.current_chat.messages.append(m)
        self.store.save()
        self._render_message(m)

    # ---- инструменты ----------------------------------------------------
    def _dispatch_tool_call(self, tool_data: Dict[str, Any]) -> None:
        if not self.current_chat:
            return
        name = str(tool_data.get("name", ""))
        args = tool_data.get("args") or {}
        if not name:
            return

        def runner():
            res = self.tools.run(self.current_chat, name, args)
            text = f"tool {name} → {json.dumps(res, ensure_ascii=False)[:1500]}"
            self._ui_queue.put(("tool_result", (name, res, text)))

        threading.Thread(target=runner, daemon=True).start()
        self._append_system(f"Запускаю инструмент {name} …")

    def _handle_bot_event(self, chat_id: str, kind: str, payload: Dict[str, Any]) -> None:
        chat = self.store.chats.get(chat_id)
        if not chat:
            return
        text = ""
        if kind == "bot_started":
            chat.bot.username = payload.get("username", chat.bot.username) or chat.bot.username
            self.store.save()
            self._refresh_chat_list()
            text = f"Бот запущен: {chat.bot.username} ({payload.get('name')})"
        elif kind == "bot_error":
            text = f"Ошибка бота: {payload.get('error')}"
        elif kind == "incoming":
            text = f"📩 от {payload.get('from')}: {payload.get('text')}"
        else:
            text = f"{kind}: {json.dumps(payload, ensure_ascii=False)}"
        if self.current_chat and self.current_chat.id == chat_id:
            self._append_system(text)
        else:
            chat.messages.append(Message(role="system", content=text))
            self.store.save()

    # ---- shell confirm --------------------------------------------------
    def confirm_shell(self, cmd: str) -> bool:
        return self.messagebox.askyesno("Запустить shell?", f"Агент хочет выполнить:\n\n{cmd}")

    # ---- закрытие -------------------------------------------------------
    def on_close(self) -> None:
        if self._is_generating:
            self._gen_stop.set()
        for bm in list(self.bot_managers.values()):
            try:
                bm.stop()
            except Exception:
                log.exception("stop bot failed on close")
        try:
            self.store.save()
        except Exception:
            log.exception("save failed on close")
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ---------------------------------------------------------------------------
# Диалог нового чата
# ---------------------------------------------------------------------------


class NewChatDialog:
    def __init__(self, app: App) -> None:
        self.app = app
        tk = app.tk
        ttk = app.ttk
        self.top = tk.Toplevel(app.root)
        self.top.title("Новый чат")
        self.top.geometry("520x420")
        self.top.configure(bg=PALETTE["bg_panel"])
        self.top.transient(app.root)
        self.top.grab_set()

        body = ttk.Frame(self.top, style="Panel.TFrame")
        body.pack(fill="both", expand=True, padx=20, pady=20)

        def row(label_text: str, default: str = "", show: Optional[str] = None):
            ttk.Label(body, text=label_text, style="Title.TLabel").pack(anchor="w", pady=(8, 2))
            ent = tk.Entry(body, bg=PALETTE["bg_alt"], fg=PALETTE["fg"], insertbackground=PALETTE["fg"], relief="flat")
            ent.insert(0, default)
            if show:
                ent.config(show=show)
            ent.pack(fill="x", ipady=6)
            return ent

        self.name_e = row("Название чата", "Новый AI-чат")
        self.token_e = row("Токен бота", DEFAULT_BOT_TOKEN, show="*")
        self.user_e = row("Юз бота (можно оставить пустым — определится автоматически)", DEFAULT_BOT_USERNAME)
        self.owner_e = row("Юз создателя", OWNER_USERNAME)
        self.desc_e = row("Описание / контекст для агента", "Главный AI-чат")

        bar = ttk.Frame(body, style="Panel.TFrame")
        bar.pack(fill="x", pady=(16, 0))
        ttk.Button(bar, text="Отмена", command=self.top.destroy).pack(side="right", padx=(8, 0))
        ttk.Button(bar, text="Создать", style="Primary.TButton", command=self._create).pack(side="right")

    def _create(self) -> None:
        token = self.token_e.get().strip()
        if not token:
            self.app.messagebox.showerror("Чат", "Не задан токен бота.")
            return
        chat = Chat(
            id=uuid.uuid4().hex,
            name=self.name_e.get().strip() or "Новый чат",
            bot=BotConfig(
                token=token,
                username=self.user_e.get().strip(),
                owner=self.owner_e.get().strip() or OWNER_USERNAME,
                description=self.desc_e.get().strip(),
            ),
        )
        self.app.add_chat(chat)
        self.top.destroy()


# ---------------------------------------------------------------------------
# Самопроверки (offline)
# ---------------------------------------------------------------------------


def self_test() -> int:
    print("self-test: parse_response …", end=" ")
    parts = parse_response("hello\n```python\nprint(1)\n```\n```agent\n{\"type\":\"buttons\",\"items\":[]}\n```")
    types = [p["type"] for p in parts]
    assert types[0] == "text"
    assert "code" in types and "agent" in types
    code_part = next(p for p in parts if p["type"] == "code")
    agent_part = next(p for p in parts if p["type"] == "agent")
    assert code_part["lang"] == "python"
    assert agent_part["data"]["type"] == "buttons"
    print("ok")

    print("self-test: estimate_tokens …", end=" ")
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world") >= 1
    print("ok")

    print("self-test: BotConfig roundtrip …", end=" ")
    bc = BotConfig(token="t", username="@u", owner="@o", description="d")
    assert BotConfig.from_dict(bc.to_dict()) == bc
    print("ok")

    print("self-test: Chat roundtrip …", end=" ")
    c = Chat(id="1", name="x", bot=BotConfig(token="t"))
    c.messages.append(Message(role="user", content="hi"))
    c2 = Chat.from_dict(c.to_dict())
    assert c2.name == "x" and len(c2.messages) == 1
    print("ok")

    print("self-test: ChatStore tmp roundtrip …", end=" ")
    with tempfile.TemporaryDirectory() as tmp:
        store = ChatStore(Path(tmp) / "c.json")
        store.add(c)
        store2 = ChatStore(Path(tmp) / "c.json")
        assert "1" in store2.chats
    print("ok")

    print("self-test: syntax rules compile …", end=" ")
    for lang, rules in SYNTAX_RULES.items():
        for _, rgx in rules:
            assert rgx.pattern
    print("ok")

    print("ALL OK")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=APP_NAME)
    p.add_argument("--headless", action="store_true", help="импорт-смоук без запуска GUI")
    p.add_argument("--self-test", action="store_true", help="внутренние оффлайн-проверки")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    example = ensure_secrets_template()
    if not CHAT_MODELS:
        log.warning(
            "Нет ни одного OpenRouter-ключа. Скопируйте %s в %s и заполните, "
            "либо экспортируйте OPENROUTER_KEY_<имя>.",
            example, DATA_DIR / "secrets.json",
        )
    if args.self_test:
        return self_test()
    if args.headless:
        print(f"{APP_NAME} v{APP_VERSION} ok (headless smoke)")
        print(f"Chat models configured: {[m['name'] for m in CHAT_MODELS]}")
        print(f"Secrets file: {DATA_DIR / 'secrets.json'} (template: {example})")
        return 0
    try:
        app = App()
    except Exception as exc:  # noqa: BLE001
        log.exception("Не удалось запустить GUI")
        print("Не удалось запустить GUI:", exc, file=sys.stderr)
        return 2
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
