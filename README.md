# AI Chat + Telegram Bot Agent

Однофайловое (`app.py`) приложение на Python 3:

* GUI на Tkinter — чаты-вкладки, подсветка синтаксиса нескольких языков,
  code-блоки с кнопками «Запустить / Скопировать / Сохранить», опросы,
  чеклисты, стриминг печати, кнопка «Стоп», блокировка ввода во время
  генерации, счётчик токенов, импорт/экспорт чатов в JSON.
* ИИ-агент поверх OpenRouter с фолбэком по нескольким моделям
  (MiniMax M2, Tencent Hunyuan A13B, Qwen3 30B A3B). Дополнительно
  доступны Whisper (audio), Gemini Image / FLUX / Seedream (image),
  Veo (video), Kokoro (TTS), Cohere Rerank, GTE-Base (embed).
* Telegram-бот на `python-telegram-bot==20.7` запускается в фоновом
  потоке и **полностью управляется агентом**: метаданные бота, реальные
  команды/инлайн-кнопки/опросы, и редактирование самого `app.py` без
  рестарта. Полный набор инструментов:
  * метаданные — `rename_bot`, `set_bot_description`,
    `set_bot_short_desc`, `set_bot_commands` (меню «/»),
    `set_chat_menu_button`;
  * сообщения — `send_telegram`, `send_telegram_buttons`
    (`callback_data` / `url`), `send_poll`;
  * **динамическая логика бота** —
    `register_handler(name, kind=command|message|callback, trigger, code, owner_only)`,
    `unregister_handler`, `list_handlers`, `bot_status`,
    `eval_in_bot` (исполнить корутину в loop'е бота), `restart_bot`;
  * файлы и код — `read_file`/`write_file`/`list_dir` с
    `scope="workspace"|"project"` (агент может править сам `app.py`),
    `run_python` (subprocess+10c таймаут в sandbox), `shell` (с
    подтверждением через GUI).
* Кастомный формат ответа агента: блоки `` ```agent ... ``` `` с JSON
  превращаются в кнопки, опросы, чеклисты и tool-вызовы прямо в чате.

## Установка и запуск

```bash
pip install requests "python-telegram-bot==20.7"
python3 app.py             # GUI
python3 app.py --self-test # оффлайн-проверки
python3 app.py --headless  # импорт-смоук без X-сервера (CI)
```

Тестировалось на Python 3.10–3.13.

## Секреты — куда класть

Никакие ключи и токены **не зашиты** в исходный код (этого не позволяет
GitHub push protection и здравый смысл). Источники секретов в порядке
приоритета:

1. Переменные окружения:
   * `TELEGRAM_BOT_TOKEN`, `TELEGRAM_BOT_USERNAME`,
     `AI_CHAT_OWNER_USERNAME`
   * `OPENROUTER_KEY_MINIMAX`, `OPENROUTER_KEY_HY3`,
     `OPENROUTER_KEY_QWEN`, `OPENROUTER_KEY_WHISPER_LARGE_V3`,
     `OPENROUTER_KEY_WHISPER_LARGE_V3_TURBO`, `OPENROUTER_KEY_WHISPER_1`,
     `OPENROUTER_KEY_GEMINI_IMAGE`, `OPENROUTER_KEY_VEO`,
     `OPENROUTER_KEY_RERANK`, `OPENROUTER_KEY_EMBED`,
     `OPENROUTER_KEY_FLUX`, `OPENROUTER_KEY_SEEDREAM`,
     `OPENROUTER_KEY_KOKORO`.
2. Файл `~/.ai_chat_agent/secrets.json` (путь меняется через
   `AI_CHAT_DATA_DIR`). При первом запуске рядом создаётся
   `secrets.json.example` с готовой схемой:

   ```json
   {
     "owner_username": "@tsuklone",
     "telegram_bot_token": "PUT-YOUR-TOKEN-HERE",
     "telegram_bot_username": "@YourBot",
     "openrouter": {
       "minimax": "sk-or-v1-...",
       "qwen": "sk-or-v1-...",
       "hy3": "sk-or-v1-..."
     }
   }
   ```

3. Ничего не задано — модель/бот будут отключены, GUI всё равно
   запустится, в чате будет видно, какие ключи отсутствуют.

## Как пользоваться

1. `python3 app.py` — открыть GUI.
2. **+ Новый чат** — заполнить токен бота, юз бота, юз создателя.
3. Нажать **Запустить бота** в шапке чата — PTB поднимется в фоне.
4. Писать агенту в чат. Агент может в ответ:
   * вызвать инструмент (`rename_bot`, `send_telegram`, …) —
     результат прилетает system-сообщением;
   * предложить кнопки/опросы/чеклисты под ответом;
   * дать code-блок Python, который вы запустите кнопкой «Запустить».
5. Кнопка **Стоп** прерывает текущую генерацию. Пока агент пишет,
   поле ввода заблокировано.
6. **Файл → Импорт/Экспорт** — JSON-дамп всех чатов.

## Структура одного файла

```
app.py
 ├── константы и загрузка секретов
 ├── OpenRouterClient (стриминг + фолбэк)
 ├── AIAgent (системный промпт + сборка messages)
 ├── TelegramBotManager (PTB v20 в отдельном loop)
 ├── ToolRunner (rename_bot/send_telegram/run_python/…)
 ├── Tk-GUI: App, NewChatDialog, рендер сообщений и code-блоков
 └── self-test и headless smoke
```

## Безопасность

* Никаких секретов в исходнике/PR.
* `run_python` — в подпроцессе с таймаутом 10 c в изолированной
  директории `~/.ai_chat_agent/workspace/`.
* `shell` всегда требует подтверждения через GUI-диалог.
