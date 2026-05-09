# TsukCat AI · мобильный однофайловый AI-чат

Один файл (`app.py`) — самодостаточное приложение на Python 3:
веб-сервер на стандартной библиотеке + встроенный мобильный фронтенд
(HTML/CSS/JS) + многомодельный оркестратор поверх OpenRouter.

* Интерфейс — **полностью мобильный**, в стиле Claude/Telegram/GPT:
  тёмная тема, drawer-список чатов с свайпом, bottom-sheet (настройки,
  файлы, модели), длинный тап → контекстное меню, ripple-анимации,
  плавный SSE-стрим ответа.
* **Оркестратор из 13 моделей** OpenRouter: planner → 3 параллельных
  reasoner-а → coder → synthesizer (стрим) → verifier. Видно стадии,
  «Размышления», «Мнения моделей» и финальную сверку.
* **Кастомные блоки** (` ```agent ` JSON-fences) — `plan`, `poll`,
  `buttons`, `file`, `run`, `tree`, `edit`, `test`, `image`, `thinking`.
  Сообщение разворачивается в интерактивные виджеты, без перегруза.
* Подсветка синтаксиса для Python/JS/TS/HTML/JSON и пр. (своя, без CDN).
  Под каждым блоком — кнопки **Копировать / Запустить / Сохранить**.
* **Файловый менеджер** в jail-каталоге `~/.tsukcat_ai/files/`: дерево,
  `mkdir`, создать/открыть/редактировать/скачать/удалить, drag-загрузка,
  ИИ может прикладывать файлы к ответу (`{"type":"file","name":"…"}`).
* Запуск кода Python/Bash в подпроцессе с таймаутом, stdout/stderr
  показываются прямо под блоком кода.
* Импорт/экспорт чатов в JSON+ZIP, поиск по чатам, опросы под
  сообщениями (с сохранением на сервере), pin/удаление чатов,
  редактирование названий, retry/copy/quote из контекстного меню.
* Никаких внешних веб-фреймворков и CDN: только стандартная
  библиотека Python (опционально `requests` — есть авто-fallback на
  `urllib.request`).

## Установка и запуск

```bash
# (опционально) ускорить сетевые вызовы
pip install --user requests

python3 app.py                 # запустить веб-сервер на http://localhost:7860
python3 app.py --port 8000     # другой порт
python3 app.py --self-test     # оффлайн-юнит-проверки (БД, парсеры, runner)
python3 app.py --headless      # CI-смоук: поднять сервер, проверить REST, выйти
```

Открой `http://localhost:7860` на телефоне (та же сеть) или в Chrome
DevTools → Toggle device toolbar для мобильного режима.

Тестировалось на Python 3.10–3.13, Linux/macOS/Windows.

## Секреты OpenRouter

Никакие ключи **не зашиты** в код. Порядок поиска ключа на каждую
модель:

1. Переменная окружения `OPENROUTER_KEY_<MODEL_ID>` (см. `MODELS` в
   `app.py`, например `OPENROUTER_KEY_QWEN_CODER`,
   `OPENROUTER_KEY_COBUDDY` и т. д.).
2. Файл `~/.tsukcat_ai/secrets.json` секции `openrouter`:

   ```json
   {
     "openrouter": {
       "cobuddy":     "sk-or-v1-…",
       "gemma":       "sk-or-v1-…",
       "qwen3-next":  "sk-or-v1-…",
       "gpt-oss-1":   "sk-or-v1-…",
       "qwen-coder":  "sk-or-v1-…",
       "owl-alpha":   "sk-or-v1-…",
       "laguna":      "sk-or-v1-…",
       "gpt-oss-2":   "sk-or-v1-…",
       "flux2":       "sk-or-v1-…",
       "riverflow":   "sk-or-v1-…",
       "nemotron-embed":"sk-or-v1-…",
       "lyria":       "sk-or-v1-…",
       "rerank":      "sk-or-v1-…"
     }
   }
   ```

   При первом запуске рядом создаётся `secrets.example.json`.
3. Каталог данных можно переопределить через `TSUKCAT_DATA_DIR`.
4. Если ключа нет — модель просто помечается `no_key` в шапке и в
   bottom-sheet «Настройки → Модели». Остальные продолжают работать.

## Архитектура одного файла

```
app.py
├── константы, MODELS, secrets loader
├── SQLite (chats / messages / files / settings / model_health)
├── OpenRouter HTTP-клиент (stream SSE + reasoning_details)
├── Многостадийный оркестратор:
│     plan → think×N (parallel) → code → synthesize (stream) → verify
├── parse_agent_blocks() — разбор кастомных JSON-блоков в `agent`-fences
├── safe_path() — jail для файлов (защита от path-traversal)
├── code_run() — запуск python/bash в подпроцессе с таймаутом
├── REST + SSE сервер (http.server.ThreadingHTTPServer)
├── self_test() — оффлайн-юнит-проверки
├── headless_smoke() — CI-смоук
└── INDEX_HTML + FRONTEND_JS — встроенный мобильный UI
```

## Кастомные блоки в ответе ИИ

Внутри ответа модели могут быть JSON-блоки `agent`, например:

````
```agent
{"type":"plan","title":"Сделать игру","steps":[
   {"text":"Скелет HTML","status":"done"},
   {"text":"Логика на JS","status":"active"},
   {"text":"Стили","status":"pending","kind":"css"}
]}
{"type":"poll","question":"Какой движок выбрать?","options":["Phaser","PixiJS","Canvas API"]}
{"type":"buttons","buttons":[
   {"label":"Запустить","action":"run-snippet","style":"primary"},
   {"label":"Что дальше?","action":"ask","prompt":"Что дальше?"}
]}
{"type":"file","name":"index.html","lang":"html","path":"games/snake/index.html","content":"<!doctype html>…"}
```
````

Они рендерятся как интерактивные виджеты (планы с прогрессом, опросы,
кнопки-действия, прикреплённые файлы с возможностью открыть/запустить
/скачать).

## Безопасность

* Все файлы — только в `~/.tsukcat_ai/files/` (jail с `safe_path`).
* Запуск кода — в подпроцессе, изолированно от каталога файлов, с
  таймаутом (по умолчанию 10 с).
* Никаких секретов в репозитории — `secrets.json` в `.gitignore`-области
  (вне рабочего каталога), пример с пустыми значениями только для
  ориентира.
* CSP: фронтенд встроен в один документ, без внешних `<script>`/CDN —
  не зависим от чужой инфраструктуры.

## Лицензия

MIT, владелец: **@tsuklone**.
