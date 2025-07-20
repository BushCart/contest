## Как решал задачу

- Взял заготовленный RAG-пайплайн из проекта MFDP и адаптировал его под учебные планы ИТМО:  
  `parse_docs.py → embed_chunks.py → build_faiss_index.py → query_engine_llm.py → Gradio UI`.  
- Скачал PDF-планы программ AI и AI Product через прямые ссылки (использовал `requests` + `BeautifulSoup`), пропустил их через `parse_docs.py` для извлечения текста.  
- Провёл векторизацию через `intfloat/multilingual-e5-base`, построил FAISS `IndexFlatL2`, затем применил LLM-based rerank и генерацию.

## Чем пользовался

- **Парсинг:** `requests`, `beautifulsoup4`, `pdfminer.six`, `python-docx`, `python-pptx`  
- **Эмбеддинги:** `sentence-transformers` (`intfloat/multilingual-e5-base`)  
- **ANN-поиск:** FAISS (`IndexFlatL2`)  
- **Rerank & генерация:** локальная LLM — модель **Mistral** через **Ollama**  
- **UI:** Gradio  
- **Инструменты DevOps:** Docker, GitHub Actions (линт, smoke-тесты)

## Как принимал решения

- **Фокус на MVP:** оставил готовый Gradio-чат, чтобы не тратить время на разработку Telegram-бота.  
- **ChatGPT в помощь:** попросил подсказки по парсеру HTML-ссылок и формулировке системного промпта.  
- **Минимум изменений:** не трогал логику генерации и интерфейс, быстро интегрировал только парсинг и системный промпт.
