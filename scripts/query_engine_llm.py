from sentence_transformers import SentenceTransformer
import faiss
import pickle
from pathlib import Path
from ollama import Client
import tiktoken

INDEX_PATH = Path("vector_store/faiss.index")
META_PATH = Path("vector_store/metadata.pkl")
MAX_TOKENS = 1500

ollama = Client(host="http://localhost:11434")
embedder = SentenceTransformer('intfloat/multilingual-e5-base')
index = faiss.read_index(str(INDEX_PATH))
enc = tiktoken.get_encoding("cl100k_base")

with META_PATH.open("rb") as f:
    metadata = pickle.load(f)


def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    scored = []
    system_prompt = (
        "Ты — автоматическая система оценки релевантности. "
        "Отвечай СТРОГО только числом от 0 до 1 (дробное число с точкой), ничего кроме числа. "
        "Никаких объяснений, комментариев или других символов. "
        "Если фрагмент не содержит полезной информации — 0. Если очень релевантен — 1."
    )

    for candidate in candidates:
        prompt = (
            f"Вопрос: {query}\n\n"
            f"Текст: {candidate['text']}\n\n"
            f"Оценка релевантности (0-1):"
        )
        try:
            resp = ollama.chat(
                model="mistral",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            score_str = resp['message']['content'].strip()
            score = float(score_str)
        except Exception as e:
            print(f"[!] Ошибка при rerank: {e}")
            score = 0.0
        
        candidate_with_score = candidate.copy()
        candidate_with_score["score"] = score
        scored.append(candidate_with_score)
    
    scored_sorted = sorted(scored, key=lambda x: x["score"], reverse=True)
    return scored_sorted[:top_k]


def get_confidence(filtered: list[dict]) -> str:
    min_dist = min(r["distance"] for r in filtered)
    if min_dist < 0.35:
        return "🟢 Уверенность: высокая"
    elif min_dist < 0.55:
        return "🟡 Уверенность: средняя"
    else:
        return "🔴 Уверенность: низкая"

def build_context(filtered: list[dict]) -> tuple[str, list[dict]]:
    context_chunks = []
    token_total = 0
    used = []

    for r in filtered:
        tokens = count_tokens(r["text"])
        if token_total + tokens > MAX_TOKENS:
            break
        context_chunks.append(r["text"])
        token_total += tokens
        used.append({
            "page": r["page"],
            "id":   r["id"],
            "source": r["source"]
        })

    return "\n---\n".join(context_chunks), used

def search(query: str, top_k: int = 5) -> list[dict]:
    query_vec = embedder.encode(["query: " + query]).astype("float32")
    distances, ids = index.search(query_vec, top_k)

    results = []
    for idx, dist in zip(ids[0], distances[0]):
        meta = metadata[idx]
        results.append({
            "id":       idx,
            "distance": float(dist),
            "text":     meta.get("text", ""),
            "source":   meta.get("source", ""),
            "page":     meta.get("page", "?")
        })
    return results


def generate_answer(
    question: str,
    search_k: int = 10,
    rerank_k: int = 5
):
    try:
        results = search(question, top_k=search_k)
        reranked = rerank(question, results, top_k=rerank_k)

        if not reranked:
            print("🔴 Подходящих фрагментов не найдено. Ответ не сформирован.")
            return

        confidence = get_confidence(reranked)
        context, used_chunks = build_context(reranked)
        system = (
            """
            Вы — приёмная комиссия ИТМО. Ваши знания ограничены только двумя магистерскими программами:
            1) «Искусственный интеллект» (AI)
            2) «AI Product»

            ПОРЯДОК ДИАЛОГА:
            1. Если в запросе не указана программа — уточните: «Какую программу вы рассматриваете: AI или AI Product?»
            2. Как только программа выбрана, спросите: «Расскажите вкратце о вашем образовании или опыте.»
            3. После получения бэкграунда — дайте рекомендацию по 3–5 выборным дисциплинам из этой программы на основании схожести описаний курсов и рассказанного фона.
            4. Затем отвечайте на любые остальные вопросы только по содержимому этих учебных планов.
            Если вопрос выходит за рамки этих двух программ, вежливо отвечайте:  
            «Извините, я могу помочь только с вопросами по учебным планам AI и AI Product.»
            """
        )
        prompt = f"Контекст:\n{context}\n\nВопрос: {question}\n\nОтвет:"

        resp = ollama.chat(
            model="mistral",
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ]
        )
        answer = resp['message']['content'].strip()

        sources = [
            f"{chunk['source']} — стр. {chunk['page']}"
            for chunk in used_chunks
        ]

        return answer, confidence, sources

    except Exception:
        return (
            "[✗] Ошибка при обращении к LLM. Попробуйте позже.",
            "🔴 Уверенность: неизвестна",
            []
        )


if __name__ == "__main__":
    q = input("Введите вопрос: ")
    ans, conf, srcs = generate_answer(q)
    print(f"\n🔍 Ответ ({conf}):\n{ans}\n")
    if srcs:
        print("📚 Использованные источники:")
        for s in srcs:
            print(f"- {s}")


