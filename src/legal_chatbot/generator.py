from __future__ import annotations

from typing import List, Dict

import requests

from .prompts import SYSTEM_PROMPT


def _format_context(passages: List[Dict]) -> str:
    formatted = []
    for idx, item in enumerate(passages, start=1):
        formatted.append(
            f"[{idx}] {item['title']} ({item['source']}):\n{item['content']}\n"
        )
    return "\n".join(formatted)


def _source_bullets(passages: List[Dict]) -> str:
    bullets = []
    for idx, item in enumerate(passages, start=1):
        snippet = item["content"].strip().replace("\n", " ")
        snippet = snippet[:300].rsplit(" ", 1)[0] + "..." if len(snippet) > 300 else snippet
        bullets.append(f"- Source [{idx}] ({item['source']}): {snippet}")
    return "\n".join(bullets)


def _extractive_answer(question: str, passages: List[Dict]) -> str:
    intro = (
        "Grounded summary generated in extractive mode. "
        "Please verify against the cited source materials.\n\n"
    )
    conclusion = (
        f"\n\nQuestion addressed: {question}\n"
        "This answer is based only on retrieved passages and is not legal advice."
    )
    return intro + _source_bullets(passages) + conclusion


def low_confidence_answer(question: str, passages: List[Dict], top1: float, threshold: float) -> str:
    """Response used when retrieval is too weak to trust a generated answer.

    The language model is not called. The user gets the closest sourced extracts and a
    clear statement that the match is weak, so they can judge the sources themselves.
    """
    if not passages:
        return (
            "No relevant passages were found for this question, so no answer was generated. "
            "Try rephrasing it, or check that the topic is covered by the indexed documents."
        )
    header = (
        f"Low retrieval confidence: the best matching passage scored {top1:.2f}, below the "
        f"{threshold:.2f} threshold. Rather than generate an answer the sources may not support, "
        "here are the closest extracts. Please check them before relying on anything.\n\n"
    )
    footer = (
        f"\n\nQuestion asked: {question}\n"
        "The indexed documents may not cover this topic. This is not legal advice."
    )
    return header + _source_bullets(passages) + footer


def generate_answer(
    provider: str,
    question: str,
    passages: List[Dict],
    openai_api_key: str = "",
    openai_model: str = "",
    openai_base_url: str = "",
    ollama_model: str = "",
    ollama_base_url: str = "",
) -> str:
    provider = provider.lower().strip()
    if not passages:
        return "No relevant passages were retrieved. Please broaden the query or update the corpus."

    context = _format_context(passages)
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Prepare a concise answer with in-text source markers such as [1] and [2]."
    )

    if provider == "extractive":
        return _extractive_answer(question, passages)

    if provider == "ollama":
        response = requests.post(
            f"{ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": ollama_model,
                "prompt": f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        return payload.get("response", "").strip()

    if provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai package is required for LLM_PROVIDER=openai") from exc

        client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        completion = client.chat.completions.create(
            model=openai_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return completion.choices[0].message.content.strip()

    raise ValueError(f"Unsupported LLM provider: {provider}")
