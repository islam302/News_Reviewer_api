from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from django.conf import settings
from django.db import transaction

from docx import Document
from openai import OpenAI

from .models import DocumentChunk


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_COMPLETION_MODEL = "gpt-4.1-mini"
MAX_CHUNK_CHAR_LENGTH = 1800


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    similarity: float


def _get_openai_client() -> OpenAI:
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var must be set before calling OpenAI APIs.")
    return OpenAI(api_key=api_key)


def _extract_text_segments(file_obj) -> List[str]:
    """
    Read a DOCX file-like object and return non-empty text segments.
    """
    document = Document(file_obj)
    segments = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if text:
            segments.append(text)
    return segments


def _batch_segments(segments: Sequence[str], max_chars: int = MAX_CHUNK_CHAR_LENGTH) -> List[str]:
    """
    Merge sequential segments into chunks that respect the max_chars limit.
    """
    batches: List[str] = []
    current: List[str] = []
    current_len = 0

    for segment in segments:
        if not segment:
            continue
        candidate_len = current_len + len(segment) + (1 if current else 0)
        if candidate_len > max_chars and current:
            batches.append("\n".join(current))
            current = [segment]
            current_len = len(segment)
        else:
            current.append(segment)
            current_len = candidate_len

    if current:
        batches.append("\n".join(current))

    return batches


def _embed_texts(texts: Sequence[str], *, model: str = DEFAULT_EMBEDDING_MODEL) -> List[List[float]]:
    client = _get_openai_client()
    response = client.embeddings.create(model=model, input=list(texts))
    return [item.embedding for item in response.data]


def ingest_docx(
    *,
    file_obj,
    document_type: DocumentChunk.DocumentType,
    title: str | None = None,
    replace_existing: bool = True,
) -> List[DocumentChunk]:
    """
    Parse, chunk, embed, and persist DOCX content for the provided document type.
    """
    source_name = getattr(file_obj, "name", "") or ""
    file_obj.seek(0)
    segments = _extract_text_segments(file_obj)
    if not segments:
        raise ValueError("لم يتم العثور على نص داخل الملف المرفوع.")

    batches = _batch_segments(segments)
    embeddings = _embed_texts(batches)

    if len(batches) != len(embeddings):
        raise RuntimeError("Embedding count mismatch while processing DOCX.")

    with transaction.atomic():
        if replace_existing:
            DocumentChunk.objects.filter(document_type=document_type).delete()

        created_chunks: List[DocumentChunk] = []
        for idx, (text, embedding) in enumerate(zip(batches, embeddings)):
            chunk = DocumentChunk.objects.create(
                document_type=document_type,
                title=title or source_name or document_type,
                source_name=source_name,
                order=idx,
                text=text,
                embedding=embedding,
                metadata={"segment_count": len(text.splitlines())},
            )
            created_chunks.append(chunk)

    return created_chunks


def _cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    dot_product = 0.0
    mag_a = 0.0
    mag_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot_product += a * b
        mag_a += a * a
        mag_b += b * b
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot_product / ((mag_a ** 0.5) * (mag_b ** 0.5))


def retrieve_similar_chunks(
    *,
    query_text: str,
    document_type: DocumentChunk.DocumentType,
    limit: int | None = 3,
) -> List[RetrievedChunk]:
    chunks = list(DocumentChunk.objects.filter(document_type=document_type))
    if not chunks:
        return []

    query_embedding = _embed_texts([query_text])[0]
    scored = [
        RetrievedChunk(chunk=chunk, similarity=_cosine_similarity(query_embedding, chunk.embedding))
        for chunk in chunks
    ]
    scored.sort(key=lambda item: item.similarity, reverse=True)
    if limit is None or limit >= len(scored):
        return scored
    return scored[:limit]


def build_review_prompt(news_text: str, guidelines: Iterable[RetrievedChunk], examples: Iterable[RetrievedChunk]) -> List[dict]:
    guideline_section_lines = []
    for idx, item in enumerate(guidelines, start=1):
        guideline_section_lines.append(f"{idx}. {item.chunk.text}")
    guideline_section = "\n".join(guideline_section_lines) if guideline_section_lines else "No guidelines available."

    example_section_lines = []
    for idx, item in enumerate(examples, start=1):
        example_section_lines.append(f"### Example {idx}\n{item.chunk.text}")
    example_section = "\n\n".join(example_section_lines) if example_section_lines else "No examples available."

    user_prompt = (
        "### Editorial Guidelines\n"
        f"{guideline_section}\n\n"
        "### Reference News Examples\n"
        f"{example_section}\n\n"
        "### Article Requiring Review\n"
        f"{news_text}\n\n"
        "Rewrite the news article while honoring the spirit of the guidelines and examples without copying them verbatim. "
        "Treat the examples as illustrative references for tone and quality only; do not match their structure exactly. "
        "Return only the final revised news article in Arabic with no additional analysis, bullet points, or meta commentary."
    )

    return [
        {
            "role": "system",
            "content": (
                "You are an experienced Arabic-language news editor acting as a quality reviewer. "
                "Use the guidelines as flexible direction and draw inspiration from the examples without copying their structure. "
                "Deliver only the revised article text in Arabic; do not add analysis, explanations, or any sections beyond the improved story."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def generate_review(
    *,
    news_text: str,
    guideline_chunks: Iterable[RetrievedChunk],
    example_chunks: Iterable[RetrievedChunk],
    model: str = DEFAULT_COMPLETION_MODEL,
) -> str:
    client = _get_openai_client()
    messages = build_review_prompt(news_text, guideline_chunks, example_chunks)
    if hasattr(client, "responses"):
        response = client.responses.create(model=model, input=messages)
        return response.output_text.strip()

    chat_client = getattr(client, "chat", None)
    if chat_client and hasattr(chat_client, "completions"):
        response = chat_client.completions.create(model=model, messages=messages)
        if response.choices:
            return response.choices[0].message.content.strip()

    raise RuntimeError("OpenAI client does not support responses or chat completions API.")

