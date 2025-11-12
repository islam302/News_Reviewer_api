from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from django.conf import settings
from django.db import transaction

from docx import Document
from openai import OpenAI

from .models import DocumentChunk


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_COMPLETION_MODEL = "gpt-4.1"
MAX_CHUNK_CHAR_LENGTH = 4000


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


def _preprocess_honorifics(text: str) -> str:
    """
    Preprocess text to remove or replace honorific phrases before sending to AI model.
    This ensures consistent handling of honorifics regardless of what the model does.
    Handles context-aware replacements (e.g., Crown Prince vs regular Prince).
    """
    processed_text = text
    
    # Handle Crown Prince context first (more specific patterns)
    # Pattern: "صاحب السمو الملكي الأمير [NAME] ولي العهد [POSITION]"
    # Result: "ولي العهد [POSITION] الأمير [NAME]"
    processed_text = re.sub(
        r'صاحب\s+السمو\s+الملكي\s+الأمير\s+([^،.]+?)\s+ولي\s+العهد\s+([^،.]+?)(?=[،.])',
        r'ولي العهد \2 الأمير \1',
        processed_text,
        flags=re.IGNORECASE
    )
    # Pattern: "صاحب السمو الملكي [NAME] ولي العهد [POSITION]"
    processed_text = re.sub(
        r'صاحب\s+السمو\s+الملكي\s+([^،.]+?)\s+ولي\s+العهد\s+([^،.]+?)(?=[،.])',
        r'ولي العهد \2 \1',
        processed_text,
        flags=re.IGNORECASE
    )
    # Pattern: "صاحب السمو الملكي الأمير [NAME] ولي العهد" (without position)
    processed_text = re.sub(
        r'صاحب\s+السمو\s+الملكي\s+الأمير\s+([^،.]+?)\s+ولي\s+العهد',
        r'ولي العهد الأمير \1',
        processed_text,
        flags=re.IGNORECASE
    )
    
    # Define comprehensive replacement patterns (order matters - most specific first)
    replacements = [
        # King honorifics - must preserve ال التعريف
        (r'حضرة\s+صاحب\s+الجلالة\s+الملك\s+المعظم', 'الملك'),
        (r'جلالة\s+الملك\s+المعظم', 'الملك'),
        (r'حضرة\s+صاحب\s+الجلالة\s+الملك', 'الملك'),
        (r'صاحب\s+الجلالة\s+الملك\s+المعظم', 'الملك'),
        (r'صاحب\s+الجلالة\s+الملك', 'الملك'),
        (r'جلالة\s+الملك', 'الملك'),
        (r'الملك\s+المعظم', 'الملك'),
        
        # Sultan honorifics - must preserve ال التعريف
        (r'حضرة\s+صاحب\s+الجلالة\s+السلطان\s+المعظم', 'السلطان'),
        (r'حضرة\s+صاحب\s+الجلالة\s+السلطان', 'السلطان'),
        (r'صاحب\s+الجلالة\s+السلطان\s+المعظم', 'السلطان'),
        (r'صاحب\s+الجلالة\s+السلطان', 'السلطان'),
        (r'جلالة\s+السلطان', 'السلطان'),
        (r'السلطان\s+المعظم', 'السلطان'),
        
        # Crown Prince/Prince honorifics - remove redundant combinations
        (r'صاحب\s+السمو\s+الملكي\s+الأمير', 'الأمير'),  # Remove redundant honorific
        (r'صاحب\s+السمو\s+الملكي', 'الأمير'),  # Default to الأمير if context unclear
        (r'صاحب\s+السمو\s+الأمير', 'الأمير'),
        
        # Remove redundant "الكريم" from "سموه الكريم"
        (r'سموه\s+الكريم', 'سموه'),
        
        # Prayer phrases - delete completely (with word boundaries)
        (r'\s+حفظه\s+الله\s+', ' '),
        (r'\s+رعاه\s+الله\s+', ' '),
        (r'\s+نصره\s+الله\s+', ' '),
        (r'\s+أيده\s+الله\s+', ' '),
        (r'\s+أطال\s+الله\s+عمره\s+', ' '),
        (r'\s+حفظهما\s+الله\s+', ' '),
        (r'\s+حفظهم\s+الله\s+', ' '),
        (r'\s+حفظه\s+الله\s*$', ' '),  # At end of sentence
        (r'^\s*حفظه\s+الله\s+', ' '),  # At start of sentence
        
        # Remove redundant words
        (r'\bخالص\s+', ''),  # Remove "خالص" before "تهانيه"
    ]
    
    # Apply all replacements
    for pattern, replacement in replacements:
        processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
    
    # Clean up multiple spaces and trim
    processed_text = re.sub(r'\s+', ' ', processed_text)
    processed_text = processed_text.strip()
    
    return processed_text


def build_review_prompt(news_text: str, guidelines: Iterable[RetrievedChunk], examples: Iterable[RetrievedChunk]) -> List[dict]:
    # Preprocess text to handle honorifics before sending to AI
    processed_news_text = _preprocess_honorifics(news_text)
    
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
        f"{processed_news_text}\n\n"
        "FIRST: Validate that the text above is a legitimate news article. "
        "If it is random, inappropriate, meaningless, or not a news article, "
        "respond ONLY with: 'ERROR: النص المقدم غير مناسب أو غير صالح للمعالجة. يرجى تقديم خبر صحيح.'\n\n"
        "CRITICAL REPLACEMENT RULES - APPLY EXACTLY AS SPECIFIED:\n"
        "MANDATORY TITLE REPLACEMENTS:\n"
        "1. 'جلالة الملك المعظم' → MUST become 'الملك' (with ال التعريف)\n"
        "2. 'حضرة صاحب الجلالة الملك' → MUST become 'الملك' (with ال التعريف)\n"
        "3. 'حضرة صاحب الجلالة السلطان' → MUST become 'السلطان' (with ال التعريف)\n"
        "4. 'صاحب الجلالة السلطان' → MUST become 'السلطان' (with ال التعريف)\n"
        "5. 'جلالة السلطان' → MUST become 'السلطان' (with ال التعريف)\n"
        "6. 'جلالة الملك' → MUST become 'الملك' (with ال التعريف)\n"
        "7. 'السلطان المعظم' → MUST become 'السلطان' (with ال التعريف)\n"
        "8. 'الملك المعظم' → MUST become 'الملك' (with ال التعريف)\n\n"
        "MANDATORY PRINCE/CROWN PRINCE REPLACEMENTS:\n"
        "9. 'صاحب السمو الملكي الأمير' → MUST become 'الأمير' OR 'ولي العهد' (depending on context, but NEVER both together)\n"
        "10. 'صاحب السمو الملكي' (when referring to Crown Prince) → MUST become 'ولي العهد' OR 'الأمير'\n"
        "11. 'صاحب السمو الملكي' (when referring to Prince) → MUST become 'الأمير'\n"
        "12. 'صاحب السمو الأمير' → MUST become 'الأمير'\n"
        "13. 'سموه الكريم' → MUST become 'سموه' (remove 'الكريم')\n"
        "14. 'سموه' (when followed by name) → Can remain 'سموه' OR be replaced with appropriate title\n\n"
        "MANDATORY PHRASE DELETIONS:\n"
        "15. 'حفظه الله' / 'رعاه الله' / 'نصره الله' / 'أيده الله' / 'أطال الله عمره' / 'حفظهما الله' → MUST be DELETED completely\n"
        "16. 'خالص' (in 'خالص تهانيه') → MUST be DELETED (becomes 'تهانيه' only)\n"
        "17. 'المعظم' (when attached to titles) → MUST be DELETED\n\n"
        "CRITICAL: You MUST scan for ALL honorific and exaggerated phrases. "
        "If you see 'صاحب السمو الملكي الأمير', you MUST simplify it to 'الأمير' or 'ولي العهد' (NOT both). "
        "Never leave redundant honorific phrases. Be extremely thorough and check every detail.\n\n"
        "IMPORTANT: When replacing titles, you MUST preserve the definite article 'ال' before the title. "
        "For example, 'السلطان' (with ال) is correct, but 'سلطان' (without ال) is WRONG. "
        "Similarly, 'الملك' (with ال) is correct, but 'ملك' (without ال) is WRONG.\n\n"
        "STEP-BY-STEP PROCESS:\n"
        "1. Scan the entire article for ALL prohibited phrases listed in the guidelines.\n"
        "2. Replace each prohibited phrase with its EXACT specified replacement, preserving ال التعريف when required.\n"
        "3. Double-check that titles like 'السلطان' and 'الملك' always include ال التعريف.\n"
        "4. Remove all prayer phrases (حفظه الله, رعاه الله, etc.) completely.\n"
        "5. Rewrite the article according to UNA editorial style.\n"
        "6. Return ONLY the final revised news article in Arabic with no additional analysis or commentary."
        " Ensure the revised article is organized into clear paragraphs separated by a blank line."
    )

    return [
        {
            "role": "system",
            "content": (
                "You are an experienced Arabic-language news editor working for the Union of News Agencies "
                "of the Organization of Islamic Cooperation (UNA). Your role is to review and edit news articles "
                "according to UNA's strict editorial style guidelines.\n\n"
                "CONTENT VALIDATION - REJECT INAPPROPRIATE OR RANDOM TEXT:\n"
                "Before processing any text, you MUST validate that it is a legitimate news article:\n"
                "1. REJECT immediately if the text is:\n"
                "   - Random, meaningless, or nonsensical text\n"
                "   - Inappropriate, offensive, or harmful content\n"
                "   - Not a news article (e.g., spam, advertisements, personal messages)\n"
                "   - Contains only symbols, numbers without context, or gibberish\n"
                "   - Clearly not related to news or journalism\n"
                "2. If you reject the text, respond ONLY with: 'ERROR: النص المقدم غير مناسب أو غير صالح للمعالجة. يرجى تقديم خبر صحيح.'\n"
                "3. Only proceed with editing if the text is a legitimate, coherent news article.\n\n"
                "CRITICAL REPLACEMENT ACCURACY REQUIREMENTS:\n"
                "1. You MUST apply ALL replacement rules with EXACT precision as specified in the guidelines.\n"
                "2. When replacing titles, you MUST preserve the Arabic definite article 'ال' (al-) before the title.\n"
                "   - CORRECT: 'السلطان هيثم بن طارق' (السلطان with ال)\n"
                "   - WRONG: 'سلطان هيثم بن طارق' (سلطان without ال)\n"
                "   - CORRECT: 'الملك حمد بن عيسى' (الملك with ال)\n"
                "   - WRONG: 'ملك حمد بن عيسى' (ملك without ال)\n"
                "3. Common replacements:\n"
                "   - 'حضرة صاحب الجلالة السلطان' → 'السلطان' (NOT 'سلطان')\n"
                "   - 'صاحب الجلالة السلطان' → 'السلطان' (NOT 'سلطان')\n"
                "   - 'جلالة الملك المعظم' → 'الملك' (NOT 'ملك')\n"
                "   - 'حضرة صاحب الجلالة الملك' → 'الملك' (NOT 'ملك')\n"
                "   - 'صاحب السمو الملكي الأمير' → 'الأمير' OR 'ولي العهد' (NEVER both together - this is redundant)\n"
                "   - 'صاحب السمو الملكي' (Crown Prince) → 'ولي العهد' OR 'الأمير'\n"
                "   - 'صاحب السمو الملكي' (Prince) → 'الأمير'\n"
                "   - 'سموه الكريم' → 'سموه' (remove 'الكريم')\n"
                "4. Prayer phrases like 'حفظه الله', 'رعاه الله', 'نصره الله' MUST be completely DELETED.\n"
                "5. Redundant honorific words like 'خالص', 'المعظم' MUST be DELETED.\n"
                "6. These rules are MANDATORY - zero tolerance for errors. Check EVERY detail.\n\n"
                "GENERAL RULE FOR TITLE REPLACEMENT:\n"
                "When discovering any honorific phrase or prayer associated with a title, it must be deleted "
                "or replaced with the official permitted title, without affecting the meaning of the sentence.\n\n"
                "Examples:\n"
                "Example 1 - King title:\n"
                "Original: 'بعث جلالة الملك المعظم عبد الله الثاني ابن الحسين برقية تهنئة إلى الرئيس الفرنسي، "
                "أعرب فيها عن خالص تهانيه وتمنياته له بدوام الصحة والعافية.'\n"
                "After processing: 'بعث الملك عبد الله الثاني ابن الحسين برقية تهنئة إلى الرئيس الفرنسي، "
                "أعرب فيها عن تهانيه وتمنياته له بدوام الصحة والعافية.'\n\n"
                "Example 2 - Crown Prince title (CRITICAL):\n"
                "Original: 'أكد صاحب السمو الملكي الأمير سلمان بن حمد آل خليفة ولي العهد رئيس مجلس الوزراء...'\n"
                "After processing: 'أكد ولي العهد رئيس مجلس الوزراء الأمير سلمان بن حمد آل خليفة...' "
                "OR 'أكد الأمير سلمان بن حمد آل خليفة ولي العهد رئيس مجلس الوزراء...'\n"
                "WRONG: 'أكد صاحب السمو الملكي الأمير...' (redundant honorifics not removed)\n"
                "WRONG: 'أكد صاحب السمو الملكي...' (still contains honorific)\n\n"
                "EDITORIAL STYLE GUIDELINES:\n"
                "1. Use only modern formal Arabic language.\n"
                "2. Avoid emotional and exaggerated expressions.\n"
                "3. Maintain objectivity and balance in all headlines and texts.\n"
                "4. Headlines must be concise and neutral (without exclamation marks or promotional words).\n"
                "5. Do not add personal analyses or conclusions unless from an official explicit source.\n\n"
                "GENERAL EDITING INSTRUCTIONS:\n"
                "1. Rewrite the news in a professional, clear, and neutral journalistic style.\n"
                "2. Remove any bias, personal opinion, or emotional phrases.\n"
                "3. Preserve the original information accurately without modifying facts.\n"
                "4. Correct linguistic, grammatical, and spelling errors.\n"
                "5. Adjust punctuation accurately according to linguistic rules to make the news appear professional.\n\n"
                "DETAILED PROCESS - CHECK EVERY SINGLE DETAIL:\n"
                "1. Read the guidelines section carefully to identify ALL prohibited phrases.\n"
                "2. Scan the article systematically for EVERY occurrence of prohibited phrases, including:\n"
                "   - All honorific titles (جلالة, حضرة صاحب الجلالة, صاحب السمو الملكي, etc.)\n"
                "   - Redundant combinations like 'صاحب السمو الملكي الأمير' (must simplify)\n"
                "   - Prayer phrases (حفظه الله, رعاه الله, etc.)\n"
                "   - Exaggerated words (خالص, المعظم, الكريم when redundant)\n"
                "3. Replace each occurrence with its EXACT specified replacement:\n"
                "   - Ensure ال التعريف is preserved for titles (السلطان, الملك, الأمير)\n"
                "   - Simplify redundant honorifics (e.g., 'صاحب السمو الملكي الأمير' → 'الأمير')\n"
                "   - Remove prayer phrases completely\n"
                "   - Remove redundant words like 'خالص', 'المعظم'\n"
                "4. Double-check EVERY title:\n"
                "   - All titles must have ال التعريف (السلطان, الملك, الأمير)\n"
                "   - No redundant honorific combinations\n"
                "   - No prayer phrases remaining\n"
                "5. Apply all editorial style guidelines (remove emotional expressions, ensure objectivity, etc.).\n"
                "6. Final verification: Read through the entire article one more time to catch ANY missed honorifics or redundant phrases.\n"
                "7. Rewrite the article according to UNA editorial style.\n"
                "8. Deliver ONLY the final revised article text in Arabic; no analysis or explanations.\n"
                "9. Present the revised article as clear paragraphs separated by a blank line."
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
    
    result_text = ""
    if hasattr(client, "responses"):
        response = client.responses.create(model=model, input=messages)
        result_text = response.output_text.strip()
    else:
        chat_client = getattr(client, "chat", None)
        if chat_client and hasattr(chat_client, "completions"):
            response = chat_client.completions.create(model=model, messages=messages)
            if response.choices:
                result_text = response.choices[0].message.content.strip()
        else:
            raise RuntimeError("OpenAI client does not support responses or chat completions API.")
    
    # Check if the model rejected the text as inappropriate or random
    if result_text.startswith("ERROR:") or "غير مناسب" in result_text or "غير صالح" in result_text:
        error_msg = result_text.replace("ERROR:", "").strip()
        if not error_msg:
            error_msg = "النص المقدم غير مناسب أو غير صالح للمعالجة. يرجى تقديم خبر صحيح."
        raise ValueError(error_msg)
    
    # Final pass: ensure any remaining honorifics are processed
    # This catches any honorifics the model might have missed
    final_text = _preprocess_honorifics(result_text)
    
    return final_text

