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
    Preprocess text to remove excessive honorifics while keeping official titles.

    KEEP (Official Titles):
    - خادم الحرمين الشريفين (official title for Saudi King)
    - صاحب السمو الملكي (official title for Royal Highness)

    REMOVE (Exaggerated Phrases):
    - جلالة الملك المعظم أيده الله → الملك
    - فخامة الرئيس حفظه الله → الرئيس
    - Prayer phrases: حفظه الله، أيده الله، رعاه الله
    - Exaggerated adjectives: المعظم، الجليل
    """
    processed_text = text

    # Define comprehensive replacement patterns (order matters - most specific first)
    replacements = [
        # King honorifics - REMOVE exaggerated parts, keep simple title
        # "جلالة الملك المعظم" → "الملك" (remove exaggeration)
        (r'جلالة\s+الملك\s+المعظم', 'الملك'),
        (r'جلالة\s+الملك', 'الملك'),
        (r'حضرة\s+صاحب\s+الجلالة\s+الملك\s+المعظم', 'الملك'),
        (r'حضرة\s+صاحب\s+الجلالة\s+الملك', 'الملك'),
        (r'صاحب\s+الجلالة\s+الملك\s+المعظم', 'الملك'),
        (r'صاحب\s+الجلالة\s+الملك', 'الملك'),
        (r'الملك\s+المعظم', 'الملك'),

        # Sultan honorifics - REMOVE exaggerated parts
        (r'حضرة\s+صاحب\s+الجلالة\s+السلطان\s+المعظم', 'السلطان'),
        (r'حضرة\s+صاحب\s+الجلالة\s+السلطان', 'السلطان'),
        (r'صاحب\s+الجلالة\s+السلطان\s+المعظم', 'السلطان'),
        (r'صاحب\s+الجلالة\s+السلطان', 'السلطان'),
        (r'جلالة\s+السلطان', 'السلطان'),
        (r'السلطان\s+المعظم', 'السلطان'),

        # President honorifics - REMOVE "فخامة" but this is less common in Gulf news
        (r'فخامة\s+الرئيس', 'الرئيس'),

        # Prince titles - KEEP "صاحب السمو الملكي" as it's an official title
        # But remove exaggerated adjectives like "الجليل" or "الكريم" when standalone
        (r'سموه\s+الكريم', 'سموه'),
        (r'سموه\s+الجليل', 'سموه'),

        # Prayer phrases - DELETE completely (these are pure exaggeration, not titles)
        (r'\s+حفظه\s+الله\s+', ' '),
        (r'\s+حفظها\s+الله\s+', ' '),
        (r'\s+رعاه\s+الله\s+', ' '),
        (r'\s+رعاها\s+الله\s+', ' '),
        (r'\s+نصره\s+الله\s+', ' '),
        (r'\s+أيده\s+الله\s+', ' '),
        (r'\s+أطال\s+الله\s+عمره\s+', ' '),
        (r'\s+أدام\s+الله\s+عزه\s+', ' '),
        (r'\s+حفظهما\s+الله\s+', ' '),
        (r'\s+حفظهم\s+الله\s+', ' '),
        (r'\s+حفظه\s+الله\s*$', ' '),  # At end of sentence
        (r'^\s*حفظه\s+الله\s+', ' '),  # At start of sentence

        # Exaggerated adjectives - DELETE
        (r'\bخالص\s+', ''),  # "خالص تهانيه" → "تهانيه"
        (r'\bالجليل\s+', ''),  # When used as standalone adjective
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
        "⚠️ CRITICAL OUTPUT FORMAT REQUIREMENT ⚠️\n"
        "Your output MUST be divided into MULTIPLE paragraphs separated by blank lines.\n"
        "DO NOT write the article as one continuous paragraph.\n"
        "Each paragraph = ONE main idea.\n"
        "Separate paragraphs with double newlines (\\n\\n).\n\n"
        "### Editorial Guidelines\n"
        f"{guideline_section}\n\n"
        "### Reference News Examples\n"
        f"{example_section}\n\n"
        "### Article Requiring Review\n"
        f"{processed_news_text}\n\n"
        "FIRST: Validate that the text above is a legitimate news article. "
        "If it is random, inappropriate, meaningless, or not a news article, "
        "respond ONLY with: 'ERROR: النص المقدم غير مناسب أو غير صالح للمعالجة. يرجى تقديم خبر صحيح.'\n\n"
        "⚠️⚠️⚠️ CRITICAL RULE: UNDERSTAND THE DIFFERENCE BETWEEN OFFICIAL TITLES AND EXAGGERATION ⚠️⚠️⚠️\n\n"
        "🔴 ABSOLUTE RULE - READ THIS CAREFULLY:\n"
        "There is a HUGE difference between:\n"
        "1. OFFICIAL STATE TITLES (الألقاب الرسمية للدولة) = These are REAL titles, NOT exaggeration → MUST KEEP\n"
        "2. EXAGGERATED PHRASES (التفخيم والتعظيم) = These are praise phrases, NOT titles → MUST REMOVE\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "✅ WHAT TO KEEP - THESE ARE OFFICIAL TITLES (DO NOT TOUCH!):\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "1. ✅ 'خادم الحرمين الشريفين' - Official title of Saudi King (like a last name)\n"
        "   Why? This is his OFFICIAL STATE TITLE, not exaggeration!\n"
        "   ⚠️ NEVER remove this! It's like removing someone's official job title!\n\n"
        "2. ✅ 'صاحب السمو الملكي' - Official Royal Highness title (government-recognized)\n"
        "   Why? This is the OFFICIAL PROTOCOL title for princes, grandson of the King!\n"
        "   ⚠️ NEVER remove this! It's their official designation in the state!\n\n"
        "3. ✅ 'ولي العهد' - Official Crown Prince position\n"
        "4. ✅ 'رئيس مجلس الوزراء' - Official Prime Minister position\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "❌ WHAT TO REMOVE - THESE ARE EXAGGERATION (DELETE OR SIMPLIFY!):\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "1. ❌ 'جلالة الملك المعظم أيده الله' → ✅ 'الملك'\n"
        "   Why? 'جلالة' and 'المعظم' and 'أيده الله' are EXAGGERATION, not official titles!\n\n"
        "2. ❌ 'جلالة الملك' → ✅ 'الملك'\n"
        "3. ❌ 'حضرة صاحب الجلالة الملك' → ✅ 'الملك'\n"
        "4. ❌ 'صاحب الجلالة السلطان' → ✅ 'السلطان'\n"
        "5. ❌ 'فخامة الرئيس' → ✅ 'الرئيس'\n"
        "6. ❌ Prayer phrases: 'حفظه الله', 'أيده الله', 'رعاه الله', 'نصره الله' → DELETE COMPLETELY\n"
        "7. ❌ Exaggeration words: 'المعظم', 'الجليل', 'خالص' → DELETE\n"
        "8. ❌ 'سموه الكريم' → ✅ 'سموه'\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📌 KEY DISTINCTION (READ THIS 10 TIMES!):\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "• 'خادم الحرمين الشريفين' = Official title (like saying 'Dr.' or 'President') → KEEP!\n"
        "• 'صاحب السمو الملكي' = Official royal protocol title → KEEP!\n"
        "• 'جلالة الملك المعظم' = Exaggerated praise → REMOVE, simplify to 'الملك'\n"
        "• 'حفظه الله' 'أيده الله' = Prayer/supplication → DELETE COMPLETELY\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📚 MANDATORY EXAMPLES - STUDY THESE CAREFULLY:\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Example 1 - Removing exaggeration, keeping simple title:\n"
        "❌ BEFORE: 'بعث جلالة الملك المعظم حمد بن عيسى آل خليفة حفظه الله برقية تهنئة خالصة'\n"
        "✅ AFTER:  'بعث الملك حمد بن عيسى آل خليفة برقية تهنئة'\n"
        "What we removed: جلالة (exaggeration), المعظم (exaggeration), حفظه الله (prayer), خالصة (exaggeration)\n"
        "What we kept: الملك (simple title)\n\n"
        "Example 2 - KEEPING official 'صاحب السمو الملكي' because it's OFFICIAL:\n"
        "❌ BEFORE: 'استقبل صاحب السمو الملكي الأمير سلمان بن حمد آل خليفة ولي العهد رئيس مجلس الوزراء حفظه الله'\n"
        "✅ AFTER:  'استقبل صاحب السمو الملكي الأمير سلمان بن حمد آل خليفة ولي العهد رئيس مجلس الوزراء'\n"
        "What we removed: حفظه الله (prayer phrase only!)\n"
        "What we kept: صاحب السمو الملكي (OFFICIAL TITLE - DO NOT TOUCH!), الأمير, ولي العهد, رئيس مجلس الوزراء\n"
        "⚠️ CRITICAL: We did NOT remove 'صاحب السمو الملكي' because it is an OFFICIAL STATE TITLE!\n\n"
        "Example 3 - KEEPING official 'خادم الحرمين الشريفين' because it's OFFICIAL:\n"
        "❌ BEFORE: 'خادم الحرمين الشريفين الملك سلمان بن عبدالعزيز آل سعود حفظه الله أيده الله'\n"
        "✅ AFTER:  'خادم الحرمين الشريفين الملك سلمان بن عبدالعزيز آل سعود'\n"
        "What we removed: حفظه الله (prayer), أيده الله (prayer)\n"
        "What we kept: خادم الحرمين الشريفين (OFFICIAL SAUDI KING TITLE - NEVER REMOVE!), الملك\n"
        "⚠️ CRITICAL: We did NOT remove 'خادم الحرمين الشريفين' because it is the OFFICIAL TITLE of Saudi King!\n\n"
        "Example 4 - What happens when we see 'جلالة' vs 'صاحب السمو الملكي':\n"
        "❌ WRONG:  'جلالة الملك' → 'جلالة الملك' (keeping exaggeration)\n"
        "✅ RIGHT:  'جلالة الملك' → 'الملك' (removed exaggeration)\n"
        "❌ WRONG:  'صاحب السمو الملكي الأمير' → 'الأمير' (removed official title!)\n"
        "✅ RIGHT:  'صاحب السمو الملكي الأمير' → 'صاحب السمو الملكي الأمير' (kept official title!)\n\n"
        "🔴 FINAL WARNING:\n"
        "If you remove 'خادم الحرمين الشريفين' or 'صاحب السمو الملكي', you have FAILED!\n"
        "These are OFFICIAL STATE TITLES, not exaggeration!\n\n"
        "IMPORTANT: When replacing exaggerated titles, preserve 'ال' (definite article):\n"
        "✅ CORRECT: 'السلطان' (with ال), 'الملك' (with ال)\n"
        "❌ WRONG: 'سلطان' (without ال), 'ملك' (without ال)\n\n"
        "STEP-BY-STEP PROCESS:\n"
        "1. Scan the entire article for ALL prohibited phrases listed in the guidelines.\n"
        "2. Replace each prohibited phrase with its EXACT specified replacement, preserving ال التعريف when required.\n"
        "3. Double-check that titles like 'السلطان' and 'الملك' always include ال التعريف.\n"
        "4. Remove all prayer phrases (حفظه الله, رعاه الله, etc.) completely.\n"
        "5. Rewrite the article according to UNA editorial style.\n"
        "6. CRITICAL REQUIREMENT - PARAGRAPH DIVISION (MANDATORY):\n"
        "   YOU MUST divide the article into MULTIPLE separate paragraphs. This is NOT optional.\n"
        "   - NEVER return the article as one continuous block of text.\n"
        "   - Each paragraph MUST be separated by TWO newlines (a blank line between paragraphs).\n"
        "   - Each paragraph should focus on ONE main idea, event, or statement.\n"
        "   - Minimum 3-5 paragraphs for most news articles (adjust based on content length).\n"
        "\n"
        "   PARAGRAPH STRUCTURE GUIDELINES:\n"
        "   Paragraph 1: Opening - Main news announcement with key facts (who, what, when, where)\n"
        "   Paragraph 2: Context/Details - Background information or event details\n"
        "   Paragraph 3: Statements/Quotes - What officials said or actions taken\n"
        "   Paragraph 4: Additional Information - Secondary details, attendees, or related information\n"
        "   Paragraph 5: Conclusion - Closing remarks, future implications, or wrap-up\n"
        "\n"
        "   EXAMPLE FORMAT (note the blank lines between paragraphs):\n"
        "   المنامة في 10 نوفمبر / بنا / أكد ولي العهد رئيس مجلس الوزراء الأمير سلمان بن حمد آل خليفة...\n"
        "\n"
        "   جاء ذلك لدى لقاء سموه اليوم في قصر القضيبية...\n"
        "\n"
        "   وأشاد سموه بما تشهده سلطنة عُمان الشقيقة من نهضةٍ تنمويةٍ متواصلة...\n"
        "\n"
        "   كما جرى خلال اللقاء استعراض القضايا ذات الاهتمام المشترك...\n"
        "\n"
        "   من جانبه، أعرب وزير الداخلية بسلطنة عُمان الشقيقة عن شكره...\n"
        "\n"
        "7. Return ONLY the final revised news article in Arabic with PROPERLY SEPARATED PARAGRAPHS. No analysis or commentary.\n\n"
        "⚠️⚠️⚠️ FINAL REMINDER - READ THIS BEFORE OUTPUTTING ⚠️⚠️⚠️\n"
        "Your response must contain AT LEAST 3-5 separate paragraphs with blank lines between them.\n"
        "If you return the article as ONE paragraph, you have FAILED the task.\n"
        "Format: Paragraph1\\n\\nParagraph2\\n\\nParagraph3\\n\\nParagraph4\\n\\nParagraph5\n"
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
                "⚠️⚠️⚠️ CRITICAL DISTINCTION: OFFICIAL STATE TITLES vs EXAGGERATION ⚠️⚠️⚠️\n\n"
                "🔴 ABSOLUTE RULE YOU MUST UNDERSTAND:\n"
                "Some phrases are OFFICIAL STATE TITLES (like job titles) - these are NOT exaggeration!\n"
                "Other phrases are EXAGGERATED PRAISE - these must be removed!\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "✅ PRESERVE THESE - OFFICIAL STATE TITLES (NEVER REMOVE!):\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "1. ✅ 'خادم الحرمين الشريفين' - OFFICIAL title of Saudi King\n"
                "   This is NOT exaggeration! It's like saying 'President' or 'Prime Minister'!\n"
                "   ⚠️ NEVER EVER remove this phrase! It's his official state designation!\n\n"
                "2. ✅ 'صاحب السمو الملكي' - OFFICIAL Royal Highness title\n"
                "   This is NOT exaggeration! It's the government-recognized protocol title!\n"
                "   ⚠️ NEVER EVER remove this phrase! It's their official rank in the state!\n\n"
                "3. ✅ 'ولي العهد' - Crown Prince (official position)\n"
                "4. ✅ 'رئيس مجلس الوزراء' - Prime Minister (official position)\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "❌ REMOVE THESE - EXAGGERATED PRAISE (NOT OFFICIAL!):\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "1. ❌ 'جلالة الملك المعظم' → ✅ 'الملك' (this IS exaggeration!)\n"
                "2. ❌ 'جلالة الملك' → ✅ 'الملك'\n"
                "3. ❌ 'حضرة صاحب الجلالة الملك' → ✅ 'الملك'\n"
                "4. ❌ 'صاحب الجلالة السلطان' → ✅ 'السلطان'\n"
                "5. ❌ 'فخامة الرئيس' → ✅ 'الرئيس'\n"
                "6. ❌ Prayer phrases: 'حفظه الله', 'أيده الله', 'رعاه الله' → DELETE\n"
                "7. ❌ Exaggeration words: 'المعظم', 'الجليل', 'خالص' → DELETE\n\n"
                "🔑 KEY DIFFERENCE:\n"
                "• 'خادم الحرمين الشريفين' = Like saying 'President Obama' → KEEP!\n"
                "• 'صاحب السمو الملكي' = Like saying 'His Royal Highness' → KEEP!\n"
                "• 'جلالة الملك المعظم' = Like saying 'His Glorious Majesty' → REMOVE!\n"
                "• 'حفظه الله' = Prayer/blessing → REMOVE!\n\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📚 MANDATORY EXAMPLES - FOLLOW THESE EXACTLY:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "Example 1 - Remove exaggeration, keep simple title:\n"
                "Original: 'بعث جلالة الملك المعظم عبد الله الثاني ابن الحسين حفظه الله برقية تهنئة خالصة'\n"
                "After: 'بعث الملك عبد الله الثاني ابن الحسين برقية تهنئة'\n"
                "Removed: جلالة, المعظم, حفظه الله, خالصة\n"
                "Kept: الملك\n\n"
                "Example 2 - KEEP 'صاحب السمو الملكي' (OFFICIAL TITLE!):\n"
                "Original: 'استقبل صاحب السمو الملكي الأمير سلمان بن حمد آل خليفة ولي العهد رئيس مجلس الوزراء حفظه الله'\n"
                "After: 'استقبل صاحب السمو الملكي الأمير سلمان بن حمد آل خليفة ولي العهد رئيس مجلس الوزراء'\n"
                "Removed: حفظه الله ONLY!\n"
                "Kept: صاحب السمو الملكي (OFFICIAL!), الأمير, ولي العهد, رئيس مجلس الوزراء\n"
                "⚠️ Notice: We did NOT remove 'صاحب السمو الملكي' - it's an OFFICIAL title!\n\n"
                "Example 3 - KEEP 'خادم الحرمين الشريفين' (OFFICIAL TITLE!):\n"
                "Original: 'خادم الحرمين الشريفين الملك سلمان بن عبدالعزيز آل سعود حفظه الله أيده الله'\n"
                "After: 'خادم الحرمين الشريفين الملك سلمان بن عبدالعزيز آل سعود'\n"
                "Removed: حفظه الله, أيده الله ONLY!\n"
                "Kept: خادم الحرمين الشريفين (OFFICIAL SAUDI TITLE!), الملك\n"
                "⚠️ Notice: We did NOT remove 'خادم الحرمين الشريفين' - it's the OFFICIAL title!\n\n"
                "Example 4 - Understanding the difference:\n"
                "❌ WRONG: 'صاحب السمو الملكي الأمير' → 'الأمير' (you removed official title!)\n"
                "✅ RIGHT: 'صاحب السمو الملكي الأمير' → 'صاحب السمو الملكي الأمير' (kept it!)\n"
                "❌ WRONG: 'جلالة الملك' → 'جلالة الملك' (you kept exaggeration!)\n"
                "✅ RIGHT: 'جلالة الملك' → 'الملك' (removed exaggeration!)\n\n"
                "🔴 If you remove 'خادم الحرمين الشريفين' or 'صاحب السمو الملكي', YOU FAILED!\n\n"
                "IMPORTANT RULE: Always preserve 'ال' (definite article) when simplifying titles:\n"
                "✅ CORRECT: 'السلطان' (with ال), 'الملك' (with ال)\n"
                "❌ WRONG: 'سلطان' (without ال), 'ملك' (without ال)\n\n"
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
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "📋 STEP-BY-STEP EDITING PROCESS:\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                "Step 1 - Scan for OFFICIAL TITLES to PRESERVE:\n"
                "   ✅ Do you see 'خادم الحرمين الشريفين'? → KEEP IT!\n"
                "   ✅ Do you see 'صاحب السمو الملكي'? → KEEP IT!\n"
                "   ✅ Do you see 'ولي العهد'? → KEEP IT!\n"
                "   ✅ Do you see 'رئيس مجلس الوزراء'? → KEEP IT!\n"
                "   These are OFFICIAL STATE TITLES - like job titles - NOT exaggeration!\n\n"
                "Step 2 - Scan for EXAGGERATION to REMOVE:\n"
                "   ❌ Do you see 'جلالة الملك'? → Change to 'الملك'\n"
                "   ❌ Do you see 'صاحب الجلالة السلطان'? → Change to 'السلطان'\n"
                "   ❌ Do you see 'حفظه الله' or 'أيده الله'? → DELETE completely\n"
                "   ❌ Do you see 'المعظم', 'الجليل', 'خالص'? → DELETE\n"
                "   These are EXAGGERATED PRAISE - not official titles!\n\n"
                "Step 3 - Apply changes CAREFULLY:\n"
                "   • NEVER remove: خادم الحرمين الشريفين, صاحب السمو الملكي\n"
                "   • ALWAYS remove: جلالة, صاحب الجلالة, فخامة\n"
                "   • DELETE prayer phrases: حفظه الله, أيده الله, رعاه الله\n"
                "   • When simplifying, preserve ال: السلطان (not سلطان), الملك (not ملك)\n\n"
                "Step 4 - Final verification checklist:\n"
                "   ✅ Is 'خادم الحرمين الشريفين' still there (if it was in original)?\n"
                "   ✅ Is 'صاحب السمو الملكي' still there (if it was in original)?\n"
                "   ✅ Are all prayer phrases ('حفظه الله', etc.) deleted?\n"
                "   ✅ Are all exaggerations ('جلالة', 'المعظم', etc.) removed?\n"
                "   ✅ Is ال التعريف preserved in simplified titles?\n\n"
                "Step 5 - Apply editorial style guidelines (objectivity, clarity, etc.).\n\n"
                "Step 6 - Rewrite the article according to UNA editorial style.\n\n"
                "8. MANDATORY PARAGRAPH DIVISION - THIS IS CRITICAL AND NON-NEGOTIABLE:\n"
                "   IMPORTANT: You MUST divide the article into multiple separate paragraphs.\n"
                "   DO NOT write the article as a single continuous paragraph.\n"
                "\n"
                "   a) Analyze the article content to identify distinct topics, ideas, or events.\n"
                "   b) Create separate paragraphs for each distinct idea:\n"
                "      - Paragraph 1: Main announcement/opening (who did what, when, where)\n"
                "      - Paragraph 2: Event context and details (background, setting, attendees)\n"
                "      - Paragraph 3: Main statements or actions (what was said or done)\n"
                "      - Paragraph 4: Additional information (related details, secondary statements)\n"
                "      - Paragraph 5: Conclusion (wrap-up, future implications, or closing remarks)\n"
                "   c) Each paragraph should be 2-4 sentences maximum.\n"
                "   d) CRITICAL: Separate each paragraph with TWO newline characters (\\n\\n) to create a blank line.\n"
                "   e) Ensure logical flow and smooth transitions between paragraphs.\n"
                "   f) The article MUST have at least 3-5 paragraphs unless the content is extremely short.\n"
                "\n"
                "   EXAMPLE OF CORRECT OUTPUT FORMAT:\n"
                "   المنامة في 10 نوفمبر / بنا / أكد ولي العهد...\n"
                "   [BLANK LINE]\n"
                "   جاء ذلك لدى لقاء سموه اليوم...\n"
                "   [BLANK LINE]\n"
                "   وأشاد سموه بما تشهده سلطنة عُمان...\n"
                "   [BLANK LINE]\n"
                "   كما جرى خلال اللقاء استعراض...\n"
                "\n"
                "9. Deliver ONLY the final revised article text in Arabic with PROPERLY DIVIDED PARAGRAPHS; no analysis or explanations."
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

