from __future__ import annotations
import re
import logging
from typing import List
from django.conf import settings
from openai import OpenAI
from serpapi import GoogleSearch
from .models import Instruction, NewsExample


logger = logging.getLogger(__name__)

# Models
BEST_MODEL = "gpt-4o"
FAST_MODEL = "gpt-4o-mini"
DEFAULT_COMPLETION_MODEL = BEST_MODEL


def _get_openai_client() -> OpenAI:
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY env var must be set before calling OpenAI APIs.")
    return OpenAI(api_key=api_key)


# =============================================================================
# INSTRUCTION PROCESSING - Translate & Summarize for better AI understanding
# =============================================================================

def process_instruction(content: str, title: str = "") -> str:
    """
    Process user instruction: analyze, summarize, and translate to English.

    This helps the review model understand and execute instructions more accurately
    because AI models perform better with clear English instructions.

    Args:
        content: Original instruction in Arabic
        title: Instruction title for context

    Returns:
        Processed instruction in English (clear, concise, actionable)
    """
    client = _get_openai_client()

    messages = [
        {
            "role": "system",
            "content": """You are an expert at converting Arabic editorial instructions into clear, actionable English rules.

Your task:
1. Understand the Arabic instruction completely
2. Extract ALL rules and requirements (don't miss any)
3. Convert to clear, concise English
4. Format as actionable rules the AI can follow

Output format:
- Use bullet points for each rule
- Be specific and unambiguous
- Include examples where helpful
- Keep it concise but complete

IMPORTANT: Do NOT lose any meaning or rule from the original Arabic text."""
        },
        {
            "role": "user",
            "content": f"""Convert this Arabic editorial instruction to clear English rules:

Title: {title}

Instruction:
{content}

Return ONLY the English rules, no explanations."""
        }
    ]

    try:
        response = client.chat.completions.create(
            model=FAST_MODEL,
            messages=messages,
            temperature=0.3,  # Low temperature for accuracy
            max_tokens=1000
        )

        if response.choices:
            return response.choices[0].message.content.strip()
        return ""
    except Exception as e:
        logger.error(f"Error processing instruction: {e}")
        return ""


def process_all_user_instructions(user) -> int:
    """
    Process all instructions for a user.

    Args:
        user: The user whose instructions to process

    Returns:
        Number of instructions processed
    """
    instructions = Instruction.objects.filter(user=user)
    count = 0

    for instruction in instructions:
        processed = process_instruction(instruction.content, instruction.title)
        if processed:
            instruction.processed_content = processed
            instruction.save(update_fields=['processed_content'])
            count += 1

    return count


# =============================================================================
# TEXT PREPROCESSING
# =============================================================================

def _preprocess_honorifics(text: str) -> str:
    """
    Preprocess text to remove excessive honorifics while keeping official titles.
    Preserves quoted text unchanged.
    """
    # Extract and preserve quoted text
    quote_patterns = [
        r'"([^"]+)"',
        r'«([^»]+)»',
        r'"([^"]+)"',
        r"'([^']+)'",
    ]

    quoted_texts = []
    processed_text = text

    for pattern in quote_patterns:
        matches = re.finditer(pattern, processed_text)
        for match in matches:
            placeholder = f"<<<QUOTE_{len(quoted_texts)}>>>"
            quoted_texts.append(match.group(0))
            processed_text = processed_text.replace(match.group(0), placeholder, 1)

    # Replacement patterns
    replacements = [
        (r'جلالة\s+الملك\s+المعظم', 'الملك'),
        (r'جلالة\s+الملك', 'الملك'),
        (r'حضرة\s+صاحب\s+الجلالة\s+الملك\s+المعظم', 'الملك'),
        (r'حضرة\s+صاحب\s+الجلالة\s+الملك', 'الملك'),
        (r'صاحب\s+الجلالة\s+الملك\s+المعظم', 'الملك'),
        (r'صاحب\s+الجلالة\s+الملك', 'الملك'),
        (r'الملك\s+المعظم', 'الملك'),
        (r'حضرة\s+صاحب\s+الجلالة\s+السلطان\s+المعظم', 'السلطان'),
        (r'حضرة\s+صاحب\s+الجلالة\s+السلطان', 'السلطان'),
        (r'صاحب\s+الجلالة\s+السلطان\s+المعظم', 'السلطان'),
        (r'صاحب\s+الجلالة\s+السلطان', 'السلطان'),
        (r'جلالة\s+السلطان', 'السلطان'),
        (r'السلطان\s+المعظم', 'السلطان'),
        (r'فخامة\s+الرئيس', 'الرئيس'),
        (r'سموه\s+الكريم', 'سموه'),
        (r'سموه\s+الجليل', 'سموه'),
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
        (r'\s+حفظه\s+الله\s*$', ' '),
        (r'^\s*حفظه\s+الله\s+', ' '),
        (r'\bخالص\s+', ''),
        (r'\bالجليل\s+', ''),
    ]

    for pattern, replacement in replacements:
        processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)

    processed_text = re.sub(r'\s+', ' ', processed_text)
    processed_text = processed_text.strip()

    # Restore quoted texts
    for i, quoted_text in enumerate(quoted_texts):
        placeholder = f"<<<QUOTE_{i}>>>"
        processed_text = processed_text.replace(placeholder, quoted_text)

    return processed_text


def _split_into_paragraphs(text: str) -> str:
    """Post-process AI output to ensure proper paragraph separation."""
    if "\n\n" in text and text.count("\n\n") >= 3:
        return text

    text = text.replace("\n\n", "<<<PARAGRAPH_BREAK>>>")
    text = text.replace("\n", " ")
    text = text.replace("<<<PARAGRAPH_BREAK>>>", "\n\n")

    city_agency_pattern = r'^(.+?)\s+((?:مدريد|المنامة|الرياض|عمّان|عمان|القاهرة|دمشق|بغداد|مسقط|الكويت|الدوحة|أبوظبي|أبو ظبي|بيروت|تونس|الجزائر|الرباط|طرابلس|نواكشوط|صنعاء|الخرطوم|تل أبيب)(?:\s+في\s+\d|\s*\(يونا))'
    match = re.search(city_agency_pattern, text)

    if match:
        title = match.group(1).strip()
        body = text[match.start(2):].strip()
        if not any(city in title[-20:] for city in ['بغداد', 'الرياض', 'القاهرة', 'دمشق', 'عمان', 'مسقط']):
            text = f"{title}\n\n{body}"

    paragraph_starters = [
        r'([.؟!])\s+(وأشاد)', r'([.؟!])\s+(وأكد)', r'([.؟!])\s+(وقال)',
        r'([.؟!])\s+(وأضاف)', r'([.؟!])\s+(جاء ذلك)', r'([.؟!])\s+(وجاء)',
        r'([.؟!])\s+(كما)', r'([.؟!])\s+(وشهد)', r'([.؟!])\s+(وشملت)',
        r'([.؟!])\s+(وتمثّلت)', r'([.؟!])\s+(تمثلت)', r'([.؟!])\s+(شملت)',
        r'([.؟!])\s+(ووقع)', r'([.؟!])\s+(وقّع)', r'([.؟!])\s+(وقع)',
        r'([.؟!])\s+(وعن)', r'([.؟!])\s+(من جانبه)', r'([.؟!])\s+(من جهته)',
        r'([.؟!])\s+(من جانبها)', r'([.؟!])\s+(بدوره)',
    ]

    for pattern in paragraph_starters:
        text = re.sub(pattern, r'\1\n\n\2', text)

    text = re.sub(r'([.؟!])\s*(\(انتهى\))', r'\1\n\n\2', text)
    text = re.sub(r'([.؟!])\s*(العُمانية/)', r'\1\n\n\2', text)
    text = re.sub(r'(\S)\s+(\(انتهى\))', r'\1\n\n\2', text)
    text = re.sub(r'(\S)\s+(العُمانية/)', r'\1\n\n\2', text)

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


# =============================================================================
# FACT CHECKING
# =============================================================================

def _search_google_for_fact_check(query: str) -> str:
    """Search Google using SERP API for fact-checking."""
    serpapi_key = settings.SERPAPI_KEY
    if not serpapi_key:
        return "No search results available."

    try:
        serpapi_key = serpapi_key.strip('"\'')
        params = {
            "q": query,
            "api_key": serpapi_key,
            "num": 5,
            "hl": "ar",
            "gl": "sa",
        }

        search = GoogleSearch(params)
        results = search.get_dict()
        organic_results = results.get("organic_results", [])

        if not organic_results:
            return "No search results found."

        formatted_results = []
        for idx, result in enumerate(organic_results[:5], 1):
            title = result.get("title", "")
            snippet = result.get("snippet", "")
            formatted_results.append(f"{idx}. {title}\n   {snippet}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Search error: {str(e)}"


def check_and_correct_text_between_hashtags(
    *,
    text: str,
    model: str = DEFAULT_COMPLETION_MODEL,
    full_context: str = None,
) -> str:
    """Extract text between ##text## markers and check/correct factual errors."""
    pattern = r'##(.+?)##'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        raise ValueError("No text found between ## markers.")

    text_to_check = matches[0].strip()
    if not text_to_check:
        raise ValueError("Empty text found between ## markers.")

    context_text = full_context if full_context else text

    search_query = text_to_check
    if "يونا" in context_text or "UNA" in context_text.upper():
        search_query = "UNA OIC news agency director general 2025"
    elif "المدير" in context_text or "الرئيس" in context_text or "الوزير" in context_text:
        search_query = f"{context_text[:200]} who is"
    else:
        search_query = f"{text_to_check} fact check"

    search_results = _search_google_for_fact_check(search_query)
    client = _get_openai_client()

    messages = [
        {
            "role": "system",
            "content": "You are a fact-checker. Verify and correct factual errors using the search results. Return only the corrected text."
        },
        {
            "role": "user",
            "content": f"Text to verify:\n{text_to_check}\n\nContext:\n{context_text[:500]}\n\nSearch results:\n{search_results}\n\nReturn corrected text only:"
        },
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    if response.choices:
        return response.choices[0].message.content.strip()

    raise RuntimeError("Failed to get response from OpenAI.")


# =============================================================================
# REVIEW PROMPT BUILDING - Uses processed (English) instructions
# =============================================================================

def build_user_instructions_prompt(news_text: str, user) -> List[dict]:
    """
    Build a review prompt using user's processed instructions.

    Uses processed_content (English) for better AI understanding,
    falls back to original content if processed not available.
    """
    instructions = Instruction.objects.filter(
        user=user,
        is_active=True
    ).order_by('order')

    if not instructions.exists():
        raise ValueError("No active instructions. Please add instructions from the dashboard.")

    # Get examples
    examples = NewsExample.objects.filter(
        user=user,
        is_active=True
    ).order_by('-created_at')[:5]

    # Preprocess news text
    processed_news_text = _preprocess_honorifics(news_text)

    # Build instructions section - USE PROCESSED CONTENT (English)
    instructions_lines = []
    for idx, instruction in enumerate(instructions, start=1):
        # Use processed_content if available, otherwise fallback to original
        content = instruction.processed_content if instruction.processed_content else instruction.content
        instructions_lines.append(f"### Rule {idx}: {instruction.title}\n{content}")

    instructions_section = "\n\n".join(instructions_lines)

    # Build examples section
    examples_section = ""
    if examples.exists():
        examples_lines = []
        for idx, example in enumerate(examples, start=1):
            example_text = (
                f"### Example {idx}: {example.title}\n"
                f"**Original:**\n{example.original_text}\n\n"
                f"**Expected Output:**\n{example.processed_text}"
            )
            if example.notes:
                example_text += f"\n\n**Notes:** {example.notes}"
            examples_lines.append(example_text)

        examples_section = (
            "\n\n" + "="*60 + "\n"
            "EXAMPLES - Follow this exact style:\n"
            + "="*60 + "\n\n"
            + "\n\n---\n\n".join(examples_lines)
        )

    # System prompt in English for better AI performance
    system_prompt = f"""You are a professional Arabic news editor. Your task is to review and edit the Arabic news text according to the specific rules below.

CRITICAL RULES:
- Apply ALL rules in order - do not skip any
- Do NOT add information not in the original
- Do NOT rewrite or paraphrase - only apply the specific edits
- Return ONLY the edited Arabic text, no explanations

{"="*60}
EDITING RULES TO APPLY:
{"="*60}

{instructions_section}
{examples_section}

{"="*60}

OUTPUT FORMAT:
1. Title on first line (with nationality added if needed)
2. Blank line
3. City (UNA/Source) - News body in paragraphs
4. Blank line between each paragraph
5. (انتهى) on final line"""

    user_prompt = f"""Edit this Arabic news article according to ALL the rules above:

{processed_news_text}

Return the edited article:"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# =============================================================================
# MAIN REVIEW FUNCTION
# =============================================================================

def generate_review(
    *,
    news_text: str,
    user,
    model: str = BEST_MODEL,
) -> str:
    """
    Generate a review of news text using user's processed instructions.

    Uses GPT-4o for high-quality output with English instructions
    for better accuracy.
    """
    # Check for ##text## markers
    pattern = r'##(.+?)##'
    matches = re.findall(pattern, news_text, re.DOTALL)

    if matches:
        processed_text = news_text
        for match in matches:
            corrected = check_and_correct_text_between_hashtags(
                text=f"##{match}##",
                model=model,
                full_context=news_text
            )
            processed_text = processed_text.replace(f"##{match}##", corrected, 1)
        news_text = processed_text

    client = _get_openai_client()
    messages = build_user_instructions_prompt(news_text, user)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,  # Lower temperature for more consistent output
    )

    if not response.choices:
        raise RuntimeError("Failed to get response from OpenAI.")

    result_text = response.choices[0].message.content.strip()

    # Check for rejection
    if result_text.startswith("ERROR:") or "غير مناسب" in result_text or "غير صالح" in result_text:
        error_msg = result_text.replace("ERROR:", "").strip()
        if not error_msg:
            error_msg = "النص المقدم غير مناسب أو غير صالح للمعالجة."
        raise ValueError(error_msg)

    # Post-processing
    final_text = _preprocess_honorifics(result_text)
    final_text = _split_into_paragraphs(final_text)

    return final_text
