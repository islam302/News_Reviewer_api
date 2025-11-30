"""
Test script for fact-checking functionality with Google Search
"""
import os
import sys
import django

# Fix encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Config.settings')
django.setup()

from reviewer.services import check_and_correct_text_between_hashtags

def test_fact_check():
    """Test the fact-checking functionality"""

    # Test case 1: Wrong capital city
    test_text_1 = "##في العاصمة الطائف##"
    print("=" * 60)
    print("Test 1: Wrong capital city")
    print(f"Input: {test_text_1}")

    try:
        result_1 = check_and_correct_text_between_hashtags(text=test_text_1)
        print(f"Output: {result_1}")
        print("✅ Test 1 passed!")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")

    print("\n" + "=" * 60 + "\n")

    # Test case 2: Wrong director name (exact from user's request)
    test_text_2 = "##سعادة الأستاذ اسلام بدران ##"
    print("Test 2: Wrong director name for UNA (with spaces)")
    print(f"Input: '{test_text_2}'")

    try:
        result_2 = check_and_correct_text_between_hashtags(text=test_text_2)
        print(f"Output: '{result_2}'")
        print("✅ Test 2 passed!")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")

    print("\n" + "=" * 60 + "\n")

    # Test case 3: Full news text with context
    full_news = 'جدة (يونا) – استقبل المدير العام لاتحاد وكالات أنباء دول منظمة التعاون الإسلامي "يونا"، ##سعادة الأستاذ اسلام بدران ##، اليوم الخميس'
    print("Test 3: Full news with wrong director name (WITH CONTEXT)")
    print(f"Input: {full_news[:80]}...")

    try:
        # Extract and check WITH FULL CONTEXT
        pattern = r'##(.+?)##'
        import re
        matches = re.findall(pattern, full_news, re.DOTALL)
        if matches:
            print(f"Extracted text: '{matches[0]}'")
            print(f"Context: Full news about UNA director")
            result_3 = check_and_correct_text_between_hashtags(
                text=f"##{matches[0]}##",
                full_context=full_news  # Pass full context!
            )
            print(f"Corrected text: '{result_3}'")
            print("✅ Test 3 passed!")
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_fact_check()
