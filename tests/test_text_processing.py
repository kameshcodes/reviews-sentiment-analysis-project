from src.utils import data_preprocessing, expand_slang_and_short_forms, load_slang_dictionary

# Initialize slang_dict correctly by calling the function
slang_dict = load_slang_dictionary()


# Tests for data_preprocessing function
def test_clean_text():
    input_text = '<p>This is a <a href="http://example.com">sample</a> text!</p> You are a #$%&!'
    expected_output = 'sample text'
    assert data_preprocessing(input_text) == expected_output


def test_clean_text_with_special_characters():
    input_text = "Hello @everyone, *welcome* to #fun times!"
    expected_output = "hello welcome fun time"
    assert data_preprocessing(input_text) == expected_output


def test_clean_text_with_html_entities():
    input_text = "This text &amp; that text with &lt;html&gt; tags!"
    # Decode HTML entities within data_preprocessing
    expected_output = "text amp text lt html gt tag"
    assert data_preprocessing(input_text) == expected_output


# Tests for expand_slang_and_short_forms function
def test_expand_slang():
    input_text = "brb, idk what to do!"
    expected_output = "be right back, i don't know what to do!"
    assert expand_slang_and_short_forms(input_text, slang_dict).lower() == expected_output.lower()


def test_expand_slang_with_multiple_slangs():
    input_text = "fomo rn because yolo!"
    expected_output = "fear of missing out right now because you only live once!"
    assert expand_slang_and_short_forms(input_text, slang_dict).lower() == expected_output.lower()
