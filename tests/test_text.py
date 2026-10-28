import pytest

from paperqa.text import chunk_words, clean_text, split_sentences, tokenize


def test_clean_text_collapses_hard_wraps():
    assert clean_text("  Deep\n  learning\tfor\n graphs ") == "Deep learning for graphs"


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The Transformer is all you need for self-attention") == [
        "transformer", "need", "self-attention"
    ]


def test_chunk_words_short_text_is_single_chunk():
    assert chunk_words("one two three", size=5, overlap=1) == ["one two three"]


def test_chunk_words_overlap_and_full_coverage():
    words = [f"w{i}" for i in range(10)]
    chunks = chunk_words(" ".join(words), size=4, overlap=2)
    assert chunks[0] == "w0 w1 w2 w3"
    assert chunks[1] == "w2 w3 w4 w5"  # 2-word overlap
    assert chunks[-1].split()[-1] == "w9"  # nothing dropped at the end
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words)


def test_chunk_words_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_words("a b c", size=3, overlap=3)


def test_split_sentences():
    text = "We propose X. It beats Y by 3%. (Code is public.) Done! e.g. lowercase stays."
    # Splits only after . ! ? followed by an uppercase letter, digit, or "(" -- so
    # abbreviations like "e.g. lowercase" and text after a closing ")" stay together.
    assert split_sentences(text) == [
        "We propose X.",
        "It beats Y by 3%.",
        "(Code is public.) Done! e.g. lowercase stays.",
    ]
