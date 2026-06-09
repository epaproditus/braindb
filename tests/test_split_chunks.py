"""
Pure-function tests for the ingest watcher's chunk splitter.

Covers: default chunk size, custom size, overlap, word boundaries, edge cases
(empty, single word, exact chunk-size boundary). Also tests
``split_chunks_with_offsets`` — the offset-aware variant whose byte ranges
get stored in fact notes so the recall agent can locate the exact source
slice via ``get_entity(offset, limit)`` instead of guessing from chunk number.
"""
from braindb.ingest_watcher import (
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    split_chunks,
    split_chunks_with_offsets,
)


def test_empty_text():
    assert split_chunks("") == []
    assert split_chunks("   \n\t  ") == []


def test_single_word():
    out = split_chunks("hello")
    assert out == ["hello"]


def test_short_text_fits_in_one_chunk():
    text = " ".join(f"w{i}" for i in range(100))   # 100 words, under 600
    out = split_chunks(text)
    assert len(out) == 1
    assert out[0] == text


def test_exact_chunk_size_boundary():
    # Exactly CHUNK_WORDS words — should be one chunk, no empty second chunk
    text = " ".join(f"w{i}" for i in range(CHUNK_WORDS))
    out = split_chunks(text)
    assert len(out) == 1
    assert len(out[0].split()) == CHUNK_WORDS


def test_one_more_than_chunk_size():
    # CHUNK_WORDS + 1 words → should produce 2 chunks (the second one small)
    total = CHUNK_WORDS + 1
    text = " ".join(f"w{i}" for i in range(total))
    out = split_chunks(text)
    assert len(out) == 2
    # First chunk is CHUNK_WORDS long
    assert len(out[0].split()) == CHUNK_WORDS
    # Second chunk starts at step = CHUNK_WORDS - CHUNK_OVERLAP
    step = CHUNK_WORDS - CHUNK_OVERLAP
    assert out[1].split()[0] == f"w{step}"
    # And contains the final word
    assert out[1].split()[-1] == f"w{total - 1}"


def test_overlap_is_as_documented():
    # Verify the configured overlap actually happens between adjacent chunks
    total = CHUNK_WORDS * 2 + 50
    text = " ".join(f"w{i}" for i in range(total))
    out = split_chunks(text)
    # At minimum the first two chunks should share CHUNK_OVERLAP words at the boundary
    first = out[0].split()
    second = out[1].split()
    # Last CHUNK_OVERLAP words of first chunk == first CHUNK_OVERLAP words of second chunk
    assert first[-CHUNK_OVERLAP:] == second[:CHUNK_OVERLAP]


def test_custom_chunk_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(30))
    out = split_chunks(text, chunk_words=10, overlap=2)
    # step = 8, so starts: 0, 8, 16, 24 — last chunk grabs what's left
    starts = [chunk.split()[0] for chunk in out]
    assert starts == ["w0", "w8", "w16", "w24"]


def test_overlap_equal_or_greater_than_chunk_falls_back_to_zero():
    # If someone misconfigures overlap >= chunk_words, the splitter must still
    # make forward progress (no infinite loop, no empty chunks)
    text = " ".join(f"w{i}" for i in range(50))
    out = split_chunks(text, chunk_words=10, overlap=15)   # nonsense config
    # Should degrade to non-overlapping 10-word chunks: 5 chunks
    assert len(out) == 5
    # And every chunk should have content
    assert all(c.strip() for c in out)


def test_no_empty_trailing_chunk():
    """Regression: splitter used to sometimes emit an empty last chunk."""
    text = " ".join(f"w{i}" for i in range(200))
    out = split_chunks(text, chunk_words=50, overlap=10)
    assert all(c.strip() for c in out), "empty chunk found"


def test_words_are_preserved_exactly():
    """Split is whitespace-based — no word should ever be cut mid-word."""
    text = "one two three four five six seven eight nine ten"
    out = split_chunks(text, chunk_words=3, overlap=1)
    # Reconstruct every word referenced in any chunk
    seen = set()
    for chunk in out:
        for word in chunk.split():
            seen.add(word)
    assert seen == {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"}


# ---------- split_chunks_with_offsets ----------------------------------------


def test_with_offsets_empty():
    assert split_chunks_with_offsets("") == []
    assert split_chunks_with_offsets("   \n\t  ") == []


def test_with_offsets_single_chunk_basic():
    text = "hello world there"
    out = split_chunks_with_offsets(text, chunk_words=10, overlap=0)
    assert len(out) == 1
    chunk_text, b_start, b_end = out[0]
    assert chunk_text == "hello world there"
    assert b_start == 0
    assert b_end == len(text)


def test_with_offsets_recover_first_and_last_word():
    """For every chunk, text[b_start:b_end] must START with the first chunk
    word and END with the last chunk word. This is the core guarantee the
    recall agent will rely on when paging by byte offset."""
    # Use varied whitespace (multiple spaces, newlines, tabs) to make sure
    # offsets track the original text, not the joined-words form.
    text = (
        "alpha  beta\nGamma\tdelta   epsilon "
        "zeta eta theta iota kappa "
        "lambda mu nu xi omicron pi rho "
        "sigma tau upsilon phi chi psi omega"
    )
    out = split_chunks_with_offsets(text, chunk_words=5, overlap=1)
    assert len(out) >= 4
    for chunk_text, b_start, b_end in out:
        slice_ = text[b_start:b_end]
        first_word = chunk_text.split()[0]
        last_word = chunk_text.split()[-1]
        assert slice_.lstrip().startswith(first_word), (
            f"slice {slice_!r} does not start with {first_word!r}"
        )
        assert slice_.rstrip().endswith(last_word), (
            f"slice {slice_!r} does not end with {last_word!r}"
        )


def test_with_offsets_byte_end_in_bounds():
    """No byte_end should ever exceed the length of the original text."""
    text = " ".join(f"w{i}" for i in range(200))
    out = split_chunks_with_offsets(text, chunk_words=50, overlap=10)
    n = len(text)
    for chunk_text, b_start, b_end in out:
        assert 0 <= b_start <= b_end <= n
        # And byte_end - byte_start is non-trivial
        assert b_end > b_start


def test_with_offsets_starts_advance_monotonically():
    """Each successive chunk's byte_start must be > the previous chunk's
    byte_start (we may overlap into the previous chunk's bytes, but we always
    advance forward by at least one word)."""
    text = " ".join(f"w{i}" for i in range(500))
    out = split_chunks_with_offsets(text, chunk_words=50, overlap=10)
    starts = [b_start for (_, b_start, _) in out]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts), "duplicate byte_starts found"


def test_split_chunks_back_compat_with_offsets():
    """split_chunks must return exactly the chunk-text components of
    split_chunks_with_offsets for the same inputs — proving the back-compat
    wrapper has zero behavioural drift from the offset-aware function.
    """
    cases = [
        "",
        "hello",
        " ".join(f"w{i}" for i in range(CHUNK_WORDS + 5)),
        " ".join(f"w{i}" for i in range(CHUNK_WORDS * 3 + 17)),
    ]
    for text in cases:
        a = split_chunks(text)
        b = [t for (t, _, _) in split_chunks_with_offsets(text)]
        assert a == b, f"divergence on text len={len(text)}"
