from backend.app.services.chunking import RecursiveCharacterTextSplitter

def test_empty_string():
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2)
    assert splitter.split_text("") == []

def test_basic_splitting():
    splitter = RecursiveCharacterTextSplitter(chunk_size=15, chunk_overlap=3)
    text = "Hello World! This is a simple test."
    chunks = splitter.split_text(text)
    
    # Verify that all chunks are at or below the threshold
    for chunk in chunks:
        assert len(chunk) <= 15
    # Verify that we extracted elements
    assert len(chunks) > 1
    assert "Hello World!" in chunks[0]

def test_overlap_retention():
    splitter = RecursiveCharacterTextSplitter(chunk_size=20, chunk_overlap=10)
    # The splitter should carry over words fitting inside overlap
    text = "WordOne WordTwo WordThree WordFour"
    chunks = splitter.split_text(text)
    
    # We expect some overlap between sequential chunks
    # Let's inspect that consecutive chunks share text
    assert len(chunks) >= 2
    # Verify the overlap is active: the end of chunk 0 should overlap with chunk 1
    chunk0_words = set(chunks[0].split())
    chunk1_words = set(chunks[1].split())
    intersection = chunk0_words.intersection(chunk1_words)
    assert len(intersection) >= 1
