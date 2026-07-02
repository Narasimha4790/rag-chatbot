from typing import List

class RecursiveCharacterTextSplitter:
    """
    Splits text recursively based on a list of separators (e.g. paragraph newlines,
    line newlines, spaces, characters) to keep related contextual units intact
    while staying under a specified chunk size and maintaining text overlap.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Splits the input text recursively using separators and merges chunks with overlap."""
        if not text:
            return []
        
        # Step 1: Split text into small atom-like parts recursively
        splits = self._split_text_recursive(text, self.separators)
        
        # Step 2: Merge the split parts into final chunks with the defined size and overlap
        chunks = []
        current_doc = []
        current_len = 0
        
        for split in splits:
            split_len = len(split)
            # If a single split item is larger than chunk_size, output it directly
            if split_len > self.chunk_size:
                if current_doc:
                    chunks.append("".join(current_doc))
                    current_doc = []
                    current_len = 0
                chunks.append(split)
                continue
                
            # If adding this split exceeds chunk_size, save current chunk and apply overlap
            if current_len + split_len > self.chunk_size:
                chunks.append("".join(current_doc))
                
                # Apply overlap: keep trailing parts of current_doc that fit within the overlap
                overlap_doc = []
                overlap_len = 0
                # Iterate backwards to find splits that fit into overlap
                for item in reversed(current_doc):
                    if overlap_len + len(item) <= self.chunk_overlap:
                        overlap_doc.insert(0, item)
                        overlap_len += len(item)
                    else:
                        break
                current_doc = overlap_doc
                current_len = overlap_len
                
            current_doc.append(split)
            current_len += split_len
            
        if current_doc:
            chunks.append("".join(current_doc))
            
        # Clean chunks: strip whitespaces and ignore empty chunks
        return [c.strip() for c in chunks if c.strip()]

    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Splits text using the first separator in the list, then recursively processes with remaining."""
        if not separators:
            # Base case: if no separators left, split character by character if still too large
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            
        separator = separators[0]
        splits = []
        
        # If separator is empty string, split into characters
        if separator == "":
            return list(text)
            
        # Split by separator
        parts = text.split(separator)
        
        # Re-attach the separator to all but the last part to preserve spacing structure
        for i, part in enumerate(parts):
            if i < len(parts) - 1:
                part_with_sep = part + separator
            else:
                part_with_sep = part
                
            if len(part_with_sep) <= self.chunk_size:
                splits.append(part_with_sep)
            else:
                # Recursively split this large part using the next separators
                splits.extend(self._split_text_recursive(part_with_sep, separators[1:]))
                
        return splits
