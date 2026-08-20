from collections import OrderedDict


def word_count(text: str) -> int:
    """
    Simple retrieval-oriented word count.

    This is deliberately tokenizer-independent. We will use the
    BGE-M3 tokenizer later when building embeddings if necessary.
    """
    return len(text.split())


def group_sentences(sentences: list[dict]):
    """
    Group sentences by document, page, and PyMuPDF block while
    preserving their existing order.

    Returns:

        (document_id, page_index, block_no)
            -> [sentence, sentence, ...]
    """
    groups = OrderedDict()

    for sentence in sentences:
        key = (
            sentence["document_id"],
            sentence["page_index"],
            sentence["logical_block_index"],
        )

        if key not in groups:
            groups[key] = []

        groups[key].append(sentence)

    return groups


def make_passage(
    sentences: list[dict],
    passage_index: int,
) -> dict:
    """
    Create one passage record from consecutive sentences.
    """
    if not sentences:
        raise ValueError("Cannot create an empty passage")

    first = sentences[0]
    last = sentences[-1]

    source_text = " ".join(sentence["text"].strip() for sentence in sentences)

    normalized_text = " ".join(
        sentence["text_normalized"].strip() for sentence in sentences
    )

    logical_block_index = first["logical_block_index"]

    sentence_ids = [sentence["sentence_id"] for sentence in sentences]

    passage_id = (
        f"{first['document_id']}:"
        f"p{first['page_number']:04d}:"
        f"lb{logical_block_index:04d}:"
        f"c{passage_index:03d}"
    )

    normalized_word_count = word_count(normalized_text)

    return {
        "passage_id": passage_id,
        "document_id": first["document_id"],
        "page_index": first["page_index"],
        "page_number": first["page_number"],
        "logical_block_index": logical_block_index,
        "block_passage_index": passage_index,
        "sentence_ids": sentence_ids,
        "first_page_sentence_index": (first["page_sentence_index"]),
        "last_page_sentence_index": (last["page_sentence_index"]),
        "sentence_count": len(sentences),
        "text": source_text,
        "text_normalized": normalized_text,
        "word_count": normalized_word_count,
        # Useful later for diagnosing noisy retrieval results.
        "retrieval_eligible": (normalized_word_count >= 8),
    }


def build_passages_for_block(
    sentences: list[dict],
    max_words: int,
    overlap_sentences: int,
) -> list[dict]:
    """
    Split consecutive sentences from one PDF block into passages.

    Passages never cross a block or page boundary.

    If a block exceeds max_words, consecutive passages overlap by
    overlap_sentences.
    """
    if not sentences:
        return []

    passages = []

    start = 0
    passage_index = 0

    while start < len(sentences):
        end = start
        current_words = 0

        while end < len(sentences):
            sentence_words = word_count(sentences[end]["text_normalized"])

            # Always include at least one sentence, even if that
            # sentence itself exceeds max_words.
            if end > start and current_words + sentence_words > max_words:
                break

            current_words += sentence_words
            end += 1

        passage_sentences = sentences[start:end]

        passages.append(
            make_passage(
                sentences=passage_sentences,
                passage_index=passage_index,
            )
        )

        passage_index += 1

        # Finished the block.
        if end >= len(sentences):
            break

        # Overlap the next passage by the requested number
        # of complete sentences.
        next_start = max(
            start + 1,
            end - overlap_sentences,
        )

        start = next_start

    return passages
