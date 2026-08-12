"""Dependency-light helpers shared by data preparation and unit tests."""
from __future__ import annotations


def pack_token_sequences(token_sequences, eos_token_id, block_size, total_blocks):
    """Pack document token sequences with EOS using bounded memory."""
    buffer: list[int] = []
    offset = 0
    emitted = 0
    for token_ids in token_sequences:
        buffer.extend(token_ids)
        buffer.append(eos_token_id)
        while len(buffer) - offset >= block_size and emitted < total_blocks:
            end = offset + block_size
            yield {"input_ids": buffer[offset:end]}
            offset = end
            emitted += 1
        if emitted >= total_blocks:
            return
        # Compact geometrically instead of shifting the whole list per block.
        if offset and (offset >= 65_536 or offset * 2 >= len(buffer)):
            buffer = buffer[offset:]
            offset = 0
