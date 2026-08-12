import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "our"))

from data_utils import pack_token_sequences


class DataUtilsTest(unittest.TestCase):
    def test_pack_inserts_eos_and_preserves_order(self):
        documents = [[1, 2], [3], [4, 5, 6]]
        blocks = list(pack_token_sequences(documents, 99, block_size=4, total_blocks=2))
        self.assertEqual(
            blocks,
            [
                {"input_ids": [1, 2, 99, 3]},
                {"input_ids": [99, 4, 5, 6]},
            ],
        )

    def test_pack_honors_block_budget(self):
        blocks = list(
            pack_token_sequences([[1, 2, 3, 4], [5, 6, 7, 8]], 9, 4, 1)
        )
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["input_ids"], [1, 2, 3, 4])

    def test_pack_handles_one_large_document_without_losing_tokens(self):
        document = list(range(100_000))
        blocks = list(pack_token_sequences([document], 100_000, 1024, 97))
        self.assertEqual(len(blocks), 97)
        self.assertEqual(blocks[0], {"input_ids": list(range(1024))})
        self.assertEqual(blocks[-1]["input_ids"][-1], 97 * 1024 - 1)


if __name__ == "__main__":
    unittest.main()
