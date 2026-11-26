import unittest

from app import batch_rows


class BatchSizeChange(unittest.TestCase):
    def test_index_uses_four_rows(self):
        self.assertEqual(batch_rows("index", list("abcdefg")), [list("abcd"), list("efg")])

    def test_accounting_keeps_its_batch_boundaries(self):
        self.assertEqual(batch_rows("accounting", list("abcdefg")), [list("abc"), list("def"), ["g"]])


if __name__ == "__main__":
    unittest.main()
