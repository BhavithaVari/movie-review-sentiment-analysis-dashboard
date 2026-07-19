import unittest

from csv_columns import find_text_column


class FindTextColumnTests(unittest.TestCase):
    def test_prefers_existing_review_column(self):
        self.assertEqual(find_text_column(["text", "review"]), "review")

    def test_detects_normalized_xquik_text_column(self):
        self.assertEqual(find_text_column(["created_at", " Full_Text "]), " Full_Text ")

    def test_returns_none_without_supported_column(self):
        self.assertIsNone(find_text_column(["created_at", "author"]))


if __name__ == "__main__":
    unittest.main()
