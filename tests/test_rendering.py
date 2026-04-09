from __future__ import annotations

import unittest

from mkpi_app.rendering import normalize_math_markdown, split_markdown_and_math


class RenderingTests(unittest.TestCase):
    def test_normalize_bracket_aligned_block_for_streamlit(self) -> None:
        source = r"""
Шаг 1.

[ \begin{aligned} 37 \times 24 &= (30 + 7) \times 24 \ &= 30 \times 24 + 7 \times 24 \ &= 720 + 168 \ &= 888. \end{aligned} ]
"""
        normalized = normalize_math_markdown(source)
        self.assertIn("$$", normalized)
        self.assertIn(r"\begin{aligned}", normalized)
        self.assertIn(r"\\ &=", normalized)

    def test_split_markdown_and_math_detects_display_formula(self) -> None:
        source = r"Итог: \[ \boxed{730} \]"
        parts = split_markdown_and_math(source)
        self.assertEqual(parts[0][0], "text")
        self.assertEqual(parts[1][0], "math")
        self.assertIn(r"\boxed{730}", parts[1][1])


if __name__ == "__main__":
    unittest.main()
