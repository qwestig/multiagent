from __future__ import annotations

import unittest

from mkpi_app.utils import keyword_coverage, marker_group_coverage, matches_marker


class TextMatchingTests(unittest.TestCase):
    def test_matches_marker_handles_russian_inflections(self) -> None:
        answer = "Назначить ответственного преподавателя и провести проверку готовности."
        self.assertTrue(matches_marker(answer, "ответственный"))
        self.assertTrue(matches_marker(answer, "проверка"))
        self.assertTrue(matches_marker(answer, "преподаватель"))

    def test_keyword_coverage_matches_multiword_variants(self) -> None:
        answer = "План включает подготовить кейсы, провести проверку качества и резервный сценарий."
        found_count, total, found = keyword_coverage(
            answer,
            ["подготовка кейсов", "проверка качества", "резервный сценарий"],
        )
        self.assertEqual(total, 3)
        self.assertEqual(found_count, 3)
        self.assertEqual(len(found), 3)

    def test_marker_group_coverage_uses_fuzzy_matching(self) -> None:
        answer = "Вывод: сначала назначить ответственного, затем провести контрольную точку готовности."
        matched, total, groups = marker_group_coverage(
            answer,
            {
                "owner": ["ответственный", "координатор"],
                "checkpoint": ["контрольная точка", "критерий готовности"],
            },
        )
        self.assertEqual(total, 2)
        self.assertEqual(matched, 2)
        self.assertEqual(groups, ["owner", "checkpoint"])


if __name__ == "__main__":
    unittest.main()
