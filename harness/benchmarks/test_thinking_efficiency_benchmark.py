from __future__ import annotations

import unittest

from thinking_efficiency_benchmark import (
    detect_loop,
    detect_repeated_ngram,
    extract_response,
    summarize_rows,
)


class LoopDetectionTests(unittest.TestCase):
    def test_detects_three_non_overlapping_ngrams(self) -> None:
        block = "we need to reconsider this assumption before choosing the next valid step now"
        reasoning = f"First attempt. {block}. Another attempt. {block}. Final attempt. {block}."

        signal = detect_repeated_ngram(reasoning, ngram_size=12, min_repeats=3)

        self.assertIsNotNone(signal)
        self.assertGreaterEqual(signal["count"], 3)

    def test_does_not_flag_short_normal_reasoning(self) -> None:
        reasoning = "Compute the product, check the arithmetic, and return the result."

        result = detect_loop(reasoning)

        self.assertFalse(result["detected"])
        self.assertEqual(result["signals"], [])

    def test_detects_repeated_sentence_span(self) -> None:
        sentence = "The current branch contradicts the second constraint and must be discarded."
        reasoning = f"{sentence}\n{sentence}\n{sentence}"

        result = detect_loop(reasoning, ngram_size=20)

        self.assertTrue(result["detected"])
        self.assertTrue(any(signal["kind"] == "repeated_span" for signal in result["signals"]))


class ResponseExtractionTests(unittest.TestCase):
    def test_prefers_lucebox_thinking_accounting(self) -> None:
        response = {
            "choices": [
                {
                    "message": {"content": "42", "reasoning_content": "six times seven"},
                    "finish_reason": "stop",
                    "finish_details": {
                        "close_kind": "natural",
                        "thinking_tokens": 9,
                        "content_tokens": 2,
                    },
                }
            ],
            "usage": {"completion_tokens": 11},
        }

        result = extract_response(response)

        self.assertEqual(result["reasoning_text"], "six times seven")
        self.assertEqual(result["content"], "42")
        self.assertEqual(result["thinking_tokens"], 9)
        self.assertEqual(result["content_tokens"], 2)
        self.assertEqual(result["thinking_token_source"], "server")

    def test_extracts_inline_think_fallback(self) -> None:
        response = {
            "choices": [
                {
                    "message": {"content": "<think>check twice</think>The answer is 4."},
                    "finish_reason": "stop",
                }
            ]
        }

        result = extract_response(response)

        self.assertEqual(result["reasoning_text"], "check twice")
        self.assertEqual(result["content"], "The answer is 4.")
        self.assertEqual(result["thinking_token_source"], "approx_words")


class SummaryTests(unittest.TestCase):
    def test_max_token_failure_is_not_automatically_a_loop(self) -> None:
        rows = [
            {
                "thinking_tokens": 7000,
                "elapsed_s": 10.0,
                "doom_loop_detected": False,
                "overthinking_detected": True,
                "hit_max_tokens": True,
                "has_final_answer": False,
                "gold_correct": None,
            }
        ]

        summary = summarize_rows(rows)

        self.assertEqual(summary["doom_loops"], 0)
        self.assertEqual(summary["doom_loop_rate"], 0.0)
        self.assertEqual(summary["hit_max_tokens"], 1)
        self.assertEqual(summary["overthinking"], 1)


if __name__ == "__main__":
    unittest.main()
