from __future__ import annotations

import math
import unittest

import torch

from baseline.common import BILOU_TAGS
from baseline.predict import _decode_records
from baseline.transition_confidence import (
    add_span_confidence,
    estimate_transition_priors,
    parse_transition_priors,
    viterbi_decode,
)


class TransitionConfidenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.id2label = dict(enumerate(BILOU_TAGS))
        self.tag_to_id = {tag: index for index, tag in self.id2label.items()}

    def test_viterbi_keeps_bilou_path_legal(self) -> None:
        probabilities = torch.full((2, len(BILOU_TAGS)), 1e-4)
        probabilities[0, self.tag_to_id["B-ORG"]] = 0.9
        probabilities[1, self.tag_to_id["O"]] = 0.9
        probabilities[1, self.tag_to_id["L-ORG"]] = 0.1

        path = viterbi_decode(probabilities, self.id2label)

        self.assertEqual(
            [self.id2label[index] for index in path], ["B-ORG", "L-ORG"]
        )

    def test_learned_start_end_priors_can_break_ambiguous_tie(self) -> None:
        payload = estimate_transition_priors(
            [["U-ORG"]] * 10 + [["O"]], BILOU_TAGS, smoothing=1.0
        )
        priors = parse_transition_priors(payload)
        probabilities = torch.full((1, len(BILOU_TAGS)), 1e-6)
        probabilities[0, self.tag_to_id["O"]] = 0.55
        probabilities[0, self.tag_to_id["U-ORG"]] = 0.45

        plain_path = viterbi_decode(probabilities, self.id2label)
        prior_path = viterbi_decode(
            probabilities, self.id2label, priors=priors, prior_scale=1.0
        )

        self.assertEqual(self.id2label[plain_path[0]], "O")
        self.assertEqual(self.id2label[prior_path[0]], "U-ORG")

    def test_estimator_rejects_illegal_gold_transition(self) -> None:
        with self.assertRaisesRegex(ValueError, "illegal transition"):
            estimate_transition_priors(
                [["B-ORG", "O"]], BILOU_TAGS, smoothing=1.0
            )

    def test_span_confidence_uses_selected_entity_tokens(self) -> None:
        offsets = [(0, 3), (4, 7), (8, 10)]
        label_ids = [
            self.tag_to_id["B-NAME"],
            self.tag_to_id["L-NAME"],
            self.tag_to_id["O"],
        ]
        probabilities = []
        for label_id, confidence in zip(label_ids, (0.81, 0.49, 0.95), strict=True):
            row = torch.zeros(len(BILOU_TAGS))
            row[label_id] = confidence
            probabilities.append(row)

        result = add_span_confidence(
            [{"label": "NAME", "start": 0, "end": 7}],
            offsets,
            probabilities,
            label_ids,
            self.id2label,
        )

        self.assertEqual(result[0]["token_count"], 2)
        self.assertAlmostEqual(result[0]["_confidence"], 0.49, places=6)
        self.assertAlmostEqual(result[0]["mean_token_confidence"], 0.65, places=6)
        self.assertAlmostEqual(
            result[0]["geometric_mean_token_confidence"],
            math.sqrt(0.81 * 0.49),
            places=6,
        )

    def test_decode_keeps_confidence_out_of_official_predictions(self) -> None:
        row = torch.full((len(BILOU_TAGS),), 1e-6)
        row[self.tag_to_id["U-NAME"]] = 0.9
        confidence_records: list[dict[str, object]] = []

        predictions = _decode_records(
            [{"hash": "doc", "text": "Ali"}],
            [{(0, 3): (row, 1.0)}],
            self.id2label,
            "viterbi",
            confidence_records=confidence_records,
        )

        self.assertEqual(
            predictions,
            [
                {
                    "hash": "doc",
                    "entities": [{"label": "NAME", "start": 0, "end": 3}],
                }
            ],
        )
        self.assertNotIn("_confidence", predictions[0]["entities"][0])
        self.assertAlmostEqual(
            confidence_records[0]["entities"][0]["_confidence"], 0.9, places=6
        )


if __name__ == "__main__":
    unittest.main()
