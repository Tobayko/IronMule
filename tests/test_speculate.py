import unittest

from friday_hardware import (
    Generation,
    ProfileError,
    accepted_prefix,
    find_continuation,
)
from tests.test_hardware_profile import profile


class ContinuationTests(unittest.TestCase):
    def test_takes_the_most_recent_match(self):
        # Two candidates; the later one is the one the text is currently repeating.
        self.assertEqual(
            find_continuation([1, 2, 3, 7, 0, 0, 1, 2, 3, 8, 0, 1, 2, 3], 3, 1), [8]
        )

    def test_returns_what_followed_the_match(self):
        self.assertEqual(find_continuation([1, 2, 3, 9, 9, 1, 2, 3], 3, 2), [9, 9])

    def test_proposes_nothing_without_a_match(self):
        self.assertEqual(find_continuation([5, 6, 7, 8], 3, 3), [])

    def test_truncates_rather_than_padding_near_the_end(self):
        self.assertEqual(find_continuation([1, 2, 3, 1, 2, 3], 3, 5), [1, 2, 3])

    def test_proposes_nothing_when_asked_for_nothing_or_given_too_little(self):
        self.assertEqual(find_continuation([1, 2, 3, 4], 3, 0), [])
        self.assertEqual(find_continuation([1, 2], 3, 2), [])

    def test_a_longer_window_never_matches_where_a_shorter_one_did_not(self):
        tokens = [4, 1, 2, 3, 9, 4, 1, 2, 3]
        short = find_continuation(tokens, 2, 1)
        long = find_continuation(tokens, 4, 1)
        self.assertTrue(short, "the short window should match here")
        self.assertTrue(long, "and so should the long one on this text")
        # Same continuation: both anchor on the same recent occurrence.
        self.assertEqual(short, long)

    def test_refuses_nonsensical_arguments(self):
        for bad in ((0, 2), (3, -1)):
            with self.assertRaises(ValueError):
                find_continuation([1, 2, 3, 4], *bad)


class AcceptanceTests(unittest.TestCase):
    def test_counts_the_agreeing_prefix_only(self):
        self.assertEqual(accepted_prefix([1, 2, 3], [1, 2, 3]), 3)
        self.assertEqual(accepted_prefix([1, 2, 3], [1, 9, 3]), 1)
        self.assertEqual(accepted_prefix([1, 2, 3], [9, 2, 3]), 0)

    def test_a_later_agreement_after_a_miss_does_not_count(self):
        # Splicing [3] onto a prefix the model rejected would change the answer.
        self.assertEqual(accepted_prefix([7, 8, 9], [7, 0, 9]), 1)

    def test_empty_inputs_accept_nothing(self):
        self.assertEqual(accepted_prefix([], [1, 2]), 0)
        self.assertEqual(accepted_prefix([1, 2], []), 0)


class GenerationRecordTests(unittest.TestCase):
    def test_acceptance_is_undefined_when_nothing_was_drafted(self):
        g = Generation(tokens=[1, 2], seconds=0.1, steps=2, drafted=0, accepted=0,
                       ngram=3, draft_length=0)
        self.assertIsNone(g.acceptance)
        self.assertAlmostEqual(g.tokens_per_step, 1.0)

    def test_acceptance_and_yield_are_reported_together(self):
        g = Generation(tokens=list(range(9)), seconds=0.5, steps=4, drafted=8,
                       accepted=6, ngram=3, draft_length=2)
        self.assertAlmostEqual(g.acceptance, 0.75)
        self.assertAlmostEqual(g.tokens_per_step, 2.25)


class ProfileLookupTests(unittest.TestCase):
    def test_a_one_token_window_is_refused(self):
        # Measured at 0.987x on agent context: it matches everywhere and predicts
        # almost nothing, below the break-even the width curve sets.
        with self.assertRaises(ProfileError):
            profile(lookup_ngram=1)

    def test_lookup_parameters_survive_a_round_trip(self):
        p = profile(lookup_ngram=8, lookup_draft=3)
        restored = type(p).from_dict(p.as_dict())
        self.assertEqual(restored.lookup_ngram, 8)
        self.assertEqual(restored.lookup_draft, 3)

    def test_refuses_a_negative_draft(self):
        with self.assertRaises(ProfileError):
            profile(lookup_draft=-1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class RewindReportingTests(unittest.TestCase):
    def test_fallback_steps_are_reported(self):
        # Speculation needs the rejected tokens taken back out of the cache. Gemma 3
        # keeps most layers in a rotating cache that stops being rewindable past its
        # window (512 tokens on the 1B, 1024 on the 4B), and the generator falls back
        # to plain decoding there rather than corrupting the answer. The count has to
        # surface, or a run that never drafted looks the same as one that chose not to.
        g = Generation(tokens=[1, 2, 3], seconds=0.3, steps=3, drafted=0, accepted=0,
                       ngram=3, draft_length=2, unrewindable_steps=3)
        self.assertEqual(g.unrewindable_steps, 3)
        self.assertIsNone(g.acceptance)

    def test_a_run_that_could_draft_reports_no_fallbacks(self):
        g = Generation(tokens=[1, 2, 3], seconds=0.3, steps=2, drafted=2, accepted=2,
                       ngram=3, draft_length=2)
        self.assertEqual(g.unrewindable_steps, 0)
        self.assertAlmostEqual(g.acceptance, 1.0)
