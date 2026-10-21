"""Unit tests for `TPTBox.segmentation.nnUnet_utils.predictor`.

The predictor is a TPTBox-maintained fork of nnU-Net's inference engine. No weights
and no GPU are involved here: the "network" is a plain callable, which is enough to
pin the test-time-augmentation contract.
"""

from __future__ import annotations

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "requires torch")
class TestMaybeMirrorAndPredict(unittest.TestCase):
    """`_internal_maybe_mirror_and_predict` averages the prediction over all axis flips."""

    @staticmethod
    def _predictor(use_mirroring: bool = True):
        import torch  # noqa: PLC0415

        from TPTBox.segmentation.nnUnet_utils.predictor import nnUNetPredictor  # noqa: PLC0415

        p = nnUNetPredictor(device=torch.device("cpu"), use_mirroring=use_mirroring, allow_tqdm=False)
        p.allowed_mirroring_axes = (0, 1, 2)
        return p

    @staticmethod
    def _x():
        import torch  # noqa: PLC0415

        return torch.arange(2 * 4 * 5 * 6, dtype=torch.float32).reshape(1, 2, 4, 5, 6)

    def test_uses_the_network_argument_for_every_tta_term(self):
        # Regression: the first forward pass used the `network` argument while the seven
        # mirrored passes used `self.network`. Both are the same object on every code path
        # today, so results were unaffected -- but the argument was silently ignored, which
        # is exactly what `loaded_networks` (one network per fold) is meant to exercise.
        import torch  # noqa: PLC0415

        p = self._predictor()
        p.network = lambda t: torch.zeros_like(t)  # must NOT be consulted
        x = self._x()
        out = p._internal_maybe_mirror_and_predict(x, network=lambda t: t.clone())
        torch.testing.assert_close(out, x)

    def test_mirroring_averages_eight_flips(self):
        import torch  # noqa: PLC0415

        p = self._predictor()
        x = self._x()
        # A network that returns a constant c makes every one of the 8 terms c.
        out = p._internal_maybe_mirror_and_predict(x, network=lambda t: torch.full_like(t, 3.0))
        torch.testing.assert_close(out, torch.full_like(x, 3.0))

    def test_asymmetric_network_matches_hand_symmetrised_average(self):
        import itertools  # noqa: PLC0415

        import torch  # noqa: PLC0415

        def net(t):
            return t * 2.0 + 1.0

        p = self._predictor()
        x = self._x()
        out = p._internal_maybe_mirror_and_predict(x, network=net)

        axes = [2, 3, 4]
        combos = [c for i in range(len(axes)) for c in itertools.combinations(axes, i + 1)]
        expected = net(x)
        for c in combos:
            expected = expected + torch.flip(net(torch.flip(x, c)), c)
        expected = expected / (len(combos) + 1)
        torch.testing.assert_close(out, expected)

    def test_mirroring_disabled_is_a_single_forward(self):
        import torch  # noqa: PLC0415

        p = self._predictor(use_mirroring=False)
        p.network = lambda t: torch.zeros_like(t)
        x = self._x()
        out = p._internal_maybe_mirror_and_predict(x, network=lambda t: t * 5.0)
        torch.testing.assert_close(out, x * 5.0)


if __name__ == "__main__":
    unittest.main()
