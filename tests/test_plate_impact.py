from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cfemp.benchmark import run_benchmark
from cfemp.plate_impact import (
    CFEMPPlateImpact1D,
    PlateImpactConfig,
    analytical_contact_stress,
    analytical_separation_time,
)


class TestPaperPlateImpact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.metrics = run_benchmark(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_paper_analytical_values(self):
        config = PlateImpactConfig()
        self.assertAlmostEqual(
            analytical_separation_time(config) * 1e6,
            8.6389101877,
            places=8,
        )
        self.assertAlmostEqual(
            analytical_contact_stress(config) / 1e6,
            -1336.9741957,
            places=6,
        )

    def test_initial_total_momentum_is_zero(self):
        solver = CFEMPPlateImpact1D()
        scale = (
            solver.config.density
            * solver.config.area
            * solver.config.length
            * solver.config.impact_speed
        )
        self.assertLess(abs(solver.total_momentum()) / scale, 1e-14)

    def test_stress_profile_matches_analytical_solution(self):
        self.assertLess(self.metrics["stress_relative_error"], 0.03)

    def test_separation_time_matches_paper(self):
        self.assertLess(self.metrics["separation_relative_error"], 0.02)

    def test_momentum_and_energy_regression(self):
        self.assertLess(
            self.metrics["max_normalized_momentum_error"], 1e-12
        )
        self.assertLess(self.metrics["max_relative_energy_error"], 0.055)

    def test_reproduction_artifacts_are_created(self):
        for name in (
            "stress_profile_3us.png",
            "energy_evolution.png",
            "contact_separation.png",
            "metrics.json",
            "history.npz",
        ):
            path = self.output / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(path.stat().st_size, 0, name)


if __name__ == "__main__":
    unittest.main()
