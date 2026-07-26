from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cfemp.benchmark_3d import run_benchmark_3d
from cfemp.plate_impact_3d import (
    PlateImpact3DConfig,
    _hex8_mesh,
    _hex8_quadrature,
    analytical_contact_stress_3d,
    analytical_separation_time_3d,
)


class TestThreeDimensionalMesh(unittest.TestCase):
    def test_hex8_geometry(self):
        config = PlateImpact3DConfig()
        nodes, elements, surface, lock_y, lock_z, _ = _hex8_mesh(config)
        gradients, det_j = _hex8_quadrature(nodes, elements)
        self.assertEqual(nodes.shape, (2107, 3))
        self.assertEqual(elements.shape, (1512, 8))
        self.assertEqual(surface.shape, (49,))
        self.assertEqual(gradients.shape, (1512, 8, 8, 3))
        self.assertTrue(np.all(det_j > 0.0))
        self.assertAlmostEqual(
            float(np.sum(det_j)),
            config.length * config.width * config.depth,
            places=15,
        )
        self.assertTrue(np.any(lock_y))
        self.assertTrue(np.any(lock_z))


class TestPaperPlateImpact3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.metrics = run_benchmark_3d(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_paper_values(self):
        config = PlateImpact3DConfig()
        self.assertAlmostEqual(
            analytical_separation_time_3d(config) * 1e6,
            8.6389101877,
            places=8,
        )
        self.assertAlmostEqual(
            analytical_contact_stress_3d(config) / 1e6,
            -1336.9741957,
            places=6,
        )

    def test_taichi_three_dimensional_state(self):
        self.assertEqual(self.metrics["dimensions"], 3)
        self.assertEqual(self.metrics["backend"], "taichi-cpu")
        self.assertEqual(self.metrics["fem_nodes"], 2107)
        self.assertEqual(self.metrics["fem_hex8_elements"], 1512)
        self.assertEqual(self.metrics["mpm_particles"], 12096)
        self.assertEqual(self.metrics["velocity_components"], 3)
        self.assertEqual(self.metrics["stress_components"], 6)
        self.assertEqual(self.metrics["element_nodes"], 8)

    def test_plate_impact_errors(self):
        self.assertLess(self.metrics["stress_relative_error"], 0.03)
        self.assertLess(self.metrics["separation_relative_error"], 0.01)
        self.assertLess(self.metrics["max_relative_energy_error"], 0.035)
        self.assertLess(self.metrics["max_penetration_um"], 5.0)

    def test_interface_balance(self):
        self.assertLess(
            self.metrics["max_normalized_momentum_error"], 1.0e-11
        )
        self.assertLess(
            self.metrics["max_contact_impulse_balance_kg_m_s"], 1.0e-15
        )

    def test_transverse_plane_strain(self):
        self.assertLess(
            self.metrics["max_transverse_velocity_m_s"], 1.0e-9
        )
        self.assertLess(
            self.metrics["max_transverse_stress_spread"], 1.0e-10
        )

    def test_artifacts(self):
        for name in (
            "discretization_3d.png",
            "stress_slice_3us.png",
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
