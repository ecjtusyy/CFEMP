from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cfemp.benchmark import run_benchmark
from cfemp.contact import project_normal_momentum
from cfemp.plate_impact import (
    CFEMPPlateImpact2D,
    MaterialPointPlate,
    PlateImpactConfig,
    Quad4Plate,
    analytical_contact_stress,
    analytical_separation_time,
)


class TestTwoDimensionalKernels(unittest.TestCase):
    def test_q4_affine_patch(self):
        config = PlateImpactConfig()
        plate = Quad4Plate(config)
        a, b, shear = 1.2, -0.4, 0.7
        velocity = np.column_stack(
            (
                a * plate.x0[:, 0] + shear * plate.x0[:, 1],
                b * plate.x0[:, 1],
            )
        )
        dt = 1.0e-6
        plate.update_stress(velocity, dt)
        expected_strain = dt * np.array([a, b, shear])
        expected_stress = config.constitutive_matrix @ expected_strain
        np.testing.assert_allclose(
            plate.strain,
            np.broadcast_to(expected_strain, plate.strain.shape),
            rtol=1.0e-12,
            atol=1.0e-15,
        )
        np.testing.assert_allclose(
            plate.stress,
            np.broadcast_to(expected_stress, plate.stress.shape),
            rtol=1.0e-12,
            atol=1.0e-6,
        )

    def test_mpm_affine_patch(self):
        config = PlateImpactConfig()
        plate = MaterialPointPlate(config)
        grid_position = np.array(
            [
                [x, y]
                for x in plate.grid_x
                for y in plate.grid_y
            ]
        )
        a, b, shear = 0.8, -0.3, 0.5
        velocity = np.column_stack(
            (
                a * grid_position[:, 0] + shear * grid_position[:, 1],
                b * grid_position[:, 1],
            )
        )
        dt = 1.0e-6
        plate.update_stress(velocity, dt)
        expected = config.constitutive_matrix @ (
            dt * np.array([a, b, shear])
        )
        np.testing.assert_allclose(
            plate.stress,
            np.broadcast_to(expected, plate.stress.shape),
            rtol=1.0e-12,
            atol=1.0e-6,
        )

    def test_oblique_projection(self):
        normal = np.array([1.0, 1.0]) / np.sqrt(2.0)
        fem_mass, mpm_mass = 2.0, 3.0
        fem_momentum = fem_mass * np.array([2.0, 0.5])
        mpm_momentum = mpm_mass * np.array([-1.0, -0.25])
        result = project_normal_momentum(
            fem_mass,
            fem_momentum,
            mpm_mass,
            mpm_momentum,
            normal,
        )
        self.assertTrue(result.active)
        fem_after = fem_momentum - result.impulse_on_mpm
        mpm_after = mpm_momentum + result.impulse_on_mpm
        relative = fem_after / fem_mass - mpm_after / mpm_mass
        self.assertAlmostEqual(float(np.dot(relative, normal)), 0.0, places=13)
        np.testing.assert_allclose(
            fem_after + mpm_after,
            fem_momentum + mpm_momentum,
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_state_dimensions(self):
        solver = CFEMPPlateImpact2D()
        self.assertEqual(solver.fem.x.shape, (301, 2))
        self.assertEqual(solver.fem.elements.shape, (252, 4))
        self.assertEqual(solver.fem.stress.shape, (252, 4, 3))
        self.assertEqual(solver.mpm.x.shape, (1008, 2))
        self.assertEqual(solver.mpm.stress.shape, (1008, 3))


class TestPaperPlateImpact2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name)
        cls.metrics = run_benchmark(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_paper_values(self):
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

    def test_two_dimensional_discretization(self):
        self.assertEqual(self.metrics["dimensions"], 2)
        self.assertEqual(self.metrics["fem_q4_elements"], 252)
        self.assertEqual(self.metrics["mpm_particles"], 1008)

    def test_plate_impact_errors(self):
        self.assertLess(self.metrics["stress_relative_error"], 0.03)
        self.assertLess(self.metrics["separation_relative_error"], 0.01)
        self.assertLess(self.metrics["max_relative_energy_error"], 0.035)

    def test_interface_balance(self):
        self.assertLess(
            self.metrics["max_normalized_momentum_error"], 1.0e-12
        )
        self.assertLess(
            self.metrics["max_contact_impulse_balance_kg_m_s"], 1.0e-15
        )

    def test_plane_strain_symmetry(self):
        self.assertLess(
            self.metrics["max_transverse_velocity_m_s"], 1.0e-9
        )
        self.assertLess(
            self.metrics["max_transverse_stress_spread"], 1.0e-10
        )

    def test_time_refinement(self):
        self.assertGreater(self.metrics["time_refinement_order"], 0.7)
        self.assertLess(self.metrics["time_refinement_order"], 1.3)

    def test_artifacts(self):
        for name in (
            "discretization_2d.png",
            "stress_field_3us.png",
            "stress_profile_3us.png",
            "energy_evolution.png",
            "contact_separation.png",
            "time_refinement.png",
            "time_refinement.csv",
            "metrics.json",
            "history.npz",
        ):
            path = self.output / name
            self.assertTrue(path.exists(), name)
            self.assertGreater(path.stat().st_size, 0, name)


if __name__ == "__main__":
    unittest.main()
