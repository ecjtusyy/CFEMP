from __future__ import annotations

import unittest

import numpy as np

from mpm_fem.common import hat_weights_and_grads
from mpm_fem.contact import PenaltyContact, solve_contact_equilibrium
from mpm_fem.fem import FEMWallBeam
from mpm_fem.mpm import SmallStrainMPM2D


def one_particle_problem():
    mpm = SmallStrainMPM2D(
        (0.0, 0.0),
        0.1,
        11,
        11,
        density=1800.0,
        young=2.0e6,
        poisson=0.3,
        gravity=0.0,
    )
    radius = mpm.add_rect_particles(
        0.52,
        0.521,
        0.05,
        0.051,
        particles_per_cell=2,
    )
    wall = FEMWallBeam(
        0.5,
        0.0,
        1.0,
        10,
        young=30.0e9,
        inertia=0.3**3 / 12.0,
    )
    contact = PenaltyContact(
        2.0e6,
        radius,
        damping=0.0,
    )
    return mpm, wall, contact


class TestPenaltyContact(unittest.TestCase):
    def test_shape_functions_form_partition_of_unity(self):
        nodes, weights, grads = hat_weights_and_grads(
            np.array([0.37, 0.62]),
            np.array([0.0, 0.0]),
            0.1,
            11,
            11,
        )
        self.assertEqual(len(nodes), 4)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=14)
        np.testing.assert_allclose(np.sum(grads, axis=0), 0.0, atol=1e-14)

    def test_contact_pushes_particle_away_from_wall(self):
        mpm, wall, contact = one_particle_problem()
        particle_force, result = contact.evaluate(mpm, wall)
        self.assertEqual(result.active_particles, 1)
        self.assertGreater(particle_force[0, 0], 0.0)
        self.assertLess(result.wall_resultant[0], 0.0)
        self.assertLess(result.balance_error, 1e-10)

    def test_subiterations_do_not_accumulate_particle_force(self):
        mpm, wall, contact = one_particle_problem()
        one_force, _ = contact.evaluate(mpm, wall)
        result = solve_contact_equilibrium(
            mpm, wall, contact, iterations=5
        )
        # 墙很刚，迭代后的力与一次计算同量级，绝不能变成五倍。
        ratio = mpm.contact_force[0, 0] / one_force[0, 0]
        self.assertGreater(ratio, 0.8)
        self.assertLess(ratio, 1.2)
        self.assertLess(result.balance_error, 1e-9)

    def test_internal_force_is_self_equilibrated(self):
        mpm, _, _ = one_particle_problem()
        mpm.stress[0] = np.array([[-2.0e5, 0.0], [0.0, 0.0]])
        mpm.p2g(gravity_scale=0.0)
        np.testing.assert_allclose(
            np.sum(mpm.grid_internal_force, axis=0),
            0.0,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
