from __future__ import annotations

import unittest

import numpy as np

from examples.render_single_fcc_cell import basis_indices
from md_Helpers.lattices import build_fcc_lattice


class SingleFCCCellExampleTests(unittest.TestCase):
    def test_each_basis_particle_gets_a_unique_color_index(self):
        lattice = build_fcc_lattice(n_cells=1, density=0.3)
        indices = basis_indices(lattice.positions, lattice.box_length)

        np.testing.assert_array_equal(np.sort(indices), np.arange(4))


if __name__ == "__main__":
    unittest.main()
