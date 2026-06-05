#!/usr/bin/env python

############################################################################
#
# MODULE:       test_v_flexure
# AUTHOR:       Andrew Wickert
# PURPOSE:      Tests for v.flexure (point-load flexural isostasy)
# COPYRIGHT:    (C) 2026 by Andrew Wickert and the GRASS Development Team
#
#               This program is free software under the GNU General Public
#               License (>=v2). Read the file COPYING that comes with GRASS
#               for details.
#
############################################################################

"""
Tests for v.flexure.

Creates a synthetic single-point load vector (no NC dataset required) and
exercises the SAS_NG solver with scalar Te values. Skips automatically when
gFlex >= 2.0.0 is not installed.

Run inside a GRASS session (e.g., with --tmp-location XY):
    python -m grass.gunittest.main
"""

import unittest

import grass.script as grass
from grass.gunittest.case import TestCase
from grass.gunittest.main import test


def _gflex_ok():
    """Return True if gFlex is importable."""
    try:
        import gflex  # noqa: F401

        return True
    except ImportError:
        return False


@unittest.skipUnless(_gflex_ok(), "gFlex not available")
class TestVFlexure(TestCase):
    """Test v.flexure with a synthetic single-point load (no NC dataset required)."""

    loads = "test_vflex_loads"
    output = "test_vflex_rout"
    vector_output = "test_vflex_vout"

    @classmethod
    def setUpClass(cls):
        cls.use_temp_region()
        # 10×10 grid at 100 m resolution (1 km × 1 km)
        cls.runModule("g.region", n=1000, s=0, e=1000, w=0, res=100)
        # Single point at domain center (500, 500), loaded as a point force [N]
        # v.in.ascii format=point: default assigns cat=1 to first point
        cls.runModule(
            "v.in.ascii",
            format="point",
            input="-",
            stdin_="500 500",
            separator="space",
            output=cls.loads,
        )
        # Attach an attribute table, then add and set the load column
        cls.runModule("v.db.addtable", map=cls.loads)
        cls.runModule(
            "v.db.addcolumn",
            map=cls.loads,
            columns="q double precision",
        )
        cls.runModule(
            "v.db.update",
            map=cls.loads,
            column="q",
            value="1e15",
            where="cat=1",
        )

    @classmethod
    def tearDownClass(cls):
        cls.del_temp_region()
        cls.runModule(
            "g.remove",
            flags="f",
            type="vector",
            name=cls.loads,
            quiet=True,
        )

    def tearDown(self):
        self.runModule(
            "g.remove", flags="f", type="raster", name=self.output, quiet=True
        )
        self.runModule(
            "g.remove", flags="f", type="vector", name=self.vector_output, quiet=True
        )

    def test_basic_deflection(self):
        """SAS_NG solver produces a valid raster output."""
        self.assertModule(
            "v.flexure",
            input=self.loads,
            column="q",
            te="10000",
            te_units="m",
            output=self.output,
        )
        self.assertRasterExists(self.output)
        # All 100 output cells should be non-null (grid covers full region)
        self.assertRasterFitsUnivar(
            raster=self.output, reference={"n": 100}, precision=0
        )
        # Interface-layer sign check: deflection under a downward load must be
        # negative, and must be physically plausible (not orders of magnitude
        # off due to a Te unit conversion bug or wrong grid-spacing scale).
        stats = grass.parse_command("r.univar", map=self.output, flags="g")
        min_w = float(stats["min"])
        self.assertLess(min_w, 0,
                        "Deflection under a downward load must be negative")
        self.assertGreater(min_w, -1000,
                           "Deflection magnitude must be physically plausible (< 1 km)")

    def test_vector_output(self):
        """vector_output option produces a valid vector alongside the raster."""
        self.assertModule(
            "v.flexure",
            input=self.loads,
            column="q",
            te="10000",
            te_units="m",
            output=self.output,
            vector_output=self.vector_output,
        )
        self.assertRasterExists(self.output)
        self.assertVectorExists(self.vector_output)

    def test_te_km_units(self):
        """Te in km produces output without error."""
        self.assertModule(
            "v.flexure",
            input=self.loads,
            column="q",
            te="10",
            te_units="km",
            output=self.output,
        )
        self.assertRasterExists(self.output)

    def test_custom_material_params(self):
        """Non-default Young's modulus and mantle density are accepted."""
        self.assertModule(
            "v.flexure",
            input=self.loads,
            column="q",
            te="10000",
            te_units="m",
            output=self.output,
            ym="70E9",
            rho_m="3200",
        )
        self.assertRasterExists(self.output)

    def test_multi_point_loads(self):
        """Two load points are processed correctly; exercises the SQL attribute loop.

        Interface-layer test: the SQL UPDATE loop in v.flexure writes one row
        per output grid point. Using two input loads (rather than one) ensures
        both the coordinate-pair iteration and multi-row SQL batches are covered.
        """
        loads2 = "test_vflex_loads2"
        output2 = "test_vflex_rout2"
        try:
            self.runModule(
                "v.in.ascii",
                format="point",
                input="-",
                stdin_="400 400\n600 600",
                separator="space",
                output=loads2,
            )
            self.runModule("v.db.addtable", map=loads2)
            self.runModule(
                "v.db.addcolumn", map=loads2, columns="q double precision"
            )
            self.runModule(
                "v.db.update", map=loads2, column="q", value="1e15", where="cat=1"
            )
            self.runModule(
                "v.db.update", map=loads2, column="q", value="8e14", where="cat=2"
            )
            self.assertModule(
                "v.flexure",
                input=loads2,
                column="q",
                te="10000",
                te_units="m",
                output=output2,
            )
            self.assertRasterExists(output2)
        finally:
            self.runModule(
                "g.remove", flags="f", type="vector", name=loads2, quiet=True,
            )
            self.runModule(
                "g.remove", flags="f", type="raster", name=output2, quiet=True,
            )


if __name__ == "__main__":
    test()
