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

from grass.gunittest.case import TestCase
from grass.gunittest.main import test


def _gflex_ok():
    """Return True if gFlex >= 2.0.0 is importable."""
    try:
        import gflex

        def _ver(v):
            try:
                return tuple(int(x) for x in v.split(".")[:3])
            except (ValueError, AttributeError):
                return (0, 0, 0)

        return _ver(gflex.__version__) >= (2, 0, 0)
    except ImportError:
        return False


@unittest.skipUnless(_gflex_ok(), "gFlex >= 2.0.0 not available")
class TestVFlexure(TestCase):
    """Test v.flexure with a synthetic single-point load (no NC dataset required)."""

    loads = "test_vflex_loads"
    output = "test_vflex_out"
    raster_output = "test_vflex_rout"

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
            "g.remove", flags="f", type="vector", name=self.output, quiet=True
        )
        self.runModule(
            "g.remove", flags="f", type="raster", name=self.raster_output, quiet=True
        )

    def test_basic_deflection(self):
        """SAS_NG solver produces a valid deflection vector output."""
        self.assertModule(
            "v.flexure",
            input=self.loads,
            column="q",
            te="10000",
            te_units="m",
            output=self.output,
        )
        self.assertVectorExists(self.output)

    def test_raster_output(self):
        """raster_output option produces a valid raster alongside the vector."""
        self.assertModule(
            "v.flexure",
            input=self.loads,
            column="q",
            te="10000",
            te_units="m",
            output=self.output,
            raster_output=self.raster_output,
        )
        self.assertVectorExists(self.output)
        self.assertRasterExists(self.raster_output)
        # All 100 output cells should be non-null (grid covers full region)
        self.assertRasterFitsUnivar(
            raster=self.raster_output, reference={"n": 100}, precision=0
        )

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
        self.assertVectorExists(self.output)

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
        self.assertVectorExists(self.output)


if __name__ == "__main__":
    test()
