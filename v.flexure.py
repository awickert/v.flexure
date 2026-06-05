#!/usr/bin/env python
############################################################################
#
# MODULE:       v.flexure
#
# AUTHOR(S):    Andrew Wickert
#
# PURPOSE:      Calculate flexure of the lithosphere under a specified
#               set of loads and with a given elastic thickness (scalar)
#
# COPYRIGHT:    (c) 2014, 2015, 2026 Andrew Wickert
#
#               This program is free software under the GNU General Public
#               License (>=v2). Read the file COPYING that comes with GRASS
#               for details.
#
#############################################################################
#
# REQUIREMENTS:
#      -  gFlex: https://github.com/awickert/gFlex

# More information
# Started 20 Jan 2015 to add GRASS GIS support for distributed point loads
# and their effects on lithospheric flexure

# %module
# % description: Lithospheric flexure: gridded deflections from scattered point loads
# % keyword: vector
# % keyword: geophysics
# %end

# %option G_OPT_V_INPUT
# %  key: input
# %  description: Vector map of loads (thickness * area * density * g) [N]
# %  guidependency: layer,column
# %end

# %option G_OPT_V_FIELD
# %  key: layer
# %  description: Layer containing load values
# %  guidependency: column
# %end

# %option G_OPT_DB_COLUMNS
# %  key: column
# %  description: Column containing load values [N]
# %  required : yes
# %end

# %option
# %  key: te
# %  type: double
# %  description: Elastic thickness: scalar; units chosen in "te_units"
# %  required : yes
# %end

# %option
# %  key: te_units
# %  type: string
# %  description: Units for elastic thickness
# %  options: m, km
# %  required : yes
# %end

# %option G_OPT_V_OUTPUT
# %  key: output
# %  description: Output vector points map of vertical deflections [m]
# %  required : yes
# %end

# %option G_OPT_R_OUTPUT
# %  key: raster_output
# %  description: Output raster map of vertical deflections [m]
# %  required : no
# %  guisection: Output
# %end

# %option
# %  key: g
# %  type: double
# %  description: gravitational acceleration at surface [m/s^2]
# %  answer: 9.8
# %  required : no
# %  guisection: Material properties
# %end

# %option
# %  key: ym
# %  type: double
# %  description: Young's Modulus [Pa]
# %  answer: 65E9
# %  required : no
# %  guisection: Material properties
# %end

# %option
# %  key: nu
# %  type: double
# %  description: Poisson's ratio
# %  answer: 0.25
# %  required : no
# %  guisection: Material properties
# %end

# %option
# %  key: rho_fill
# %  type: double
# %  description: Density of material that fills flexural depressions [kg/m^3]
# %  answer: 0
# %  required : no
# %  guisection: Material properties
# %end

# %option
# %  key: rho_m
# %  type: double
# %  description: Mantle density [kg/m^3]
# %  answer: 3300
# %  required : no
# %  guisection: Material properties
# %end


##################
# IMPORT MODULES #
##################

# PYTHON
import os
import tempfile
import warnings

import numpy as np

# GRASS
import grass.script as grass


####################
# UTILITY FUNCTION #
####################


def get_points_xy(vect_name):
    """Return (x, y) coordinate arrays for all points in a vector map."""
    out = grass.read_command(
        "v.out.ascii", input=vect_name, format="point",
        separator="space", quiet=True
    )
    rows = [line.split() for line in out.strip().splitlines() if line.strip()]
    coords = np.array([[float(r[0]), float(r[1])] for r in rows], dtype=float)
    return coords[:, 0], coords[:, 1]  # x, y


############################
# PASS VARIABLES AND SOLVE #
############################


def main():
    """
    Superposition of analytical solutions in gFlex for flexural isostasy in
    GRASS GIS
    """

    options, flags = grass.parser()
    # if just interface description is requested, it will not get to this point
    # so gflex will not be needed

    # Import gFlex only after we know we will actually do the computation
    try:
        import gflex
    except ImportError:
        grass.fatal(
            _(
                "Cannot import gFlex. Install it from source with:\n"
                "  pip install -e /path/to/gFlex\n"
                "or see https://github.com/awickert/gFlex for details."
            )
        )

    _gver = tuple(
        int(x.split("a")[0].split("b")[0].split("rc")[0])
        for x in gflex.__version__.split(".")[:3]
    )
    if _gver < (2, 0, 0):
        grass.fatal(
            _("v.flexure requires gFlex >= 2.0.0; installed: ")
            + gflex.__version__
        )

    ##########
    # SET-UP #
    ##########

    # This code is for 2D flexural isostasy
    flex = gflex.F2D()
    # And show that it is coming from GRASS GIS
    flex.grass = True

    # Method
    flex.method = "sas_ng"

    # Parameters that are often changed for the solution
    ######################################################

    # x, y, q
    flex.x, flex.y = get_points_xy(options["input"])
    # xw, yw: gridded output
    if len(grass.parse_command("g.list", type="vect", pattern=options["output"])):
        if not grass.overwrite():
            grass.fatal(
                _("Vector map <%s> already exists. Use '--o' to overwrite.")
                % options["output"]
            )
    # Just check raster at the same time if it exists
    if len(
        grass.parse_command("g.list", type="rast", pattern=options["raster_output"])
    ):
        if not grass.overwrite():
            grass.fatal(
                _("Raster map <%s> already exists. Use '--o' to overwrite.")
                % options["raster_output"]
            )
    grass.run_command(
        "v.mkgrid",
        map=options["output"],
        type="point",
        overwrite=grass.overwrite(),
        quiet=True,
    )
    grass.run_command(
        "v.db.addcolumn",
        map=options["output"],
        columns="w double precision",
        quiet=True,
    )
    flex.xw, flex.yw = get_points_xy(options["output"])  # gridded output coordinates
    vect_db = grass.vector_db_select(options["input"])
    col_names = np.array(vect_db["columns"])
    q_col = col_names == options["column"]
    if np.sum(q_col):
        col_values = np.array(list(vect_db["values"].values())).astype(float)
        flex.q = col_values[:, q_col].reshape(-1)  # always 1-D, even for a single point
    else:
        grass.fatal(
            _("Column <%s> not found in vector map <%s>.")
            % (options["column"], options["input"])
        )
    # Elastic thickness
    flex.T_e = float(options["te"])
    if options["te_units"] == "km":
        flex.T_e *= 1000
    elif options["te_units"] == "m":
        pass
    else:
        grass.fatal(_("Inappropriate te_units; this should not be reachable."))
    flex.rho_fill = float(options["rho_fill"])

    # Parameters that often stay at their default values
    ######################################################
    flex.g = float(options["g"])
    flex.E = float(
        options["ym"]
    )  # Can't just use "E" because reserved for "east", I think
    flex.nu = float(options["nu"])
    flex.rho_m = float(options["rho_m"])

    # Set verbosity
    if grass.verbosity() >= 2:
        flex.verbose = True
    if grass.verbosity() >= 3:
        flex.debug = True
    elif grass.verbosity() == 0:
        flex.quiet = True

    # Check if lat/lon and let user know if verbosity is True
    if grass.region_env()[6] == "3":
        flex.latlon = True
        flex.planetary_radius = float(grass.parse_command("g.proj", flags="j")["+a"])
        if flex.verbose:
            grass.message(_("Latitude/longitude grid."))
            grass.message(_("Based on r_Earth = 6371 km"))
            grass.message(
                _("Computing distances between load points using great circle paths")
            )

    ##########
    # SOLVE! #
    ##########

    grass.message(_("Computing flexural deflections..."))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flex.initialize()
        flex.run()
        # finalize() deletes flex.w in gFlex v2, so capture it first
        w_out = list(flex.w)
        flex.finalize()
    for warninfo in caught:
        grass.warning(str(warninfo.message))

    # Write deflection values to the output vector's attribute table.
    # v.mkgrid assigns sequential cats (1, 2, ...) in row-major order,
    # matching the order of flex.w returned by gFlex.
    table_name = options["output"].split("@")[0]
    sql_lines = [
        "UPDATE {t} SET w = {val} WHERE cat = {cat};".format(
            t=table_name, val=float(w_out[i]), cat=i + 1
        )
        for i in range(len(w_out))
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        f.write("\n".join(sql_lines))
        sql_file = f.name
    try:
        grass.run_command("db.execute", input=sql_file, quiet=True)
    finally:
        os.unlink(sql_file)
    grass.run_command("v.build", map=options["output"], quiet=True)

    # And raster export
    # "w" vector defined by raster resolution, so can do direct v.to.rast
    # though if this option isn't selected, the user can do a finer-grained
    # interpolation, which shouldn't introduce much error so long as these
    # outputs are spaced at << 1 flexural wavelength.
    if options["raster_output"]:
        grass.run_command(
            "v.to.rast",
            input=options["output"],
            output=options["raster_output"],
            use="attr",
            attribute_column="w",
            type="point",
            overwrite=grass.overwrite(),
            quiet=True,
        )
        # And create a nice colormap!
        grass.run_command(
            "r.colors", map=options["raster_output"], color="differences", quiet=True
        )


if __name__ == "__main__":
    main()
