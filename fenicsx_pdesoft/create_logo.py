# Meshing script from
# https://github.com/jpdean/mixed_domain_demos/blob/main/lagrange_multiplier.py
# by Joseph Dean
# SPDX license identifier: MIT
# Modified by Jørgen S. Dokken

import gmsh
from mpi4py import MPI

from dolfinx import io, mesh

def create_mesh(comm, h, d, order, vol_ids, bound_ids):
    """
    Create a mesh with the FEniCS logo if d == 2 and a box containing
    a sphere in 3D.
        Parameters:
            comm: MPI communicator
            h: maximum cell diameter
            d: geometric dimension
            order: Geometric order
            vol_ids: Dictionary with tags for volumes
            bound_ids: Dictionary with tags for boundaries
    """

    gmsh.initialize()
    if comm.rank == 0:
        if d == 2:
            gmsh.model.add("square_with_fenics_logo")

            factory = gmsh.model.geo

            # Corners of the square
            square_points = [
                factory.addPoint(0.0, 0.0, 0.0, h),
                factory.addPoint(1.0, 0.0, 0.0, h),
                factory.addPoint(1.0, 1.0, 0.0, h),
                factory.addPoint(0.0, 1.0, 0.0, h)
            ]

            # Points for the first closed loop in the FEniCS logo
            logo_points_0 = [
                factory.addPoint(0.6017391304347826,
                                 0.7981132075471697, 0.0, h),
                factory.addPoint(0.5782608695652174,
                                 0.7584905660377359, 0.0, h),
                factory.addPoint(0.56, 0.730188679245283, 0.0, h),
                factory.addPoint(0.5417391304347826,
                                 0.6924528301886792, 0.0, h),
                factory.addPoint(0.5417391304347826,
                                 0.6584905660377358, 0.0, h),
                factory.addPoint(0.5730434782608695,
                                 0.6320754716981132, 0.0, h),
                factory.addPoint(0.6252173913043478,
                                 0.5962264150943397, 0.0, h),
                factory.addPoint(0.6643478260869564,
                                 0.5641509433962264, 0.0, h),
                factory.addPoint(0.7008695652173913,
                                 0.5320754716981132, 0.0, h),
                factory.addPoint(0.7373913043478262,
                                 0.5037735849056604, 0.0, h),
                factory.addPoint(0.7660869565217392,
                                 0.469811320754717, 0.0, h),
                factory.addPoint(0.7895652173913044,
                                 0.4377358490566038, 0.0, h),
                factory.addPoint(0.8, 0.4037735849056604, 0.0, h),
                factory.addPoint(0.8052173913043479,
                                 0.369811320754717, 0.0, h),
                factory.addPoint(0.7843478260869565,
                                 0.34150943396226413, 0.0, h),
                factory.addPoint(0.7530434782608695,
                                 0.3150943396226415, 0.0, h),
                factory.addPoint(0.708695652173913,
                                 0.29245283018867924, 0.0, h),
                factory.addPoint(0.6434782608695652,
                                 0.26037735849056604, 0.0, h),
                factory.addPoint(0.586086956521739,
                                 0.23962264150943396, 0.0, h),
                factory.addPoint(0.5234782608695652,
                                 0.22075471698113208, 0.0, h),
                factory.addPoint(0.4791304347826087,
                                 0.20566037735849058, 0.0, h),
                factory.addPoint(0.4191304347826087,
                                 0.19811320754716977, 0.0, h),
                factory.addPoint(0.46608695652173915,
                                 0.23962264150943396, 0.0, h),
                factory.addPoint(0.5026086956521739,
                                 0.27169811320754716, 0.0, h),
                factory.addPoint(0.5365217391304348,
                                 0.3018867924528302, 0.0, h),
                factory.addPoint(0.557391304347826,
                                 0.3339622641509434, 0.0, h),
                factory.addPoint(0.557391304347826,
                                 0.36415094339622645, 0.0, h),
                factory.addPoint(0.528695652173913,
                                 0.4018867924528302, 0.0, h),
                factory.addPoint(0.4921739130434783,
                                 0.44716981132075473, 0.0, h),
                factory.addPoint(0.46608695652173915,
                                 0.4811320754716981, 0.0, h),
                factory.addPoint(0.44, 0.5150943396226415, 0.0, h),
                factory.addPoint(0.4165217391304348,
                                 0.5509433962264151, 0.0, h),
                factory.addPoint(0.39304347826086955,
                                 0.5867924528301887, 0.0, h),
                factory.addPoint(0.3826086956521739,
                                 0.6113207547169812, 0.0, h),
                factory.addPoint(0.3826086956521739,
                                 0.6396226415094339, 0.0, h),
                factory.addPoint(0.4165217391304348,
                                 0.6735849056603773, 0.0, h),
                factory.addPoint(0.45565217391304347,
                                 0.7037735849056603, 0.0, h),
                factory.addPoint(0.5078260869565218,
                                 0.7415094339622641, 0.0, h),
                factory.addPoint(0.5469565217391305,
                                 0.7641509433962264, 0.0, h)
            ]

            # Points for the second closed loop in the FEniCS logo
            logo_points_1 = [
                factory.addPoint(0.30695652173913046,
                                 0.5792452830188679, 0.0, h),
                factory.addPoint(0.2808695652173913,
                                 0.5509433962264151, 0.0, h),
                factory.addPoint(0.24434782608695654,
                                 0.5075471698113208, 0.0, h),
                factory.addPoint(0.22086956521739132,
                                 0.47358490566037736, 0.0, h),
                factory.addPoint(0.20782608695652174,
                                 0.41132075471698115, 0.0, h),
                factory.addPoint(0.20521739130434782,
                                 0.3754716981132076, 0.0, h),
                factory.addPoint(0.23391304347826086,
                                 0.3452830188679245, 0.0, h),
                factory.addPoint(0.2782608695652174,
                                 0.3169811320754717, 0.0, h),
                factory.addPoint(0.3173913043478261,
                                 0.2962264150943396, 0.0, h),
                factory.addPoint(0.3669565217391304,
                                 0.27735849056603773, 0.0, h),
                factory.addPoint(0.3956521739130435,
                                 0.2679245283018868, 0.0, h),
                factory.addPoint(0.4217391304347826,
                                 0.28679245283018867, 0.0, h),
                factory.addPoint(0.44, 0.30377358490566037, 0.0, h),
                factory.addPoint(0.48434782608695653,
                                 0.33962264150943394, 0.0, h),
                factory.addPoint(0.5026086956521739,
                                 0.36415094339622645, 0.0, h),
                factory.addPoint(0.4973913043478261,
                                 0.3905660377358491, 0.0, h),
                factory.addPoint(0.4608695652173913,
                                 0.4339622641509434, 0.0, h),
                factory.addPoint(0.4295652173913044,
                                 0.4660377358490566, 0.0, h),
                factory.addPoint(0.3956521739130435,
                                 0.4962264150943396, 0.0, h),
                factory.addPoint(0.3669565217391304,
                                 0.5245283018867924, 0.0, h),
                factory.addPoint(0.3356521739130435,
                                 0.5547169811320755, 0.0, h)
            ]

            # Boundary of square
            square_lines = [
                factory.addLine(square_points[0], square_points[1]),
                factory.addLine(square_points[1], square_points[2]),
                factory.addLine(square_points[2], square_points[3]),
                factory.addLine(square_points[3], square_points[0])
            ]

            # Approximate the first closed loop in the logo as two splines
            logo_lines_0 = []
            logo_lines_0.append(factory.addSpline(logo_points_0[0:22]))
            logo_lines_0.append(factory.addSpline(
                logo_points_0[21:] + [logo_points_0[0]]))

            # Approximate the second closed loop in the logo as two splines
            logo_lines_1 = []
            logo_lines_1.append(factory.addSpline(logo_points_1[0:11]))
            logo_lines_1.append(factory.addSpline(
                logo_points_1[10:] + [logo_points_1[0]]))

            # Create curves
            square_curve = factory.addCurveLoop(square_lines)
            logo_curve_0 = factory.addCurveLoop(logo_lines_0)
            logo_curve_1 = factory.addCurveLoop(logo_lines_1)

            # Create surfaces
            square_surface = factory.addPlaneSurface(
                [square_curve, logo_curve_0, logo_curve_1])
            circle_surface = factory.addPlaneSurface([logo_curve_0])
            logo_surface_1 = factory.addPlaneSurface([logo_curve_1])

            factory.synchronize()

            # Add 2D physical groups
            gmsh.model.addPhysicalGroup(
                2, [square_surface], vol_ids["omega_0"])
            gmsh.model.addPhysicalGroup(
                2, [circle_surface, logo_surface_1], vol_ids["omega_1"])

            # Add 1D physical groups
            gmsh.model.addPhysicalGroup(1, square_lines, bound_ids["gamma"])
            gmsh.model.addPhysicalGroup(1, logo_lines_0 + logo_lines_1,
                                        bound_ids["gamma_i"])
        elif d == 3:
            gmsh.model.add("box_with_sphere")
            box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
            sphere = gmsh.model.occ.addSphere(0.5, 0.5, 0.5, 0.25)
            ov, ovv = gmsh.model.occ.fragment([(3, box)], [(3, sphere)])

            gmsh.model.occ.synchronize()

            # Add physical groups
            gmsh.model.addPhysicalGroup(3, [ov[0][1]], vol_ids["omega_0"])
            gmsh.model.addPhysicalGroup(3, [ov[1][1]], vol_ids["omega_1"])
            gamma_dim_tags = gmsh.model.getBoundary([ov[0], ov[1]])
            gamma_i_dim_tags = gmsh.model.getBoundary([ov[0]])
            gmsh.model.addPhysicalGroup(
                2, [surface[1] for surface in gamma_dim_tags],
                bound_ids["gamma"])
            gmsh.model.addPhysicalGroup(
                2, [surface[1] for surface in gamma_i_dim_tags],
                bound_ids["gamma_i"])

            # Assign a mesh size to all the points:
            gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

        gmsh.model.mesh.generate(d)
        gmsh.model.mesh.setOrder(order)

    partitioner = mesh.create_cell_partitioner(mesh.GhostMode.none)
    msh, ct, ft = io.gmshio.model_to_mesh(
        gmsh.model, comm, 0, gdim=d, partitioner=partitioner)
    gmsh.finalize()
    return msh, ct, ft


if __name__ == "__main__":

    # Tags for volumes and boundaries
    vol_ids = {"omega_0": 0,
            "omega_1": 1}
    bound_ids = {"gamma": 2,  # Boundary
                "gamma_i": 3}  # Interface
    h = 0.05
    d = 2
    order = 2
    assert MPI.COMM_WORLD.size == 1, "Script should only be executed in serial"
    msh, ct, ft = create_mesh(MPI.COMM_SELF, h, d, order, vol_ids, bound_ids)
    with io.XDMFFile(MPI.COMM_SELF, "logo.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
        xdmf.write_meshtags(ct, msh.geometry)
        xdmf.write_meshtags(ft, msh.geometry)