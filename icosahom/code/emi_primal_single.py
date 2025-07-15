from mpi4py import MPI
from petsc4py import PETSc
import pandas
import dolfinx
import pathlib
from ufl import (
    inner,
    grad,
    TestFunctions,
    TrialFunctions,
    FacetNormal,
    MixedFunctionSpace,
    sin,
    pi,
    extract_blocks,
    Measure,
    SpatialCoordinate,
    cos,
    div,
)
import scifem
import numpy as np
import numpy.typing as npt
from packaging.version import Version
import time


x_L = 0.25
x_U = 0.75
y_L = 0.25
y_U = 0.75
z_L = 0.25
z_U = 0.75

from generate_mesh import generate_mesh

interior_marker = 2
exterior_marker = 3
interface_marker = 4
boundary_marker = 5


def solve_problem(
    num_refinements: int, degree: int, gdim: int
) -> dict[str, float | int | list[float]]:
    if gdim == 2:
        z_bounds = None
    elif gdim == 3:
        z_bounds = (z_L, z_U)
    else:
        raise ValueError(f"Invalid geometric dimension: {gdim}. Must be 2 or 3.")
    mesh, ct, ft = generate_mesh(
        MPI.COMM_WORLD,
        0,
        (x_L, x_U),
        (y_L, y_U),
        interior_marker=interior_marker,
        exterior_marker=exterior_marker,
        interface_marker=interface_marker,
        dirichlet_marker=boundary_marker,
        num_refinements=num_refinements,
        z_bounds=z_bounds,
    )

    num_cells_local = (
        mesh.topology.index_map(mesh.topology.dim).size_local
        + mesh.topology.index_map(mesh.topology.dim).num_ghosts
    )
    MPI.COMM_WORLD.Barrier()
    start_sm = time.perf_counter()
    omega_i, interior_to_parent, _, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim, ct.find(interior_marker)
    )
    omega_e, exterior_to_parent, e_vertex_to_parent, _ = dolfinx.mesh.create_submesh(
        mesh, mesh.topology.dim, ct.find(exterior_marker)
    )
    end_sm = time.perf_counter()
    submesh_creation = end_sm - start_sm

    # Integration measures for volumes
    dx = Measure("dx", domain=mesh, subdomain_data=ct)
    dxI = dx(interior_marker)
    dxE = dx(exterior_marker)

    # Create integration measure for interface
    # Interior marker is considered as ("+") restriction
    def compute_interface_data(
        cell_tags: dolfinx.mesh.MeshTags, facet_indices: npt.NDArray[np.int32]
    ) -> npt.NDArray[np.int32]:
        """
        Compute interior facet integrals that are consistently ordered according to the `cell_tags`,
        such that the data `(cell0, facet_idx0, cell1, facet_idx1)` is ordered such that
        `cell_tags[cell0]`<`cell_tags[cell1]`, i.e the cell with the lowest cell marker is considered the
        "+" restriction".

        Args:
            cell_tags: MeshTags that must contain an integer marker for all cells adjacent to the `facet_indices`
            facet_indices: List of facets (local index) that are on the interface.
        Returns:
            The integration data.
        """
        # Future compatibilty check
        integration_args: tuple[int] | tuple
        if Version("0.10.0") <= Version(dolfinx.__version__):
            integration_args = ()
        else:
            fdim = cell_tags.dim - 1
            integration_args = (fdim,)
        idata = dolfinx.cpp.fem.compute_integration_domains(
            dolfinx.fem.IntegralType.interior_facet,
            cell_tags.topology,
            facet_indices,
            *integration_args,
        )
        ordered_idata = idata.reshape(-1, 4).copy()
        switch = (
            cell_tags.values[ordered_idata[:, 0]]
            > cell_tags.values[ordered_idata[:, 2]]
        )
        if True in switch:
            ordered_idata[switch, :] = ordered_idata[switch][:, [2, 3, 0, 1]]
        return ordered_idata

    ordered_integration_data = compute_interface_data(ct, ft.find(interface_marker))
    parent_cells_plus = ordered_integration_data[:, 0]
    parent_cells_minus = ordered_integration_data[:, 2]
    mesh_to_interior = np.full(num_cells_local, -1, dtype=np.int32)
    mesh_to_interior[interior_to_parent] = np.arange(
        len(interior_to_parent), dtype=np.int32
    )
    mesh_to_exterior = np.full(num_cells_local, -1, dtype=np.int32)
    mesh_to_exterior[exterior_to_parent] = np.arange(
        len(exterior_to_parent), dtype=np.int32
    )

    entity_maps = {
        omega_i: mesh_to_interior,
        omega_e: mesh_to_exterior,
    }
    entity_maps[omega_i][parent_cells_minus] = entity_maps[omega_i][parent_cells_plus]
    entity_maps[omega_e][parent_cells_plus] = entity_maps[omega_e][parent_cells_minus]
    gamma_marker = 83
    dGamma = Measure(
        "dS",
        domain=mesh,
        subdomain_data=[(gamma_marker, ordered_integration_data.flatten())],
        subdomain_id=gamma_marker,
    )

    element = ("Lagrange", degree)
    Vi = dolfinx.fem.functionspace(omega_i, element)
    Ve = dolfinx.fem.functionspace(omega_e, element)
    W = MixedFunctionSpace(Vi, Ve)
    vi, ve = TestFunctions(W)
    ui, ue = TrialFunctions(W)

    sigma_e = dolfinx.fem.Constant(omega_e, 2.0)
    sigma_i = dolfinx.fem.Constant(omega_i, 1.0)
    Cm = dolfinx.fem.Constant(mesh, 1.0)
    dt = dolfinx.fem.Constant(mesh, 1.0e-2)

    # Setup variational form
    tr_ui = ui("+")
    tr_ue = ue("-")
    tr_vi = vi("+")
    tr_ve = ve("-")

    if mesh.geometry.dim == 2:
        x, y = SpatialCoordinate(mesh)
        ue_exact = sin(pi * (x + y))
        ui_exact = sigma_e / sigma_i * ue_exact + cos(pi * (x - x_L) * (x - x_U)) * cos(
            pi * (y - y_L) * (y - y_U)
        )
    else:
        x, y, z = SpatialCoordinate(mesh)
        ue_exact = sin(pi * (x + y + z))
        ui_exact = sigma_e / sigma_i * ue_exact + cos(pi * (x - x_L) * (x - x_U)) * cos(
            pi * (y - y_L) * (y - y_U)
        ) * cos(pi * (z - z_L) * (z - z_U))

    n = FacetNormal(mesh)
    n_e = n("-")
    Im = sigma_e * inner(grad(ue_exact), n_e)
    T = Cm / dt
    f = ui_exact - ue_exact - 1 / T * Im

    a = sigma_e * inner(grad(ue), grad(ve)) * dxE
    a += sigma_i * inner(grad(ui), grad(vi)) * dxI
    a += T * (tr_ue - tr_ui) * tr_ve * dGamma
    a += T * (tr_ui - tr_ue) * tr_vi * dGamma
    L = T * inner(f, (tr_vi - tr_ve)) * dGamma
    L -= div(sigma_e * grad(ue_exact)) * ve * dxE
    L -= div(sigma_i * grad(ui_exact)) * vi * dxI
    a_compiled = dolfinx.fem.form(extract_blocks(a), entity_maps=entity_maps)
    L_compiled = dolfinx.fem.form(extract_blocks(L), entity_maps=entity_maps)

    sub_tag, _ = scifem.mesh.transfer_meshtags_to_submesh(
        ft, omega_e, e_vertex_to_parent, exterior_to_parent
    )
    omega_e.topology.create_connectivity(omega_e.topology.dim - 1, omega_e.topology.dim)
    bc_dofs = dolfinx.fem.locate_dofs_topological(
        Ve, omega_e.topology.dim - 1, sub_tag.find(boundary_marker)
    )
    u_bc = dolfinx.fem.Function(Ve)
    u_bc.interpolate(
        lambda x: np.sin(np.pi * sum(x[i] for i in range(mesh.geometry.dim)))
    )

    bc = dolfinx.fem.dirichletbc(u_bc, bc_dofs)
    MPI.COMM_WORLD.Barrier()
    start_assembly = time.perf_counter()
    A = dolfinx.fem.petsc.assemble_matrix(a_compiled, kind="mpi", bcs=[bc])
    A.assemble()
    b = dolfinx.fem.petsc.assemble_vector(L_compiled, kind="mpi")
    bcs1 = dolfinx.fem.bcs_by_block(
        dolfinx.fem.extract_function_spaces(a_compiled, 1), [bc]
    )
    dolfinx.fem.petsc.apply_lifting(b, a_compiled, bcs=bcs1)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bcs0 = dolfinx.fem.bcs_by_block(
        dolfinx.fem.extract_function_spaces(L_compiled), [bc]
    )
    dolfinx.fem.petsc.set_bc(b, bcs0)

    P = sigma_e * inner(grad(ue), grad(ve)) * dxE
    P += sigma_i * inner(grad(ui), grad(vi)) * dxI
    P += inner(ui, vi) * dxI
    P_compiled = dolfinx.fem.form(extract_blocks(P), entity_maps=entity_maps)
    bc_P = dolfinx.fem.dirichletbc(0.0, bc_dofs, Ve)
    B = dolfinx.fem.petsc.assemble_matrix(P_compiled, kind="mpi", bcs=[bc_P])
    B.assemble()
    end_assembly = time.perf_counter()
    MPI.COMM_WORLD.Barrier()
    ksp = PETSc.KSP().create(mesh.comm)
    ksp.setOperators(A, B)
    ksp.setType("cg")
    ksp.getPC().setType("hypre")
    ksp.getPC().setHYPREType("boomeramg")
    ksp.setErrorIfNotConverged(True)
    # ksp.getPC().setType("lu")
    # ksp.getPC().setFactorSolverType("mumps")
    ksp.setTolerances(1e-8, 1e-8)
    # ksp.setMonitor(
    #     lambda ksp, its, rnorm: PETSc.Sys.Print(f"Iteration: {its}, residual: {rnorm}")
    # )
    ksp.setNormType(PETSc.KSP.NormType.NORM_PRECONDITIONED)
    ksp.setErrorIfNotConverged(True)

    ui = dolfinx.fem.Function(Vi)
    ue = dolfinx.fem.Function(Ve)
    x = b.duplicate()

    start_solve = time.perf_counter()
    ksp.solve(b, x)
    end_solve = time.perf_counter()

    x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    dolfinx.fem.petsc.assign(x, [ui, ue])
    num_iterations = ksp.getIterationNumber()
    converged_reason = ksp.getConvergedReason()
    print(f"Solver converged in: {num_iterations} with reason {converged_reason}")
    assert converged_reason > 0
    # with dolfinx.io.VTXWriter(omega_i.comm, "uh_i.bp", [ui], engine="BP5") as bp:
    #     bp.write(0.0)
    # with dolfinx.io.VTXWriter(omega_i.comm, "uh_e.bp", [ue], engine="BP5") as bp:
    #     bp.write(0.0)

    error_ui = dolfinx.fem.form(
        inner(ui - ui_exact, ui - ui_exact) * dxI, entity_maps=entity_maps
    )
    error_ue = dolfinx.fem.form(
        inner(ue - ue_exact, ue - ue_exact) * dxE, entity_maps=entity_maps
    )
    local_ui = dolfinx.fem.assemble_scalar(error_ui)
    local_ue = dolfinx.fem.assemble_scalar(error_ue)
    global_ui = np.sqrt(mesh.comm.allreduce(local_ui, op=MPI.SUM))
    global_ue = np.sqrt(mesh.comm.allreduce(local_ue, op=MPI.SUM))

    num_dofs_i = Vi.dofmap.index_map.size_global * Vi.dofmap.index_map_bs
    num_dofs_e = Ve.dofmap.index_map.size_global * Ve.dofmap.index_map_bs

    local_h = mesh.h(mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32))
    min_local_h = np.min(local_h)
    glob_min_h = mesh.comm.allreduce(min_local_h, op=MPI.MIN)

    if MPI.COMM_WORLD.rank == 0:
        print(
            f"Num dofs {num_dofs_e + num_dofs_i} {glob_min_h=:.2e} L2(ui): {global_ui:.2e}\n L2(ue): {global_ue:.2e}"
        )
    sm_times = MPI.COMM_WORLD.allgather(submesh_creation)
    print(sm_times)
    MPI.COMM_WORLD.Barrier()
    as_times = MPI.COMM_WORLD.allgather(end_assembly - start_assembly)
    MPI.COMM_WORLD.Barrier()
    sl_times = MPI.COMM_WORLD.allgather(end_solve - start_solve)
    MPI.COMM_WORLD.Barrier()

    ksp.destroy()
    A.destroy()
    B.destroy()
    b.destroy()
    x.destroy()

    return {
        "num_procs": mesh.comm.size,
        "num_dofs": num_dofs_e + num_dofs_i,
        "h": glob_min_h,
        "L2(ui)": float(global_ui),
        "L2(ue)": float(global_ue),
        "submesh_min": min(sm_times),
        "submesh_max": max(sm_times),
        "submesh_avg": sum(sm_times) / len(sm_times),
        "assembly_min": min(as_times),
        "assembly_max": max(as_times),
        "assembly_avg": sum(as_times) / len(as_times),
        "solve_min": min(sl_times),
        "solve_max": max(sl_times),
        "solve_avg": sum(sl_times) / len(sl_times),
        "num_iterations": num_iterations,
        "degree": degree,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-n",
        "--num-refinements",
        type=int,
        default=0,
        help="Number of mesh refinements",
    )
    parser.add_argument("-d", "--degree", type=int, default=1, help="Polynomial degree")
    parser.add_argument(
        "--gdim",
        type=int,
        default=3,
        choices=[2, 3],
        help="Geometric dimension of the problem",
    )
    args = parser.parse_args()
    results = {
        "num_procs": [],
        "num_dofs": [],
        "h": [],
        "L2(ui)": [],
        "L2(ue)": [],
        "submesh_min": [],
        "submesh_max": [],
        "submesh_avg": [],
        "assembly_min": [],
        "assembly_max": [],
        "assembly_avg": [],
        "solve_min": [],
        "solve_max": [],
        "solve_avg": [],
        "num_iterations": [],
        "degree": [],
    }
    for i in range(args.num_refinements + 1):
        MPI.COMM_WORLD.Barrier()
        data = solve_problem(i, args.degree, args.gdim)
        for key in results.keys():
            results[key].append(data[key])
    df = pandas.DataFrame.from_dict(results)

    result_file = pathlib.Path("results.csv")
    if MPI.COMM_WORLD.rank == 0:
        if result_file.exists():
            df.to_csv(result_file, mode="a", header=False, index=False)
        else:
            df.to_csv(result_file, mode="w", index=False)
        new_df = pandas.read_csv(result_file)
        PETSc.Sys.Print(new_df)
