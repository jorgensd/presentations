
# # Create a DG scheme coupling two subdomains with different function spaces
# # using interior penalty method (Nitsche's method) on the interface.
# Authors: Jørgen S. Dokken and Joseph P. Dean
# SPDX-License-Identifier: MIT

from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
import numpy.typing as npt
import scifem


with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "logo.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
    ct = xdmf.read_meshtags(mesh, name="cell_tags")
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    ft = xdmf.read_meshtags(mesh, name="facet_tags")

# Split domain in half and set an interface tag of 5

boundary_facets = ft.find(2)
gdim = mesh.geometry.dim
tdim = mesh.topology.dim
fdim = tdim - 1

submesh_b, submesh_b_to_mesh, b_v_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(0)
)[0:3]
submesh_t, submesh_t_to_mesh, t_v_map = dolfinx.mesh.create_submesh(
    mesh, tdim, ct.find(1)
)[0:3]
fmap = mesh.topology.index_map(fdim)
num_facets_local = fmap.size_local + fmap.num_ghosts
cellmap = mesh.topology.index_map(mesh.topology.dim)
num_cells_local = cellmap.size_local + cellmap.num_ghosts

# We need to modify the cell maps, as for `dS` integrals of interfaces between submeshes, there is no entity to map to.
# We use the entity on the same side to fix this (as all restrictions are one-sided)

# Transfer meshtags to submesh
ft_b, b_facet_to_parent = scifem.transfer_meshtags_to_submesh(
    ft, submesh_b, b_v_map, submesh_b_to_mesh
)
ft_t, t_facet_to_parent = scifem.transfer_meshtags_to_submesh(
    ft, submesh_t, t_v_map, submesh_t_to_mesh
)


V_0 = dolfinx.fem.functionspace(submesh_b, ("Lagrange", 4))
V_1 = dolfinx.fem.functionspace(submesh_t, ("Lagrange", 2))
W = ufl.MixedFunctionSpace(V_0, V_1)

u_0 = dolfinx.fem.Function(V_0, name="u_b")
u_1 = dolfinx.fem.Function(V_1, name="u_t")
v_0, v_1 = ufl.TestFunctions(W)

dx = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

F = ufl.inner(ufl.grad(u_0), ufl.grad(v_0)) * dx(0) - 2.0 * v_0 * dx(0)
F += ufl.inner(ufl.grad(u_1), ufl.grad(v_1)) * dx(1) - 5.0 * v_1 * dx(1)


# Add coupling term to the interface
# Get interface markers on submesh b
idata = scifem.compute_interface_data(ct, ft.find(3))
dInterface = ufl.Measure("dS", domain=mesh, subdomain_data=[(3, idata.flatten())], subdomain_id=3)
b_res = "+"
t_res = "-"

v_b = v_0(b_res)
v_t = v_1(t_res)
u_b = u_0(b_res)
u_t = u_1(t_res)


def mixed_term(u, v, nh):
    return ufl.dot(ufl.grad(u), nh) * v


nh = ufl.FacetNormal(mesh)
n_b = nh(b_res)
n_t = nh(t_res)
cr = ufl.Circumradius(mesh)
Q = dolfinx.fem.functionspace(mesh, ("DG", 0))
cr = dolfinx.fem.Function(Q)

cr.x.array[:] = mesh.h(mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32))
h_b = 2 * cr(b_res)
h_t = 2 * cr(t_res)
gamma = 100.0



F += (
    -0.5 * mixed_term((u_b + u_t), v_b, n_b) * dInterface
    - 0.5 * mixed_term(v_b, (u_b - u_t), n_b) * dInterface
)

F += (
    +0.5 * mixed_term((u_b + u_t), v_t, n_b) * dInterface
    - 0.5 * mixed_term(v_t, (u_b - u_t), n_b) * dInterface
)
F += 2 * gamma / (h_b + h_t) * (u_b - u_t) * v_b * dInterface
F+= -2 * gamma / (h_b + h_t) * (u_b - u_t) * v_t * dInterface



b_bc = dolfinx.fem.Function(u_0.function_space)
b_bc.x.array[:] = 0.
submesh_b.topology.create_connectivity(
    submesh_b.topology.dim - 1, submesh_b.topology.dim
)
bc_b = dolfinx.fem.dirichletbc(
    b_bc, dolfinx.fem.locate_dofs_topological(u_0.function_space, fdim, ft_b.find(2))
)


t_bc = dolfinx.fem.Function(u_1.function_space)
t_bc.x.array[:] = 0.05
submesh_t.topology.create_connectivity(
    submesh_t.topology.dim - 1, submesh_t.topology.dim
)
bc_t = dolfinx.fem.dirichletbc(
    t_bc, dolfinx.fem.locate_dofs_topological(u_1.function_space, fdim, ft_t.find(2))
)
bcs = [bc_b, bc_t]

solver = dolfinx.fem.petsc.NonlinearProblem(
    ufl.extract_blocks(F),
    u=[u_0, u_1],
    bcs=bcs,
    entity_maps=[submesh_b_to_mesh, submesh_t_to_mesh],
    petsc_options_prefix="logo_dg_",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_monitor": None,
        "snes_rtol": 1e-10,
        "snes_linesearch_type": "none",
        "snes_type": "newtonls",
        "ksp_monitor_true_residual": None,
        "snes_error_if_not_converged": True,
        "ksp_error_if_not_converged": True,
    },
)
solver.solve()

bp = dolfinx.io.VTXWriter(mesh.comm, "u_b.bp", [u_0], engine="BP4")
bp.write(0)
bp.close()
bp = dolfinx.io.VTXWriter(mesh.comm, "u_t.bp", [u_1], engine="BP4")
bp.write(0)
bp.close()
