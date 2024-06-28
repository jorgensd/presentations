# docker run -ti -v $(pwd):/root/shared -w /root/shared jpdean/mixed_domain


# SPDX-License-Identifier: MIT

from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc


class NewtonSolver:
    max_iterations: int
    bcs: list[dolfinx.fem.DirichletBC]
    A: PETSc.Mat
    b: PETSc.Vec
    J: dolfinx.fem.Form
    b: dolfinx.fem.Form
    dx: PETSc.Vec

    def __init__(
        self,
        F: list[dolfinx.fem.form],
        J: list[list[dolfinx.fem.form]],
        w: list[dolfinx.fem.Function],
        bcs: list[dolfinx.fem.DirichletBC] | None = None,
        max_iterations: int = 5,
        petsc_options: dict[str, str | float | int | None] = None,
    ):
        self.max_iterations = max_iterations
        self.bcs = [] if bcs is None else bcs
        self.b = dolfinx.fem.petsc.create_vector_block(F)
        self.F = F
        self.J = J
        self.A = dolfinx.fem.petsc.create_matrix_block(J)
        self.dx = self.A.createVecLeft()
        self.w = w
        self.x = dolfinx.fem.petsc.create_vector_block(F)

        # Set PETSc options
        opts = PETSc.Options()
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v

        # Define KSP solver
        self._solver = PETSc.KSP().create(self.b.getComm().tompi4py())
        self._solver.setOperators(self.A)
        self._solver.setFromOptions()

        # Set matrix and vector PETSc options
        self.A.setFromOptions()
        self.b.setFromOptions()

    def solve(self, tol=1e-6, beta=1.0):
        i = 0

        while i < self.max_iterations:
            dolfinx.cpp.la.petsc.scatter_local_vectors(
                self.x,
                [si.x.petsc_vec.array_r for si in self.w],
                [
                    (
                        si.function_space.dofmap.index_map,
                        si.function_space.dofmap.index_map_bs,
                    )
                    for si in self.w
                ],
            )
            self.x.ghostUpdate(
                addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
            )

            # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
            with self.b.localForm() as b_local:
                b_local.set(0.0)

            dolfinx.fem.petsc.assemble_vector_block(
                self.b, self.F, self.J, bcs=self.bcs, x0=self.x, scale=-1.0
            )

            self.b.ghostUpdate(
                PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD
            )

            # Assemble Jacobian
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
            self.A.assemble()

            self._solver.solve(self.b, self.dx)
            # self._solver.view()
            assert (
                self._solver.getConvergedReason() > 0
            ), "Linear solver did not converge"
            offset_start = 0
            for s in self.w:
                num_sub_dofs = (
                    s.function_space.dofmap.index_map.size_local
                    * s.function_space.dofmap.index_map_bs
                )
                s.x.petsc_vec.array_w[:num_sub_dofs] -= (
                    beta * self.dx.array_r[offset_start : offset_start + num_sub_dofs]
                )
                s.x.petsc_vec.ghostUpdate(
                    addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
                )
                offset_start += num_sub_dofs
            # Compute norm of update

            correction_norm = self.dx.norm(0)
            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            i += 1

    def __del__(self):
        self.A.destroy()
        self.b.destroy()
        self.dx.destroy()
        self._solver.destroy()
        self.x.destroy()


def transfer_meshtags_to_submesh(
    mesh, entity_tag, submesh, sub_vertex_to_parent, sub_cell_to_parent
):
    """
    Transfer a meshtag from a parent mesh to a sub-mesh.
    """

    tdim = mesh.topology.dim
    cell_imap = mesh.topology.index_map(tdim)
    num_cells = cell_imap.size_local + cell_imap.num_ghosts
    mesh_to_submesh = np.full(num_cells, -1)
    mesh_to_submesh[sub_cell_to_parent] = np.arange(
        len(sub_cell_to_parent), dtype=np.int32
    )
    sub_vertex_to_parent = np.asarray(sub_vertex_to_parent)

    submesh.topology.create_connectivity(entity_tag.dim, 0)

    num_child_entities = (
        submesh.topology.index_map(entity_tag.dim).size_local
        + submesh.topology.index_map(entity_tag.dim).num_ghosts
    )
    submesh.topology.create_connectivity(submesh.topology.dim, entity_tag.dim)

    c_c_to_e = submesh.topology.connectivity(submesh.topology.dim, entity_tag.dim)
    c_e_to_v = submesh.topology.connectivity(entity_tag.dim, 0)

    child_markers = np.full(num_child_entities, 0, dtype=np.int32)

    mesh.topology.create_connectivity(entity_tag.dim, 0)
    mesh.topology.create_connectivity(entity_tag.dim, mesh.topology.dim)
    p_f_to_v = mesh.topology.connectivity(entity_tag.dim, 0)
    p_f_to_c = mesh.topology.connectivity(entity_tag.dim, mesh.topology.dim)
    sub_to_parent_entity_map = np.full(num_child_entities, -1, dtype=np.int32)
    for facet, value in zip(entity_tag.indices, entity_tag.values):
        facet_found = False
        for cell in p_f_to_c.links(facet):
            if facet_found:
                break
            if (child_cell := mesh_to_submesh[cell]) != -1:
                for child_facet in c_c_to_e.links(child_cell):
                    child_vertices = c_e_to_v.links(child_facet)
                    child_vertices_as_parent = sub_vertex_to_parent[child_vertices]
                    is_facet = np.isin(
                        child_vertices_as_parent, p_f_to_v.links(facet)
                    ).all()
                    if is_facet:
                        child_markers[child_facet] = value
                        facet_found = True
                        sub_to_parent_entity_map[child_facet] = facet
    tags = dolfinx.mesh.meshtags(
        submesh,
        entity_tag.dim,
        np.arange(num_child_entities, dtype=np.int32),
        child_markers,
    )
    tags.name = entity_tag.name
    return tags, sub_to_parent_entity_map


with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "logo.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh(ghost_mode=dolfinx.mesh.GhostMode.shared_facet)
    ct = xdmf.read_meshtags(mesh, name="Cell tags")
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    ft = xdmf.read_meshtags(mesh, name="Facet tags")

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
parent_to_sub_b = np.full(num_cells_local, -1, dtype=np.int32)
parent_to_sub_b[submesh_b_to_mesh] = np.arange(len(submesh_b_to_mesh), dtype=np.int32)
parent_to_sub_t = np.full(num_facets_local, -1, dtype=np.int32)
parent_to_sub_t[submesh_t_to_mesh] = np.arange(len(submesh_t_to_mesh), dtype=np.int32)    

# We need to modify the cell maps, as for `dS` integrals of interfaces between submeshes, there is no entity to map to.
# We use the entity on the same side to fix this (as all restrictions are one-sided)

# Transfer meshtags to submesh
ft_b, b_facet_to_parent = transfer_meshtags_to_submesh(
    mesh, ft, submesh_b, b_v_map, submesh_b_to_mesh
)
ft_t, t_facet_to_parent = transfer_meshtags_to_submesh(
    mesh, ft, submesh_t, t_v_map, submesh_t_to_mesh
)

t_parent_to_facet = np.full(num_facets_local, -1)
t_parent_to_facet[t_facet_to_parent] = np.arange(len(t_facet_to_parent), dtype=np.int32)


# Hack, as we use one-sided restrictions, pad dS integral with the same entity from the same cell on both sides
mesh.topology.create_connectivity(fdim, tdim)
f_to_c = mesh.topology.connectivity(fdim, tdim)
for facet in ft.find(3):
    cells = f_to_c.links(facet)
    b_map = parent_to_sub_b[cells]
    t_map = parent_to_sub_t[cells]

    parent_to_sub_b[cells] = max(b_map)
    parent_to_sub_t[cells] = max(t_map)

entity_maps = {submesh_b: parent_to_sub_b, submesh_t: parent_to_sub_t}
# entity_maps = {submesh_b._cpp_object: parent_to_sub_b, submesh_t._cpp_object: parent_to_sub_t}




def define_interior_eq(mesh,degree,  submesh, submesh_to_mesh, value):
    # Compute map from parent entity to submesh cell
    codim = mesh.topology.dim - submesh.topology.dim
    ptdim = mesh.topology.dim - codim
    num_entities = (
        mesh.topology.index_map(ptdim).size_local
        + mesh.topology.index_map(ptdim).num_ghosts
    )
    mesh_to_submesh = np.full(num_entities, -1)
    mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh), dtype=np.int32)

    V = dolfinx.fem.functionspace(submesh, ("Lagrange", degree))
    u = dolfinx.fem.Function(V)
    v = ufl.TestFunction(V)
    ct_r = dolfinx.mesh.meshtags(mesh, mesh.topology.dim, submesh_to_mesh, np.full_like(submesh_to_mesh, 1, dtype=np.int32))
    val = dolfinx.fem.Constant(submesh, value)
    dx_r = ufl.Measure("dx", domain=mesh, subdomain_data=ct_r, subdomain_id=1)
    F = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_r - val * v * dx_r
    return u, F, mesh_to_submesh


u_0, F_00, m_to_b = define_interior_eq(mesh, 4, submesh_b, submesh_b_to_mesh, 2.0)
u_1, F_11, m_to_t = define_interior_eq(mesh, 2, submesh_t, submesh_t_to_mesh, 3.0)
u_0.name = "u_b"
u_1.name = "u_t"


# Add coupling term to the interface
# Get interface markers on submesh b
dInterface = ufl.Measure("dS", domain=mesh, subdomain_data=ft, subdomain_id=3)
b_res = "+"
t_res = "-"

v_b = ufl.TestFunction(u_0.function_space)(b_res)
v_t = ufl.TestFunction(u_1.function_space)(t_res)
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



F_0 = (
    -0.5 * mixed_term((u_b + u_t), v_b, n_b) * dInterface
    - 0.5 * mixed_term(v_b, (u_b - u_t), n_b) * dInterface
)

F_1 = (
    +0.5 * mixed_term((u_b + u_t), v_t, n_b) * dInterface
    - 0.5 * mixed_term(v_t, (u_b - u_t), n_b) * dInterface
)
F_0 += 2 * gamma / (h_b + h_t) * (u_b - u_t) * v_b * dInterface
F_1 += -2 * gamma / (h_b + h_t) * (u_b - u_t) * v_t * dInterface

F_0 += F_00
F_1 += F_11

#F_1 = F_11
jac00 = ufl.derivative(F_0, u_0)

jac01 = ufl.derivative(F_0, u_1)

jac10 = ufl.derivative(F_1, u_0)
jac11 = ufl.derivative(F_1, u_1)
J00 = dolfinx.fem.form(jac00, entity_maps=entity_maps)


J01 = dolfinx.fem.form(jac01, entity_maps=entity_maps)
J10 = dolfinx.fem.form(jac10, entity_maps=entity_maps)
J11 = dolfinx.fem.form(jac11, entity_maps=entity_maps)
J = [[J00, J01], [J10, J11]]
F = [
    dolfinx.fem.form(F_0, entity_maps=entity_maps),
    dolfinx.fem.form(F_1, entity_maps=entity_maps),
]
# foo = F[1]
# print(foo._cpp_object.domains(dolfinx.fem.IntegralType.cell, 1))
# dolfinx.fem.assemble.pack_coefficients(foo._cpp_object)
# #print(foo._cpp_object.domains(dolfinx.fem.IntegralType.interior_facet, 5))
# exit()
# dolfinx.fem.petsc.assemble_vector(foo)


b_bc = dolfinx.fem.Function(u_0.function_space)
b_bc.x.array[:] = 0.2
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

solver = NewtonSolver(
    F,
    J,
    [u_0, u_1],
    bcs=bcs,
    max_iterations=2,
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
)
solver.solve(1e-5)

bp = dolfinx.io.VTXWriter(mesh.comm, "u_b.bp", [u_0], engine="BP4")
bp.write(0)
bp.close()
bp = dolfinx.io.VTXWriter(mesh.comm, "u_t.bp", [u_1], engine="BP4")
bp.write(0)
bp.close()
