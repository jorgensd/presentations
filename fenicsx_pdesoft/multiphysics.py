# SPDX-License-Identifier: MIT
# Author: Jørgen S. Dokken

from create_multiphysics_mesh import top_val, contact_val
from mpi4py import MPI
import dolfinx
import dolfinx.fem.petsc
import ufl
import numpy as np
from petsc4py import PETSc
import basix.ufl


class NewtonSolver():
    max_iterations: int
    bcs: list[dolfinx.fem.DirichletBC]
    A: PETSc.Mat
    b: PETSc.Vec
    J: dolfinx.fem.Form
    b: dolfinx.fem.Form
    dx: PETSc.Vec
    def __init__(self, F:list[dolfinx.fem.form], J:list[list[dolfinx.fem.form]], w: list[dolfinx.fem.Function], 
                 bcs: list[dolfinx.fem.DirichletBC]|None=None, max_iterations:int=5,
                 petsc_options: dict[str, str|float|int|None]=None):
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

            # Assemble F(u_{i-1}) - J(u_D - u_{i-1}) and set du|_bc= u_D - u_{i-1}
            self.b.zeroEntries()
            dolfinx.fem.petsc.assemble_vector_block(self.b, self.F,self.J, bcs=self.bcs,x0=self.x, scale=-1.0)
            self.b.ghostUpdate(PETSc.InsertMode.INSERT_VALUES, PETSc.ScatterMode.FORWARD)
            # Assemble Jacobian
            self.A.zeroEntries()
            dolfinx.fem.petsc.assemble_matrix_block(self.A, self.J, bcs=self.bcs)
            self.A.assemble()
            
            
            self._solver.solve(self.b, self.dx)
            assert self._solver.getConvergedReason() > 0, "Linear solver did not converge"
            #breakpoint()
            offset_start = 0
            for s in self.w:
                num_sub_dofs = s.function_space.dofmap.index_map.size_local * s.function_space.dofmap.index_map_bs
                s.x.array[:num_sub_dofs] -= beta*self.dx.array_r[offset_start:offset_start+num_sub_dofs]
                s.x.scatter_forward()
                offset_start += num_sub_dofs
            # Compute norm of primal space diff
            norm = self.b.copy()
            norm.zeroEntries()

            dolfinx.cpp.la.petsc.scatter_local_vectors(
                self.x,
                [si.x.petsc_vec.array_r for si in self.w],
                [
                    (si.function_space.dofmap.index_map, si.function_space.dofmap.index_map_bs)
                    for si in self.w
                ])
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

            correction_norm = self.dx.norm(0)
            
            print(f"Iteration {i}: Correction norm {correction_norm}")
            if correction_norm < tol:
                break
            i += 1




with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "r") as xdmf:
    mesh = xdmf.read_mesh()
    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    ft = xdmf.read_meshtags(mesh, name="Facet tags")

degree = 1

top_facets= ft.find(top_val)
contact_facets = ft.find(contact_val)

fdim = mesh.topology.dim - 1
assert fdim == ft.dim
submesh, submesh_to_mesh = dolfinx.mesh.create_submesh(mesh, fdim, contact_facets)[0:2]

E, nu = 2.e4, 0.40
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

def epsilon(w):
    return ufl.sym(ufl.grad(w))

def sigma(w, gdim):
    return 2.0 * mu * epsilon(w) + lmbda * ufl.tr(ufl.grad(w)) * ufl.Identity(gdim)

# enriched_element = basix.ufl.enriched_element(
#     [basix.ufl.element("Lagrange", mesh.topology.cell_name(), degree),
#      basix.ufl.element("Bubble", mesh.topology.cell_name(), degree+mesh.geometry.dim)])
# FIXME: We need facet bubbles
#element_u = basix.ufl.blocked_element(enriched_element, shape=(mesh.geometry.dim, ))
#element_p = basix.ufl.element("Lagrange", submesh.topology.cell_name(), degree-1, discontinuous=True)

element_u = basix.ufl.element("Lagrange", mesh.topology.cell_name(), degree, shape=(mesh.geometry.dim, ))
V = dolfinx.fem.functionspace(mesh, element_u)

u = dolfinx.fem.Function(V, name="displacement")
v = ufl.TestFunction(V)


element_p = basix.ufl.element("Lagrange", submesh.topology.cell_name(), degree)
W = dolfinx.fem.functionspace(submesh, element_p)

v = ufl.TestFunction(V)
psi = dolfinx.fem.Function(W)
psi_k = dolfinx.fem.Function(W)
w = ufl.TestFunction(W)
facet_imap = mesh.topology.index_map(fdim)
num_facets = facet_imap.size_local + facet_imap.num_ghosts
mesh_to_submesh = np.full(num_facets, -1)
mesh_to_submesh[submesh_to_mesh] = np.arange(len(submesh_to_mesh))
entity_maps = {submesh: mesh_to_submesh}

ds = ufl.Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=contact_val, metadata={"quadrature_degree": 10})
n = ufl.FacetNormal(mesh)
alpha = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(1.0))
f = dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type((0.0, 0.0)))
x = ufl.SpatialCoordinate(mesh)
g = x[1]+dolfinx.fem.Constant(mesh, dolfinx.default_scalar_type(0.05))

F00 = alpha*ufl.inner(sigma(u, mesh.geometry.dim), ufl.sym(ufl.grad(v))) * ufl.dx(domain=mesh) - alpha * ufl.inner(f, v) * ufl.dx(domain=mesh)
F01 = -ufl.inner(psi-psi_k, ufl.dot(v, n)) * ds
F10 = ufl.inner(ufl.dot(u, n), w)  * ds
F11 = ufl.inner(ufl.exp(psi), w)  * ds - ufl.inner(g, w)  * ds
F0 = F00 + F01 
F1 = F10 + F11

F0 = F00 + F01 
F1 = F10 + F11
residual_0 = dolfinx.fem.form(F0, entity_maps=entity_maps)
residual_1 = dolfinx.fem.form(F1, entity_maps=entity_maps)
jac00 = ufl.derivative(F0, u)
jac01 = ufl.derivative(F0, psi)
jac10 = ufl.derivative(F1, u)
jac11 = ufl.derivative(F1, psi)
J00 = dolfinx.fem.form(jac00, entity_maps=entity_maps)
J01 = dolfinx.fem.form(jac01, entity_maps=entity_maps)
J10 = dolfinx.fem.form(jac10, entity_maps=entity_maps)
J11 = dolfinx.fem.form(jac11, entity_maps=entity_maps)

J = [[J00, J01], [J10, J11]]
F = [residual_0, residual_1]

u_bc = dolfinx.fem.Function(V)
disp = -0.12
u_bc.interpolate(lambda x: (np.full(x.shape[1], 0.0), np.full(x.shape[1], disp)))
V0, V0_to_V = V.sub(1).collapse()
bc = dolfinx.fem.dirichletbc(u_bc, dolfinx.fem.locate_dofs_topological(V, fdim, top_facets))
bcs = [bc]


solver = NewtonSolver(F, J, [u, psi], bcs=bcs, max_iterations=25, petsc_options={"ksp_type":"preonly", "pc_type":"lu", 
"pc_factor_mat_solver_type":"mumps"})

bp = dolfinx.io.VTXWriter(mesh.comm, "uh.bp", [u], engine="BP4")
bp_psi = dolfinx.io.VTXWriter(mesh.comm, "psi.bp", [psi], engine="BP4")


s = sigma(u, mesh.geometry.dim) - 1. / 3 * ufl.tr(sigma(u,mesh.geometry.dim)) * ufl.Identity(len(u))
von_Mises = ufl.sqrt(3. / 2 * ufl.inner(s, s))
V_DG = dolfinx.fem.functionspace(mesh, ("DG", 1, (mesh.geometry.dim, )))
V_von_mises, _ = V_DG.sub(0).collapse()
stress_expr = dolfinx.fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
vm = dolfinx.fem.Function(V_von_mises, name="VonMises")
u_vm = dolfinx.fem.Function(V_DG, name="u")
bp_vm = dolfinx.io.VTXWriter(mesh.comm, "von_mises.bp", [vm, u_vm])

Qe = basix.ufl.quadrature_element(submesh.topology.cell_name(), degree=4)
Q = dolfinx.fem.functionspace(submesh, Qe)
q_q = dolfinx.fem.Function(Q)
expr = dolfinx.fem.Expression(ufl.dot(u, n)-g, Q.element.interpolation_points())
ents = dolfinx.cpp.fem.compute_integration_domains(dolfinx.fem.IntegralType.exterior_facet, mesh.topology, submesh_to_mesh, mesh.topology.dim-1)

Qo = dolfinx.fem.functionspace(submesh, ("DG", 2))
p, q = ufl.TrialFunction(Qo), ufl.TestFunction(Qo)
a = ufl.inner(p, q) * ufl.dx
L = ufl.inner(q_q, q) * ufl.dx
out = dolfinx.fem.Function(Qo)
problem_q = dolfinx.fem.petsc.LinearProblem(a, L, u=out, bcs=[], petsc_options={"ksp_type":"preonly", "pc_type":"lu", "pc_factor_mat_solver_type":"mumps"})
bp_derived = dolfinx.io.VTXWriter(mesh.comm, "un.bp", [out], engine="BP4")

M = 10
for it in range(M):
    print(f"{it}/{M}")
    #u_bc.x.array[V0_to_V] = (it+1)/M * disp
    u_bc.x.array[V0_to_V] = disp
 
    #print((it+1)/M * disp)
    alpha.value += 1
    #alpha.value = 1
    #try:
    solver.solve(1e-8, (it+1)/M)
    # except AssertionError as e:
    #     bp.close()
    #     print(e)
    #     exit(1)
    psi_k.x.array[:] = psi.x.array
    bp.write(it)
    bp_psi.write(it)
    u_vm.interpolate(u)
    vm.interpolate(stress_expr)
    bp_vm.write(it)

    q_q.x.array[:] = expr.eval(mesh, ents).reshape(-1)
    problem_q.solve()
    bp_derived.write(it)

bp_derived.close()
bp.close()
bp_psi.close()
bp_vm.close()