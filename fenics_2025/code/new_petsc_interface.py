from mpi4py import MPI
import dolfinx.fem.petsc
import ufl

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 96, 96, dolfinx.mesh.CellType.triangle
)

V0 = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))
V1 = V0.clone()
W = ufl.MixedFunctionSpace(V0, V1)
u, p = ufl.TrialFunctions(W)
v, q = ufl.TestFunctions(W)
a_mono = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx + p * q * ufl.dx
a_blocked = ufl.extract_blocks(a_mono)
a = dolfinx.fem.form(a_blocked)
A = dolfinx.fem.petsc.create_matrix(a, kind="mpi")
A_nest = dolfinx.fem.petsc.create_matrix(a, kind="nest")
dolfinx.fem.petsc.assemble_matrix(A, a)
A.assemble()
dolfinx.fem.petsc.assemble_matrix(A_nest, a)
A_nest.assemble()
