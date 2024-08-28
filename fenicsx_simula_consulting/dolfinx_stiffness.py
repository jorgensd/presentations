from mpi4py import MPI
import dolfinx
import ufl
import time

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

start_c = time.perf_counter()
a_compiled = dolfinx.fem.form(a)
end_c = time.perf_counter()
print(f"Compilation: {end_c-start_c:.2e}")

for i in range(3):
    start = time.perf_counter()
    A = dolfinx.fem.assemble_matrix(a_compiled)
    A.scatter_reverse()
    end = time.perf_counter()
    print(f"{i}: {end-start:.2e}")