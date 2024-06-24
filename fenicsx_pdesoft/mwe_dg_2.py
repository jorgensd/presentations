from mpi4py import MPI
import dolfinx
import ufl
from basix.ufl import element
import numpy as np

# Create discrete domain
cell = dolfinx.mesh.CellType.triangle
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 7, cell)

el = element("Lagrange", cell.name, 3, discontinuous=True)
V = dolfinx.fem.functionspace(mesh, el)

# Define problem specific variables
h = 2 * ufl.Circumradius(mesh)
n = ufl.FacetNormal(mesh)
x, y = ufl.SpatialCoordinate(mesh)
g = ufl.sin(2 * ufl.pi * x) + ufl.cos(y)
f = dolfinx.fem.Function(V)
f.interpolate(lambda x: x[0] + 2 * np.sin(x[1]))
alpha = dolfinx.fem.Constant(mesh, 25.0)
gamma = dolfinx.fem.Constant(mesh, 25.0)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define variational formulation
ds = ufl.Measure("ds", domain=mesh)
dx = ufl.Measure("dx", domain=mesh)
dS = ufl.Measure("dS", domain=mesh)

F = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx - f * v * dx


# Nitsche terms
def flux_term(u, v):
    return -ufl.dot(n, ufl.grad(u)) * v


F += flux_term(v, u) * ds + alpha / h * u * v * ds + flux_term(u, v) * ds
F -= flux_term(v, g) * ds + alpha / h * g * v * ds


# Interior penalty/DG terms
def dg_flux(u, v):
    return -ufl.dot(ufl.avg(ufl.grad(u)), ufl.jump(v, n))


F += dg_flux(u, v) * dS + dg_flux(v, u) * dS
F += gamma / ufl.avg(h) * ufl.inner(ufl.jump(v, n), ufl.jump(u, n)) * dS

a, L = ufl.system(F)
a_form = dolfinx.fem.form(a, dtype=np.float64)
L_form = dolfinx.fem.form(L, dtype=np.float64)


# Solve linear problem
import dolfinx.fem.petsc
uh = dolfinx.fem.Function(V)
solver_options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}
problem = dolfinx.fem.petsc.LinearProblem(
    a_form, L_form, u=uh, petsc_options=solver_options
)
problem.solve()
print(f"Solver converged with {problem.solver.getConvergedReason()}")

# Store solution to disk
with dolfinx.io.VTXWriter(mesh.comm, "solution_2.bp", [uh], engine="BP4") as bp:
    bp.write(0.0)
