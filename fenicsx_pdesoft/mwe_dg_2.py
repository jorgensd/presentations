from mpi4py import MPI
import dolfinx
import ufl
import numpy as np

# Create discrete domain and function space
dtype = np.float32
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 7,
                                       dolfinx.mesh.CellType.triangle,
                                       dtype=dtype)

Vh = dolfinx.fem.functionspace(mesh, ("DG", 3))

# Define problem specific variables
h = 2 * ufl.Circumradius(mesh)
n = ufl.FacetNormal(mesh)
x, y = ufl.SpatialCoordinate(mesh)
g = ufl.sin(2 * ufl.pi * x) + ufl.cos(y)
f = dolfinx.fem.Function(Vh, dtype=dtype)
f.interpolate(lambda x: x[0] + 2 * np.sin(x[1]))
alpha = dolfinx.fem.Constant(mesh, dtype(25.0))
gamma = dolfinx.fem.Constant(mesh, dtype(25.0))
u = ufl.TrialFunction(Vh)
v = ufl.TestFunction(Vh)

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
a_form = dolfinx.fem.form(a, dtype=dtype)
L_form = dolfinx.fem.form(L, dtype=dtype)


# Solve linear problem
import dolfinx.fem.petsc
uh = dolfinx.fem.Function(Vh, name="uh", dtype=dtype)
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
with dolfinx.io.VTXWriter(mesh.comm, "dg_solution.bp", [uh], engine="BP4") as bp:
    bp.write(0.0)
