# Slide 0
from mpi4py import MPI
import dolfinx

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 15, 15)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 3))

# Slide 1
dt = dolfinx.fem.Constant(mesh, 0.01)
dt.value = 0.005  # Direct assignment


def k_func(t):
    return 0.1 if t < 0.5 else 0.05


t = 0
k = dolfinx.fem.Constant(mesh, k_func(t))
while t < 1:
    # Update t
    t += float(dt)
    # Update k
    k.value = k_func(t)


# Slide 2

import ufl

x, y = ufl.SpatialCoordinate(mesh)
condition = ufl.lt(x, 0.5)
t_c = dolfinx.fem.Constant(mesh, 0.0)
true_statement = 0.4 * y
false_statement = 0.5 * t_c
f = ufl.conditional(condition, true_statement, false_statement)
while float(t_c) < 1:
    # Update t_c (implicit update of f)
    t_c.value += float(dt)


# Slide 3

uh = dolfinx.fem.Function(V)
u_n = dolfinx.fem.Function(V)
dudt = (uh - u_n) / dt
v = ufl.TestFunction(V)
dx = ufl.Measure("dx", domain=mesh)
F = dudt * v * dx + k * ufl.inner(ufl.grad(uh), ufl.grad(v)) * dx - f * v * dx


# Slide 4
import numpy as np


def u_init(x, sigma=0.1, mu=0.3):
    """
    The input function x is a (3, number_of_points) numpy array, which is then
    evaluated with the vectorized numpy functions for efficiency
    """
    return (
        1.0
        / (2 * np.pi * sigma)
        * np.exp(-0.5 * ((x[0] - mu) / sigma) ** 2)
        * np.exp(-0.5 * ((x[1] - mu) / sigma) ** 2)
    )


u_n.interpolate(u_init)

# Slide 5
import dolfinx.fem.petsc
import dolfinx.nls.petsc

problem = dolfinx.fem.petsc.NonlinearProblem(F, u=uh, bcs=[])
solver = dolfinx.nls.petsc.NewtonSolver(mesh.comm, problem)
ksp = solver.krylov_solver
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("lu")
pc.setFactorSolverType("mumps")

# Slide 6
bp_file = dolfinx.io.VTXWriter(mesh.comm, "u_nonlinear.bp", [uh], engine="BP4")
t = 0
while t < 1:
    t += float(dt)
    k.value = k_func(t)
    solver.solve(uh)
    # Update previous solution
    u_n.x.array[:] = uh.x.array
    bp_file.write(t)
bp_file.close()
