# Slide 0
from mpi4py import MPI
import dolfinx

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 15, 15)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 3))

# Slide 1
dt = dolfinx.fem.Constant(mesh, 0.01)
dt.value = 0.005 # Direct assignment

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
t_c = dolfinx.fem.Constant(mesh, 0.)
true_statement = 0.4 * y
false_statement = 0.5 * t_c
f = ufl.conditional(condition, true_statement, false_statement)
while float(t_c) < 1:
    # Update t_c (implicit update of f)
    t_c.value += float(dt)



# Slide 3

u = ufl.TrialFunction(V)
u_n = dolfinx.fem.Function(V)
dudt = (u - u_n) / dt
v = ufl.TestFunction(V)
dx = ufl.Measure("dx", domain=mesh)
F = dudt * v * dx + k * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx - f * v * dx
a, L = ufl.system(F)
a_compiled = dolfinx.fem.form(a)
L_compiled = dolfinx.fem.form(L)


# Slide 4
import numpy as np
def u_init(x, sigma=0.1, mu=0.3):
    """
    The input function x is a (3, number_of_points) numpy array, which is then
    evaluated with the vectorized numpy functions for efficiency
    """
    return 1./(2 * np.pi * sigma)*np.exp(-0.5*((x[0]-mu)/sigma)**2)*np.exp(-0.5*((x[1]-mu)/sigma)**2)

u_n.interpolate(u_init)

# Slide 5
import dolfinx.fem.petsc
uh = dolfinx.fem.Function(V, name="uh")
petsc_options = {"ksp_type": "preonly",
                 "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}
problem = dolfinx.fem.petsc.LinearProblem(
    a_compiled, L_compiled, u=uh, bcs=[], petsc_options=petsc_options)


# Slide 6
bp_file = dolfinx.io.VTXWriter(mesh.comm, "u.bp", [uh], engine="BP4")
t = 0
while t < 1:
    t += float(dt)
    k.value = k_func(t)
    problem.solve()
    # Update previous solution
    u_n.x.array[:] = uh.x.array
    bp_file.write(t)
bp_file.close()