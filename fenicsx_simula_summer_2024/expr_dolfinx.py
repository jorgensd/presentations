from mpi4py import MPI
import dolfinx
import ufl

mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, 10)
x = ufl.SpatialCoordinate(mesh)
N = dolfinx.fem.Constant(mesh, 7.)
f = ufl.sin(N * ufl.pi* x[0])
compiled_form = dolfinx.fem.form(f*ufl.dx)

print(dolfinx.fem.assemble_scalar(compiled_form))

# Reassign N to a new value
N.value = 3
print(dolfinx.fem.assemble_scalar(compiled_form))


import numpy as np
num_cells_local = mesh.topology.index_map(mesh.topology.dim).size_local
midpoints = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim, np.arange(num_cells_local, dtype=np.int32))
first_cell_index = np.argsort(midpoints[:, 0])[0]
cells = np.array([first_cell_index], dtype=np.int32)

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 3))
u = dolfinx.fem.Function(V)
u.interpolate(lambda x: 3*x[0]**3)
grad_u_squared = ufl.dot(ufl.grad(u), ufl.grad(u))
point_in_reference_element = np.array([0.5])
compiled_expression = dolfinx.fem.Expression(grad_u_squared, point_in_reference_element)
print(compiled_expression.eval(mesh, cells))
# Exact solution
print((9*midpoints[first_cell_index]**2)**2)

