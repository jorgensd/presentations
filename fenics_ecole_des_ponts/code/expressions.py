from mpi4py import MPI
import dolfinx 
import numpy as np
import ufl


mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3, dolfinx.cpp.mesh.CellType.quadrilateral)
cells = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim, lambda x: x[0]<0.4)  
midpoints = dolfinx.mesh.compute_midpoints(mesh, mesh.topology.dim, cells)

# evaluate an expression at a point

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim, )))
u = dolfinx.fem.Function(V)

def f(x):
    return 1/2*x[0]**2 - 2 *x[1]**2, -3/2*x[0]**2 + 1/2*x[1]**2

u.interpolate(f)


ref_x = np.array([[0.5, 0.5]])
gradu_squared = dolfinx.fem.Expression(ufl.inner(ufl.grad(u), ufl.grad(u)), ref_x, comm=mesh.comm)
values = gradu_squared.eval(mesh,cells)


def dfdx(x):
    return np.hstack((x[0], -4*x[1],- 3 * x[0],  x[1]))

for midpoint, value in zip(midpoints, values):
    dfdx_ = dfdx(midpoint.reshape(3, -1))
    np.testing.assert_allclose(np.dot(dfdx_, dfdx_), value)

print(cells,  values)


# Use expression for interpolation

Q = dolfinx.fem.functionspace(mesh, ("DQ", 1))
q = dolfinx.fem.Function(Q)

expr = u[1].dx(0)
compiled_expression = dolfinx.fem.Expression(
    expr, Q.element.interpolation_points())
q.interpolate(compiled_expression)


ref_q = dolfinx.fem.Function(Q)
ref_q.interpolate(lambda x:-3 *x[0])

np.testing.assert_allclose(q.x.array, ref_q.x.array, atol=1e-13)


# Facet expression evaluations

n = ufl.FacetNormal(mesh)
flux = ufl.dot(u, n)

x_ref_facet = np.array([[0.2], [0.7]])

flux_expr = dolfinx.fem.Expression(flux, x_ref_facet, comm=mesh.comm)
left_facets = dolfinx.mesh.locate_entities_boundary(mesh, mesh.topology.dim -1,
                                                    lambda x: x[0]<1e-14)
integration_entities = dolfinx.fem.compute_integration_domains(dolfinx.fem.IntegralType.exterior_facet, mesh.topology,
                                                                left_facets, mesh.topology.dim-1)
flux_values = flux_expr.eval(mesh, integration_entities)

import basix.ufl
from dolfinx.fem import Expression, IntegralType, functionspace, Function, compute_integration_domains
def move_to_facet_quadrature(ufl_expr, mesh, sub_facets, scheme="default", degree=6):
    fdim = mesh.topology.dim - 1
    # Create submesh
    bndry_mesh, entity_map, _, _ = dolfinx.mesh.create_submesh(mesh, fdim, sub_facets)
    # Create quadrature space on submesh
    q_el = basix.ufl.quadrature_element(bndry_mesh.basix_cell(), ufl_expr.ufl_shape , scheme, degree)
    Q = functionspace(bndry_mesh, q_el)

    # Compute where to evaluate expression per submesh cell
    integration_entities = compute_integration_domains(IntegralType.exterior_facet, mesh.topology, entity_map, fdim)
    compiled_expr = Expression(ufl_expr, Q.element.interpolation_points())

    # Evaluate expression
    q = Function(Q)
    q.x.array[:] = compiled_expr.eval(mesh, integration_entities).reshape(-1)
    return q


u = Function(V, name="Velocity")
u.interpolate(lambda x: (np.sin(x[1])*np.cos(x[0]), x[0]+x[1]**2))

exterior_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
flux = ufl.dot(ufl.grad(u), n)
q = move_to_facet_quadrature(flux, mesh, exterior_facets)
q.name = "dot(grad(u), n)"

import scifem
scifem.xdmf.create_pointcloud("flux.xdmf", [q])
with dolfinx.io.VTXWriter(mesh.comm, "flux.bp", [u]) as bp:
    bp.write(0.0)