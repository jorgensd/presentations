# slide 1
from mpi4py import MPI
import dolfinx
import ufl

mesh = dolfinx.mesh.create_unit_square(
    MPI.COMM_WORLD, 25, 25, dolfinx.mesh.CellType.quadrilateral
)

V = dolfinx.fem.functionspace(mesh, ("Lagrange", 3, (2,)))
subdomain_cells = dolfinx.mesh.locate_entities(
    mesh, mesh.topology.dim, lambda x: (x[0] > 0.8) & (x[1] > 0.2)
)
uh = dolfinx.fem.Function(V)
uh.interpolate(lambda x: (0.05 * x[1] ** 2, 0 * x[0]), cells=subdomain_cells)


# slide 2
def epsilon(u):
    return ufl.sym(ufl.grad(u))


def sigma(u, lmbda=2, mu=0.5):
    return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)


s = sigma(uh) - 1.0 / 3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
von_Mises = ufl.sqrt(3.0 / 2 * ufl.inner(s, s))

Q = dolfinx.fem.functionspace(mesh, ("DQ", 2))
compiled_expr = dolfinx.fem.Expression(von_Mises, Q.element.interpolation_points())
q = dolfinx.fem.Function(Q, name="VonMises")
q.interpolate(compiled_expr)

bp_file = dolfinx.io.VTXWriter(mesh.comm, "von_mises.bp", [q], engine="BP4")
bp_file.write(0.0)
bp_file.close()


bp_file = dolfinx.io.VTXWriter(mesh.comm, "displacement.bp", [uh], engine="BP4")
bp_file.write(0.0)
bp_file.close()
