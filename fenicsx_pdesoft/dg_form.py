import ufl
from basix.ufl import element

# Create a symbolic representation of a mesh
cell = "triangle"
c_el = element("Lagrange", cell, 1, shape=(2,))
domain = ufl.Mesh(c_el)

# Create a symbolic representation of a function space
el = element("Lagrange", cell, 3, discontinuous=True)
V = ufl.FunctionSpace(domain, el)

# Define problem specific variables
h = 2 * ufl.Circumradius(domain)
n = ufl.FacetNormal(domain)
x, y = ufl.SpatialCoordinate(domain)
g = ufl.sin(2 * ufl.pi * x) + ufl.cos(y)
f = ufl.Coefficient(V)
alpha = ufl.Constant(domain)
gamma = ufl.Constant(domain)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Define variational formulation
ds = ufl.Measure("ds", domain=domain)
dx = ufl.Measure("dx", domain=domain)
dS = ufl.Measure("dS", domain=domain)

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
