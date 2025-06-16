# This is a simplified version of the DOLFINx demo for the mixed Poisson problem
# https://github.com/FEniCS/dolfinx/blob/c85068be81741d6ce9b8dac7c2142a2a14b7fdfd/python/demo/demo_mixed-poisson.py
# SPDX-License-Identifier: LGPL-3.0-or-later


from mpi4py import MPI
from petsc4py import PETSc
import dolfinx.fem.petsc
import numpy as np
import ufl
from dolfinx import fem, mesh
from dolfinx.fem.petsc import discrete_gradient, interpolation_matrix
from dolfinx.mesh import CellType, create_unit_square

msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle)
k = 1
V = fem.functionspace(msh, ("RT", k))
W = fem.functionspace(msh, ("Discontinuous Lagrange", k - 1))
# -

# +
Q = ufl.MixedFunctionSpace(V, W)
sigma, u = ufl.TrialFunctions(Q)
tau, v = ufl.TestFunctions(Q)

# The source function is set to be $f = 10\exp(-((x_{0} - 0.5)^2 +
# (x_{1} - 0.5)^2) / 0.02)$:

# +
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)) / 0.02)
# -

# We now declare the blocked bilinear and linear forms. We use `ufl.extract_blocks`
# to extract the block structure of the bi-linear and linear form.
# For the first block of the right-hand-side, we provide a form that efficiently is 0.
# We do this to preserve knowledge of the test space in the block. *Note that the defined `L`
# corresponds to $u_{0} = 0$ on $\Gamma_{D}$.*

# +
dx = ufl.Measure("dx", msh)

a = ufl.extract_blocks(
    ufl.inner(sigma, tau) * dx
    + ufl.inner(u, ufl.div(tau)) * dx
    + ufl.inner(ufl.div(sigma), v) * dx
)
L = [ufl.ZeroBaseForm((tau,)), -ufl.inner(f, v) * dx]


F = (
    ufl.inner(sigma, tau) * dx
    + ufl.inner(u, ufl.div(tau)) * dx
    + ufl.inner(ufl.div(sigma), v) * dx
    + sum(L)
)
a, L = ufl.system(F)
breakpoint()
# -


# In preparation for Dirichlet boundary conditions, we use the function
# `locate_entities_boundary` to locate mesh entities (facets) with which
# degree-of-freedoms to be constrained are associated with, and then use
# `locate_dofs_topological` to get the  degree-of-freedom indices. Below
# we identify the degree-of-freedom in `V` on the (i) top ($x_{1} = 1$)
# and (ii) bottom ($x_{1} = 0$) of the mesh/domain.

# +
fdim = msh.topology.dim - 1
facets_top = mesh.locate_entities_boundary(msh, fdim, lambda x: np.isclose(x[1], 1.0))
facets_bottom = mesh.locate_entities_boundary(
    msh, fdim, lambda x: np.isclose(x[1], 0.0)
)
dofs_top = fem.locate_dofs_topological(V, fdim, facets_top)
dofs_bottom = fem.locate_dofs_topological(V, fdim, facets_bottom)
# -

# Now, we create Dirichlet boundary objects for the condition $\sigma
# \cdot n = \sin(5 x_(0)$ on the top and bottom boundaries:

# +
cells_top_ = mesh.compute_incident_entities(msh.topology, facets_top, fdim, fdim + 1)
cells_bottom = mesh.compute_incident_entities(
    msh.topology, facets_bottom, fdim, fdim + 1
)
g = fem.Function(V)
g.interpolate(
    lambda x: np.vstack((np.zeros_like(x[0]), np.sin(5 * x[0]))), cells0=cells_top_
)
g.interpolate(
    lambda x: np.vstack((np.zeros_like(x[0]), -np.sin(5 * x[0]))), cells0=cells_bottom
)
bcs = [fem.dirichletbc(g, dofs_top), fem.dirichletbc(g, dofs_bottom)]
# -

# Rather than solving the linear system $A x = b$, we will solve the
# preconditioned problem $P^{-1} A x = P^{-1} b$. Commonly $P = A$, but
# this does not lead to efficient solvers for saddle point problems.
#
# For this problem, we introduce the preconditioner
# $$
# a_p((\sigma, u), (\tau, v))
# = \begin{bmatrix} \int_{\Omega} \sigma \cdot \tau + (\nabla \cdot
# \sigma) (\nabla \cdot \tau) \ {\rm d} x  & 0 \\ 0 &
# \int_{\Omega} u \cdot v \ {\rm d} x \end{bmatrix}
# $$
# and assemble it into the matrix `P`:

# +
a_p = ufl.extract_blocks(
    ufl.inner(sigma, tau) * dx
    + ufl.inner(ufl.div(sigma), ufl.div(tau)) * dx
    + ufl.inner(u, v) * dx
)

# -

# We create finite element functions that will hold the $\sigma$ and $u$
# solutions:

# +
sigma, u = fem.Function(V), fem.Function(W)
# -

# We now create a linear problem solver for the mixed problem.
# As we will use different preconditions for the individual blocks of the saddle point problem,
# we specify the matrix kind to be "nest", so that we can use # [`fieldsplit`](https://petsc.org/release/manual/ksp/#sec-block-matrices)
# (block) type and set the 'splits' between the $\sigma$ and $u$ fields.


# +

problem = fem.petsc.LinearProblem(
    a,
    L,
    u=[sigma, u],
    P=a_p,
    kind="nest",
    bcs=bcs,
    petsc_options={
        "ksp_type": "gmres",
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",
        "ksp_rtol": 1e-8,
        "ksp_gmres_restart": 100,
    },
)
# -


# +
# We get the nest index sets for the two fields
nested_IS = problem.A.getNestISs()
ksp = problem.solver
ksp.setMonitor(lambda ksp, its, rnorm: print(f"Iteration: {its}, residual: {rnorm}"))
ksp.getPC().setFieldSplitIS(("sigma", nested_IS[0][0]), ("u", nested_IS[0][1]))
ksp_sigma, ksp_u = ksp.getPC().getFieldSplitSubKSP()
# -

# For the $P_{11}$ block, which is the discontinuous Lagrange mass
# matrix, we let the preconditioner be the default, which is incomplete
# LU factorisation and which can solve the block exactly in one
# iteration. The $P_{00}$ requires careful handling as $H({\rm div})$
# problems require special preconditioners to be efficient.
#
# If PETSc has been configured with Hypre, we use the Hypre `Auxiliary
# Maxwell Space` (AMS) algebraic multigrid preconditioner. We can use
# AMS for this $H({\rm div})$-type problem in two-dimensions because
# $H({\rm div})$ and $H({\rm curl})$ spaces are effectively the same in
# two-dimensions, just rotated by $\pi/2.

# +
pc_sigma = ksp_sigma.getPC()

pc_sigma.setType("hypre")
pc_sigma.setHYPREType("ams")

opts = PETSc.Options()
opts[f"{ksp_sigma.prefix}pc_hypre_ams_cycle_type"] = 7
opts[f"{ksp_sigma.prefix}pc_hypre_ams_relax_times"] = 2

# Construct and set the 'discrete gradient' operator, which maps
# grad H1 -> H(curl), i.e. the gradient of a scalar Lagrange space
# to a H(curl) space
V_H1 = fem.functionspace(msh, ("Lagrange", k))
V_curl = fem.functionspace(msh, ("N1curl", k))
G = discrete_gradient(V_H1, V_curl)
G.assemble()
pc_sigma.setHYPREDiscreteGradient(G)

assert k > 0, "Element degree must be at least 1."
if k == 1:
    # For the lowest order base (k=1), we can supply interpolation
    # of the '1' vectors in the space V. Hypre can then construct
    # the required operators from G and the '1' vectors.
    cvec0, cvec1 = fem.Function(V), fem.Function(V)
    cvec0.interpolate(lambda x: np.vstack((np.ones_like(x[0]), np.zeros_like(x[1]))))
    cvec1.interpolate(lambda x: np.vstack((np.zeros_like(x[0]), np.ones_like(x[1]))))
    pc_sigma.setHYPRESetEdgeConstantVectors(cvec0.x.petsc_vec, cvec1.x.petsc_vec, None)
else:
    # For high-order spaces, we must provide the (H1)^d -> H(div)
    # interpolation operator/matrix
    V_H1d = fem.functionspace(msh, ("Lagrange", k, (msh.geometry.dim,)))
    Pi = interpolation_matrix(V_H1d, V)  # (H1)^d -> H(div)
    Pi.assemble()
    pc_sigma.setHYPRESetInterpolations(msh.geometry.dim, None, None, Pi, None)

    # High-order elements generally converge less well than the
    # lowest-order case with algebraic multigrid, so we perform
    # extra work at the multigrid stage
    opts[f"{ksp_sigma.prefix}pc_hypre_ams_tol"] = 1e-12
    opts[f"{ksp_sigma.prefix}pc_hypre_ams_max_iter"] = 3

    ksp_sigma.setFromOptions()

# -


# Once we have set the preconditioners for the two blocks, we can
# solve the linear system. The `LinearProblem` class will
# automatically assemble the linear system, apply the boundary
# conditions and call the Krylov solver and update the solution
# vectors `u` and `sigma`.

# +
problem.solve()

reason = ksp.getConvergedReason()
assert reason > 0, f"Krylov solver has not converged {reason}."
ksp.view()


# -

# We save the solution `u` in VTX format:

# +
try:
    from dolfinx.io import VTXWriter

    u.name = "u"
    with VTXWriter(msh.comm, "output_mixed_poisson.bp", u, "bp4") as f:
        f.write(0.0)
except ImportError:
    print("ADIOS2 required for VTX output.")
