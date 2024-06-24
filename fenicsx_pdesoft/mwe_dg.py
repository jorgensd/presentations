from mpi4py import MPI
import dolfinx.fem.petsc
import numpy as np

from dg_form import a, L, alpha, gamma, el, f

d_cell = dolfinx.mesh.to_type(el.cell_type.name)
mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 6, 7, d_cell)
Vh = dolfinx.fem.functionspace(mesh, el)

compiler_options = {"scalar_type": dolfinx.default_scalar_type}
compiled_a = dolfinx.fem.compile_form(
    mesh.comm,
    a,
    form_compiler_options=compiler_options,
)
compiled_L = dolfinx.fem.compile_form(
    mesh.comm,
    L,
    form_compiler_options=compiler_options,
)


# Create coefficients and constants
alp = dolfinx.fem.Constant(mesh, 25.0)
gam = dolfinx.fem.Constant(mesh, 25.0)
fh = dolfinx.fem.Function(Vh)
fh.interpolate(lambda x: x[0] + 2 * np.sin(x[1]))

# Compile variational forms
a_form = dolfinx.fem.create_form(
    compiled_a, [Vh, Vh], mesh, {}, {alpha: alp, gamma: gam}
)
L_form = dolfinx.fem.create_form(compiled_L, [Vh], mesh, {f: fh}, {alpha: alp})

# Solve linear problem
uh = dolfinx.fem.Function(Vh)
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
with dolfinx.io.VTXWriter(mesh.comm, "solution.bp", [uh], engine="BP4") as bp:
    bp.write(0.0)
