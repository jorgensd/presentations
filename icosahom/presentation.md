---
marp: true
class: lead
paginate: true
math: katex
theme: uncover
style: |

  section {

  background-color: #ccc;
  letter-spacing: 1px;
  text-align: left;

  }
  h1 {

  font-size: 1.3em;
  text-align: center;
  color: #f15922;
  }
  h2 {

  font-size: 1.5em;
  text-align: left;
  color: #f15922;

  }
  h3 {

  font-size: 1em;

  text-align: center;
  font-weight: normal;
  letter-spacing: 1px;
  color: #f15922;


  }
  h6 {

  text-align: center;
  font-weight: normal;
  letter-spacing: 1px;
  color: #f15922;

  }
  p{

  text-align: left;
  font-size: 0.75em;
  letter-spacing: 0px;

  }
  img[src$="centerme"] {
  font-size: 0.8em; 
  display:block; 
  margin: 0 auto; 
  }
  footer{

  color: black;
  text-align: left;

  }
  ul {

  padding: 10;
  margin: 0;

  }
  ul li {

  color: black;
  margin: 5px;
  font-size: 30px;

  }
  /* Code */
  pre, code, tt {

  font-size: 0.98em;
  font-size: 25px;
  font-family: Consolas, Courier, Monospace;
  color: white;
  background-color: #D1CFCC;
  }
  code , tt{

  margin: 0px;
  padding: 2px;
  white-space: nowrap;
  border: 1px solid #eaeaea;
  border-radius: 3px;
  color: white;
  background: 	#D1CFCC;
  }

  /* code blocks */
  pre {

  padding: 6px 10px;
  border-radius: 3px;
  color: black;
  background: #D1CFCC;

  }

  /* Code blocks */
  pre code, pre tt {

  background-color: transparent;
  border: none;
  margin: 0;
  padding: 1;
  white-space: pre;
  border: none;
  background: transparent;
  color: black;
  }

  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }

  .skewed-columns {
    display: grid;
    grid-template-columns: minmax(0, 55fr) minmax(0, 35fr);
  }

backgroundImage: url('./logos/simula.png')
backgroundSize: 150px
backgroundPosition: bottom+10px left+10px
---

# Mixed-domain and multi-physics modelling in the FEniCS framework

<center>
Jørgen S. Dokken
<center/>

<center>
<b> dokken@simula.no </b>
<center/>

<center>
<a href="https://jsdokken.com">https://jsdokken.com</a>
<center/>

<center>
<div>
<img src="./logos/fenics.png" width=150px>
</div>
<div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="./logos/wellcome.png" height=100px>
<img src="./logos/simula.png" height=100px>
<img src="./logos/batcat.png" height=100px>
</div>
<center/>

---

# What is FEniCS(x)?

---

<!-- footer: Baratta, I. A. et al. (2023). DOLFINx: The next generation FEniCS problem solving environment. DOI: 10.5281/zenodo.10447666
 -->

<center>
<img src="./images/workflow.png" width=750px>
<br>
<center/>

---

<!-- footer: -->

# The Poisson equation

![bg contain right:30%](images/uh.png)

```python
from mpi4py import MPI
import dolfinx.fem.petsc
import ufl
import numpy as np

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 5))























```

---

# The Poisson equation

![bg contain right:30%](images/uh.png)

```python
from mpi4py import MPI
import dolfinx.fem.petsc
import ufl
import numpy as np

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 5))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
x, y = ufl.SpatialCoordinate(mesh)
f = x * ufl.sin(y * ufl.pi)
L = ufl.inner(f, v) * ufl.dx

















```

---

# The Poisson equation

```python
from mpi4py import MPI
import dolfinx.fem.petsc
import ufl
import numpy as np

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 3, 3)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 5))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
x, y = ufl.SpatialCoordinate(mesh)
f = x * ufl.sin(y * ufl.pi)
L = ufl.inner(f, v) * ufl.dx

boundary_dofs = dolfinx.fem.locate_dofs_geometrical(
    V, lambda x: np.isclose(x[0], 0) | np.isclose(x[0], 1)
)
bcs = [dolfinx.fem.dirichletbc(0.0, boundary_dofs, V)]
options = {
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "ksp_error_if_not_converged": True,
}
problem = dolfinx.fem.petsc.LinearProblem(
  a, L, bcs=bcs, petsc_options=options)
uh = problem.solve()
with dolfinx.io.VTXWriter(mesh.comm, "uh.bp", [uh]) as bp:
    bp.write(0.0)

```

![bg contain right:30%](images/uh.png)

---

---

# A wrapper for writing blocked problems

## `ufl.MixedFunctionSpace`

```python
msh = create_unit_square(MPI.COMM_WORLD, 96, 96, CellType.triangle)
k = 1
V = fem.functionspace(msh, ("RT", k))
W = fem.functionspace(msh, ("Discontinuous Lagrange", k - 1))

Q = ufl.MixedFunctionSpace(V, W)
sigma, u = ufl.TrialFunctions(Q)
tau, v = ufl.TestFunctions(Q)
```

---

# Example of potential API rewrite

```python
a = [
    [ufl.inner(sigma, tau) * dx, ufl.inner(u, ufl.div(tau)) * dx],
    [ufl.inner(ufl.div(sigma), v) * dx, None],
]
```

<div data-marpit-fragment>

`ufl.extract_blocks`

```python
a_mono = ufl.inner(sigma, tau) * dx + ufl.inner(u, ufl.div(tau)) * dx \
  + ufl.inner(ufl.div(sigma), v) * dx
a = ufl.extract_blocks(a_mono)
```

</div>

---

# Why do we need this?

```python
gamma, gamma_to_omega = dolfinx.mesh.create_submesh(omega, fdim, ft.find(tag))[
        0:2]
V = dolfinx.fem.functionspace(omega, ("Lagrange", 1, (omega.geometry.dim, )))
Q = dolfinx.fem.functionspace(gamma, ("Lagrange", 1))
W = ufl.MixedFunctionSpace(V, Q)
```

<div data-marpit-fragment>

For more information about mixed-dimensional assembly see: [https://fenicsproject.org/blog/v0.9.0/#mixed-assembly](https://fenicsproject.org/blog/v0.9.0/#mixed-assembly)

</div>

<!-- footer: An example where this is used can be found at: https://jsdokken.com/FEniCS-workshop/src/multiphysics/coupling.html <br><br>
 -->

---

# Why do we need this (part 2)?

```python
interior_spaces = [Vi.clone() for _ in range(num_ions+1)]
exterior_spaces = [Ve.clone() for _ in range(num_ions+1)]
spaces = (interior_spaces+exterior_spaces)
W = MixedFunctionSpace(*spaces)
```

<!-- footer: These lines of code is based on the KNP-EMI model in DOLFINx by Halvor Herlyng: https://github.com/scientificcomputing/fenics-in-the-wild/pull/8 <br><br>
 -->

---

<!-- footer: <br> -->

# Supported UFL operations

- `ufl.derivative(F, [sigma, u], [ds, du])`
- `ufl.lhs`, `ufl.rhs`, `ufl.system`: [UFL #350](https://github.com/FEniCS/ufl/pull/350)
- `ufl.action`: [UFL 351](https://github.com/FEniCS/ufl/pull/351)
- `ufl.adjoint`: [UFL #352](https://github.com/FEniCS/ufl/pull/352)

---

# Simplified interface

```python
a_blocked = ufl.extract_blocks(a_mono)
a = dolfinx.fem.form(a_blocked)
```

<div data-marpit-fragment>

Blocked matrices

```python
A = dolfinx.fem.petsc.create_matrix(a, kind="mpi")
```

</div>

<div data-marpit-fragment>

Nest matrices

```python
A = dolfinx.fem.petsc.create_matrix(a, kind="nest")
```

</div>

<div data-marpit-fragment>

Unified assembly

```python
dolfinx.fem.petsc.assemble_matrix(A, a)
A.assemble()
```

</div>

---

# Blocked vectors needed a major redesign

Special handling of Dirichlet boundary conditions (lifting)

```python
b = assemble_vector_block(L_blocked, a_blocked, bcs=bcs)
```

<div data-marpit-fragment>

Unified logic for all assembly modes

```python
b = assemble_vector(L_blocked, kind=PETSc.Vec.Type.MPI)
bcs1 = fem.bcs_by_block(fem.extract_function_spaces(a_blocked, 1), bcs)
apply_lifting(b, a_blocked, bcs=bcs1)
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bcs0 = fem.bcs_by_block(fem.extract_function_spaces(L_blocked), bcs)
set_bc(b, bcs0)
```

</div>

---

# Interacting with PETSc vectors

```python
x = dolfinx.fem.petsc.create_vector(L_block, kind="mpi")
assign((u, p), x) # Transfer data from DOLFINx functions to PETSc Vec
assign(x, (u, p)) # Transfer data from PETSc Vec to DOLFINx functions
```

---

# What caused all this redesign?

<div data-marpit-fragment>
<center>
Simpler solver interface
</center>
</div>

---

# Linear problems

```python
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
    },
)
```

<div data-marpit-fragment>

```python
nested_IS = problem.A.getNestISs()
ksp = problem.solver
ksp.getPC().setFieldSplitIS(("sigma", nested_IS[0][0]), ("u", nested_IS[0][1]))
ksp_sigma, ksp_u = ksp.getPC().getFieldSplitSubKSP()
```

</div>

---

# Non-linear problems

- `dolfinx.nls.petsc.NewtonSolver` is being deprecated
- `dolfinx.fem.petsc.NonlinearProblem` is being deprecated
  - renamed to `dolfinx.fem.petsc.NewtonSolverNonlinearProblem`

<div data-marpit-fragment>

- `dolfinx.fem.petsc.NonlinearProblem` encapsulates the `PETSc.SNES` solver

```python
options = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "none",
    "ksp_type": "preonly",
    "pc_type": "lu",
    "snes_monitor": None,
}
problem = NonlinearProblem(F, u, petsc_options=options)
_, converged_reason, num_iterations = problem.solve()
```

</div>
