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

  }
  h2 {

  font-size: 1.5em;
  text-align: left;

  }
  h3 {

  font-size: 1em;

  text-align: center;
  font-weight: normal;
  letter-spacing: 1px;

  }
  h6 {

  text-align: center;
  font-weight: normal;
  letter-spacing: 1px;

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
  background: 	#222222;

  }
  code , tt{

  margin: 0px;
  padding: 2px;
  white-space: nowrap;
  border: 1px solid #eaeaea;
  border-radius: 3px;
  color: white;
  background: 	#222222;

  }

  pre {

  padding: 6px 10px;
  border-radius: 3px;
  background-color:  #D1CFCC;
  color: black;
  }

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
---

# DOLFINx: The next generation FEniCS problem solving environment

### Baratta, I. A., Dean, J. P., Dokken, J. S., Habera, M., Hale, J. S., Richardson, C. N., Rognes, M. E., Scroggs, M. W., Sime, N., & Wells, G. N.

###### dokken@simula.no

### [Zenodo: 10.5281/zenodo.10447666](https://doi.org/10.5281/zenodo.10447666)

---

# Brief history of finite elements

* 1940s: Rayleigh-Ritz/Ritz Galerkin method
* 1970s: General purpose finite element software
* 1990s: Object orientation
* 2000s: User-friendliness
* 2020s: High performance computing

---

# Brief history of the FEniCS project


* 2003: Initiated in Netherlands, Sweden and USA
* 2006-2016: Hans Petter era: CBC
* 2017-Present: Development of DOLFINx

---

# Motivation for a new API

* Age: Code 15+ years old
* Maintainability
* Scalability
* Extendability


---

# Implicitness in DOLFIN
```python
from dolfin import *
import time
mesh = UnitSquareMesh(1, 1)
V = FunctionSpace(mesh, "Lagrange", 5)
u, v = TrialFunction(V), TestFunction(V)
a = inner(grad(u), grad(v))*dx
for i in range(3):
    start = time.perf_counter()
    assemble(a)
    end = time.perf_counter()
    print(f"{i}: {end-start:.2e}")
```
Output:
```bash
0: 3.30e+00
1: 3.94e-04
2: 3.10e-04
```

---

# Explicit version in DOLFINx
```python
from mpi4py import MPI
import dolfinx
import ufl
import time

mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 1, 1)
V = dolfinx.fem.functionspace(mesh, ("Lagrange", 1))

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx

start_c = time.perf_counter()
a_compiled = dolfinx.fem.form(a)
end_c = time.perf_counter()
print(f"Compilation: {end_c-start_c:.2e}")

for i in range(3):
    start = time.perf_counter()
    dolfinx.fem.assemble_matrix(a_compiled)
    end = time.perf_counter()
    print(f"{i}: {end-start:.2e}")
```

---
# Explicit version continued

```bash
Compilation: 1.58e-01
0: 2.26e-04
1: 5.69e-05
2: 4.18e-05
```
# Same experiment for P1
<div class="columns">
<div>

## DOLFIN
```bash

0: 3.27e+00
1: 4.29e-04
2: 3.04e-04
```
</div>
<div>

## DOLFINx
```
Compilation: 1.30e-01
0: 2.11e-04
1: 7.79e-05
2: 5.05e-05
```
</div>
<div>

---

