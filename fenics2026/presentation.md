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
    grid-template-columns: minmax(0, 50fr) minmax(0, 35fr);
  }
  .right-skewed-columns {
    display: grid;
    grid-template-columns: minmax(0, 35fr) minmax(0, 50fr);
  }

  {
  box-sizing: border-box;
  }

  body {
    background-color: #474e5d;
    font-family: Helvetica, sans-serif;
  }

  .qr-code {
  position: absolute;
  top: 30px;
  right: 30px;
  width: 120px; /* Adjust size as needed */


  }


backgroundImage: url('logos/simula.png')
backgroundSize: 150px
backgroundPosition: bottom+10px left+10px
---

# Evolving FEniCS: The Extension Ecosystem at Simula Scientific Computing

<center>
FEniCS 2026 at University of Chicago in Paris
<br>
<b>Jørgen S. Dokken</b> 
<br/>
<b>Henrik N.T. Finsberg</b>
<br>
<img src="logos/simula.png" vspace=20px width=150px>

<br/>
<div style="display: grid; grid-template-columns: repeat(3, 1fr); align-items: center; justify-items: center; width: 100%;">
  <img src="logos/wellcome.png" alt="Wellcome Logo" style="margin: 20px 0; width: 120px;">
  <img src="logos/fenics.png" alt="Fenics Logo" style="margin: 20px 0; width: 150px;">
  <img src="logos/batcat.png" alt="Batcat Logo" style="margin: 20px 0; width: 300px; max-width: 100%;">
</div>

</center>

---

# It all started over 21 years ago


<div class="skewed-columns">
<div>
<img src="./acknowledged_screenshots/hpl2005.png" vspace=0px width=600px>
<figcaption style="font-size: 50%; padding-top: 10px;">
FEniCS 05 (Chicago) - <i>Tools for Multi-Physics Simulation</i><br>  by Hans Petter Langtangen (Simula/UiO)
</figcaption>
</div>
<div>
<div data-marpit-fragment>

Packages developed or maintained

- UFL/FFC(x)/DOLFIN(x)
- DOLFINx_MPC
- <b>scifem</b>
- <b>io4dolfinx</b> (<s>adios4dolfinx</s>)
- <b>fenicsx_ii</b>
- <b>DOLFIN(x)-adjoint</b>
- <b>networks_fenicsx</b>

</div>

</div>
</div>

---

<h1> Scifem - FEM prototyping playground </h1>

<!-- <div class="columns"> -->

<div>
<center>
Not all ideas are good ideas <span data-marpit-fragment> <b>in the beginning</b></span>

<img src="qr_codes/scifem_qr.png" class="qr-code" vspace=0px width=200px>
</div>
<br>
<div data-marpit-fragment>
<b>Examples that are now in DOLFINx</b>

</center>
<!-- Real spaces -->
<div class="columns">
<div data-marpit-fragment>
  <ul>
 <li>Real function spaces <code>scifem.create_real_functionspace</code>
  </ul>
</div>
<div data-marpit-fragment>
  <pre is="marp-pre" data-auto-scaling="downscale-only">
  <code class="language-python"
>r_el = basix.ufl.real_element(mesh.basix_cell(), shape=(2, 3))
R = dolfinx.fem.functionspace(mesh, r_el)
</code></pre>
</div>
</div>
<!-- Blocked solvers -->
<div class="skewed columns">
<div data-marpit-fragment>
  <ul>
 <li>Blocked Newton solvers <code>scifem.BlockedNewtonSolver</code>
  </ul>
</div>
<div data-marpit-fragment>
  <pre is="marp-pre" data-auto-scaling="downscale-only">
  <code class="language-python"
  >dolfinx.fem.petsc.NonlinearProblem
</code></pre>
</div>
</div>

<!-- Transfer tags -->
<div class="columns">
<div data-marpit-fragment>
  <ul>
 <li>Transfer tags to submesh <code>scifem.transfer_meshtags_to_submesh</code>
  </ul>
</div>
<div data-marpit-fragment>
  <pre is="marp-pre" data-auto-scaling="downscale-only">
  <code class="language-python"
  >dolfinx.mesh.transfer_meshtags_to_submesh
</code></pre>
</div>
</div>

---

<img src="qr_codes/scifem_qr.png" class="qr-code" vspace=0px width=200px>

# What is next?

<center>
<code>scifem.create_space_of_simple_functions</code>
</center>

<div class="skewed-columns">
<div>
<pre is="marp-pre" data-auto-scaling="downscale-only" style="margin-bottom: 0; padding-bottom: 0; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
<code class="language-python">mesh = dolfinx.mesh.create_unit_square(comm, 10, 10)
tdim = mesh.topology.dim
tol = 1e-14
</code></pre>
<div data-marpit-fragment style="margin: 0; padding: 0;">
<pre is="marp-pre" data-auto-scaling="downscale-only" style="margin-top: 0; margin-bottom: 0; padding-top: 0; padding-bottom: 0; border-radius: 0;">
<code class="language-python"><span style="color: #2e8b57;"># Divide cell into three regions</span>
tags = (4,5,8)
cell_map = mesh.topology.index_map(tdim)
num_cells_local = cell_map.size_local + cell_map.num_ghosts
markers = np.full(num_cells_local, tags[0],  dtype=np.int32)
markers[dolfinx.mesh.locate_entities(
    mesh, tdim, lambda x: x[0] <= 0.5+tol)] = tags[1]
markers[dolfinx.mesh.locate_entities(
    mesh, tdim, lambda x: x[1] <= 0.5+tol)] = tags[2]
cells = np.arange(num_cells_local, dtype=np.int32)
ct = dolfinx.mesh.meshtags(mesh, tdim, cells, markers)
</code></pre>
</div>
<div data-marpit-fragment style="margin: 0; padding: 0;">
<pre is="marp-pre" data-auto-scaling="downscale-only" style="margin-top: 0; padding-top: 0; border-top-left-radius: 0; border-top-right-radius: 0;">
<code class="language-python"><span style="color: #2e8b57;"># Create a piecewise constant (per region) function space</span>
V = create_space_of_simple_functions(mesh, ct, tags)
u = dolfinx.fem.Function(V)
u.x.array[0] = 3.2
u.x.array[1] = 5.5
u.x.array[2] = 4.2
assert len(u.x.array) == 3
</code></pre>
</div>
</div>
<div>
<img src="simple_function.png" vspace=0px width=500>
</div>
</div>
</div>

---


<img src="qr_codes/scifem_qr.png" class="qr-code" vspace=0px width=200px>

<h1> What is next?</h1>
<div class="skewed-columns">
<div>
<br>
<pre is="marp-pre" data-auto-scaling="downscale-only" style="margin-bottom: 0; padding-bottom: 0; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
<code class="language-python">from scifem import closest_point_projection
points, ref_points = closest_point_projection(
    mesh,
    closest_cells,
    points,
    tol_x=1e-7,
)
</code></pre>
<div font_size=10>
Based on simplex projections<sup>2,3,4</sup>
</div>
</div>
<div>
<img src="cp.png" vspace=0px width=500>
<figcaption style="font-size: 50%; padding-top: 0px;">
<center>
Figure from the morning tutorial<sup>1</sup>
</center>
</figcaption>
</div>
</div>
</div>


<!--  footer: <sup>1</sup> Dokken, J.S <a href="https://a-latyshev.github.io/fenics26-tutorials/grid-mapping/">https://a-latyshev.github.io/fenics26-tutorials/grid-mapping/</a><br><sup>2</sup>Held, Wolfe, & Crowder (1974). DOI: <a href="https://doi.org/10.1007/BF01580223">10.1007/bf01580223</a><br><sup>3</sup>Bertsekas, (1976). DOI: <a href="https://doi.org/10.1109/tac.1976.1101194">10.1109/tac.1976.1101194</a> <br> <sup>4</sup>Condat, L. (2015). DOI: <a href="https://doi.org/10.1007/s10107-015-0946-6">10.1007/s10107-015-0946-6</a><br><br> -->


---


<!--  footer: <sup>5</sup> Habera, Demarle, Hale, Richardson, Zilian , <i>XDMF and Paraview Checkpointing format</i>), FEniCS'18 <br><br> -->

# IO4DOLFINx - a unified IO?

<div class="skewed-columns">
<div>
<br>
<center>
Visualization and checkpointing (write/read functions) has diverged due to N+1 different file formats and finite elements.
</center>
</div>
<div>
<center>
<br>
<img src="./acknowledged_screenshots/habera2017.png" vspace=0px width=450>
<figcaption style="font-size: 50%; padding-top: 10px;">
From M. Habera's presentation<sup>5</sup> at FEniCS 2018 on the XDMF format
</figcaption>
</center>
</div>


<div class="columns">

<div>

<br>
<img src="qr_codes/io4dolfinx_qr.png" class="qr-code" vspace=0px width=200px>

</div>

<div>

</div>
</div>


---




<!--  footer: <sup>6</sup>Dokken, J. S., (2024). <i>ADIOS4DOLFINx: A framework for checkpointing in FEniCS</i>. JOSS, DOI:<a href="https://doi.org/10.21105/joss.06451">10.21105/joss.06451</a> <br> <sup>7</sup>Dokken J.S (2023) <i>Checkpointing in FEniCSx</i>. FEniCS'23 <br><br> -->

# IO4DOLFINx - a unified IO?

<img src="qr_codes/io4dolfinx_qr.png" class="qr-code" vspace=0px width=200px>

<div class="right-skewed-columns">
<div>
<center>
ADIOS4DOLFINx<sup>6</sup> introduced a specific split between readable and visualizable functions.
</center>
</div>
<div>
<center>
<figure>
<img src="./acknowledged_screenshots/adios4dolfinx2023_1.png" vspace=0px width=300>
<img src="./acknowledged_screenshots/adios4dolfinx2023_2.png" vspace=0px width=270>
<figcaption style="font-size: 50%; padding-top: 0px;">
Snapshots from the FEniCS 2023 presentation on checkpointing<sup>7</sup>.
</figcaption>
</figure>
</center>
</div>
</div>

---

<!--  footer: <br><br> -->

# IO4DOLFINx - a unified IO?

<img src="qr_codes/io4dolfinx_qr.png" class="qr-code" vspace=0px width=200px>

Reality is that most users use iso-parameteric finite elements (often P1).

IO4DOLFINx is a <b>backend agnositic</b> interface to many mesh formats.

- `gmsh`, `PyVista`, `XDMF`, `VTKHDF`
  - `{read/write}_{point/mesh}_data`  
- `adios2`, `h5py`
  - `{read/write}_checkpoint`


---


# FEniCSx_ii

<div class="columns">

<div>

<br>
<img src="qr_codes/FEniCSx_ii_qr.png" class="qr-code" vspace=0px width=200px>

</div>

<div>

</div>
</div>


---

# DOLFINx-adjoint

<div class="columns">

<div>

<br>
<img src="qr_codes/DOLFINx_adjoint_qr.png" class="qr-code" vspace=0px width=200px>

</div>

<div>

</div>
</div>


---

# Networks_FEniCSx

<div class="columns">

<div>

<br>
<img src="qr_codes/networks_fenicsx_qr.png" class="qr-code" vspace=0px width=200px>

</div>

<div>

</div>
</div>


---

# Whats next?



---

<img src="qr_codes/irksome_qr.png" vspace=0px width=200px>

---

<img src="qr_codes/FEniCSx_JAX_qr.png" class="qr-code"  vspace=0px width=200px>

