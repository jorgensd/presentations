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

backgroundImage: url('Simula_logo.png')
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
<img src="simula.png" vspace=20px width=150px>

<br/>
<div>
<img src="fenics_logo.png" vspace=20px width=150px>
</div>

</center>

---

# It all started over 21 years ago


<div class="skewed-columns">
<div>
<img src="hpl2005.png" vspace=0px width=600px>
<figcaption style="font-size: 50%; padding-top: 10px;">
FEniCS 05 (Chicago) - <i>Tools for a-Physics Simulation</i><br>  by Hans Petter Langtangen (Simula/UiO)
</figcaption>
</div>
<div>
<div data-marpit-fragment>

Packages developed or maintained at SRL

- UFL
- DOLFINx_MPC
- adios4dolfinx
- scifem
- io4dolfinx
- fenicsx_ii
- DOLFIN(x)-adjoint 

</div>
</div>
</div>

---

