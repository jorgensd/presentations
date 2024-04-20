from dolfin import *
import time
mesh = UnitSquareMesh(1, 1)
V = FunctionSpace(mesh, "Lagrange", 5)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
for i in range(3):
    start = time.perf_counter()
    assemble(a)
    end = time.perf_counter()
    print(f"{i}: {end-start:.2e}")