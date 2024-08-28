import time

import dolfin as df

degree = 5
N = 7

mesh = df.UnitIntervalMesh(10)
f = df.Expression("sin(N*pi*x[0])", N=N, degree=degree, domain=mesh)
int_f = df.assemble(f*df.dx)

x = df.SpatialCoordinate(mesh)
g = df.sin(df.Constant(N)*df.pi*x[0])
int_g = df.assemble(g*df.dx)

print(f"{degree=}, {N=}, {int_f=:.2e}, {int_g=:.2e}, {abs(int_f - int_g)/abs(int_g)*100:.2f}% difference")


x = df.SpatialCoordinate(mesh)
g_slow = df.sin(N*df.pi*x[0])
int_slow = df.assemble(g_slow*df.dx)
print(f"{degree=}, {N=}, {int_g=:.2e}, {int_slow=:.2e}, {abs(int_g - int_slow)/abs(int_slow)*100:.2f}% difference")
