import numpy as np
from mpi4py import MPI
assert MPI.COMM_WORLD.size >= 2, "This example requires at least 2 MPI processes"

import basix.ufl
import dolfinx
import ufl

if (rank:=MPI.COMM_WORLD.rank) == 0:
    x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    cells = np.array([[0, 1, 3, 4]], dtype=np.int64) 
elif rank == 1:
    x = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
    cells = np.array([[1, 2, 4, 5]], dtype=np.int64)
else:
    x = np.empty((0, 2), dtype=np.float32)
    cells = np.empty((0, 4), dtype=np.int64)
coordinate_element = basix.ufl.element("Lagrange", "quadrilateral", 1,
                                       shape=(x.shape[1],))
msh = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, cells, x, ufl.Mesh(coordinate_element))
print(msh.comm.allreduce(dolfinx.fem.assemble_scalar(dolfinx.fem.form(1*ufl.dx(domain=msh))), op=MPI.SUM))
