import numpy as np
from mpi4py import MPI

import basix.ufl
import dolfinx
import ufl

x = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
              [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
cells = np.array([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=np.int64)
coordinate_element = basix.ufl.element("Lagrange", "quadrilateral", 1,
                                       shape=(x.shape[1],))
msh = dolfinx.mesh.create_mesh(MPI.COMM_SELF, cells, x, ufl.Mesh(coordinate_element))


