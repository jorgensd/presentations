import numpy as np
from mpi4py import MPI
import dolfinx
import ufl
import basix.ufl

nodes = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
        [0, 0.5],
        [0.5, 0],
        [1.0, 1.0],
        [1.1, 0.5],
        [0.5, 1.15],
    ],
    dtype=np.float32,
)
connectivity = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 6, 8, 7, 5]], dtype=np.int64)

c_el = basix.ufl.element("Lagrange", "triangle", 2, shape=(nodes.shape[1],))
domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, connectivity, nodes, ufl.Mesh(c_el))

with dolfinx.io.VTXWriter(domain.comm, "mesh.bp", domain, engine="BP4") as bp:
    bp.write(0.0)
