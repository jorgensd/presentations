import subprocess
from pathlib import Path

import dolfinx.io.gmshio
cwd = Path.cwd().absolute().as_posix()
fname = "lv-mesh.msh"
cmd = f"geo lv-ellipsoid {fname}"

subprocess.run(cmd.split(" "))

from mpi4py import MPI
import dolfinx
mesh, ct, ft = dolfinx.io.gmshio.read_from_msh(fname, comm=MPI.COMM_WORLD, rank=0)
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "lv-ellipsoid.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_meshtags(ct, mesh.geometry)
    xdmf.write_meshtags(ft, mesh.geometry)