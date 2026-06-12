from mpi4py import MPI
import dolfinx
import numpy as np
from scifem import create_space_of_simple_functions
comm = MPI.COMM_WORLD

mesh = dolfinx.mesh.create_unit_square(comm, 10, 10)
tdim = mesh.topology.dim
tol = 1e-14
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

V = create_space_of_simple_functions(mesh, ct, tags)
u = dolfinx.fem.Function(V)
u.x.array[0] = 3.2
u.x.array[1] = 5.5
u.x.array[2] = 4.2
assert len(u.x.array) == 3

import pyvista
sargs = dict(
    vertical=True,    
    width=0.06,       # Make it slightly thinner (was 0.1)
    height=0.6,       
    position_x=0.90,  # Push it further right (was 0.8)
    position_y=0.2,   
    title_font_size=80,
    label_font_size=55,
    fmt="%.1f",       
    shadow=False,     
    color="black"     
)
pl = pyvista.Plotter(off_screen=True)
msh = pyvista.UnstructuredGrid(*dolfinx.plot.vtk_mesh(mesh))
msh.cell_data["u"] = u.x.array[V.dofmap.list].flatten()
pl.add_mesh(msh, scalars="u", show_edges=True, scalar_bar_args=sargs,
            reset_camera=True)
pl.camera.zoom(1.2)  # A slight zoom to make the fit tighter
pl.view_xy()
image_path = "simple_function.png"
pl.screenshot(image_path, transparent_background=True, window_size=[2000, 2000])
