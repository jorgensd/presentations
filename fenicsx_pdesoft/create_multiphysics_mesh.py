# SPDX-License-Identifier: MIT
# Author: Jørgen S. Dokken
from mpi4py import MPI
import gmsh
import numpy as np


__all__ = ["create_gmsh_model", "top_val", "contact_val", "side_val"]

top_val = 1
contact_val = 2
side_val = 3

def create_gmsh_model(order:int, res:float, comm: MPI.Intracomm, rank:int=0)->gmsh.model:
    gmsh.initialize()
    gmsh.model.add("contact_problem")
    if comm.rank == rank:
        c_x = 0.5
        r = 1
        H = max(0.5, r)
        obstacle = gmsh.model.occ.addDisk(c_x, r, 0, r, r)
        rectangle = gmsh.model.occ.addRectangle(c_x-r, r, 0, 2*r, H)
        gmsh.model.occ.fuse([(2, obstacle)], [(2, rectangle)])
        gmsh.model.occ.synchronize()
        surf = gmsh.model.getEntities(dim=2)
    
        top, contact, side = [], [], []
        boundaries = gmsh.model.getBoundary(surf, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [c_x-r, r+H / 2, 0]):
                side.append(boundary[1])
            elif np.allclose(center_of_mass, [c_x+r, r+H / 2, 0]):
                side.append(boundary[1])
            
            elif np.allclose(center_of_mass, [c_x, r+H, 0]):
                top.append(boundary[1])
            else:
                contact.append(boundary[1])
        
        gmsh.model.addPhysicalGroup(1, side, side_val)
        gmsh.model.addPhysicalGroup(1, top, top_val)
        gmsh.model.addPhysicalGroup(1, contact, contact_val)
        gmsh.model.addPhysicalGroup(2, [s[1] for s in surf], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", res)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(order)
    return gmsh.model


if __name__ == "__main__":
    comm = MPI.COMM_WORLD

    import argparse
    import dolfinx
    from pathlib import Path

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Generate mesh for contact example")
    parser.add_argument("--output", "-o", dest="output", type=Path, default=Path("mesh.xdmf"), help="Name of output file")
    parser.add_argument("--order", dest="order", type=int, default=1, help="Order of mesh geometry")
    parser.add_argument("--res", dest="res", type=float, default=0.05, help="Resolution of mesh")
    args = parser.parse_args()
    gmsh_model = create_gmsh_model(args.order, args.res, comm)

    fname = args.output.with_suffix(".xdmf")
    mesh, ct, ft = dolfinx.io.gmshio.model_to_mesh(gmsh_model, comm, 0, gdim=2)
    with dolfinx.io.XDMFFile(comm, fname, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct, mesh.geometry)
        xdmf.write_meshtags(ft, mesh.geometry)

    gmsh.finalize()
