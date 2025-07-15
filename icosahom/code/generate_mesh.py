from mpi4py import MPI
import gmsh
from dolfinx.cpp.mesh import create_cell_partitioner
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import GhostMode, refine, transfer_meshtag, RefinementOption
import pathlib
from dolfinx.io import XDMFFile


def generate_mesh(
    comm,
    rank: int,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    interior_marker: int,
    exterior_marker: int,
    interface_marker: int,
    dirichlet_marker: int,
    num_refinements: int,
    z_bounds: tuple[float, float] | None = None,
):
    gmsh.initialize()
    x_L, x_U = x_bounds
    y_L, y_U = y_bounds
    assert x_L > 0 and x_U < 1
    assert y_L > 0 and y_U < 1
    if z_bounds is None:
        tdim = 2
    else:
        tdim = 3
        z_L, z_U = z_bounds
        assert z_L > 0 and z_U < 1
    if comm.rank == rank:
        if tdim == 2:
            outer = gmsh.model.occ.addRectangle(0, 0, 0, 1, 1, tag=1)
            inner = gmsh.model.occ.addRectangle(
                x_L, y_L, 0, x_U - x_L, y_U - y_L, tag=2
            )
        else:
            outer = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1, tag=1)
            inner = gmsh.model.occ.addBox(
                x_L, y_L, z_L, x_U - x_L, y_U - y_L, z_U - z_L, tag=2
            )
        gmsh.model.occ.synchronize()
        whole_domain, map_to_input = gmsh.model.occ.fragment(
            [(tdim, outer)], [(tdim, inner)]
        )
        gmsh.model.occ.synchronize()

        rec_inner = [idx for (dim, idx) in map_to_input[1] if dim == tdim]

        rec_outer = [
            idx
            for (dim, idx) in map_to_input[0]
            if dim == tdim and idx not in rec_inner
        ]
        gmsh.model.addPhysicalGroup(tdim, rec_inner, tag=interior_marker)
        gmsh.model.addPhysicalGroup(tdim, rec_outer, tag=exterior_marker)

        inner_boundary = gmsh.model.getBoundary(
            [(tdim, e) for e in rec_inner], recursive=False, oriented=False
        )
        outer_boundary = gmsh.model.getBoundary(
            [(tdim, e) for e in rec_outer], recursive=False, oriented=False
        )

        interface = [idx for (dim, idx) in inner_boundary if dim == tdim - 1]
        ext_boundary = [
            idx
            for (dim, idx) in outer_boundary
            if idx not in interface and dim == tdim - 1
        ]
        gmsh.model.addPhysicalGroup(tdim - 1, interface, tag=interface_marker)
        gmsh.model.addPhysicalGroup(tdim - 1, ext_boundary, tag=dirichlet_marker)

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.1)
        gmsh.model.mesh.generate(tdim)

    partitioner = create_cell_partitioner(GhostMode.none)

    mesh_data = model_to_mesh(
        gmsh.model, comm, rank, gdim=tdim, partitioner=partitioner
    )
    gmsh.finalize()
    mesh = mesh_data.mesh
    ct = mesh_data.cell_tags
    ft = mesh_data.facet_tags

    for _ in range(num_refinements):
        mesh.topology.create_entities(1)
        mesh, parent_cell, parent_facet = refine(
            mesh, partitioner=None, option=RefinementOption.parent_cell_and_facet
        )
        cn = ct.name
        fn = ft.name
        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
        ct = transfer_meshtag(ct, mesh, parent_cell, parent_facet)
        ct.name = cn

        mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
        ft = transfer_meshtag(ft, mesh, parent_cell, parent_facet)
        ft.name = fn
    tmp_path = pathlib.Path("tmp_mesh.xdmf")
    if comm.rank == rank:
        if tmp_path.exists():
            tmp_path.unlink()
    comm.Barrier()
    with XDMFFile(comm, tmp_path, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct, mesh.geometry)
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        xdmf.write_meshtags(ft, mesh.geometry)

    with XDMFFile(comm, tmp_path, "r") as xdmf:
        mesh = xdmf.read_mesh(ghost_mode=GhostMode.shared_facet)
        ct = xdmf.read_meshtags(mesh, name="cell_tags")
        mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
        ft = xdmf.read_meshtags(mesh, name="facet_tags")
    comm.Barrier()
    if comm.rank == rank:
        tmp_path.unlink()
        tmp_path.with_suffix(".h5").unlink()
    return mesh, ct, ft


if __name__ == "__main__":
    mesh, ct, ft = generate_mesh(
        MPI.COMM_WORLD, 0, (0.2, 0.8), (0.3, 0.7), 1, 2, 3, 4, 1, z_bounds=(0.3, 0.4)
    )
    from dolfinx.io import XDMFFile

    with XDMFFile(mesh.comm, "initial_mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(ct, mesh.geometry)
        xdmf.write_meshtags(ft, mesh.geometry)
