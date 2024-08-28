from mpi4py import MPI
import numpy as np
import basix.ufl
import dolfinx
import ufl
import matplotlib.pyplot as plt


def saw_tooth(x):
    f = 4 * abs(x[0] - 0.43)
    for _ in range(8):
        f = abs(f - 0.3)
    return f



def approximate_sawtooth(N: int, M: int, variant: basix.LagrangeVariant)-> tuple[float, float]:

    msh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, M,
                                          cell_type=dolfinx.mesh.CellType.quadrilateral)

    x = ufl.SpatialCoordinate(msh)
    u_exact = saw_tooth(x)
    ufl_element = basix.ufl.element(basix.ElementFamily.P, 
                                    msh.topology.cell_name(), 10, variant)
    V = dolfinx.fem.functionspace(msh, ufl_element)
    uh = dolfinx.fem.Function(V)
    uh.interpolate(lambda x: saw_tooth(x))

    M = dolfinx.fem.form((u_exact - uh) ** 2 * ufl.dx)
    error = np.sqrt(msh.comm.allreduce(dolfinx.fem.assemble_scalar(M), op=MPI.SUM))

    # Compute the mesh size
    num_cells = msh.topology.index_map(msh.topology.dim).size_local
    cell_sizes= msh.h(msh.topology.dim, np.arange(num_cells, dtype=np.int32))
    hs = msh.comm.allreduce(np.max(cell_sizes), op=MPI.MAX)
    return hs, error

sizes = [5, 10, 20, 80, 160]
hs = np.zeros((2, len(sizes)))
errors = np.zeros((2, len(sizes)))
variants = [basix.LagrangeVariant.equispaced, basix.LagrangeVariant.gll_warped]
for j, variant in enumerate(variants):
    for i, h in enumerate(sizes):
        hs[j, i], errors[j, i] = approximate_sawtooth(h, h, variant)





if MPI.COMM_WORLD.rank == 0:
    M = 150
    x = np.linspace(0, 1, M)
    y = np.linspace(0, 1, M)
    xv, yv = np.meshgrid(x, y)
    zz = saw_tooth(np.vstack([xv.reshape(-1), yv.reshape(-1)])).reshape(xv.shape)



    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    surf = ax.plot_surface(xv, yv, zz, cmap='viridis',rcount=M, ccount=M)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sawtooth function")
    plt.savefig("Sawtooth.png", bbox_inches='tight', pad_inches=0.0, transparent=True)


    fig = plt.figure()
    plt.title("Interpolation of sawtooth into different Lagrange-variants")
    plt.grid(True, which="both")
    plt.loglog(hs[0, :], errors[0, :], "-ro", label=variants[0].name)
    plt.loglog(hs[1, :], errors[1, :], "-bs", label=variants[1].name)
    plt.legend()
    plt.xlabel("Mesh size")
    plt.ylabel(r"$L^2$ error")
    plt.savefig("Errors_2D.png", transparent=True)