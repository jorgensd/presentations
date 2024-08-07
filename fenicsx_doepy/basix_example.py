import matplotlib.patches
import matplotlib.pyplot
import numpy as np
import basix.ufl
import dolfinx.io

def plot_dof_positions(el, name):
    geom = el.reference_geometry
    perm = dolfinx.cpp.io.perm_vtk(dolfinx.cpp.mesh.to_type(el.cell.cellname()), geom.shape[0])
    geom = geom[perm]
    cell = matplotlib.patches.Polygon(geom, closed=True, fill=False)
    nodes = np.vstack(el._x[0])
    for i in range(1, len(el._x)):
        nodes_i = [xj for xj in el._x[i]]
        nodes_i.append(nodes)
        nodes = np.vstack(nodes_i)

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111)
    ax.add_patch(cell)
    ax.plot(nodes[:, 0], nodes[:, 1], 'o')
    ax.set_title(el.lagrange_variant.name)
    matplotlib.pyplot.savefig(name, transparent=True)


degree = 6
lagrange = basix.ufl.element(
    "Lagrange", "quadrilateral", degree, basix.LagrangeVariant.equispaced)
lagrange_gll = basix.ufl.element(
    "Lagrange", "quadrilateral", degree, basix.LagrangeVariant.gll_warped)

plot_dof_positions(lagrange, "equispaced.png")
plot_dof_positions(lagrange_gll, "gll_warped.png")


