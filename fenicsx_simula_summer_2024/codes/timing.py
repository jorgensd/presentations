import time
import argparse
import os
import pathlib
from mpi4py import MPI
import numpy as np

class Timer:
    def __init__(self, name: str):
        self.t = 0
        self.elapsed = 0
        self.num_calls = 0
        self.name = name
        self.max = None
        self.min = None

    def start(self):
        self.t = time.perf_counter()
        self.num_calls += 1

    def stop(self):
        end = time.perf_counter()
        self.elapsed += end - self.t
        if self.min is None or end - self.t < self.min:
            self.min = end - self.t
        if self.max is None or end - self.t > self.max:
            self.max = end - self.t
        return end - self.t

    def __enter__(self):
        self.start()

    def __exit__(self, *args):
        print(f"{self.name}: {self.stop():.3e}")

class TimerContainer:
    def __init__(self):
        self.timers = {}

    def add_timer(self, timer: Timer):
        if timer.min is None:
            min_val = 0
        else:
            min_val = timer.min
        if timer.max is None:
            max_val = 0
        else:
            max_val = timer.max
        self.timers[timer.name] = {"runtime": timer.elapsed, "num_calls": timer.num_calls, "min": min_val, "max": max_val}

    def create_table(self, outfile: pathlib.Path):
        header = "Operation Min Max Avg Number of calls\n"
        content = ""
        for name, timer in self.timers.items():
            time = MPI.COMM_WORLD.gather(timer["runtime"], root=0)
            num_ops = MPI.COMM_WORLD.gather(timer["num_calls"], root=0)
            min_times = MPI.COMM_WORLD.gather(timer["min"], root=0)
            max_times = MPI.COMM_WORLD.gather(timer["max"], root=0)
            if MPI.COMM_WORLD.rank != 0:
                continue
            for op in num_ops:
                assert op == num_ops[0], "Operation called different number of times on each process"
            if num_ops[0] == 0:
                continue
            content += f"{name} {np.min(min_times)} {np.max(max_times)} {np.sum(time)/(num_ops[0]*len(time))} {num_ops[0]}\n"            
        if MPI.COMM_WORLD.rank == 0:
            outfile.parent.mkdir(exist_ok=True)
            with open(outfile, "w") as f:
                f.write(header)
                f.write(content)


def rmtree(f: pathlib.Path):
    if not f.exists():
        pass
    elif f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            rmtree(child)
        f.rmdir()

parser = argparse.ArgumentParser(description='Compilation example for old FEniCS')
parser.add_argument('--degree', "-d", type=int, default=1, help='Degree of the finite element space')
parser.add_argument("--repeats", "-r", type=int, default=5, help="Number of times to repeat the simulation")
parser.add_argument("-N", type=int, default=50, help="Number of elements in each direction of the mesh")
parser.add_argument("--backend", "-b", type=str, default="dolfinx", choices=["dolfinx", "dolfin"], help="Which backend to use")
parser.add_argument("--out_prefix", "-o", type=str, default="results", help="Prefix for output files")

if __name__ == "__main__":
    args = parser.parse_args()
    degree = args.degree
    repeat = args.repeats
    backend = args.backend
    N = args.N
    prefix = args.out_prefix

    total_t = Timer("Total")
    mesh_t = Timer("Mesh")
    V_t = Timer("FunctionSpace")
    form_compilation_t = Timer("Compile-form")
    assembly_lhs_t = Timer("Assemble-LHS")
    assembly_rhs_t = Timer("Assemble-RHS")
    solve_t = Timer("Solve")


    alpha = 0.1
    f = 0.01

    cache_dir = pathlib.Path.cwd() / (backend + "-cache")

    if MPI.COMM_WORLD.rank == 0:
        rmtree(cache_dir)
    MPI.COMM_WORLD.barrier()

    container = TimerContainer()
    if backend == "dolfin":
        os.environ["DIJITSO_CACHE_DIR"] = cache_dir.absolute().as_posix()
        import dolfin
        for i in range(repeat):
            total_t.start()
            mesh_t.start()
            mesh = dolfin.UnitCubeMesh(N, N, N)
            mesh_t.stop()

            V_t.start()
            V = dolfin.FunctionSpace(mesh, "Lagrange", degree)
            V_t.stop()

            u = dolfin.TrialFunction(V)
            v = dolfin.TestFunction(V)
            u_n = dolfin.Function(V)
            alpha_c = dolfin.Constant(alpha)
            f_c = dolfin.Constant(f)
            a = alpha_c * dolfin.dot(dolfin.grad(u), dolfin.grad(v)) * dolfin.dx + u * v * dolfin.dx
            L = u_n * v *dolfin.dx  + alpha_c * dolfin.Constant(f) * v * dolfin.dx
            

            uh = dolfin.Function(V)
            assembly_lhs_t.start()
            A = dolfin.assemble(a)
            assembly_lhs_t.stop()

            assembly_rhs_t.start()
            b = dolfin.assemble(L)
            assembly_rhs_t.stop()

            solve_t.start()
            dolfin.solve(A, uh.vector(), b, "mumps")
            solve_t.stop()
            total_t.stop()

    elif backend == "dolfinx":
        from petsc4py import PETSc
        import dolfinx.fem.petsc
        import ufl
        jit_options = {"cache_dir": cache_dir.absolute().as_posix()}
        for i in range(repeat):
            total_t.start()
            mesh_t.start()
            mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)
            mesh_t.stop()

            
            V_t.start()
            V = dolfinx.fem.functionspace(mesh, ("Lagrange", degree), jit_options=jit_options)
            V_t.stop()

            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)
            u_n = dolfinx.fem.Function(V)
            alpha_c = dolfinx.fem.Constant(mesh, alpha)
            f_c = dolfinx.fem.Constant(mesh, f)
            a = alpha_c * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + u * v * ufl.dx
            L = u_n * v *ufl.dx  + alpha_c * dolfinx.fem.Constant(mesh, f) * v * ufl.dx
            
            form_compilation_t.start()
            a = dolfinx.fem.form(a, jit_options=jit_options)
            L = dolfinx.fem.form(L, jit_options=jit_options)
            form_compilation_t.stop()

            uh = dolfinx.fem.Function(V)
            assembly_lhs_t.start()
            A = dolfinx.fem.petsc.assemble_matrix(a)
            A.assemble()
            assembly_lhs_t.stop()

            assembly_rhs_t.start()
            b = dolfinx.fem.petsc.assemble_vector(L)
            b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            assembly_rhs_t.stop()

            solve_t.start()
            ksp = PETSc.KSP().create(mesh.comm)
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            ksp.setOperators(A)
            ksp.solve(b, uh.x.petsc_vec)
            uh.x.scatter_forward()
            solve_t.stop()
            total_t.stop()

            b.destroy()
            A.destroy()
            ksp.destroy()

    container.add_timer(mesh_t)
    container.add_timer(V_t)
    container.add_timer(form_compilation_t)
    container.add_timer(assembly_lhs_t)
    container.add_timer(assembly_rhs_t)
    container.add_timer(solve_t)
    container.add_timer(total_t)

    outfile = (pathlib.Path(f"{prefix}") / f"{backend}_{N=}_{degree=}_{MPI.COMM_WORLD.size}").with_suffix(".txt")
    container.create_table(outfile)