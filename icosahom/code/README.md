# Spack installation

# spack

```bash
 . ./spack/share/spack/setup-env.sh
spack env create fenicsx-july2025
spack env activate fenicsx-july2025
spack add py-fenics-dolfinx@main fenics-dolfinx+adios2 ^adios2+python ^petsc+mumps+int64 cflags="-O3" fflags="-O3"
spack concretize

```
