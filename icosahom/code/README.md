# Spack installation

# spack

```bash
 . ./spack/share/spack/setup-env.sh
spack env create fenicsx-july2025
spack env activate fenicsx-july2025
spack add gcc@13.4.0
spack install
spack compiler find
spack add py-pip py-scikit-build-core py-nanobind cmake
spack add py-gmsh
spack add fenics-dolfinx@main+petsc+adios2 %gcc@13.4.0 py-fenics-dolfinx@main %gcc@13.4.0 ^petsc+mumps+int64 cflags="-O3" fflags="-O3" %gcc@13.4.0 ^python@3.12
spack concretize
spack install
```
