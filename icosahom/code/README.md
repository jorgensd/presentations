# Spack installation

# spack

```bash
 . ./spack/share/spack/setup-env.sh
spack env create fenicsx-july2025
spack env activate fenicsx-july2025
spack add gcc@13.4.0
spack install
spack compiler find
spack add py-fenics-dolfinx@main %gcc@13.4.0 ^py-fenics-basix %gcc@13.4.0 ^petsc+mumps+int64 cflags="-O3" fflags="-O3" ^python@3.12
spack concretize
spack install
```
