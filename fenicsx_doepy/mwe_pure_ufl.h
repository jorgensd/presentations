
// This code conforms with the UFC specification version 2018.2.0.dev0
// and was automatically generated by FFCx version 0.8.0.dev0.
//
// This code was generated with the following options:
//
//  {'epsilon': 1e-14,
//   'output_directory': '.',
//   'profile': False,
//   'scalar_type': 'float64',
//   'sum_factorization': False,
//   'table_atol': 1e-09,
//   'table_rtol': 1e-06,
//   'ufl_file': ['mwe_pure_ufl.py'],
//   'verbosity': 30,
//   'visualise': False}

#pragma once
#include <ufcx.h>

#ifdef __cplusplus
extern "C" {
#endif

extern ufcx_finite_element element_712b97cf8c20314449c31b6faa0d429d0a020bc4;

extern ufcx_finite_element element_ab3ab6d636218a22f5f7b9a8dcf5e89b39b016c0;

extern ufcx_finite_element element_f99c97188ff012ce840dfb7e251cf41c8f652cf6;

extern ufcx_dofmap dofmap_712b97cf8c20314449c31b6faa0d429d0a020bc4;

extern ufcx_dofmap dofmap_ab3ab6d636218a22f5f7b9a8dcf5e89b39b016c0;

extern ufcx_dofmap dofmap_f99c97188ff012ce840dfb7e251cf41c8f652cf6;

extern ufcx_integral integral_a80de02e2fc39315d8672b75da91b1586209cb47;

extern ufcx_form form_598da04c64d41a43354e41ad7dd3f9021354d1f9;

// Helper used to create form using name which was given to the
// form in the UFL file.
// This helper is called in user c++ code.
//
extern ufcx_form* form_mwe_pure_ufl_a;

// Helper used to create function space using function name
// i.e. name of the Python variable.
//
ufcx_function_space* functionspace_form_mwe_pure_ufl_a(const char* function_name);

#ifdef __cplusplus
}
#endif
