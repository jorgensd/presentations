import numpy
numpy.set_printoptions(precision=3, suppress=True)

import basix.ufl


dtype = numpy.float64 # Can be changed to float 32 for lower precision
discontinuous = True
element = basix.ufl.element("Lagrange", "quadrilateral", 2,
                            dtype=dtype, discontinuous=discontinuous)

points = numpy.array([[0.2, 0.5], [0.3,0.82]], dtype=dtype)
print("Basis functions:\n", element.tabulate(0, points))

print("Basis derivatives:\n", element.tabulate(1, points)[1:])