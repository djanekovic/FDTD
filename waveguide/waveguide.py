import petsc4py
import sys
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sizes = [3, 3]
    dof = 1
    stencil_width = 1
    boundary_type = None
    stencil_type = PETSc.DMDA.StencilType.STAR

    dmda = PETSc.DMDA().create(dim = len(sizes),
                               dof = dof,
                               sizes = sizes,
                               boundary_type = boundary_type,
                               stencil_type = stencil_type,
                               stencil_width = stencil_width,
                               comm = PETSc.COMM_WORLD)

    A = dmda.createMatrix()

    (xs, xe), (ys, ye) = dmda.getRanges()

    h = 1
    k = 1
    diag = - (4 - h**2 * k**2)

    #TODO: get corners za granice i umjesto ovog sizes[0]
    for y in range(ys, ye):
        for x in range(xs, xe):
            ind = y * sizes[0] + x
            print (x, y)
            #TODO: postavi samo row, col i data i onda na kraju setValues
            if y == 0:
                if x == 0:                          # kut (0, 0)
                    A.setValues(ind,
                                [ind, ind+1, ind+sizes[0]], [diag, 2, 2])
                elif x == (sizes[0] - 1):           # kut (n, 0)
                    A.setValues(ind,
                                [ind-1, ind, ind+sizes[0]], [2, diag, 2])
                else:                               # granica (x, 0)
                    A.setValues(ind,
                                [ind-1, ind, ind+1, ind+sizes[0]],
                                [1, diag, 1, 2])
            elif x == 0:
                if y == (sizes[1] -1):              # kut (0, n)
                    A.setValues(ind,
                                [ind-sizes[0], ind, ind+1], [2, diag, 2])
                else:                               # granica (0, y)
                    A.setValues(ind,
                                [ind-sizes[0], ind, ind+1, ind+sizes[0]],
                                [1, diag, 2, 1])
            elif x == (sizes[0] - 1):               # granica (n, y)
                if y == (sizes[1] - 1):
                    A.setValues(ind,
                                [ind-sizes[0], ind-1, ind],
                                [2, 2, diag])
                else:
                    A.setValues(ind,
                                [ind-sizes[0], ind-1, ind, ind+sizes[0]],
                                [1, 2, diag, 1])
            elif y == (sizes[1] - 1):               # granica (x, n)
                A.setValues(ind,
                            [ind-sizes[0], ind-1, ind, ind+1],
                            [2, 1, diag, 1])
            else:
                A.setValues(ind,
                            [ind-sizes[0], ind-1, ind, ind+1, ind+sizes[0]],
                            [1, 1, diag, 1, 1])

    A.assemblyBegin()
    A.assemblyEnd()

    A.view()

    #TODO:set uniform coordinates
    dmda.view()
