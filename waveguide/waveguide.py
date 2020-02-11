import slepc4py, sys
slepc4py.init(sys.argv)
import petsc4py
from slepc4py import SLEPc
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sizes = [4, 3]
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
    _, (xsize, ysize) = dmda.getCorners()

    h = 1/xsize
    k = 2 * np.pi
    diag = +4

    for y in range(ys, ye):
        for x in range(xs, xe):
            index = y * xsize + x

            # default stencil
            row = index
            cols = [index-xsize, index-1, index, index+1, index+xsize]
            data = [1, 1, diag, 1, 1]

            if y == 0:
                if x == 0:                          # kut (0, 0)
                    cols = [index, index + 1, index+xsize]
                    data = [diag, 2, 2]
                elif x == (xsize - 1):           # kut (n, 0)
                    cols = [index - 1, index, index + xsize]
                    data = [2, diag, 2]
                else:                               # granica (x, 0)
                    cols = [index-1, index, index+1, index + xsize]
                    data = [1, diag, 1, 2]
            elif x == 0:
                if y == (ysize -1):              # kut (0, n)
                    cols = [index-xsize, index, index+1]
                    data = [2, diag, 2]
                else:                               # granica (0, y)
                    cols = [index-xsize, index, index+1, index+xsize]
                    data = [1, diag, 2, 1]
            elif x == (xsize - 1):               # granica (n, y)
                if y == (ysize - 1):
                    cols = [index-xsize, index-1, index]
                    data = [2, 2, diag]
                else:
                    cols = [index-xsize, index-1, index, index+xsize]
                    data = [1, 2, diag, 1]
            elif y == (ysize - 1):               # granica (x, n)
                cols = [index-xsize, index-1, index, index+1]
                data = [2, 1, diag, 1]

            A.setValues(row, cols, data)

    A.assemblyBegin()
    A.assemblyEnd()

    E = SLEPc.EPS()
    E.create()

    E.setOperators(A)
    E.setProblemType(SLEPc.EPS.ProblemType.HEP)
    E.setFromOptions()
    E.solve()

    xr, xi = A.createVecs()

    print ("=" * 79)
    print ("    Našao {} svojstvene vrijednosti".format(E.getConverged()))
    print ("=" * 79)
    for i in range(E.getConverged()):
        k = E.getEigenpair(i, xr, xi)
        error = E.computeError(i)
        if k.imag != 0.0:
            print ("{}+{}i {}".format(k.real, k.imag, error))
        else:
            print (k.real, error)
