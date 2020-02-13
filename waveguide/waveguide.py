import slepc4py, sys
slepc4py.init(sys.argv)
import petsc4py
from slepc4py import SLEPc
from petsc4py import PETSc
import numpy as np
import scipy
from scipy import constants
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sizes = [100, 100]
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

    h = 1/ysize
    diag = 4

    for y in range(ys, ye):
        for x in range(xs, xe):
            index = y * xsize + x

            # default stencil
            row = index
            cols = [index-xsize, index-1, index, index+1, index+xsize]
            data = [-1, -1, diag, -1, -1]

            if y == 0 or x == 0 or x == (xsize-1) or y == (ysize - 1):
                cols = [index]
                data = [diag]

            A.setValues(row, cols, data)

    A.assemblyBegin()
    A.assemblyEnd()

    E = SLEPc.EPS()
    E.create()

    E.setOperators(A)
    E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
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

        x = np.linspace(-1, 1, xsize)
        y = np.linspace(-1, 1, ysize)
        z = xr.getArray()
        Z = z.reshape(xsize, ysize)
        X, Y = np.meshgrid(x, y)

        plt.contourf(X, Y, Z, 10)
        plt.show()
