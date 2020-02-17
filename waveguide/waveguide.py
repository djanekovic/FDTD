import slepc4py, sys
slepc4py.init(sys.argv)
import petsc4py
from slepc4py import SLEPc
from petsc4py import PETSc
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

def compute_kz(x, h, eigenvalue):
    kz2 = np.power(2 * np.pi * x/constants.c, 2) - eigenvalue / h**2
    return np.sqrt(kz2)

def plot_eigenvalue(h, eigenvalue):
    f0 = constants.c * np.sqrt(eigenvalue)/(2 * np.pi * h)

    print (" f0 = {}".format(f0))

    # +1 tako da nemam negativne brojeve pod korijenom
    x = np.linspace(f0+1, f0 + 10e9)
    y = compute_kz(x, h, eigenvalue)


    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    dim = np.array([5.817, 2.903])
    h = 0.05

    sizes = (dim / h).astype(np.int64)
    dof = 1
    stencil_width = 1
    boundary_type = None
    stencil_type = PETSc.DMDA.StencilType.STAR

    print ("=" * 79)
    print (" Problem dimenzija: {} [cm]".format(dim))
    print (" Korak diskretizacije u obje dimenzije: {}".format(h))
    print (" Problem rezultira matricom dimenzija {}".format(sizes))
    print ("=" * 79)

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
    eigenvalues = []
    for i in range(E.getConverged()):
        k = E.getEigenpair(i, xr, xi)
        error = E.computeError(i)
        assert k.imag == 0.0, "Svojstvene vrijednosti nisu realne!"

        print (" {}. Svojstvena vrijednost {}, greska {}".format(
                i+1, k.real, error))

        x = np.linspace(0, dim[0], sizes[0])
        y = np.linspace(0, dim[1], sizes[1])
        z = xr.getArray()

        Z = z.reshape(sizes[1], sizes[0])
        X, Y = np.meshgrid(x, y)
        plt.contourf(X, Y, Z, 100)
        plt.show()

        eigenvalues.append(k.real)

    plot_eigenvalue(h, eigenvalues[0])
    plot_eigenvalue(h, eigenvalues[1])
