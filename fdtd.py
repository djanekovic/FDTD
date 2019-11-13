import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

def main(simulation_step):
    a = 1
    mu = 0.5
    n = np.array(list(range(simulation_step)))
    x_data = 1 * np.exp(-((n-50)/150)**2)

    # create PETSc matrix
    nnz = [3 for i in range(simulation_step)]
    nnz[0] = nnz[-1] = 2
    A = PETSc.Mat()
    A.create(PETSc.COMM_WORLD)
    A.setSizes([simulation_step, simulation_step])
    A.setType('mpiaij')
    A.setPreallocationNNZ(nnz)

    row_data = [-a*mu/2, 1, a*mu/2]
    #TODO: cython
    # prvi i zadnji su samo 2 od 3
    Istart, Iend = A.getOwnershipRange()
    for i in range(Istart, Iend):
        if i == 0:
            A.setValues(0, [0, 1], row_data[1:])
        elif i == simulation_step - 1:
            A.setValues(simulation_step-1,
                    [simulation_step - 2, simulation_step-1],
                    row_data[:2])
        else:
            A.setValues(i, [i-1, i, i+1], row_data)
    A.assemblyBegin()
    A.assemblyEnd()

    x = PETSc.Vec()
    b = PETSc.Vec()
    x.createMPI(comm = PETSc.COMM_WORLD,
                size = A.getSize(),
                bsize = A.getBlockSize())
    b.createWithArray(x_data,
                comm = PETSc.COMM_WORLD,
                size = A.getSize(),
                bsize = A.getBlockSize())

    ksp = PETSc.KSP().create()
    ksp.setType('gmres')
    ksp.getPC().setType('none')
    ksp.setOperators(A)
    ksp.setFromOptions()

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(b.getArray())
    while True:
        ksp.solve(b, x)
        print("iterations: %d residual norm: %g" % (ksp.its, ksp.norm)) 
        line1.set_ydata(b.getArray())
        fig.canvas.draw()
        fig.canvas.flush_events()
        b = x

if __name__ == "__main__":
    simulation_step = 1000
    main(simulation_step)
