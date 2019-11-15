import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
import matplotlib.pyplot as plt

class Simulation():
    def __init__(self, a, mu, simulation_steps):
        self.A, self.x, self.b = self.__setup_petsc(a, mu, simulation_steps)

        self.ksp = PETSc.KSP().create()
        self.ksp.setType('gmres')
        self.ksp.getPC().setType('none')
        self.ksp.setOperators(self.A)
        self.ksp.setFromOptions()

    def __setup_petsc(self, a, mu, simulation_steps):
        # create PETSc matrix
        nnz = [3 for i in range(simulation_steps)]
        nnz[0] = nnz[-1] = 2
        A = PETSc.Mat()
        A.create(PETSc.COMM_WORLD)
        A.setSizes([simulation_steps, simulation_steps])
        A.setType('mpiaij')
        A.setPreallocationNNZ(nnz)

        row_data = [-a*mu/2, 1, a*mu/2]
        #TODO: cython
        # prvi i zadnji su samo 2 od 3
        Istart, Iend = A.getOwnershipRange()
        for i in range(Istart, Iend):
            if i == 0:
                A.setValues(0, [0, 1], row_data[1:])
            elif i == simulation_steps - 1:
                A.setValues(simulation_steps-1,
                        [simulation_steps - 2, simulation_steps-1],
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

        return A, x, b



    def start(self, initial_condition):
        self.b.createWithArray(initial_condition,
                    comm = PETSc.COMM_WORLD,
                    size = self.A.getSize(),
                    bsize = self.A.getBlockSize())

        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_ylim((-1, 1))
        line1, = ax.plot(self.b.getArray())
        while True:
            self.ksp.solve(self.b, self.x)
            print("iterations: %d residual norm: %g" % (self.ksp.its, self.ksp.norm))
            line1.set_ydata(self.b.getArray())
            fig.canvas.draw()
            fig.canvas.flush_events()
            self.b = self.x


if __name__ == "__main__":
    a = 2
    mu = 0.3
    simulation_step = 1000

    n = np.array(list(range(simulation_step)))
    x_data = np.where(n < 200, np.abs(np.sin(2 * np.pi * n/200)), 0)

    s = Simulation(a, mu, simulation_step)
    s.start(x_data)
