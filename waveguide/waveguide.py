import slepc4py, sys
slepc4py.init(sys.argv)
import petsc4py
from slepc4py import SLEPc
from petsc4py import PETSc
import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


"""
    Funkcija koja vraća np.array kz vrijednosti za odgovarajuću svojstvenu
    vrijednosti i korak diskretizacije h.

    kz^2 = (2 pi f/c)^2 - lambda/h^2
           ^^^^^^^^^^^^   ^^^^^^^^^^
                k^2       kx^2 + ky^2
"""
def compute_kz(f, h, eigenvalue):
    kz2 = np.power(2 * np.pi * f/constants.c, 2) - eigenvalue / h**2
    return np.sqrt(kz2)


"""
    Funkcija koja grafički prikazuje kz u ovisnosti o f.
    Prvo je potrebno izračunati cuf-off frekvenciju (f0) i zatim zadati raspon
    do kojeg se prikazuju kz. Po defaultu će prikazati kz na domeni:
    [f0 + 1, f0 + 4e8].
"""
def plot_kz(h, eigenvalue, plot_label, freq_offset=4e8):
    f0 = constants.c * np.sqrt(eigenvalue)/(2 * np.pi * h)

    print (" f0 = {} Hz".format(f0))

    # +1 tako da nemam negativne brojeve pod korijenom
    f = np.linspace(f0 + 1, f0 + 4e8, num=100)
    kz = compute_kz(f, h, eigenvalue)

    plt.title("Koeficijenti rasprostiranja k_z za različite modove")
    plt.xlabel("f [Hz]")
    plt.ylabel("k_z [rad/m]")
    plt.plot(f, kz, label=plot_label)


"""
    Funkcija koja će grafički prikazati svojstveni vektor koji odgovara
    prostornoj raspodjeli polja u valovodu. Za poziv funkcije su potrebni
    parametri koji određuju stvarne dimenzije problema kao i dimenzije
    diskretiziranog problema. Argument xr označava realnu komponentu
    svojstvenog vektora.

    Argument levels je opcionalni i određuje u koliko koraka konturnih ploha će
    biti korišteno za sliku. Što veći broj, to su prijelazi "glađi".
"""
def plot_eigenfunction(dim, sizes, xr, levels=100):
    x = np.linspace(0, dim[0], sizes[0])
    y = np.linspace(0, dim[1], sizes[1])
    z = xr.getArray()

    Z = z.reshape(sizes[1], sizes[0])
    X, Y = np.meshgrid(x, y)
    plt.title("Razdioba z-komponente električnog polja u presjeku valovoda")
    plt.xlabel("a [cm]")
    plt.ylabel("b [cm]")
    plt.contourf(X, Y, Z, levels)


if __name__ == "__main__":
    # Dimenzije problema u centimentrima
    dim = np.array([5.817, 2.903])
    # Korak diskretizacije
    h = 0.05

    # postavi dimenziju resetke
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
    # stvori matricu, DMDA brine o ispravnoj prealokaciji AIJ matrice
    A = dmda.createMatrix()

    (xs, xe), (ys, ye) = dmda.getRanges()
    _, (xsize, ysize) = dmda.getCorners()

    diag = 4

    # Algoritam popunjavanja matrice
    # ako smo na dijagonali postavi 4
    # elementi koji nisu na rubu imaju puni stencil [-1, -1, 4, -1, -1]
    # elementi koju su na rubu imaju samo jedinicu na dijagonali buduci da
    # je rubni uvjet Dirichletov
    for y in range(ys, ye):
        for x in range(xs, xe):
            index = y * xsize + x

            # default stencil
            row = index
            cols = [index-xsize, index-1, index, index+1, index+xsize]
            data = [-1, -1, diag, -1, -1]

            # ovo se moze i sa MatZeroRowsColumns (vjerojatno je efikasnije)
            if y == 0 or x == 0 or x == (xsize-1) or y == (ysize - 1):
                cols = [index]
                data = [diag]

            A.setValues(row, cols, data)

    A.assemblyBegin()
    A.assemblyEnd()

    # postavi eigenvalue solver
    E = SLEPc.EPS()
    E.create()

    E.setOperators(A)
    # matrica nije hermitska zbog rubnih uvjeta
    E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    E.setFromOptions()
    E.solve()

    # stvori vektore za rješenje
    xr, xi = A.createVecs()

    eigenvalues = []
    for i in range(E.getConverged()):
        k = E.getEigenpair(i, xr, xi)   # vrati svojstvenu vrijednost + vektor
        error = E.computeError(i)       # vrati gresku s kojom smo to dobili
        assert k.imag == 0.0, "Svojstvene vrijednosti trebaju biti realne!"

        print (" {}. Svojstvena vrijednost {}, greska {}".format(
                i+1, k.real, error))

        plot_eigenfunction(dim, sizes, xr)
        plt.show()
        eigenvalues.append(k.real)

    plot_kz(h, eigenvalues[0], "m=1, n=1")
    plot_kz(h, eigenvalues[1], "m=2, n=1")
    plot_kz(h, eigenvalues[2], "m=3, n=1")

    plt.legend()
    plt.show()
