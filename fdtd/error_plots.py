import matplotlib.pyplot as plt
import numpy as np

def dissipation_aprox(a, mu, x):
    return np.sqrt()

def dissipation(a, mu, x):
    return np.sqrt(1/(1 + a**2 * mu**2 * np.power(np.sin(x), 2)))

def main():
    x = np.linspace(0, np.pi/2)
    plt.plot(dissipation(1, 0.5, x))
    plt.show()


if __name__ == "__main__":
    main()
