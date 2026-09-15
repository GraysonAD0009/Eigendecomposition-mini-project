"""
Nuclear Physics Many-Body Problem Solver.

This module provides functions to solve the 2-body nuclear Schrödinger equation
using the Finite Difference Method (FDM) and exact eigen decomposition.
"""
from typing import Tuple
import numpy as np
import scipy.sparse as sp  # type: ignore[import-untyped]
import scipy.sparse.linalg as spla  # type: ignore[import-untyped]


def eigen_decomposition(
    points: int = 40,
    size: float = 6.0,
    V_0: float = -50.0, 
    sigma: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Solves the 2-body nuclear Schrodinger equation using the Finite Difference Method.
    Please make yourrself aware of the many body Schrodinger equation.

    Parameters
    ----------
    points : int
        Number of spatial grid points for each particle's coordinate.
    size : float
        The size of the nuclear well boundary [-L/2, L/2] in femtometers (fm).
    V_0 : float
        Strength of the short-range nucleon-nucleon interaction (MeV).
    sigma : float
        Range of the nuclear interaction (fm).

    Returns
    -------
    eigenvalues : ndarray
        The lowest 3 energy eigenvalues (MeV) of the many-body system.
    eigenvectors : ndarray
        The corresponding spatial wavefunctions (eigenvectors).
    x : ndarray
        The 1D coordinate grid vector used for the spatial discretization.
    V_2D : ndarray
        The 2D interaction potential matrix evaluated across the spatial grid.
    """
    N = points
    L = size

    # Establish the coordinate grid
    # Higher numerical accuracy will be determined, but matrice size will increase.
    x = np.linspace(-L/2, L/2, N)
    dx = x[1] - x[0]
    # Nucleon Constant
    hbar = 41.47

    # Builds the 1 Dimensional Single-Particle Kinetic Energy Matrix
    # Kinetic energy equation for quantum mechanics specifically is required for this
    # Use the 3-point central finite difference (learned this Monday, kind of)
    # Holds (-2.0 / dx**2) coefficient
    main_diagonal = np.ones(N) * (-2.0 / dx**2)
    # Holds (1.0 / dx**2) coefficient
    off_diagonal = np.ones(N - 1) * (1.0 / dx**2)
    # Use csr so we can bring out all of the 0s, it makes it look nicer, that's all
    T_1D = sp.diags([off_diagonal, main_diagonal, off_diagonal], [-1, 0, 1], format='csr')

    # Now we can use the 1D and build off of that to do 2D
    # and create the many body problem
    # Kronker sums for the 2D Kinetic Energy
    # This maps the 1D derivatives into the full 2D coordinate space (x1, x2)
    # Identity matrix
    Id_mt = sp.eye(N, format='csr')
    # sp.kron(T_1D, Id_mt) multiplies kinetic energy and the Id_mt for particle one
    # sp.kron(Id_mt, T_1D) multiplies kinetic energy and the Id_mt for particle two
    T_mb = -0.5 * hbar * (sp.kron(T_1D, Id_mt) + sp.kron(Id_mt, T_1D))

    # 2 body potential matrice
    # ij indexing is for purely matrix and tensor indexing
    X1, X2 = np.meshgrid(x, x, indexing='ij')

    # Gaussian potential for short distance nuclear force
    distance = np.abs(X1 - X2)
    # Poteential Energy
    V_2D = V_0 * np.exp(-(distance**2) / (2 * sigma**2))

    # Turn the 2D matrix into a 1D vector
    V_flat = V_2D.flatten()
    # Place that 1D vector on the main diagonal of the sparse
    V_mb = sp.diags(V_flat, 0, format='csr')

    # Sum of the Many Body Hamiltonian
    Ham = T_mb + V_mb

    # Eigen Decomposition
    # Matrices will be large so we use sparse
    # Sparse Linear Algebra, Hermitian
    eigenvalues, eigenvectors = spla.eigsh(Ham, k=3)

    return eigenvalues, eigenvectors, x, V_2D


def main() -> :
    """
    Execute the core 2-body eigen decomposition and print results.

    This function serves as the primary entry point for the module script
    execution, processing default values and showing successfully resolved energies.
    """
    # Execute decomposition
    eigenvalues, eigenvectors, x, V_2D = eigen_decomposition()


if __name__ == "__main__":
    main()
