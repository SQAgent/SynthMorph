import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import random


def young_2dm(S, phi):
    """
    Calculate 2D Young's modulus.

    Parameters:
    S : numpy.ndarray
        3x3 symmetric matrix representing elastic constants of the material.
    phi : float
        Direction angle (in radians).

    Returns:
    float
        Value of Young's modulus.
    """

    # Calculate direction cosines
    x = np.cos(phi)
    y = np.sin(phi)
    

    # Extract components of matrix S
    S11, S12, S13 = S[0, 0], S[0, 1], S[0, 2]
    S22, S23, S33 = S[1, 1], S[1, 2], S[2, 2]
    

    # Calculate Young's modulus
    denominator = (S11 * x**4 + S22 * y**4 + (S33 + 2 * S12) * x**2 * y**2 +
                   2 * (S13 * x**3 * y + S23 * x * y**3))
    E = 1.0 / denominator
    
    return E


def poisson_2dm(S, phi, young_2dm):
    """
    Calculate 2D Poisson's ratio.

    Parameters:
    S : numpy.ndarray
        3x3 symmetric matrix representing elastic constants of the material.
    phi : float
        Direction angle (in radians).
    young_2dm : function
        Function to calculate Young's modulus, accepts S and phi as parameters.

    Returns:
    float
        Value of Poisson's ratio.
    """

    # Calculate direction cosines
    x = np.cos(phi)
    y = np.sin(phi)
    

    # Calculate Young's modulus
    E = young_2dm(S, phi)
    

    # Extract components of matrix S
    S11, S12, S13 = S[0, 0], S[0, 1], S[0, 2]
    S22, S23, S33 = S[1, 1], S[1, 2], S[2, 2]
    

    # Calculate base value, amplitude change, and phase angle of Poisson's ratio
    v0 = ((S11 + S22 - S33) / 2 + 3 * S12) / 4
    rv = np.sqrt((S23 - S13)**2 + (S12 - (S11 + S22 - S33) / 2)**2) / 4
    phiv = np.arctan2((S23 - S13), (S12 - (S11 + S22 - S33) / 2))

    # Calculate Poisson's ratio
    v = -E * (v0 + rv * np.cos(4 * phi + phiv))
    
    return v


def shear_2dm(S, phi):
    """
    Calculate 2D shear modulus.

    Parameters:
    S : numpy.ndarray
        3x3 symmetric matrix representing elastic constants of the material.
    phi : float
        Direction angle (in radians).

    Returns:
    float
        Value of shear modulus.
    """

    # Calculate direction cosines
    x = np.cos(phi)
    y = np.sin(phi)

    # Extract components of matrix S
    S11, S12, S13 = S[0, 0], S[0, 1], S[0, 2]
    S22, S23, S33 = S[1, 1], S[1, 2], S[2, 2]
    

    # Calculate base value, amplitude change, and phase angle of shear modulus
    G0 = (S11 + S12 - 2 * S12 + S33) / 8
    rG = 0.25 * np.sqrt(0.25 * (S33 + 2 * S12 - S11 - S22)**2 + (S23 - S13)**2)
    phiG = np.arctan2(2 * (S13 - S23), (S33 + 2 * S12 - S11 - S22))

    # Calculate shear modulus
    G = 0.25 / (G0 + rG * np.cos(4 * phi + phiG))
    
    return G


def calc_elastic_matrix(target_E: float, target_v: float, phi: float=0): 
    """
    Infer 2D elastic property matrix S from Young's modulus and Poisson's ratio by optimization.

    Parameters:
        target_E : float
            Target Young's modulus.
        target_v : float
            Target Poisson's ratio.
        phi : float
            Direction angle (in radians), default is 0.

    Returns:
        list: Each element is a 3x3 symmetric matrix.
    """
    result_list = []

    def error(S_list, phi=phi) -> float:
        S = np.array([[S_list[0], S_list[1], S_list[2]],
                      [S_list[1], S_list[3], S_list[4]],
                      [S_list[2], S_list[4], S_list[5]]])
        E = young_2dm(S, phi)
        v = poisson_2dm(S, phi, young_2dm)
        return (E - target_E)**2 + (v - target_v)**2
    
    for i in range(3):
        # Randomly generate initial guess, third and fifth values are fixed to 0
        initial_guess = [random.uniform(0.1, 2) if j not in [2, 4] else 0 for j in range(6)]
        result = minimize(error, initial_guess, method='BFGS')
        S_tmp = np.array([[result.x[0], result.x[1], result.x[2]],
                          [result.x[1], result.x[3], result.x[4]],
                          [result.x[2], result.x[4], result.x[5]]])
        E = young_2dm(S_tmp, phi)
        print(f"Optimization {i+1}: Target E = {target_E}, Optimized E = {E}")
        v = poisson_2dm(S_tmp, phi, young_2dm)
        print(f"Optimization {i+1}: Target v = {target_v}, Optimized v = {v}")
        C = np.linalg.inv(S_tmp)
        result_list.append(C)
    return result_list

if __name__ == "__main__":
    target_E = 80.0  # Target Young's modulus
    target_v = -0.3    # Target Poisson's ratio
    phi = 0.0  # Direction angle
    result = calc_elastic_matrix(target_E, target_v, phi)
    print("Optimized Elastic Constants Matrix S:")
    for mat in result:
        print(mat)
    # print(result)