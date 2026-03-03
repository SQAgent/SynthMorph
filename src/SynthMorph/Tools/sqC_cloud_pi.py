import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
import random


def young_2dm(S, phi):
    """
    计算二维杨氏模量 (Young's Modulus)。
    
    参数:
    S : numpy.ndarray
        3x3 对称矩阵，表示材料的弹性常数。
    phi : float
        方向角（弧度制）。
    
    返回:
    float
        杨氏模量的值。
    """
    # 计算方向余弦
    x = np.cos(phi)
    y = np.sin(phi)
    
    # 提取矩阵 S 的分量
    S11, S12, S13 = S[0, 0], S[0, 1], S[0, 2]
    S22, S23, S33 = S[1, 1], S[1, 2], S[2, 2]
    
    # 计算杨氏模量
    denominator = (S11 * x**4 + S22 * y**4 + (S33 + 2 * S12) * x**2 * y**2 +
                   2 * (S13 * x**3 * y + S23 * x * y**3))
    E = 1.0 / denominator
    
    return E


def poisson_2dm(S, phi, young_2dm):
    """
    计算二维泊松比 (Poisson's Ratio)。
    
    参数:
    S : numpy.ndarray
        3x3 对称矩阵，表示材料的弹性常数。
    phi : float
        方向角（弧度制）。
    young_2dm : function
        用于计算杨氏模量的函数，接受 S 和 phi 作为参数。
    
    返回:
    float
        泊松比的值。
    """
    # 计算方向余弦
    x = np.cos(phi)
    y = np.sin(phi)
    
    # 计算杨氏模量
    E = young_2dm(S, phi)
    
    # 提取矩阵 S 的分量
    S11, S12, S13 = S[0, 0], S[0, 1], S[0, 2]
    S22, S23, S33 = S[1, 1], S[1, 2], S[2, 2]
    
    # 计算泊松比的基值、幅值变化和相位角
    v0 = ((S11 + S22 - S33) / 2 + 3 * S12) / 4
    rv = np.sqrt((S23 - S13)**2 + (S12 - (S11 + S22 - S33) / 2)**2) / 4
    phiv = np.arctan2((S23 - S13), (S12 - (S11 + S22 - S33) / 2))
    
    # 计算泊松比
    v = -E * (v0 + rv * np.cos(4 * phi + phiv))
    
    return v


def shear_2dm(S, phi):
    """
    计算二维剪切模量 (Shear Modulus)。
    
    参数:
    S : numpy.ndarray
        3x3 对称矩阵，表示材料的弹性常数。
    phi : float
        方向角（弧度制）。
    
    返回:
    float
        剪切模量的值。
    """
    # 计算方向余弦
    x = np.cos(phi)
    y = np.sin(phi)
    
    # 提取矩阵 S 的分量
    S11, S12, S13 = S[0, 0], S[0, 1], S[0, 2]
    S22, S23, S33 = S[1, 1], S[1, 2], S[2, 2]
    
    # 计算剪切模量的基值、幅值变化和相位角
    G0 = (S11 + S12 - 2 * S12 + S33) / 8
    rG = 0.25 * np.sqrt(0.25 * (S33 + 2 * S12 - S11 - S22)**2 + (S23 - S13)**2)
    phiG = np.arctan2(2 * (S13 - S23), (S33 + 2 * S12 - S11 - S22))
    
    # 计算剪切模量
    G = 0.25 / (G0 + rG * np.cos(4 * phi + phiG))
    
    return G


def calc_elastic_matrix(target_E: float, target_v: float, phi: float=0): 
    """
    从杨氏模量 通过优化推断二维弹性性质矩阵 S。
    
    参数:
        target_E : float
            目标杨氏模量。
        phi : float
            方向角（弧度制）,默认为0。
    
    返回:

    list: 每个元素都是一个3*3的矩阵，且矩阵必定对称
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
        # 随机生成初始猜测值 initial_guess，第三个和第五个值固定为 0
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
    target_E = 80.0  # 目标杨氏模量
    target_v = -0.3    # 目标泊松比
    phi = 0.0  # 方向角
    result = calc_elastic_matrix(target_E, target_v, phi)
    print("Optimized Elastic Constants Matrix S:")
    for mat in result:
        print(mat)
    # print(result)