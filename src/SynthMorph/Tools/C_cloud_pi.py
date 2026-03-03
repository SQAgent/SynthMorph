import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
import sys
from datetime import datetime

# 配置日志系统
def setup_logger(log_filename=None):
    """
    配置日志记录器
    
    参数:
    log_filename: 日志文件名，如果为None则使用时间戳生成
    """
    # 如果未指定日志文件名，使用时间戳生成
    if log_filename is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"elastic_constants_{current_time}.log"
    
    # 创建日志记录器
    logger = logging.getLogger('ElasticConstantsLogger')
    logger.setLevel(logging.INFO)
    
    # 防止重复添加处理器
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

def elastic_constants_2d(C, theta):
    """
    计算一般各向异性材料在特定方向上的弹性常数
    
    参数:
    C: 3x3弹性刚度矩阵 (一般各向异性)
    theta: 角度 (弧度)
    
    返回:
    E: 杨氏模量
    nu: 泊松比
    G: 切变模量
    """
    # 方向余弦
    l = np.cos(theta)
    m = np.sin(theta)
    
    # 计算柔度矩阵 (C的逆)
    S = np.linalg.inv(C)
    
    # 提取柔度矩阵分量
    S11, S12, S16 = S[0,0], S[0,1], S[0,2]
    S22, S26, S66 = S[1,1], S[1,2], S[2,2]
    
    # 计算变换后的柔度矩阵分量
    S11_prime = (S11*l**4 + S22*m**4 + 
                (2*S12 + S66)*l**2*m**2 + 
                2*S16*l**3*m + 2*S26*l*m**3)
    
    S12_prime = (S12*(l**4 + m**4) + 
                (S11 + S22 - S66)*l**2*m**2 + 
                S16*l*m*(l**2 - m**2) + 
                S26*l*m*(m**2 - l**2))
    
    S66_prime = (S66*(l**4 + m**4) + 
                2*(2*S11 + 2*S22 - 4*S12 - S66)*l**2*m**2 + 
                4*(S16-S26)*l*m*(l**2 - m**2))
    
    # 计算弹性常数
    E = 1.0 / S11_prime
    nu = -S12_prime / S11_prime
    G = 1.0 / S66_prime
    
    return E, nu, G

def plot_all_properties_polar(C, material_name="2D Material", logger=None, save_path=None):
    """
    绘制极坐标下的所有弹性常数云图 - 莫兰迪配色版
    """
    if logger:
        logger.info("开始绘制极坐标弹性常数云图...")
    
    # 莫兰迪配色方案
    morandi_colors = {
        'primary': {
            'dusty_rose': '#D8A496',    # 灰粉色/砖红色
            'sage_green': '#8A9B68',    # 鼠尾草绿
            'slate_blue': '#6B8E9C',    # 灰蓝色
            'muted_teal': '#7BA498',    # 柔和青绿
            'warm_gray': '#A6A6A8',     # 暖灰色
            'dusty_purple': '#9D8CA6',  # 灰紫色
        },
        'light': {
            'light_rose': '#E8C4B8',    # 浅灰粉
            'light_sage': '#B8C2A0',    # 浅鼠尾草
            'light_blue': '#A3BCCC',    # 浅灰蓝
            'light_teal': '#A8C9BC',    # 浅青绿
            'light_gray': '#D0D0D2',    # 浅灰色
        }
    }
    
    # 创建角度数组 (0到2π)
    theta = np.linspace(0, 2*np.pi, 720)

    # 计算各方向的弹性常数
    E_values = np.zeros_like(theta)
    nu_values = np.zeros_like(theta)
    G_values = np.zeros_like(theta)
    
    for i, angle in enumerate(theta):
        E, nu, G = elastic_constants_2d(C, angle)
        E_values[i] = E
        nu_values[i] = nu
        G_values[i] = G
    
    # 三个极坐标子图 (1:1 figure ratio, 2x2 layout)
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(gs[0, 0], projection='polar'),
        fig.add_subplot(gs[0, 1], projection='polar'),
        fig.add_subplot(gs[1, 0], projection='polar'),
    ]
    
    # 绘制杨氏模量 - 灰蓝色系
    axes[0].plot(theta, 1000*E_values, color=morandi_colors['primary']['slate_blue'], 
                linewidth=2.5, alpha=1.0)
    axes[0].fill(theta, 1000*E_values, alpha=0.9, 
                color=morandi_colors['light']['light_blue'])
    # axes[0].fill(theta, 1000*E_values, alpha=0.25, 
    #             color=morandi_colors['primary']['slate_blue'])
    axes[0].set_title('Young\'s Modulus (MPa)', fontsize=20, pad=10, 
                      color=morandi_colors['primary']['slate_blue'], fontweight='medium')
    axes[0].grid(True, color='gray', linestyle='-', linewidth=1.0, alpha=0.7)
    axes[0].tick_params(colors='#666666', labelsize=15, pad=7)
    
    # 绘制泊松比 - 草绿色系
    axes[1].plot(theta, nu_values, color=morandi_colors['primary']['sage_green'], 
                linewidth=2.5, alpha=1.0)
    axes[1].fill(theta, nu_values, alpha=0.9, 
                color=morandi_colors['light']['light_sage'])
    # axes[1].fill(theta, nu_values, alpha=0.25, 
    #             color=morandi_colors['primary']['sage_green'])
    axes[1].set_title('Poisson\'s Ratio', fontsize=20, pad=10, 
                      color=morandi_colors['primary']['sage_green'], fontweight='medium')
    axes[1].grid(True, color='gray', linestyle='-', linewidth=1.0, alpha=0.7)
    axes[1].tick_params(colors='#666666', labelsize=15, pad=7)
    
    # 绘制切变模量 - 灰紫色系
    axes[2].plot(theta, 1000*G_values, color=morandi_colors['primary']['dusty_purple'], 
                linewidth=2.5, alpha=1.0)
    # axes[2].fill(theta, 1000*G_values, alpha=0.9, 
    #             color=morandi_colors['primary']['dusty_purple'])
    axes[2].fill(theta, 1000*G_values, alpha=0.9, 
                color=morandi_colors['light']['light_rose'])
    axes[2].set_title('Shear Modulus (MPa)', fontsize=20, pad=10, 
                      color=morandi_colors['primary']['dusty_purple'], fontweight='medium')
    axes[2].grid(True, color='gray', linestyle='-', linewidth=1.0, alpha=0.7)
    axes[2].tick_params(colors='#666666', labelsize=15, pad=7)
    
    # 添加整体标题
    fig.suptitle(f'{material_name} - Anisotropic Elastic Properties', 
                fontsize=16, color='#444444', fontweight='bold', y=0.98)
    
    # plt.tight_layout()
    fig.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
    
    # 保存图像
    if save_path is None:
        save_path = f"{material_name.replace(' ', '_')}_polar_plot_morandi.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    # if logger:
    #     logger.info(f"图像已保存为: {save_path}")
    # else:
    #     print(f"图像已保存为: {save_path}")
    
    plt.show()
    
    return E_values, nu_values, G_values, save_path

def calculate_principal_constants(C, logger=None):
    """
    计算一般各向异性材料的主方向弹性常数
    """
    # 计算柔度矩阵
    S = np.linalg.inv(C)
    
    # 提取柔度矩阵分量
    S11, S12, S16 = S[0,0], S[0,1], S[0,2]
    S22, S26, S66 = S[1,1], S[1,2], S[2,2]
    
    # 计算主方向弹性常数（在材料坐标系中）
    E1 = 1.0 / S11
    E2 = 1.0 / S22
    nu12 = -S12 / S11
    nu21 = -S12 / S22
    G12 = 1.0 / S66
    
    if logger:
        logger.info(f"计算的主方向弹性常数:")
        logger.info(f"  E1 = {E1:.2f} GPa")
        logger.info(f"  E2 = {E2:.2f} GPa")
        logger.info(f"  nu12 = {nu12:.4f}")
        logger.info(f"  nu21 = {nu21:.4f}")
        logger.info(f"  G12 = {G12:.2f} GPa")
    
    return E1, E2, nu12, nu21, G12

def plot_2d_properties(C, image_path):
    """
    绘制2D弹性常数的极坐标图并保存到指定路径，返回计算结果字典。

    参数:
    C: 3x3弹性刚度矩阵 (一般各向异性)
    image_path: 保存图片的路径
    log_filename: 兼容旧参数，已不再使用
    """
    material_name = "2D Material"
    E1, E2, nu12, nu21, G12 = calculate_principal_constants(C)
    E_values, nu_values, G_values, save_path = plot_all_properties_polar(
        C, material_name, logger=None, save_path=image_path
    )

    return {
        "E1": E1, "E2": E2, "nu12": nu12, "nu21": nu21, "G12": G12,
        "E_range": [float(np.min(E_values)), float(np.max(E_values))],
        "nu_range": [float(np.min(nu_values)), float(np.max(nu_values))],
        "G_range": [float(np.min(G_values)), float(np.max(G_values))],
        "E_mean": float(np.mean(E_values)),
        "nu_mean": float(np.mean(nu_values)),
        "G_mean": float(np.mean(G_values)),
    } , save_path

def main():
    # 设置日志记录器
    logger, log_filename = setup_logger()
    logger.info("=" * 60)
    logger.info("弹性常数计算程序开始运行")
    logger.info("=" * 60)
    
    # 获取用户输入
    try:
        C=np.array([[0.075934921,-0.046007816,0.004416387],
                    [-0.046007816,0.08051132,-0.004410084],
                    [0.004416387,-0.004410084,0.007886424]])

        # 检查矩阵对称性
        if not np.allclose(C, C.T):
            logger.warning("刚度矩阵应该是对称的！")
            
        material_name = "2D Material"
        
        logger.info(f"\n输入的刚度矩阵 (GPa):")
        # 将矩阵转换为字符串以便记录
        matrix_str = "\n".join(["  " + " ".join([f"{val:12.6f}" for val in row]) for row in C])
        logger.info(f"\n{matrix_str}")
        
        # 计算并显示主方向弹性常数
        logger.info("\n" + "="*50)
        logger.info("计算主方向弹性常数...")
        logger.info("="*50)
        
        E1, E2, nu12, nu21, G12 = calculate_principal_constants(C, logger)
        
        logger.info("\n" + "="*50)
        logger.info("主方向弹性常数:")
        logger.info("="*50)
        logger.info(f"E1 (x方向杨氏模量): {E1:.2f} GPa")
        logger.info(f"E2 (y方向杨氏模量): {E2:.2f} GPa")
        logger.info(f"ν12 (x方向受力泊松比): {nu12:.4f}")
        logger.info(f"ν21 (y方向受力泊松比): {nu21:.4f}")
        logger.info(f"G12 (面内切变模量): {G12:.2f} GPa")
        
        # 检查泊松比互易关系
        logger.info(f"\n泊松比互易关系检查: E1*ν21 = E2*ν12")
        logger.info(f"E1*ν21 = {E1*nu21:.4f}, E2*ν12 = {E2*nu12:.4f}")
        logger.info(f"差值: {abs(E1*nu21 - E2*nu12):.6f}")
        
        # 计算各向异性比率
        anisotropy_E = max(E1, E2) / min(E1, E2) if min(E1, E2) > 0 else float('inf')
        logger.info(f"\n各向异性比率 (E_max/E_min): {anisotropy_E:.4f}")
        
        # 可视化
        logger.info("\n开始绘制极坐标图...")
        E_values, nu_values, G_values, _ = plot_all_properties_polar(C, material_name, logger)
        
        # 输出极值信息
        logger.info("\n" + "="*50)
        logger.info("各方向极值统计:")
        logger.info("="*50)
        logger.info(f"杨氏模量范围: {np.min(E_values):.2f} - {np.max(E_values):.2f} GPa")
        logger.info(f"泊松比范围: {np.min(nu_values):.4f} - {np.max(nu_values):.4f}")
        logger.info(f"切变模量范围: {np.min(G_values):.2f} - {np.max(G_values):.2f} GPa")
        
        # 计算平均值
        logger.info("\n平均值统计:")
        logger.info(f"杨氏模量平均值: {np.mean(E_values):.2f} GPa")
        logger.info(f"泊松比平均值: {np.mean(nu_values):.4f}")
        logger.info(f"切变模量平均值: {np.mean(G_values):.2f} GPa")
        
        logger.info("\n" + "="*60)
        logger.info(f"程序运行完成，日志已保存到: {log_filename}")
        logger.info("="*60)
        
    except ValueError as e:
        logger.error(f"输入值错误: {e}")
    except np.linalg.LinAlgError as e:
        logger.error(f"线性代数错误 - 刚度矩阵不可逆: {e}")
    except Exception as e:
        logger.error(f"程序运行过程中发生未知错误: {e}", exc_info=True)
    finally:
        # 关闭所有日志处理器
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

if __name__ == "__main__":
    main()