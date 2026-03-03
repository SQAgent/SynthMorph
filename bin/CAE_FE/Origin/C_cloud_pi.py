def analyze_C(C_matrix,state) -> dict:
    """对弹性刚度矩阵进行数值计算和可视化分析。"""
    def elastic_constants_2d(C, theta):
        l = np.cos(theta)
        m = np.sin(theta)
        S = np.linalg.inv(C)
        S11, S12, S16 = S[0,0], S[0,1], S[0,2]
        S22, S26, S66 = S[1,1], S[1,2], S[2,2]
        S11_prime = (S11*l**4 + S22*m**4 + (2*S12 + S66)*l**2*m**2 + 2*S16*l**3*m + 2*S26*l*m**3)
        S12_prime = (S12*(l**4 + m**4) + (S11 + S22 - S66)*l**2*m**2 + S16*l*m*(l**2 - m**2) + S26*l*m*(m**2 - l**2))
        S66_prime = (S66*(l**4 + m**4) + 2*(2*S11 + 2*S22 - 4*S12 - S66)*l**2*m**2 + 4*(S16-S26)*l*m*(l**2 - m**2))
        E = 1.0 / S11_prime
        nu = -S12_prime / S11_prime
        G = 1.0 / S66_prime
        return E, nu, G

    def calculate_principal_constants(C):
        S = np.linalg.inv(C)
        S11, S12, S16 = S[0,0], S[0,1], S[0,2]
        S22, S26, S66 = S[1,1], S[1,2], S[2,2]
        E1 = 1.0 / S11
        E2 = 1.0 / S22
        nu12 = -S12 / S11
        nu21 = -S12 / S22
        G12 = 1.0 / S66
        return E1, E2, nu12, nu21, G12
    C = np.array(C_matrix)
    E1, E2, nu12, nu21, G12 = calculate_principal_constants(C)
    theta = np.linspace(0, 2*np.pi, 720)
    E_values, nu_values, G_values = [], [], []
    for angle in theta:
        E, nu, G = elastic_constants_2d(C, angle)
        E_values.append(E)
        nu_values.append(nu)
        G_values.append(G)
    E_values, nu_values, G_values = np.array(E_values), np.array(nu_values), np.array(G_values)
    # 绘图
    fig, axes = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(18, 7))
    axes[0].plot(theta, 1000*E_values, color='#6B8E9C', linewidth=2.5, alpha=0.8)
    axes[0].fill(theta, 1000*E_values, alpha=0.25, color='#A3BCCC')
    axes[0].set_title('Young\'s Modulus (MPa)', fontsize=20, pad=10, color='#6B8E9C')
    axes[1].plot(theta, nu_values, color='#8A9B68', linewidth=2.5, alpha=0.8)
    axes[1].fill(theta, nu_values, alpha=0.25, color='#B8C2A0')
    axes[1].set_title('Poisson\'s Ratio', fontsize=20, pad=10, color='#8A9B68')
    axes[2].plot(theta, 1000*G_values, color='#9D8CA6', linewidth=2.5, alpha=0.8)
    axes[2].fill(theta, 1000*G_values, alpha=0.25, color='#E8C4B8')
    axes[2].set_title('Shear Modulus (MPa)', fontsize=20, pad=10, color='#9D8CA6')
    material_name = "2D Material"
    plt.suptitle(f'{material_name} - Anisotropic Elastic Properties', fontsize=16, color='#444444', fontweight='bold', y=1.05)
    plt.tight_layout()
    save_path = os.path.join(state["WORKDIR"], f"{material_name.replace(' ', '_')}_polar_plot_morandi.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return {
                "E1": E1, "E2": E2, "nu12": nu12, "nu21": nu21, "G12": G12,
                "E_range": [float(np.min(E_values)), float(np.max(E_values))],
                "nu_range": [float(np.min(nu_values)), float(np.max(nu_values))],
                "G_range": [float(np.min(G_values)), float(np.max(G_values))],
                "E_mean": float(np.mean(E_values)),
                "nu_mean": float(np.mean(nu_values)),
                "G_mean": float(np.mean(G_values)),
                "polar_plot": save_path
            }