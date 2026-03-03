import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import logging
import sys
from datetime import datetime

def setup_logger(log_filename=None):
    if log_filename is None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"elastic_constants_{current_time}.log"
    
    logger = logging.getLogger('ElasticConstantsLogger')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

def elastic_constants_2d(C, theta):

    l = np.cos(theta)
    m = np.sin(theta)

    S = np.linalg.inv(C)

    S11, S12, S16 = S[0,0], S[0,1], S[0,2]
    S22, S26, S66 = S[1,1], S[1,2], S[2,2]

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

    E = 1.0 / S11_prime
    nu = -S12_prime / S11_prime
    G = 1.0 / S66_prime
    
    return E, nu, G

def plot_all_properties_polar(C, material_name="2D Material", logger=None, save_path=None):

    if logger:
        logger.info("plt...")
    

    morandi_colors = {
        'primary': {
            'dusty_rose': '#D8A496',    
            'sage_green': '#8A9B68',    
            'slate_blue': '#6B8E9C',    
            'muted_teal': '#7BA498',    
            'warm_gray': '#A6A6A8',     
            'dusty_purple': '#9D8CA6',  
        },
        'light': {
            'light_rose': '#E8C4B8',    
            'light_sage': '#B8C2A0',    
            'light_blue': '#A3BCCC',    
            'light_teal': '#A8C9BC',    
            'light_gray': '#D0D0D2',    
        }
    }
    

    theta = np.linspace(0, 2*np.pi, 720)


    E_values = np.zeros_like(theta)
    nu_values = np.zeros_like(theta)
    G_values = np.zeros_like(theta)
    
    for i, angle in enumerate(theta):
        E, nu, G = elastic_constants_2d(C, angle)
        E_values[i] = E
        nu_values[i] = nu
        G_values[i] = G
    

    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(gs[0, 0], projection='polar'),
        fig.add_subplot(gs[0, 1], projection='polar'),
        fig.add_subplot(gs[1, 0], projection='polar'),
    ]
    
 
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
    

    fig.suptitle(f'{material_name} - Anisotropic Elastic Properties', 
                fontsize=16, color='#444444', fontweight='bold', y=0.98)
    

    fig.tight_layout(pad=3.0, w_pad=3.0, h_pad=3.0)
    
    if save_path is None:
        save_path = f"{material_name.replace(' ', '_')}_polar_plot_morandi.png"
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    plt.show()
    
    return E_values, nu_values, G_values, save_path

def calculate_principal_constants(C, logger=None):

    S = np.linalg.inv(C)
    

    S11, S12, S16 = S[0,0], S[0,1], S[0,2]
    S22, S26, S66 = S[1,1], S[1,2], S[2,2]

    E1 = 1.0 / S11
    E2 = 1.0 / S22
    nu12 = -S12 / S11
    nu21 = -S12 / S22
    G12 = 1.0 / S66
    
    if logger:
        logger.info(f"E:")
        logger.info(f"  E1 = {E1:.2f} GPa")
        logger.info(f"  E2 = {E2:.2f} GPa")
        logger.info(f"  nu12 = {nu12:.4f}")
        logger.info(f"  nu21 = {nu21:.4f}")
        logger.info(f"  G12 = {G12:.2f} GPa")
    
    return E1, E2, nu12, nu21, G12

def plot_2d_properties(C, image_path):

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
    # Set up logger
    logger, log_filename = setup_logger()
    logger.info("=" * 60)
    logger.info("Elastic constants calculation program started.")
    logger.info("=" * 60)
    
    # Get user input
    try:
        C=np.array([[0.075934921,-0.046007816,0.004416387],
                    [-0.046007816,0.08051132,-0.004410084],
                    [0.004416387,-0.004410084,0.007886424]])

        # Check matrix symmetry
        if not np.allclose(C, C.T):
            logger.warning("The stiffness matrix should be symmetric!")
        
        material_name = "2D Material"
        
        logger.info(f"\nInput stiffness matrix (GPa):")
        # Convert matrix to string for logging
        matrix_str = "\n".join(["  " + " ".join([f"{val:12.6f}" for val in row]) for row in C])
        logger.info(f"\n{matrix_str}")
        
        # Calculate and display principal elastic constants
        logger.info("\n" + "="*50)
        logger.info("Calculating principal elastic constants...")
        logger.info("="*50)
        
        E1, E2, nu12, nu21, G12 = calculate_principal_constants(C, logger)
        
        logger.info("\n" + "="*50)
        logger.info("Principal elastic constants:")
        logger.info("="*50)
        logger.info(f"E1 (Young's modulus in x direction): {E1:.2f} GPa")
        logger.info(f"E2 (Young's modulus in y direction): {E2:.2f} GPa")
        logger.info(f"ν12 (Poisson's ratio for x loading): {nu12:.4f}")
        logger.info(f"ν21 (Poisson's ratio for y loading): {nu21:.4f}")
        logger.info(f"G12 (in-plane shear modulus): {G12:.2f} GPa")
        
        # Check Poisson's ratio reciprocity
        logger.info(f"\nPoisson's ratio reciprocity check: E1*ν21 = E2*ν12")
        logger.info(f"E1*ν21 = {E1*nu21:.4f}, E2*ν12 = {E2*nu12:.4f}")
        logger.info(f"Difference: {abs(E1*nu21 - E2*nu12):.6f}")
        
        # Calculate anisotropy ratio
        anisotropy_E = max(E1, E2) / min(E1, E2) if min(E1, E2) > 0 else float('inf')
        logger.info(f"\nAnisotropy ratio (E_max/E_min): {anisotropy_E:.4f}")
        
        # Visualization
        logger.info("\nStart plotting polar chart...")
        E_values, nu_values, G_values, _ = plot_all_properties_polar(C, material_name, logger)
        
        # Output extreme values
        logger.info("\n" + "="*50)
        logger.info("Directional extreme statistics:")
        logger.info("="*50)
        logger.info(f"Young's modulus range: {np.min(E_values):.2f} - {np.max(E_values):.2f} GPa")
        logger.info(f"Poisson's ratio range: {np.min(nu_values):.4f} - {np.max(nu_values):.4f}")
        logger.info(f"Shear modulus range: {np.min(G_values):.2f} - {np.max(G_values):.2f} GPa")
        
        # Calculate averages
        logger.info("\nAverage statistics:")
        logger.info(f"Young's modulus mean: {np.mean(E_values):.2f} GPa")
        logger.info(f"Poisson's ratio mean: {np.mean(nu_values):.4f}")
        logger.info(f"Shear modulus mean: {np.mean(G_values):.2f} GPa")
        
        logger.info("\n" + "="*60)
        logger.info(f"Program finished, log saved to: {log_filename}")
        logger.info("="*60)
        
    except ValueError as e:
        logger.error(f"Value error: {e}")
    except np.linalg.LinAlgError as e:
        logger.error(f"Linear algebra error - stiffness matrix is not invertible: {e}")
    except Exception as e:
        logger.error(f"Unknown error occurred during program execution: {e}", exc_info=True)
    finally:
        # Close all logger handlers
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

if __name__ == "__main__":
    main()