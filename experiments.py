
import numpy as np
import matplotlib.pyplot as plt
import os

from Functions.compute_cost import compute_cost
from Functions.gradient_descent import gradient_descent


def main():
    
    # Garante que a pasta de figuras existe
    os.makedirs("Figures", exist_ok=True)

    data = np.loadtxt('Data/ex1data1.txt', delimiter=',')

    x = data[:, 0]

    y = data[:, 1]

    m = len(y)

    x_aug = np.column_stack((np.ones(m), x))

    theta_init = np.array([0, 0])

    iterations = 1500

    # 1.Teste de diferentes taxas de aprendizado
    alphas = [0.005, 0.01, 0.001]
    colors = ['b','g', 'r']
    thetas = []

    # Gráfico da convergência da função de custo
    plt.figure(figsize=(8, 5))
    for alpha, color in zip(alphas, colors):
        theta, J_history, _ = gradient_descent(x_aug, y, theta_init, alpha, iterations)
        plt.plot(np.arange(1, iterations + 1), J_history, color, label=f'α = {alpha}')
        thetas.append(theta)
    
    plt.xlabel('Iteração')
    plt.ylabel('Custo J(θ)')
    plt.title('Comparação de Convergência para diferentes valores de α')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.savefig("Figures/experimento_taxa_aprendizado.png", dpi=300, bbox_inches='tight')
    plt.savefig("Figures/experimento_taxa_aprendizado.svg", format='svg', bbox_inches='tight')
    plt.show()

    print('Valores finais de theta para diferentes taxas de aprendizado:')
    for alpha, theta in zip(alphas, thetas):
        print(f'α = {alpha}: θ = {theta}')

    # 2.Testando diferentes inicializações de theta
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    j_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
    for i, t0 in enumerate(theta0_vals):
        for j, t1 in enumerate(theta1_vals):
            j_vals[i, j] = compute_cost(x_aug, y, np.array([t0, t1]))
    j_vals = j_vals.T

    theta_inits = [
        np.array([0, 0]),
        np.array([8.5, 4]),
        np.array([-3.63, 1.16]),
        np.random.randn(2),
        np.random.randn(2),
        np.random.randn(2)
    ]

    plt.figure(figsize=(8, 6))
    plt.contour(theta0_vals, theta1_vals, j_vals, levels=np.logspace(-2, 3, 20))
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')
    plt.title('Trajetórias de Gradiente com Diferentes Inicializações')
    plt.grid(True)

    for i, init_theta in enumerate(theta_inits):
        _, _, th_hist = gradient_descent(x_aug, y, init_theta.copy(), alpha=0.01, num_iters=iterations)
        label = f'init {i+1}: {init_theta.round(2)}'
        plt.plot(th_hist[:, 0], th_hist[:, 1], marker='o', markersize=3, label=label)

    plt.legend(fontsize=8)
    plt.savefig("Figures/experimento_inicializacao_pesos.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    main()
