
import numpy as np
import matplotlib.pyplot as plt
import os

from Functions.plot_data import plot_data
from Functions.gradient_descent import gradient_descent


def main():
    
    # Garante que a pasta de figuras existe
    os.makedirs("Figures", exist_ok=True)

    data = np.loadtxt('Data/ex1data1.txt', delimiter=',')
    # Separa os dados em duas variáveis: x e y
    # x contém a população da cidade (em dezenas de milhar)
    # y contém o lucro (em dezenas de mil dólares)
    # A primeira coluna de data é a população (x), a feature
    # que será usada para prever o lucro.
    x = data[:, 0]
    # A segunda coluna de data é o lucro (y), a label ou target
    y = data[:, 1]
    # Agora, obtemos o número de exemplos de treinamento (m)
    m = len(y)

    x_aug = np.column_stack((np.ones(m), x))

    theta_init = np.array([8.5, 4.0])
    # Parâmetros da descida do gradiente
    # Define o número de iterações e a taxa de aprendizado (alpha)
    # O número de iterações determina quantas vezes os parâmetros serão atualizados.
    iterations = 1500

    # A taxa de aprendizado (alpha) controla o tamanho do passo dado em cada iteração do algoritmo de descida do gradiente.
    # Um alpha muito grande pode fazer o algoritmo divergir, enquanto um muito pequeno pode torná-lo lento.
    # Aqui, alpha é definido como 0.01, um valor comumente usado em problemas de regressão linear.
    # Você pode experimentar outros valores para ver como o algoritmo se comporta.
    alphas = [0.011, 0.01, 0.009]
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

    print(thetas)


if __name__ == '__main__':
    main()
