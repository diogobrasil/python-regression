"""
@file gradient_descent.py
@brief Implementa o algoritmo de descida do gradiente para regressão linear.
"""

import numpy as np
from Functions.compute_cost import compute_cost


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Executa a descida do gradiente para minimizar a função de custo J(θ)
    no contexto de regressão linear.

    @param X: np.ndarray
        Matriz de entrada (m amostras × n atributos), incluindo termo de bias.
    @param y: np.ndarray
        Vetor de saída esperado com dimensão (m,).
    @param theta: np.ndarray
        Vetor de parâmetros inicial (n,).
    @param alpha: float
        Taxa de aprendizado (learning rate).
    @param num_iters: int
        Número de iterações da descida do gradiente.

    @return: tuple[np.ndarray, np.ndarray]
        theta: vetor otimizado de parâmetros (n,).
        J_history: vetor com o histórico do valor da função de custo em cada iteração (num_iters,).
        theta_history: parâmetros em cada iteração (num_iters+1, n).
    """
    # Obtem o número de amostras
    m = len(y)
    # Inicializa o vetor de custo J_history para armazenar o custo em cada iteração
    J_history = np.zeros(num_iters)

    # Inicializa o vetor theta_history para armazenar os parâmetros em cada iteração
    theta_history = np.zeros((num_iters + 1, len(theta)))

    # Armazena os parâmetros iniciais no vetor theta_history
    theta_history[0] = theta

    for i in range(num_iters):
        # Calcula as previsões (hipótese) com base nos parâmetros atuais
        predictions = np.dot(X, theta)

        # Calcula o erro entre as previsões e os valores reais
        erro = predictions - y

        # Calcula o gradiente da função de custo em relação a theta
        gradient = (1 / m) * np.dot(X.T, erro)

        # Atualiza os parâmetros theta
        theta = theta - alpha * gradient

        # Armazena o custo da iteração atual para análise
        J_history[i] = compute_cost(X, y, theta)

        # Armazena os parâmetros theta da iteração atual para análise
        theta_history[i + 1] = theta

    return theta, J_history, theta_history
