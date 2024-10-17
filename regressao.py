import numpy as np
import matplotlib.pyplot as plt

# Carregar os dados
dados = np.loadtxt('aerogerador.dat', delimiter='\t')

# Organizar os dados
X = dados[:, 0].reshape(-1, 1)  # Variável regressor (velocidade do vento)
y = dados[:, 1].reshape(-1, 1)  # Variável dependente (potência gerada)

# Adicionar coluna de 1s para o intercepto
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)  # X agora é R^N x 2

# Gráfico
plt.scatter(X[:, 1], y, color='purple')
plt.title('Gráfico de Espalhamento: Velocidade do Vento vs. Potência Gerada')
plt.xlabel('Velocidade do Vento')
plt.ylabel('Potência Gerada')
plt.xlim(0, np.max(X[:, 1]) + 100)  # Ajustando os limites do eixo x
plt.ylim(0, np.max(y) + 100)        # Ajustando os limites do eixo y
plt.grid(True)
plt.show()

# Verificando as dimensões das matrizes
print("Dimensões de X:", X.shape)  # Deve ser (N, 2)
print("Dimensões de y:", y.shape)  # Deve ser (N, 1)

# Rodadas
R = 500

# Valores de lambda
lambdas = [0, 0.25, 0.5, 0.75, 1]

# Lista para armazenar os RSS de cada modelo
resultados_rss = {0: [], 'MQO Tradicional': []}  # Inicializando com as chaves corretas
for lambd in lambdas[1:]:
    resultados_rss[lambd] = []  # Inicializa as chaves para os modelos regularizados

# Simulação por Monte Carlo
for _ in range(R):
    # Particionamento dos dados em treinamento e teste (80% treino, 20% teste)
    indices = np.random.permutation(X.shape[0])
    tamanho_treino = int(0.8 * X.shape[0])
    indices_treino, indices_teste = indices[:tamanho_treino], indices[tamanho_treino:]

    X_treino, y_treino = X[indices_treino], y[indices_treino]
    X_teste, y_teste = X[indices_teste], y[indices_teste]

    # Modelo: Média da variável dependente
    y_media = np.mean(y_treino)
    rss_media = np.sum((y_teste - y_media) ** 2)
    resultados_rss[0].append(rss_media)

    # Modelo: MQO Tradicional
    b_hat_mqo = np.linalg.inv(X_treino.T @ X_treino) @ X_treino.T @ y_treino
    y_pred_mqo = X_teste @ b_hat_mqo
    rss_mqo = np.sum((y_teste - y_pred_mqo) ** 2)
    resultados_rss['MQO Tradicional'].append(rss_mqo)

    # Modelo: MQO Regularizado para diferentes valores de lambda
    for lambd in lambdas[1:]:  # Ignorando lambda = 0 que já foi feito
        b_hat_ridge = np.linalg.inv(X_treino.T @ X_treino + lambd * np.eye(X_treino.shape[1])) @ X_treino.T @ y_treino
        y_pred_ridge = X_teste @ b_hat_ridge
        rss_ridge = np.sum((y_teste - y_pred_ridge) ** 2)
        resultados_rss[lambd].append(rss_ridge)

# Cálculo das estatísticas dos resultados
estatisticas_resultados = {
    'Modelo': [],
    'Média': [],
    'Desvio Padrão': [],
    'Maior Valor': [],
    'Menor Valor': []
}

# Média da variável dependente
estatisticas_resultados['Modelo'].append('Média de variável dependente')
estatisticas_resultados['Média'].append(np.mean(resultados_rss[0]))
estatisticas_resultados['Desvio Padrão'].append(np.std(resultados_rss[0]))
estatisticas_resultados['Maior Valor'].append(np.max(resultados_rss[0]))
estatisticas_resultados['Menor Valor'].append(np.min(resultados_rss[0]))

# MQO Tradicional
estatisticas_resultados['Modelo'].append('MQO Tradicional')
estatisticas_resultados['Média'].append(np.mean(resultados_rss['MQO Tradicional']))
estatisticas_resultados['Desvio Padrão'].append(np.std(resultados_rss['MQO Tradicional']))
estatisticas_resultados['Maior Valor'].append(np.max(resultados_rss['MQO Tradicional']))
estatisticas_resultados['Menor Valor'].append(np.min(resultados_rss['MQO Tradicional']))

# MQO Regularizado
for lambd in lambdas[1:]:
    estatisticas_resultados['Modelo'].append(f'MQO Regularizado ({lambd})')
    estatisticas_resultados['Média'].append(np.mean(resultados_rss[lambd]))
    estatisticas_resultados['Desvio Padrão'].append(np.std(resultados_rss[lambd]))
    estatisticas_resultados['Maior Valor'].append(np.max(resultados_rss[lambd]))
    estatisticas_resultados['Menor Valor'].append(np.min(resultados_rss[lambd]))

# Exibir resultados em formato de tabela
print("\nResultados dos Modelos:")
print(f"{'Modelo':<30} | {'Média':<10} | {'Desvio Padrão':<15} | {'Maior Valor':<10} | {'Menor Valor':<10}")
print("-" * 80)
for i in range(len(estatisticas_resultados['Modelo'])):
    print(f"{estatisticas_resultados['Modelo'][i]:<30} | {estatisticas_resultados['Média'][i]:<10.2f} | "
          f"{estatisticas_resultados['Desvio Padrão'][i]:<15.2f} | "
          f"{estatisticas_resultados['Maior Valor'][i]:<10.2f} | "
          f"{estatisticas_resultados['Menor Valor'][i]:<10.2f}")

# Preparando os dados para o gráfico
modelos = ['Média de variável dependente', 
           'MQO Tradicional', 
           'MQO Regularizado (0.25)', 
           'MQO Regularizado (0.5)', 
           'MQO Regularizado (0.75)', 
           'MQO Regularizado (1)']
media = [5014701.21, 357422.62, 357414.84, 357430.34, 357468.61, 357529.14]
desvio_padrao = [280213.25, 78813.86, 78280.25, 77775.74, 77240.16, 76733.35]

# Gráfico de barras
plt.figure(figsize=(10, 6))
bars = plt.barh(modelos[1:], media[1:], xerr=desvio_padrao[1:], color='skyblue', edgecolor='black')  # Excluindo a média da variável dependente

# Adicionando os valores no gráfico
for bar in bars:
    plt.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f'{bar.get_width():.2f}', 
             va='center', ha='left', color='black')

plt.title('Desempenho dos Modelos (RSS Médio)')
plt.xlabel('Soma dos Resíduos Quadráticos (RSS)')
plt.grid(axis='x')
plt.xlim(0, max(media) * 1.1)  # Ajustando limite do eixo x
plt.show()

