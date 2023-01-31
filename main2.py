import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Carregando dados de média de pontos por jogo do LeBron James na temporada 21/22
df = pd.read_csv("lebron_james_21-22.csv")

# Preparando dados para a regressão linear
x = np.array(df["Jogo"]).reshape(-1, 1) # número do jogo
y = np.array(df["Pontos"]) # média de pontos

# Treinando o modelo de regressão linear
reg = LinearRegression().fit(x, y)

# Previsão da média de pontos com base no modelo
y_pred = reg.predict(x)

# Calculando média e desvio padrão da previsão
mean = np.mean(y_pred)
std = np.std(y_pred)

# Imprimindo média e desvio padrão da previsão
print("Média da previsão:", mean)
print("Desvio padrão da previsão:", std)

# Plotando gráfico de dispersão com a linha de regressão
plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.xlabel("Jogo")
plt.ylabel("Pontos")
plt.show()
