import pandas as pd
from sklearn.linear_model import LinearRegression

# Carregue os dados do jogo do Manchester United
data = pd.read_csv("manchester_united_data.csv")

# Selecione as variáveis ​​que serão usadas para prever o resultado do jogo
X = data[['goals_scored', 'shots_on_target', 'possession']]

# Selecione a variável alvo (resultado do jogo)
y = data['result']

# Crie o modelo de regressão linear
model = LinearRegression()

# Treine o modelo com os dados
model.fit(X, y)

# Faça uma previsão com os dados de um jogo futuro
game_data = [[3, 10, 55]]
prediction = model.predict(game_data)

print("Previsão do resultado do jogo:", prediction)
