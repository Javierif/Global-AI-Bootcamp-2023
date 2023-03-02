from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
from keras import backend as K

# Convertir acciones en one-hot encoding
actions = ['Zerling', 'Trabajador', 'Overlord']
num_actions = len(actions)
X = []
for game in ActionsEveryGame:
    x = np.zeros((len(game), num_actions))
    for i, action in enumerate(game):
        x[i] = to_categorical(actions.index(action), num_classes=num_actions)
    X.append(x)

# Normalizar puntuación y tiempo de ataque
Y_score = np.array(ScoresEveryGame) / max(ScoresEveryGame)
Y_time = np.array(TimeAtaqueEveryGame) / max(TimeAtaqueEveryGame)


# Crear modelo
model = Sequential()
model.add(Dense(16, input_dim=num_actions, activation='relu'))
model.add(Dense(1, activation='linear'))

# Definir función de pérdida personalizada
def custom_loss(y_true, y_pred):
    mse = K.mean(K.square(y_true[:, 0] - y_pred[:, 0]))
    score_diff = K.abs(y_true[:, 1] - y_pred[:, 1])
    return mse + 0.1 * score_diff  # Usamos un factor de penalización de 0.1

# Compilar modelo
model.compile(loss=custom_loss, optimizer='adam')

# Entrenar modelo
model.fit(np.array(X), np.array([Y_time, Y_score]).T, epochs=1000, batch_size=8)