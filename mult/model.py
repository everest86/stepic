from keras.layers import Dense
from keras.models import Sequential

model = Sequential([Dense(1, input_shape=(1,), activation='linear')])

model.compile(loss='mse', optimizer='sgd', metrics=['mae'])


def learn(x, y):
    model.fit(x, y, epochs=100)


def calculatePredict(x):
    return model.predict(x)
