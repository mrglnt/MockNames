from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import LambdaCallback
from MockNames.processing.model_utils import *
import pickle


class Model():
    def __init__(self,
                 parser,
                 hidden_states=128
                 ):
        model = Sequential()
        model.add(LSTM(hidden_states, input_shape=(parser.max_char, parser.char_dim), return_sequences=True))
        model.add(Dense(parser.char_dim, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.hidden_states = hidden_states
        self.model = model
        self.X = parser.X
        self.Y = parser.Y

    def fit(
            self,
            batch_size=10000,
            epochs=300
    ):
        name_generator = LambdaCallback(on_epoch_end=generate_name_loop)
        self.model.fit(self.X, self.Y, batch_size=batch_size, epochs=epochs, callbacks=[name_generator], verbose=1)

    def save_model(
            self,
            filename: str = 'output/model.pkl'
    ):
        pickle.dump(self.model, open(filename, "wb"))


if __name__ == '__main__':
    Model()
