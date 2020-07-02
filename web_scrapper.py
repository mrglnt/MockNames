import requests
from bs4 import BeautifulSoup
import pickle
import numpy as np
import re


class Parser:
    def __init__(self):
        self.webscrap = None
        self.names = None
        self.X = None
        self.Y = None

    def load_url(self,
                 url: str = 'https://www.uniprot.org/docs/speclist'
                 ):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.findAll('pre')[0].text
        self.webscrap = text
        return

    def fit(self):
        raw_text = self.webscrap.split('\n')
        names = []
        for i in range(len(raw_text)):
            if raw_text[i].find("N=") != -1:
                rest = raw_text[i][(raw_text[i].find("N=") + 2):].split("(", 1)[0].rstrip() + '.'
                result = ''.join(i for i in rest if not i.isdigit())
                names.append(re.sub(r"[^a-zA-Z0-9]+", ' ', result).lower())

        # let's generate dictionary index - character and viceversa
        char_to_index = dict((chr(i + 96), i) for i in range(1, 27))
        char_to_index[' '] = 0
        char_to_index['.'] = 27

        index_to_char = dict((i, chr(i + 96)) for i in range(1, 27))
        index_to_char[0] = ' '
        index_to_char[27] = '.'

        max_char = len(max(names, key=len))
        m = len(names)
        char_dim = len(char_to_index)

        # now, we generate the datasets as one hot encoded variables

        X = np.zeros((m, max_char, char_dim))
        Y = np.zeros((m, max_char, char_dim))

        for i in range(m):
            name = list(names[i])
            for j in range(len(name)):
                X[i, j, char_to_index[name[j]]] = 1
                if j < len(name) - 1:
                    Y[i, j, char_to_index[name[j + 1]]] = 1

        self.names = names
        self.X = X
        self.Y = Y

        return

    def save_data(self,
                  path: str = 'data_names.pkl'
                  ):
        pickle.dump(self.names, open(path, "wb"))
        return

    def load_data(self,
                  path: str = 'data_names.pkl'
                  ):
        self.names = pickle.load(open(path, "rb"))

        char_to_index = dict((chr(i + 96), i) for i in range(1, 27))
        char_to_index[' '] = 0
        char_to_index['.'] = 27

        index_to_char = dict((i, chr(i + 96)) for i in range(1, 27))
        index_to_char[0] = ' '
        index_to_char[27] = '.'

        max_char = len(max(self.names, key=len))
        m = len(self.names)
        char_dim = len(char_to_index)

        # now, we generate the datasets as one hot encoded variables

        X = np.zeros((m, max_char, char_dim))
        Y = np.zeros((m, max_char, char_dim))

        for i in range(m):
            name = list(self.names[i])
            for j in range(len(name)):
                X[i, j, char_to_index[name[j]]] = 1
                if j < len(name) - 1:
                    Y[i, j, char_to_index[name[j + 1]]] = 1

        self.X = X
        self.Y = Y
        return


if __name__ == '__main__':
    pass
