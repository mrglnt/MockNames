import numpy as np


def make_name(model, parser):
    name = []
    x = np.zeros((1, parser.max_char, parser.char_dim))
    end = False
    i = 0

    while end == False:
        probs = list(model.predict(x)[0, i])
        probs = probs / np.sum(probs)
        index = np.random.choice(range(parser.char_dim), p=probs)
        if i == parser.max_char - 2:
            character = '.'
            end = True
        else:
            character = parser.index_to_char[index]
        name.append(character)
        x[0, i + 1, index] = 1
        i += 1
        if character == '.':
            end = True

    print(''.join(name))
    return name


def generate_name_loop(epoch, _):
    if epoch % 5 == 0:

        print('Names generated after epoch %d:' % epoch)

        for i in range(3):
            make_name(model)

