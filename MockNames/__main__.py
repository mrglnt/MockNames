from MockNames.preprocessing import Parser


def main():
    # preprocessing
    par = Parser()
    par.load_url()
    par.fit()
    
    # processing


if __name__ == '__main__':
    main()
