import opencc
import sys

if __name__ == '__main__':
    converter = opencc.OpenCC('t2s.json')
    with open(sys.argv[1], 'r') as f:
        for line in f:
            print(converter.convert(line.strip()))
