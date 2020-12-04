#encoding=utf-8
import jieba
import sys

if __name__ == '__main__':
    f_inp = sys.argv[1]
    with open(f_inp, 'r') as f:
        for line in f:
            print(' '.join(jieba.cut(line.strip())))
