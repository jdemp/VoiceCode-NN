import numpy as np
import re
from collections import Counter

def get_vocab(path, vocab_file, size):
    f = open(path)
    vocab = Counter()
    regex_split = re.compile(r'[\-|\ |\_ |\.]+')
    regex_replace = re.compile(r'\ \. \ ')

    for line in f:
        s = regex_split.split(line.strip())
        #print(split)
        for word in s[:-1]:
            lower = word.lower()
            vocab[lower]+=1

    vocab_file = open(vocab_file,'w')
    for word in vocab.most_common(size):
        s = word[0]+"\t"+str(word[1])
        vocab_file.write("%s\n" % s)


if __name__ == '__main__':
    get_vocab("../datasets/en-django/all.code", "../vocab/code.voc",5000)
