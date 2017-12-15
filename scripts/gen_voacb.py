import numpy as np
import re
from collections import Counter

def get_vocab(path, vocab_file, size):
    f = open(path)
    vocab = Counter()
    regex_split = re.compile(r'[\-|\ |\_ |\.]+')
    regex_replace_code = re.compile(r'\ \. \ ')
    regex_repalce_anno_0 = re.compile(r'[^0-9a-zA-z]+[0-9a-zA-z]*')
    regex_replace_anno = re.compile(r'[^0-9a-zA-z]')

    for line in f:
        s = line.replace('_',' _ ')
        s = regex_split.split(line.strip())
        #print(split)
        for word in s[:-1]:
            #lower = word.lower()
            #lower = regex_repalce_anno_0.sub(repl='',string=lower)
            #lower = regex_replace_anno.sub(repl='',string=lower)

            if word!='':
                vocab[word]+=1

    vocab_file = open(vocab_file,'w')
    for word in vocab.most_common(size):
        s = word[0]+"\t"+str(word[1])
        vocab_file.write("%s\n" % s)


if __name__ == '__main__':
    get_vocab("../datasets/en-django/all.code", "../vocab/code.voc",5000)
