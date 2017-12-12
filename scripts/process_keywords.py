import numpy as np

def get_keywords(language, path):
    f = open(path)
    keywords = []
    for line in f:
        l = line.strip().split("\t")
        lang=l[0].lower()
        if(lang==language):
            keywords.append((l[1],l[2]))
    print(len(keywords))

    keywords_file = open("../datasets/"+language+"_keywords.kw", 'w')
    for k in keywords:
        s = k[0]+"\t"+k[1]
        keywords_file.write("%s\n" % s)

if __name__ == '__main__':
    get_keywords("python", "../datasets/keywords.tsv")
