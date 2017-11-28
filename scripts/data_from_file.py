#import tensorflow as tf
import re

class DataFromFile(object):
    def __init__(self,db_dir="./",language="Python"):
        self.db_dir = db_dir
        if language=="Python":
            #self.regex = re.compile(r'\-+|\ +|\_+ ')
            self.regex_split = re.compile(r'[\-|\ |\_ |\.]+')
            self.regex_replace = re.compile(r'\ \. \ ')
        else:
            self.regex_split = re.compile(r'\ ')

    def clean_up_anno(self, line):
        clean = self.regex_split.split(line)
        return clean

    def clean_up_code(self, line):
        clean = line.replace(" . ", ".")
        return clean

    def read_in_seperate(self, code, anno):
        code_file = open(self.db_dir+code)
        anno_file = open(self.db_dir+anno)
        X = []
        Y = []
        for x_raw,y_raw in zip(anno_file, code_file):

            x = self.clean_up_anno(x_raw.strip())
            y = self.clean_up_code(y_raw.strip())
            X.append(x)
            Y.append(y)
        return X,Y

    def read_in_combined(self, file):
        pass

    # bool is if they are combined
    # files is a list of files to create dbs from, don't include extentions
    def create_datasets(self, combined, files, combined_ext="annotation", seperate_code_ext="code", seperate_anno_ext="anno"):
        datasets = {}
        for f in files:
            if combined:
                x,y = self.read_in_combined(f+"."+combined_ext)
            else:
                x,y = self.read_in_seperate(f+"."+seperate_code_ext, f+"."+seperate_anno_ext)
        print(len(x))
        #print(x[0])
        print(y[0:10])



def main():
    dff = DataFromFile(db_dir="../datasets/en-django/")
    files = ["all"]
    dff.create_datasets(False, files)

if __name__ == '__main__':
    main()
