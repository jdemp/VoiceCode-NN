#import tensorflow as tf
import re
import numpy as np

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
        return clean[:-1]

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

    def string_to_bytes(self, word):
        chars = []
        for c in word:
            chars.append(ord(c))
        return chars

    def format_inputs(self,X):
        formated_inputs = []
        for i in X:
            formated_input = []
            for word in i:
                chars = self.string_to_bytes(word)
                formated_input.append(chars)
            formated_inputs.append(formated_input)
        return formated_inputs

    def format_labels(self,labels):
        formated_labels = []
        for l in labels:
            temp = self.string_to_bytes(l)
            temp.append(3)
            formated_labels.append(temp)
        return formated_labels
            #print(words)
    # bool is if they are combined
    # files is a list of files to create dbs from, don't include extentions
    def create_datasets(self, combined, files, combined_ext="annotation", seperate_code_ext="code", seperate_anno_ext="anno"):
        datasets = {}
        for f in files:
            if combined:
                x,y = self.read_in_combined(f+"."+combined_ext)
            else:
                x,y = self.read_in_seperate(f+"."+seperate_code_ext, f+"."+seperate_anno_ext)

        inputs_list = self.format_inputs(x)
        labels_list = self.format_labels(y)

        #create numpy arrays
        input_arry =

        #print(len(x))
        #print(y[0:10])
        features =
        labels =
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))


def main():
    dff = DataFromFile(db_dir="../datasets/en-django/")
    files = ["all"]
    dff.create_datasets(False, files)

if __name__ == '__main__':
    main()
