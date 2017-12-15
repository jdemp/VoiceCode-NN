import tensorflow as tf
import re
import numpy as np

class DataFromFile(object):
    def __init__(self,db_dir="./",language="Python"):
        self.db_dir = db_dir
        self.datasets ={}
        if language=="Python":
            #self.regex = re.compile(r'\-+|\ +|\_+ ')
            self.regex_split = re.compile(r'[\-|\ |\_ |\.]+')
            self.regex_replace = re.compile(r'\ \. \ ')
            self.regex_repalce_anno_0 = re.compile(r'[^0-9a-zA-z]+[0-9a-zA-z]*')
            self.regex_replace_anno = re.compile(r'[^0-9a-zA-z]')
        else:
            self.regex_split = re.compile(r'\ ')

    def next_batch(self,ds="all", size=32):
        end_of_epoch = False
        data = self.datasets[ds]
        start_index = data["batch_index"]
        end_index = start_index+size
        if(end_index<data["size"]):
            batch = data["batch_perm"][start_index:end_index]
            data["batch_index"] = end_index
        else:
            batch = data["batch_perm"][start_index:]
            data["batch_index"] = 0
            data["batch_perm"]=np.random.permutation(data["size"])
            end_of_epoch = True

        inputs = np.take(data["inputs"], batch, 0)
        inputs_len = np.take(data["inputs_len"], batch, 0)
        labels = np.take(data["labels"], batch, 0)
        labels_len = np.take(data["labels_len"], batch, 0)

        max_length = np.max(labels_len)
        return (inputs,inputs_len,labels[:,0:max_length],labels_len),  end_of_epoch

    def test_set(self, batch_size=32, ds="all"):
        data = self.datasets[ds]
        test_all = data["test_set"][0:50*batch_size]
        test_set = []
        for i in range(0,50*batch_size,batch_size):
            test = test_all[i:i+batch_size]
            inputs = np.take(data["inputs"], test, 0)
            inputs_len = np.take(data["inputs_len"], test, 0)
            labels = np.take(data["labels"], test, 0)
            labels_len = np.take(data["labels_len"], test, 0)
            max_length = np.max(labels_len)
            test_set.append((inputs,inputs_len,labels[:,0:max_length],labels_len))
        return test_set


    def clean_up_anno(self, line):
        lower = self.regex_split.split(line.lower())
        clean = []
        for word in lower:
            temp = self.regex_repalce_anno_0.sub(repl='',string=word)
            temp = self.regex_replace_anno.sub(repl='',string=temp)
            if temp!='':
                clean.append(temp)

        return clean, len(clean)

    def clean_up_code(self, line):
        clean = line.replace('_',' _ ')
        return clean.strip().split()

    def read_in_seperate(self, code, anno):
        code_file = open(self.db_dir+code)
        anno_file = open(self.db_dir+anno)
        X = []
        Y = []
        for x_raw,y_raw in zip(anno_file, code_file):
            if (len(y_raw.strip()))<126:
                X.append(x_raw.strip())
                Y.append(y_raw.strip())
                #print(y_raw.strip())
        return X,Y

    def read_in_combined(self, file):
        pass

    def format_inputs(self,X):
        longest_sequence = 0;
        formated_inputs = []
        input_lengths = []

        for line in X:
            chars, length = self.clean_up_anno(line)
            formated_inputs.append(chars)
            input_lengths.append(length)
            if length>longest_sequence:
                longest_sequence=length

        return formated_inputs, input_lengths, longest_sequence

    def format_labels(self,labels):
        formated_labels = []
        formated_labels_len = []
        for l in labels:
            clean = self.clean_up_code(l)
            clean.append("<end>")
            formated_labels.append(clean)
            formated_labels_len.append(len(clean))
        return formated_labels, formated_labels_len

    # bool is if they are combined
    # files is a list of files to create dbs from, don't include extentions
    def create_datasets(self, combined, files, input_vocab, output_vocab,combined_ext="annotation",
                        seperate_code_ext="code", seperate_anno_ext="anno"):
        for f in files:
            if combined:
                x,y = self.read_in_combined(f+"."+combined_ext)
            else:
                x,y = self.read_in_seperate(f+"."+seperate_code_ext, f+"."+seperate_anno_ext)

            inputs_list, sequence_lengths, max_sequence = self.format_inputs(x)
            labels_list, labels_len = self.format_labels(y)

            valid = len(inputs_list)==len(labels_list)==len(labels_len)==len(sequence_lengths)
            assert(valid)

            #print(sequence_lengths.count(0))

            #create inputs tensor
            inputs_tensor = np.zeros(shape=(len(inputs_list), max_sequence), dtype=np.int32)
            for i in range(len(inputs_list)):
                for w in range(len(inputs_list[i])):
                    if inputs_list[i][w].isdigit():
                        a = 2
                    elif inputs_list[i][w] in input_vocab.keys():
                        a = input_vocab[inputs_list[i][w]]
                    else:
                        a = 1
                    inputs_tensor[i,w] = a
            #print(inputs_tensor)

            input_length_tensor = np.array(sequence_lengths, dtype=np.int32).flatten()
            #print(input_length_tensor.shape)
            #create word lengths tensor
            #create labels tensor
            max_label_len = max(labels_len)

            #print (labels_list)

            labels_tensor = np.zeros(shape=(len(labels_list), max_label_len), dtype=np.int32)
            for i in range(len(labels_list)):
                for w in range(len(labels_list[i])):
                    if labels_list[i][w].isdigit():
                        a = 2
                    if labels_list[i][w] in output_vocab.keys():
                        a = output_vocab[labels_list[i][w]]
                    else:
                        a = 1
                    labels_tensor[i,w] = a

            labels_length_tensor = np.array(labels_len, dtype=np.int32).flatten()
            #print(labels_tensor.shape)
            self.datasets[f]={
                "inputs": inputs_tensor,
                "inputs_len": input_length_tensor,
                "labels": labels_tensor,
                "labels_len": labels_length_tensor,
                "size":len(inputs_list),
                "batch_index": 0,
                "batch_perm": np.random.permutation(len(inputs_list)),
                "test_set": np.random.permutation(len(inputs_list))
            }
            #print(self.datasets[f]["size"])




def main():
    dff = DataFromFile(db_dir="../datasets/en-django/")
    files = ["all"]
    dff.create_datasets(False, files)



if __name__ == '__main__':
    main()
