import numpy as np
import csv
import sys

# Returns dataset_word_lists, dictionary, dataset_labels
def get_data(traininput, dictinput):
    filename = traininput
    with open(filename) as f1:
        r1 = csv.reader(f1, delimiter='\t')
        dataset_labels = []
        dataset_words = []
        for e in r1:
            dataset_labels.append(e[0])
            dataset_words.append(e[1])
        #npdataset_labels=np.array(dataset_labels)
    dictionaryfile = dictinput
    with open(dictionaryfile) as f2:
        r2 = csv.reader(f2, delimiter='\t')
        word = []
        ind = []
        for e in r2:
            dic_objs=e[0].split(' ')
            word.append(dic_objs[0])
            ind.append(dic_objs[1])

    dictionary={}
    for w,i in zip(word,ind):
        dictionary[w]=i

    dataset_word_lists=[]
    for e in dataset_words:
        dataset_word_lists.append(e.split())
    return dataset_word_lists, dictionary,dataset_labels

# Returns 1_occur matrix
def getoneoccur(dictionary,dataset_word_lists):
    no_of_vocab_words=len(dictionary)
    no_of_dataset_words=len(dataset_word_lists)
    bag_of_word_vec=np.zeros((no_of_dataset_words,no_of_vocab_words))

    for every_list,every_bag in zip(dataset_word_lists,bag_of_word_vec):
        for every_word in every_list:
            if every_word in dictionary:
                every_bag[int(dictionary[every_word])] = 1
    print(bag_of_word_vec)
    print(np.shape(bag_of_word_vec))
    return bag_of_word_vec

# Returns 1_trim matrix
def getonetrim(dictionary,dataset_word_lists):
    t=4 # Threshold
    no_of_vocab_words=len(dictionary)

    no_of_dataset_words=len(dataset_word_lists)
    trim_bag_of_word_vec=np.zeros((no_of_dataset_words,no_of_vocab_words))

    for every_list,every_bag in zip(dataset_word_lists,trim_bag_of_word_vec):
        for every_word in every_list:
            if every_word in dictionary and every_list.count(every_word)<t:
                every_bag[int(dictionary[every_word])] = 1
    print(trim_bag_of_word_vec)
    print(np.shape(trim_bag_of_word_vec))
    return trim_bag_of_word_vec

# Returns formatted tsv datafile in "bag of words" form
def write_bag_of_words(dictionary,dataset_word_lists,formattedfilename,dataset_labels):
    writelist=[]
    #print(dictionary)
    for every_list, r in zip(dataset_word_lists, range(len(dataset_word_lists))):
        writelist.append([])

        for every_word in every_list:
            if every_word in dictionary and (dictionary[every_word] + ':' + '1') not in writelist[r]:
                if r ==2:
                    pass
                    #print(dictionary[every_word])
                    #print('list no:',r)
                    #print(every_word)

                writelist[r].append(dictionary[every_word] + ':' + '1')
    # Inserting Labels at first element of each row list

    for lab,li in zip(dataset_labels,range(len(writelist))):
        writelist[li].insert(0,lab)
    csv.register_dialect("tab",delimiter="\t")
    #print(writelist)
    with open(formattedfilename, 'w') as formattedfile:
        writer= csv.writer(formattedfile,dialect="tab")
        writer.writerows(writelist)

# Returns formatted tsv datafile in "trimmed bag of words" form
def write_trimmed_bag_of_words(dictionary,dataset_word_lists,formattedfilename,dataset_labels):
    writelist = []
    t=4 # Trim threshold
    for every_list, r in zip(dataset_word_lists, range(len(dataset_word_lists))):
        writelist.append([])
        for every_word in every_list:
            if every_word in dictionary and (dictionary[every_word] + ':' + '1') not in writelist[r] and every_list.count(every_word)<t:
                writelist[r].append(dictionary[every_word] + ':' + '1')
    # Inserting Labels at first element of each row list
    for lab, li in zip(dataset_labels, range(len(writelist))):
        writelist[li].insert(0, lab)
    csv.register_dialect("tab", delimiter="\t")

    with open(formattedfilename, 'w') as formattedfile:
        writer= csv.writer(formattedfile,dialect="tab")
        writer.writerows(writelist)

if __name__=='__main__':
    #ARGS TO feature.py
    #<traininput><validationinput><testinput>
    # <dictinput>
    # <formattedtrainout><formattedvalidationout> <formattedtestout>
    # <featureflag>.


    '''
    traininput="/Users/adi/Desktop/ML 10-601/HW4/handout/largedata/train_data.tsv"
    validationinput="/Users/adi/Desktop/ML 10-601/HW4/handout/smalldata/smallvalid_data.tsv"
    testinput="/Users/adi/Desktop/ML 10-601/HW4/handout/largedata/test_data.tsv"
    dictinput="/Users/adi/Desktop/ML 10-601/HW4/handout/dict.txt"
    formattedtrainout="formattedtrainout.tsv"
    formattedvalidationout="formattedvalidationout.tsv"
    formattedtestout="formattedtestout.tsv"
    featureflag=2
    '''


    traininput = sys.argv[1]
    validationinput = sys.argv[2]
    testinput = sys.argv[3]
    dictinput = sys.argv[4]
    formattedtrainout = sys.argv[5]
    formattedvalidationout = sys.argv[6]
    formattedtestout = sys.argv[7]
    featureflag = sys.argv[8]




    if int(featureflag) == 1:
        dataset_word_lists, dictionary, dataset_labels = get_data(traininput, dictinput)
        write_bag_of_words(dictionary, dataset_word_lists, formattedtrainout, dataset_labels)

        dataset_word_lists, dictionary, dataset_labels = get_data(validationinput, dictinput)
        write_bag_of_words(dictionary, dataset_word_lists, formattedvalidationout, dataset_labels)

        dataset_word_lists, dictionary, dataset_labels = get_data(testinput, dictinput)
        write_bag_of_words(dictionary, dataset_word_lists, formattedtestout, dataset_labels)

    if int(featureflag) == 2:
        dataset_word_lists, dictionary, dataset_labels = get_data(traininput, dictinput)
        write_trimmed_bag_of_words(dictionary, dataset_word_lists, formattedtrainout, dataset_labels)

        dataset_word_lists, dictionary, dataset_labels = get_data(validationinput, dictinput)
        write_trimmed_bag_of_words(dictionary, dataset_word_lists, formattedvalidationout, dataset_labels)

        dataset_word_lists, dictionary, dataset_labels = get_data(testinput, dictinput)
        write_trimmed_bag_of_words(dictionary, dataset_word_lists, formattedtestout, dataset_labels)

    #getoneoccur(dictionary,dataset_word_lists)
    #getonetrim(dictionary,dataset_word_lists)