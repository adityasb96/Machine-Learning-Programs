import numpy as np
import sys
import csv
import math


def read_features(formattedtraininput):
    filename = formattedtraininput
    with open(filename) as f1:
        r1 = csv.reader(f1, delimiter='\t')
        formatteddata=[]
        for e in r1:
            formatteddata.append(e)
    #formatteddata=np.genfromtxt(filename,delimiter='\t')
    #print(formatteddata[0][2].split(':')[0])
    label_list = []
    for e in formatteddata:
        label_list.append(e[0])
        del(e[0])

    #print(label_list)
    #print(len(label_list))

    return formatteddata,label_list

def dictionary(dictinput):
    with open(dictinput) as f2:
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
    return dictionary

def getoccur(formatteddata,dictionary):
    no_of_vocab_words=len(dictionary)
    no_of_dataset_words=len(formatteddata)
    bag_of_word_vec=np.zeros((no_of_dataset_words,no_of_vocab_words))

    for every_list,every_bag in zip(formatteddata,bag_of_word_vec):
        for every_pair in every_list:
            every_bag[int(every_pair.split(':')[0])] = 1

    #print(bag_of_word_vec)
    #print(np.shape(bag_of_word_vec))
    return bag_of_word_vec

def makesparse(bag_of_word_vec):
    # Add bias term into fold
    bag_of_word_vec=np.insert(bag_of_word_vec,0,1,axis=1)
    #print(bag_of_word_vec)
    #print(np.shape(bag_of_word_vec))
    X = np.array([])
    for bags in bag_of_word_vec:
        d={}
        for b, b_no in zip(bags,range(len(bags))):
            if b!= 0:
                d[b_no] = 1
        X=np.append(X,d)

    return X

# Returns final optimal parameters th
def sgd(X,label_list,dictionary,num_epoch):

    th = np.zeros(len(dictionary)+1)
    dJ_dth = np.zeros(len(dictionary)+1)
    # Objective function is: negative conditional log likelihood
    # Partial derivative (gradient) of objective function for all parameters for a each example is :
    #li=0.1  #D Setting initial learning rate to 0.1
    #lrate = li/((num_epoch - 1)*li +1)
    lrate = 0.1

    # Repeating for  n = no_of_epochs:
    for n in range(num_epoch):
        print(type(num_epoch))
        for i in range(len(X)):
            p = 0.0
            for j,v in X[i].items():
                if v == 1:
                    #print(len(th))
                    #print(j)
                    p= p + th[j]
                    #print(i)
                    #print("thTX:",p)
            #print(p)
            #print(j)
            gradient = - (int(label_list[i]) - ((math.exp(p))/(1+ (math.exp(p)))))
            for j,v in X[i].items():
                dJ_dth[j] = gradient
                #print("dJ_dth[j]:",dJ_dth[j],j)
                th[j] -= lrate * dJ_dth[j]
           #print(th[0])
        #print(th)
    return th

# Calculate label with max. likelihood for each example


def predict(th,X,label_list):
    pred_label_list = []
    for i in range(len(X)):
        #for j,v in X[i].items():
         #   exp = math.exp(th[j] * v)
        p = 0.0
        for j, v in X[i].items():
            #if v == 1:
            p = p + th[j]
    #print("pred",p)
    #for i in range(len(X)):
        # Likelihood function
        ex_likelihood_y_one = math.exp(p)/(1 + math.exp(p))
        #print(ex_likelihood_y_one)
        if ex_likelihood_y_one >= 0.5:
            pred_label_list.append(1)
        else:
            pred_label_list.append(0)
    #print(label_list,pred_label_list)
    return pred_label_list


def error(label_list,pred_label_list):
    e=0
    for l, pl in zip(label_list,pred_label_list):
        #print(l,pl)
        if int(l) != pl:
            e +=1

    err= e/len(label_list)
    print("error:",np.round(err,6))
    return err


def labels_out(lab_filename,pred_labels):
    with open(lab_filename, 'w') as f:
        for e in pred_labels:
            f.write(str(e) + '\n')
'''
def metrics_out(err,dataset_name):
    with open('metricsout', 'w') as out_f:
        out_f.write('error' + '('+ str(dataset_name) + ')'+ ':' + str(np.round(err,6)) + '\n')
'''


if __name__ == "__main__":
    #formattedtraininput > < formattedvalidationinput > < formattedtestinput >
    #< dictinput >
    #< trainout > < testout >
    #< metricsout >
    #< numepoch >
    '''
    dictinput = "/Users/adi/Desktop/ML 10-601/HW4/handout/dict.txt"
    formattedtraininput = "/Users/adi/Desktop/ML 10-601/HW4/formattedtrainout.tsv"
    formattedvalidationinput = "/Users/adi/Desktop/ML 10-601/HW4/formattedvalidationout.tsv"
    formattedtestinput = "/Users/adi/Desktop/ML 10-601/HW4/formattedtestout.tsv"
    num_epoch= 60

    '''
    formattedtraininput = sys.argv[1]
    formattedvalidationinput = sys.argv[2]
    formattedtestinput = sys.argv[3]
    dictinput = sys.argv[4]
    trainout = sys.argv[5]
    testout = sys.argv[6]
    metricsout = sys.argv[7]
    num_epoch = sys.argv[8]


    dictionary = dictionary(dictinput)

    # Run lr for train

    formatteddata,label_list_train = read_features(formattedtraininput)
    bowv=getoccur(formatteddata, dictionary)
    X=makesparse(bowv)

    th = sgd(X,label_list_train,dictionary,int(num_epoch))

    pred_label_list_train= predict(th, X, label_list_train)
    labels_out(lab_filename=trainout,pred_labels= pred_label_list_train)
    #lab_filename=


    # Run lr for Test

    formatteddata, label_list_test = read_features(formattedtestinput)
    bowv = getoccur(formatteddata, dictionary)
    X = makesparse(bowv)

    #th = sgd(X, label_list, dictionary, num_epoch)

    pred_label_list_test = predict(th, X, label_list_test)
    labels_out(lab_filename=testout, pred_labels=pred_label_list_test)
    #lab_filename=

    err1 = error(label_list_train, pred_label_list_train)
    err2 = error(label_list_test, pred_label_list_test)

    #Write Metrics file

    with open(metricsout, 'w') as out_f:
        out_f.write('error' + '(train)' + ':' + str(np.round(err1, 6)) + '\n')
        out_f.write('error' + '(test)' + ':' + str(np.round(err2, 6)))

    #metrics_out(err1, dataset_name='train')

    #metrics_out(err2,dataset_name='test')
