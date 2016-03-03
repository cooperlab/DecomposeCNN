import os
import sys
import random
import subprocess
import re
import numpy as np
import time
from sklearn.metrics import accuracy_score
import h5py
import decomposition as dc

def run_experiment(rank):
    
    # Decompose prototxt
    #dc.decompose('cp_decomposition', rank)

    # Define parameters
    #test_dset = "/home/mnalisn/tissueNet/datasets/lgg-endo-combined-test-48.h5"
    test_dset = "/home/mnalisn/tissueNet/slides/LGG/TCGA-WY-A85E-01Z-00-DX1.AA7A4C1F-99AA-490D-B6D4-280EAB1EFF56.svs.h5"
    #train_net_path = "/home/nnauata/CellNet/app/tn_train -t /home/mnalisn/tissueNet/datasets/lgg-endo-combined-train-48.h5 -p /home/nnauata/CellNet/online_caffe_model/cnn_train_val.prototxt -o /home/nnauata/CellNet/app/tn_16_layers.caffemodel -r"
    #test_net_path = "/home/nnauata/CellNet/app/tn_predict -d /home/mnalisn/tissueNet/datasets/lgg-endo-combined-test-48.h5 -p /home/nnauata/CellNet/app/cnn_test.prototxt -m /home/nnauata/CellNet/app/tn_16_layers.caffemodel -o /home/nnauata/CellNet/app/out -r"
    test_net_path = "/home/nnauata/CellNet/app/tn_predict -d " + test_dset + " -p /home/nnauata/CellNet/app/cnn_test.prototxt -m /home/nnauata/CellNet/app/tn_16_layers.caffemodel -o /home/nnauata/CellNet/app/out -r -b 1000"

    # Call processes
    #subprocess.call(train_net_path, shell=True)
    start = time.time()
    output = subprocess.check_output(test_net_path, shell=True)
    elapsed_time = time.time() - start

    # Read targets
    #d_set =  h5py.File("/home/mnalisn/tissueNet/datasets/" + test_dset,"r")
    #target = d_set["labels"][...]
    #target = np.where(target == -1, 0, 1)

    # Get accuracy
    accs = []
    #for k in range(4):
    #    with open("out_" + str(k) + ".txt") as f:
    #        accs.append([float(line.split(";")[0]) for line in f.readlines()])

    #max_accs = np.amax(accs, axis=0)
    #max_accs = np.where(max_accs > 0.5, 1, 0)
    acc = 0
    #acc = accuracy_score(max_accs.ravel(), target.ravel())

    # Remove text files
    for k in range(4):
        os.remove("/home/nnauata/CellNet/app/out_" + str(k) + ".txt")
    return acc, elapsed_time

if __name__ == "__main__":

    # PARSET
    max_rank = 3
    rep = 10
   
    # Run experiments
    accs = []
    times = []
    for i in range(2, max_rank):
        time_rank = []
        acc_rank = []
        for j in range(rep):
           acc, elapsed_time = run_experiment(i) 
           time_rank.append(elapsed_time)
           acc_rank.append(acc)
        times.append(time_rank)
        accs.append(acc_rank)

    accs = np.array(accs)
    times = np.array(times)
        
    print "Accuracy:"
    print np.mean(accs, axis=1)
    print np.var(accs, axis=1)

    print "Time:"
    print np.mean(times, axis=1)
    print np.var(times, axis=1)
