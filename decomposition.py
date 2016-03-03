################################## Requirements ######################################
# We suppose the kernels have equal height and width
# We suppose the rank is greater than one


__author__ = 'nelson'
import numpy as np
import theano
from theano import tensor as T
from theano.tensor.nnet import conv
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from numpy.linalg import svd
import cv2
from sktensor import dtensor, cp_als
from collections import Counter
import time
from random import shuffle
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
#np.set_printoptions(threshold='nan')
sys.setrecursionlimit(1000000)
caffe_root = '/home/nelson/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import handle_protobuf as hp

def regular_conv(weights, img, iter, doShuffle=False):

    # Instantiate 4D tensor for input
    input = T.tensor4(name='input')

    ## Regular Convolution ##
    weights = weights.eval()
    if doShuffle == True:
        pos = range(weights.shape[0])
        shuffle(pos)
        weights = weights[pos, :, :, :]

    conv_out = conv.conv2d(input, weights)
    f = theano.function([input], conv_out, profile=False)
    times = []
    for i in range(iter):
        start = time.time()
        filtered_img = f(img)
        done = time.time()
        times.append(done-start)
    avg1 = np.mean(times)
    return filtered_img, avg1

def fast_conv(weights, img, iter, w_shp, size, N_i, N_j, rank):

    # Define parameters
    (F, C, X, Y) = w_shp
    (N, C, H, W) = size

    # Instantiate 4D tensor for input
    input = T.tensor4(name='input')

    # Initialize shared variable for weights.
    weights = weights.eval().transpose(1, 2, 3, 0)

    # Rows Clustering
    kmeans_rows = KMeans(init='k-means++', n_clusters=N_i , n_init=10)
    W_C = np.reshape(weights, (C, X*Y*F))
    kmeans_rows.fit(W_C)
    [clusters_rows, pos_c] = map(list, zip(*sorted(zip(kmeans_rows.labels_, range(C)), key=lambda item:item[0])))
    new_W_C = W_C[pos_c, :]

    # Cols Clustering
    kmeans_cols = KMeans(init='k-means++', n_clusters=N_j , n_init=10)
    W_F = np.reshape(new_W_C, (C*X*Y, F))
    kmeans_cols.fit(W_F.T)
    [clusters_cols, pos_f] = map(list, zip(*sorted(zip(kmeans_cols.labels_, range(F)), key=lambda item:item[0])))
    new_W_F = np.reshape(W_F[:, pos_f], (C, X, Y, F))

    # Breakdown into cubes
    counter_cols = Counter(clusters_cols)
    counter_rows = Counter(clusters_rows)

    c_sum = 0
    D_CF = []
    for i, c in enumerate(list(counter_rows.values())):
        f_sum = 0
        D_F = []
        for j, f in enumerate(list(counter_cols.values())):

            # Crop cluster
            W_tensor =  dtensor(new_W_F[c_sum:c+c_sum, :, :, f_sum:f+f_sum])

            # Apply CP-Decomposition on the clustered weight tensor
            P, fit, itr, exectimes = cp_als(W_tensor, rank, init='random')
            D_F.append(P)
            f_sum += f
        D_CF.append(D_F)
        c_sum += c

    # Compute Convolution
    conv_out = theano.shared(np.zeros((N, F, H-X+1, W-Y+1)), name='out')
    c_sum = 0
    for i, c in enumerate(list(counter_rows.values())):
        f_sum = 0
        for j, f in enumerate(list(counter_cols.values())):

            # Crop cluster
            f_out = pos_f[f_sum:f+f_sum]
            c_in = pos_c[c_sum:c+c_sum]

            # Apply CP-Decomposition on the clustered weight tensor
            P = D_CF[i][j]
            for k in range(rank):

                # Create shared variables
                T_C = theano.shared(np.reshape(P.U[0][:,k], (1, c, 1, 1)), name='C_{0}_{1}_{2}'.format(k, i, j))
                T_X = theano.shared(np.reshape(P.U[1][:,k], (1, 1, X, 1)), name='X_{0}_{1}_{2}'.format(k, i, j))
                T_Y = theano.shared(np.reshape(P.U[2][:,k], (1, 1, 1, Y)), name='Y_{0}_{1}_{2}'.format(k, i, j))
                T_F = theano.shared(np.reshape(P.U[3][:,k], (f, 1, 1, 1)), name='F_{0}_{1}_{2}'.format(k, i, j))

                # Apply convolution on each dimension individually
                conv_C = conv.conv2d(input[:, c_in, :, :], T_C)
                conv_X = conv.conv2d(conv_C, T_X)
                conv_Y = conv.conv2d(conv_X, T_Y)
                conv_F = conv.conv2d(conv_Y, T_F)
                if f == 1:
                    conv_out = T.set_subtensor(conv_out[:, f_out[0], :, :], np.add(conv_out[:, f_out[0], :, :], conv_F[:, 0, :, :]))
                else:
                    conv_out = T.set_subtensor(conv_out[:, f_out, :, :], np.add(conv_out[:, f_out, :, :], conv_F))
            f_sum += f
        c_sum += c

    # Map Theano function
    f = theano.function([input], conv_out, profile=False)

    # Execute Theano function
    times = []
    for i in range(iter):
        start = time.time()
        filtered_img = f(img)
        done = time.time()
        times.append(done-start)
    avg1 = np.mean(times)
    return filtered_img, avg1

def bc_decomposotion():
    return

def cp_decomposition(weights, bias, rank):

    # Define parameters
    (F, C, X, Y) = weights.shape

    # Initialize shared variable for weights.
    W_tensor =  dtensor(weights)

    # Apply CP-Decomposition on the clustered weight tensor
    P, fit, itr, exectimes = cp_als(W_tensor, rank, init='random')

    output = []
    for k in range(rank):

        T_F = np.reshape(P.U[0][:,k], (F, 1, 1, 1))
        T_C = np.reshape(P.U[1][:,k], (1, C, 1, 1))
        T_X = np.reshape(P.U[2][:,k], (1, 1, X, 1))
        T_Y = np.reshape(P.U[3][:,k], (1, 1, 1, Y))
        output.append([T_C, T_X, T_Y, T_F, bias])
    return output

def cp_conv(weights, img, iter, w_shp, size, N_i, N_j, rank):

    # Define parameters
    (F, C, X, Y) = w_shp
    (N, C, H, W) = size

    # Instantiate 4D tensor for input
    input = T.tensor4(name='input')

    # Initialize shared variable for weights.
    weights = weights.eval()
    W_tensor =  dtensor(weights)

    # Apply CP-Decomposition on the clustered weight tensor
    P, fit, itr, exectimes = cp_als(W_tensor, rank, init='random')

    output = None
    for k in range(rank):

        T_F = theano.shared(np.reshape(P.U[0][:,k], (F, 1, 1, 1)), name='F_{0}'.format(k))
        T_C = theano.shared(np.reshape(P.U[1][:,k], (1, C, 1, 1)), name='C_{0}'.format(k))
        T_X = theano.shared(np.reshape(P.U[2][:,k], (1, 1, X, 1)), name='X_{0}'.format(k))
        T_Y = theano.shared(np.reshape(P.U[3][:,k], (1, 1, 1, Y)), name='Y_{0}'.format(k))

        # Apply convolution on each dimension individually
        conv_C = conv.conv2d(input, T_C)
        conv_X = conv.conv2d(conv_C, T_X)
        conv_Y = conv.conv2d(conv_X, T_Y)
        conv_F = conv.conv2d(conv_Y, T_F)

        output = output + conv_F if output else conv_F

    # Map Theano function
    f = theano.function([input], output, profile=False)

    # Execute Theano function
    times = []
    for i in range(iter):
        start = time.time()
        filtered_img = f(img)
        done = time.time()
        times.append(done-start)
    avg1 = np.mean(times)
    return filtered_img, avg1

def init_caffe():
    
    # Init Caffe
    return caffe.TEST

def get_params(net):
    
    # Get weights
    name = ['conv1_2']  
    weights = [[net.params[name[0]][0].data[...], net.params[name[0]][1].data[...]]]
    return weights, name

def decompose(method, rank):

    # Set param
    net_weights = "/home/nnauata/CellNet/app/tn_16_layers.caffemodel"
    new_net_weights = "/home/nnauata/CellNet/app/tn_16_layers_decomposed.caffemodel"
    net_params = "/home/nnauata/CellNet/app/cnn_test.prototxt"
    new_params_name = "/home/nnauata/CellNet/app/cnn_test_decomposed.prototxt"
    layers_name = ['conv1_2']

    # Init caffe    
    mode = init_caffe()

    # Load model
    net = hp.load_net(net_weights, net_params, mode)   

    # Get weights and names
    params, names = get_params(net)

    # Decompose weights
    new_params = []
    for weights, bias in params:
        if method == 'cp_decomposition':
            new_params.append(cp_decomposition(weights, bias, rank))

        elif method == 'biclustering':
            print 'Not Working Yet'

    # Write new prototxt
    hp.write_prototxt(layers_name, rank, net_params)
    
    # Set parameters to new model
    new_net = hp.load_net(None, new_params_name, caffe.TEST)
    hp.write_caffemodel(new_net_weights, net_params, net, new_net, layers_name, rank, new_params)
    return

def main():
    # Define parameters
    N_i = 4; N_j = 4
    iter = 1

    # Set param
    net_weights = "/home/nelson/PycharmProjects/CNN_speedup/VGG_ILSVRC_16_layers.caffemodel"
    net_params = "/home/nelson/PycharmProjects/CNN_speedup/VGG_ILSVRC_16_layers_deploy.prototxt"
    mode = caffe.TEST

    # Load model
    net = caffe.Net(net_params, net_weights, mode)
    w_shp = net.params['conv1_2'][0].data[...].shape
    weights = net.params['conv1_2'][0].data[...].astype('float64')
    (F, C, X, Y) = w_shp

    # Initialize shared variable for weights.
    weights = theano.shared(weights, name ='W')

    # Initialize input
    N = 1; H = 128; W = 128;
    size= (N, C, H, W)
    img = np.random.uniform(-1, 1, size=size)
    rank = 1

    # Compute Convolution
    reg_img, t1 = regular_conv(weights, img, iter)
    fast_img, t2 = fast_conv(weights, img, iter, w_shp, size, N_i, N_j, rank)
    cp_img, t3 = cp_conv(weights, img, iter, w_shp, size, N_i, N_j, rank)

    print "Speed Up"
    print "fast: " + str(t1/t2)
    print "cp: " + str(t1/t3)

    print "reg vs fast"
    print mean_absolute_error(reg_img.flatten(), fast_img.flatten())

    print "reg vs cp"
    print mean_absolute_error(reg_img.flatten(), cp_img.flatten())

    print "max"
    print max(reg_img.flatten())

    print "min"
    print min(reg_img.flatten())
    return

if __name__ == '__main__':
    #main()
    decompose('cp_decomposition', 2)
