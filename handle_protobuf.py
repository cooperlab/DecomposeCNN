
import sys
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/caffe/proto')
#sys.path.insert(0, '/home/nelson/caffe/python/caffe/proto')
import caffe_pb2
from google.protobuf import text_format
import unicodedata
import numpy as np
caffe_root = '/home/nelson/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe

def create_conv_layer(name, shape, type, bottom, rank=-1):

    # Define new convolutional layer
    layer = caffe_pb2.LayerParameter()
    layer.name = name
    layer.type = type
    layer.bottom.extend([bottom])
    layer.top.extend([name])

    # Set dimension
    conv_param = layer.convolution_param
    conv_param.kernel_size.extend(shape[-2:])
    conv_param.num_output = shape[0]

    if "_F_" in name and rank == 0:
    	param_1 = caffe_pb2.ParamSpec()
        param_1.lr_mult = 1
        param_2 = caffe_pb2.ParamSpec()
        param_2.lr_mult = 2
        layer.param.extend([param_1, param_2])
    return layer

def load_net(net_weights, net_params, mode):
   
    # Load model
    if net_weights is None:
        return caffe.Net(net_params, mode)
    else:
        return caffe.Net(net_params, net_weights, mode)

def write_caffemodel(new_name, proto_name, old_net, new_net, layers_name, rank, weights_net):

    # Set weights
    for i, l_name in enumerate(layers_name):
        # Loop over ranks
        for r in range(rank):
            # Loop over decomposed layers
            for j, l_ref in enumerate(['_C_', '_X_', '_Y_', '_F_']):
                d_layer_name = l_name + l_ref + str(r)
                new_net.params[d_layer_name][0].data[...] = weights_net[i][r][j]              
                if l_ref == '_F_' and r == 0:
                    # Set bias
                    new_net.params[d_layer_name][1].data[...] = weights_net[i][r][4]
        	else:
          	    b_shp = new_net.params[d_layer_name][1].data[...].shape
        	    new_net.params[d_layer_name][1].data[...] = np.zeros(b_shp)

    # Set old params
    net_params = caffe_pb2.NetParameter()
    proto_name = proto_name.replace(".prototxt", "")
    with open(proto_name + '.prototxt', 'r') as file:
        text_format.Merge(str(file.read()), net_params)
    for layer in net_params.layer:
        l_name = str(layer.name)    
        if l_name not in layers_name and l_name in new_net.params:
        	new_net.params[l_name][0].data[...] = old_net.params[l_name][0].data[...]
        	new_net.params[l_name][1].data[...] = old_net.params[l_name][1].data[...]

    # Save network
    new_net.save(new_name)
    return

def write_prototxt(layers_name, rank, proto_name):
    net_params = caffe_pb2.NetParameter()
    proto_name = proto_name.replace(".prototxt", "")

    with open(proto_name + '.prototxt', 'r') as file:
        text_format.Merge(str(file.read()), net_params)
    new_net_params = caffe_pb2.NetParameter()

    # Setup input
    new_net_params.name = net_params.name + '_decomposed'
    new_net_params.input.extend(net_params.input)
    new_net_params.input_dim.extend(net_params.input_dim)
   
    # Setup layers
    mod  = 0
    top = ""
    new_layers = []
    for layer in net_params.layer:
 
        layer_bottom = str(layer.bottom[0]) if len(layer.bottom) > 0 else ""         
        layer_top = str(layer.top[0])
        if layer.name not in layers_name:

            # Update bottom and top
            if layer_bottom in layers_name:
                layer.bottom.remove(layer_bottom)
                layer.bottom.extend(["sum_" + layer_bottom])
            if layer_top in layers_name:
                layer.top.remove(layer_top)
                layer.top.extend(["sum_" + layer_top])
            new_layers.append(layer)
            mod = 0
        else:

            # Define top and bottom
            if mod:
                bottom = top
            else:
                bottom = layer_bottom
            l_names = []
            num_output = layer.convolution_param.num_output
            kernel_size = layer.convolution_param.kernel_size[0]
            for k in range(rank):

               # Define new layers
               l_C = create_conv_layer(layer_top + '_C_' + str(k), [1, 1, 1, 1], "Convolution", bottom)
               l_X = create_conv_layer(layer_top + '_X_' + str(k), [1, 1, kernel_size, 1], "Convolution", layer_top + '_C_' + str(k))
               l_Y = create_conv_layer(layer_top + '_Y_' + str(k), [1, 1, 1, kernel_size], "Convolution", layer_top + '_X_' + str(k))
               l_F = create_conv_layer(layer_top + '_F_' + str(k), [num_output, 1, 1, 1], "Convolution", layer_top + '_Y_' + str(k), k)
               l_names.append(layer_top + '_F_' + str(k))
               new_layers += [l_C, l_X, l_Y, l_F]
            mod = 1

            # Sum all layers
            top = "sum_" + layer.name
            eltwise_layer = caffe_pb2.LayerParameter()
            eltwise_layer.name = top
            eltwise_layer.type = "Eltwise"
            eltwise_layer.top.extend([top])
            eltwise_layer.bottom.extend(l_names)
            new_layers.append(eltwise_layer)
            eltwise_param = eltwise_layer.eltwise_param
            eltwise_param.operation = 1

    #print new_layers
    new_net_params.layer.extend(new_layers)
    #print new_net_params
    with open(proto_name + '_decomposed.prototxt', 'wb') as f:
        f.write(str(new_net_params))
    return

if __name__ == '__main__':
    write_prototxt(['conv1_1'], 2, 'VGG_ILSVRC_16_layers_deploy')
    test_net = load_net(None, 'VGG_ILSVRC_16_layers_deploy_decomposed.prototxt', caffe.TEST)
    write_caffemodel('VGG_ILSVRC_16_layers_deploy', 'VGG_ILSVRC_16_layers', test_net)
