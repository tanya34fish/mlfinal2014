import numpy as np
import pickle
import graphlab as gl
from scipy.misc import imresize

def loadData(filename,type=None,ignore_indices=[]):
    data = {}
    data['label'] = []
    data['image'] = []
    input = []
    output = []
    size = 128,128

    idx = 0
    
    with open(filename ,'r') as f:
        for line in f:
            if idx in ignore_indices:
                idx += 1
                continue;
            else:
                idx += 1
                train = line.strip().split()
                if type == 'train':
                    if len(train) < 10:
                        continue
                classID = int(train[0])
                output.append(classID)
                #105*122 features
                image = []
                for i in range(12810):
                    image.append(0.0)
                for i in range(1,len(train)):
                    pixel = train[i].split(':')
                    image[int(pixel[0])-1] = float(pixel[1]) * 255

                input.append(image)

    data['label'] = gl.SArray(output)
    scaled_input = gl.SArray(input)
    img_sarray = gl.SArray.pixel_array_to_image(scaled_input, 105, 122, 1, allow_rounding = True)
    data['image'] = img_sarray
    return data

# full_training_data = gl.load_sframe('data/full_training_data_sframe')

# m = gl.load_model('model_layer2_hidden500_1')
# pred = m.classify(full_training_data)

# count = 0
# indices = []
# idx = 0
# fout = open('ignore_indices.txt', 'w')
# fout.write('Index | Prediction | Score\n')
# for (i,j) in zip(list(pred['class']),list(pred['score'])):
#     if j < 0.6 and full_training_data['label'][idx] != i:
#         count += 1
#         indices.append(idx)
#         fout.write('train_%d : %d , %f\n' % (idx+1, i, j))
#     idx += 1
# # print count
# # print indices
# fout.close()

# results = m.evaluate(full_training_data)
# print results

# train_input = loadData('ml14fall_train.dat',type='train',ignore_indices=indices)
# train_data = gl.SFrame(train_input)
# training_data, validation_data = train_data.random_split(0.8)
# training_data.save('data/training_data_sframe_06')
# validation_data.save('data/validation_data_sframe_06')
training_data = gl.load_sframe('data/training_data_sframe_06')
validation_data = gl.load_sframe('data/validation_data_sframe_06')
test_data = gl.load_sframe('data/test_data_sframe')

# conv_net = gl.deeplearning.create(training_data, target='label')
conv_net = gl.deeplearning.ConvolutionNet(num_convolution_layers=2,
       kernel_size=3, stride=2, num_channels=25, num_output_units=0)
conv_net.layers.insert(1,gl.deeplearning.layers.RectifiedLinearLayer())
conv_net.layers.insert(2,gl.deeplearning.layers.DropoutLayer(threshold=0.3))
conv_net.layers.insert(5,gl.deeplearning.layers.RectifiedLinearLayer())
conv_net.layers.insert(6,gl.deeplearning.layers.DropoutLayer(threshold=0.3))
percpt_net = gl.deeplearning.MultiLayerPerceptrons(num_hidden_layers=3, num_hidden_units=[918,189,32])
conv_net.layers.extend(percpt_net.layers)
conv_net.layers.insert(-5,gl.deeplearning.layers.RectifiedLinearLayer())
conv_net.layers.insert(-5,gl.deeplearning.layers.DropoutLayer())
del conv_net.layers[-5]
conv_net.layers.insert(-3,gl.deeplearning.layers.RectifiedLinearLayer())
conv_net.layers.insert(-3,gl.deeplearning.layers.DropoutLayer())
del conv_net.layers[-3]
conv_net.save('net_hidden918_189_drop03_input06')
print(conv_net.verify())
m = gl.neuralnet_classifier.create(training_data, target='label',
                network = conv_net,
                validation_set=validation_data,
                metric=['accuracy', 'recall@2'],
                max_iterations=100)
m.save('model_hidden918_189_drop03_input06')
# m = gl.load_model('model_layer2_hidden500_2')
pred = m.classify(test_data)
print(pred)

with open('predict_hidden918_189_drop03_input06.txt','w') as f:
    for (i,j) in zip(list(pred['class']),list(pred['score'])):
        f.write('%d %f\n' %(i,j))
