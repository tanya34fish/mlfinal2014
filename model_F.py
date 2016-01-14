import numpy as np
import pickle
import graphlab as gl
from scipy.misc import imresize
import random

def loadData(filename,type=None):
    data = {}
    data['label'] = []
    data['image'] = []
    input = []
    output = []
    size = 128,128
	# print filename
    with open(filename ,'r') as f:
        for line in f:
            train = line.strip().split()
            if type == 'train':
                if len(train) < 50:
                    continue
            classID = int(train[0])
            for j in range(3):
                output.append(classID)
            #105*122 features
                image = []
                for i in range(12810):
                    image.append(0.0) #initialize
                for i in range(1,len(train)):
                    pixel = train[i].split(':')
                    k=random.random()
                    if j==0:
                        k=1
                    if float(pixel[1])<0.1:
                        pixel[1]=0
                    elif float(pixel[1])<0.4 and k<0.3:
                        pixel[1]=0
                    else:
                        pixel[1]=1
                    image[int(pixel[0])-1] = float(pixel[1]) * 255
                input.append(image)
                if type=='test':
                    break

    data['label'] = gl.SArray(output)
    scaled_input = gl.SArray(input)
    img_sarray = gl.SArray.pixel_array_to_image(scaled_input, 105, 122, 1, allow_rounding = True)
    data['image'] = img_sarray
    return data

if __name__ == '__main__':
	train_input = loadData('ml_train.dat.txt',type='train')
	test_input = loadData('ml14fall_test2_no_answer.dat',type='test')

	train_data = gl.SFrame(train_input)
	test_data = gl.SFrame(test_input)
	training_data, validation_data = train_data.random_split(0.6)
	training_data.save('data11/training_data_sframe')
	validation_data.save('data11/validation_data_sframe')
	test_data.save('data11/test_data_sframe')
	#training_data = gl.load_sframe('data10/training_data_sframe')
	#validation_data = gl.load_sframe('data10/validation_data_sframe')
	#test_data = gl.load_sframe('data10/test_data_sframe')
	conv_net = gl.deeplearning.ConvolutionNet(num_convolution_layers=2,
		   kernel_size=3, stride=2, num_channels=30, num_output_units=0)
	conv_net.layers.insert(1,gl.deeplearning.layers.RectifiedLinearLayer())
	conv_net.layers.insert(2,gl.deeplearning.layers.DropoutLayer(threshold=0.2))
	conv_net.layers.insert(5,gl.deeplearning.layers.RectifiedLinearLayer())
	conv_net.layers.insert(6,gl.deeplearning.layers.DropoutLayer(threshold=0.3))
	percpt_net = gl.deeplearning.MultiLayerPerceptrons(num_hidden_layers=3, num_hidden_units=[894,256,32])
	conv_net.layers.extend(percpt_net.layers)
	conv_net.layers.insert(-5,gl.deeplearning.layers.RectifiedLinearLayer())
	conv_net.layers.insert(-5,gl.deeplearning.layers.DropoutLayer(threshold=0.45))
	del conv_net.layers[-5]
	conv_net.layers.insert(-3,gl.deeplearning.layers.RectifiedLinearLayer())
	conv_net.layers.insert(-3,gl.deeplearning.layers.DropoutLayer())
	del conv_net.layers[-3]

	conv_net.save('net_F_1')
	print(conv_net.verify())
	#start running
	print "start_running"
	m = gl.neuralnet_classifier.create(training_data, target='label',
					 network = conv_net,
					 validation_set=validation_data,
					 metric=['accuracy', 'recall@2'],
					 max_iterations=30)
	m.save('model_hidden_F1')
	#m = gl.load_model('model_hidden500_3')
	pred = m.classify(test_data)
	print(pred['class'])

	with open('Final_f1.txt','w') as f:
		for i in list(pred['class']):
			f.write('%d\n' %i)
