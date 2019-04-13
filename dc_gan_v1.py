import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

def generator_model(Z, reuse = None):

	with tf.variable_scope('gen', reuse = reuse):
		hidden1 = tf.layers.dense(	inputs 	 = Z,
									units	 = 7 * 7 * 256,
									use_bias = False)

		#hidden1 = tf.layers.batch_normalization(inputs 		= hidden1)
		hidden1 = tf.nn.leaky_relu(	features = hidden1)

		r_layer = tf.reshape(hidden1, [-1, 7 , 7, 256])

		hidden2 = tf.layers.conv2d_transpose(	inputs  	= r_layer,
												filters 	= 64,
												kernel_size = (5, 5),
												strides 	= (1, 1),
												padding		= 'same',
												use_bias	= False)

		#hidden2 = tf.layers.batch_normalization(inputs 		= hidden2)
		hidden2 = tf.nn.leaky_relu(	features = hidden2)

		hidden3 = tf.layers.conv2d_transpose(	inputs  	= hidden2,
												filters 	= 128,
												kernel_size = (5, 5),
												strides 	= (2, 2),
												padding		= 'same',
												use_bias	= False)	

		#hidden3 = tf.layers.batch_normalization(inputs 		= hidden3)
		hidden3 = tf.nn.leaky_relu(	features = hidden3)

		output = tf.layers.conv2d_transpose(	inputs  	= hidden3,
												filters 	= 1,
												kernel_size = (5, 5),
												strides 	= (2, 2),
												padding		= 'same',
												use_bias	= False,
												activation  = tf.nn.tanh)	

		#print(hidden4.shape)
		assert (output.shape[1:] == (28, 28, 1))
		
		#output  = tf.nn.tanh(x = hidden4)
	
		return output

def discriminator_model(X, reuse = None):
	with tf.variable_scope('dis', reuse = reuse):
		hidden1 	= tf.layers.conv2d(	inputs 		= X,
										filters 	= 64,
										kernel_size = (5, 5),
										strides 	= (2, 2),
										padding  	= 'same',
										use_bias 	= True)

		hidden1 	= tf.nn.leaky_relu(	features = hidden1)
		hidden1 	= tf.nn.dropout(hidden1, keep_prob = 0.5)

		hidden2 	= tf.layers.conv2d(	inputs		= hidden1,
										filters 	= 128,
										kernel_size = (5, 5),
										strides		= (2, 2),
										padding		= 'same',
										use_bias 	= True)

		hidden2 	= tf.nn.leaky_relu(	features = hidden2)
		hidden2 	= tf.nn.dropout(hidden2, keep_prob = 0.5)

		l_flatten 	= tf.layers.flatten( inputs 	= hidden2)

		output	 	= tf.layers.dense(	inputs 		= l_flatten,
										units 		= 1) 

		return output




Zin 		 = tf.placeholder(	dtype = tf.float32,
								shape = [None, 100],
								name  = 'Zin')

syn_img    	 = generator_model(Zin)


#init = tf.global_variables_initializer()

#with tf.Session() as sess:

#	sess.run(init)
#	noise_sample = np.random.normal(size = (1, 100))
#	img_sample 	 = sess.run(syn_img, feed_dict = {Zin : noise_sample})

#	print(img_sample.shape)
#	plt.imshow(img_sample.reshape((28, 28)), cmap = 'gray')
#	plt.show()


X = tf.placeholder(	dtype = tf.float32,
					shape = [None, 28, 28, 1],
					name  = 'X')

dis_real_image = discriminator_model(X)
dis_fake_image = discriminator_model(syn_img, reuse = True)

def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_in, labels = labels_in))

def get_loss(dis_real_image, dis_fake_image):
	real_image_loss = loss_func(dis_real_image, tf.ones_like(dis_real_image)) 
	fake_image_loss = loss_func(dis_fake_image, tf.zeros_like(dis_fake_image))

	dis_loss        = real_image_loss + fake_image_loss
	gen_loss        = loss_func(dis_fake_image, tf.ones_like(dis_fake_image))
	
	return dis_loss, gen_loss

dis_loss, gen_loss = get_loss(dis_real_image, dis_fake_image)

glob_vars = tf.trainable_variables()


dis_vars  = [var for var in glob_vars if 'dis' in var.name]
gen_vars  = [var for var in glob_vars if 'gen' in var.name]

dis_optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4)
gen_optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4)

dis_trainer   = dis_optimizer.minimize(dis_loss, var_list = dis_vars)
gen_trainer   = gen_optimizer.minimize(gen_loss, var_list = gen_vars)

epochs      = 100000
batch_size  = 100

init   = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	iters = int(mnist.train.num_examples / batch_size)

	for epoch in range(1, epochs + 1):
		dis_loss_t = 0.0
		gen_loss_t = 0.0
		for _ in range(iters):
			images_feed, _ = mnist.train.next_batch(batch_size)
			images_feed    = images_feed.reshape((-1, 28, 28, 1))
			#images_feed    = images_feed * 2.0 - 1.0


			#z 			   = np.random.normal(size = (batch_size, 100))
			z              = np.random.uniform(low = -1.0,  high = 1.0, size = (batch_size, 100))
			_, dis_loss_ep = sess.run([dis_trainer, dis_loss], feed_dict = {X : images_feed, Zin : z})
			_, gen_loss_ep = sess.run([gen_trainer, gen_loss], feed_dict = {Zin : z})

			dis_loss_t += dis_loss_ep / float(iters)
			gen_loss_t += gen_loss_ep / float(iters)

		print('epoch {:4d}, dis_loss : {:.7f}, gen_loss : {:.7f}'.format(epoch, dis_loss_t, gen_loss_t))

		sample_z   = np.random.uniform(low = -1.0, high = 1.0, size = (1, 100))
		#sample_z   = np.random.normal(size = (1, 100))
		gen_sample = sess.run(generator_model(Zin, reuse = True), feed_dict = {Zin : sample_z})

		print(gen_sample.shape)
		syn_sample = np.reshape(gen_sample, (28, 28))
		print(syn_sample.shape)
		print(syn_sample)
		plt.imshow(syn_sample)
		plt.show()