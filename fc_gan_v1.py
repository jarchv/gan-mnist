import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot = True)

def generator_model(Z, reuse = None):
	# tf.name_scope 	:  will add scope as a prefix to all operations
	# tf.variable_scope :  will add scope as a prefix to all variables and operations
	with tf.variable_scope('gen', reuse = reuse):
		hidden1 = tf.layers.dense(	inputs 		= Z,
									units  		= 128, # 1200
									activation 	= tf.nn.leaky_relu)

		hidden2 = tf.layers.dense(	inputs 		= hidden1,
									units		= 128, # 1200
									activation	= tf.nn.leaky_relu)

		output 	= tf.layers.dense( 	inputs 		= hidden2,
									units 		= 784,
									activation  = tf.nn.tanh)

	return output

def discriminator_model(X, reuse = None):
	with tf.variable_scope('dis', reuse = reuse):
		hidden1 = tf.layers.dense(	inputs 		= X,
									units  		= 256, # 240
									activation 	= tf.nn.leaky_relu)

		hidden1 = tf.nn.dropout(hidden1, keep_prob = 0.5)
		#hidden1 = tf.contrib.layers.maxout( inputs  	= hidden1,
		#									num_units	= 5)

		hidden2 = tf.layers.dense(	inputs 		= hidden1,
									units		= 256, # 240
									activation	= tf.nn.leaky_relu)

		hidden2 = tf.nn.dropout(hidden2, keep_prob = 0.5)
		#hidden2 = tf.contrib.layers.maxout( inputs  	= hidden2,
		#									num_units	= 5)

		output 	= tf.layers.dense( 	inputs 		= hidden2,
									units 		= 1,
									activation  = None)

	return output	

tf.reset_default_graph()

Zin = tf.placeholder(	dtype = tf.float32,
						shape = [None, 100],
						name  = 'Zin')

X   = tf.placeholder(	dtype = tf.float32,
						shape = [None, 784],
						name  = 'X')

G = generator_model(Zin)

dis_real = discriminator_model(X)
dis_fake = discriminator_model(G, True)

def loss_func(logits_in, labels_in):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits_in, labels = labels_in))

def get_loss(dis_real_image, dis_fake_image):
	real_image_loss = loss_func(dis_real_image, tf.ones_like(dis_real_image)) 
	fake_image_loss = loss_func(dis_fake_image, tf.zeros_like(dis_fake_image))

	#real_image_loss = -tf.reduce_mean(tf.math.log(dis_real_image + 1e-12),		axis = 0)
	#fake_image_loss = -tf.reduce_mean(tf.math.log(1 - dis_fake_image + 1e-12), axis = 0)

	dis_loss       =   real_image_loss + fake_image_loss
	
	gen_loss       = loss_func(dis_fake_image, tf.ones_like(dis_fake_image))
	#gen_loss       = -tf.reduce_mean(tf.math.log(dis_fake_image + 1e-12), axis = 0)

	return dis_loss, gen_loss

dis_loss, gen_loss = get_loss(dis_real, dis_fake)

glob_vars = tf.trainable_variables()


dis_vars  = [var for var in glob_vars if 'dis' in var.name]
gen_vars  = [var for var in glob_vars if 'gen' in var.name]

dis_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
gen_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)

dis_trainer   = dis_optimizer.minimize(dis_loss, var_list = dis_vars)
gen_trainer   = gen_optimizer.minimize(gen_loss, var_list = gen_vars)
 

epochs = 100000
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
			images_feed    = images_feed * 2.0 - 1.0
			z              = np.random.uniform( low = -1.0,  high = 1.0, size = (batch_size, 100))
			_, dis_loss_ep = sess.run([dis_trainer, dis_loss], feed_dict = {X : images_feed, Zin : z})
			_, gen_loss_ep = sess.run([gen_trainer, gen_loss], feed_dict = {Zin : z})

			dis_loss_t += dis_loss_ep / float(iters)
			gen_loss_t += gen_loss_ep / float(iters)

		print('epoch {:4d}, dis_loss : {:.4f}, gen_loss : {:.4f}'.format(epoch, dis_loss_t, gen_loss_t))

		sample_z   = np.random.uniform(low = -1.0, high = 1.0, size = (1, 100))
		gen_sample = sess.run(generator_model(Zin, reuse = True), feed_dict = {Zin : sample_z})

		if epoch % 40 == 0:
			plt.imshow(gen_sample.reshape(28, 28), cmap='gray')
			plt.show()