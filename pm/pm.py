import tensorflow as tf
import numpy as np
import math, sys


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)


class PM(object):
	def __init__(self):
		self.hour  = 9
		self.dim   = 18
		self.epoch = 30
		self.lamda = 1
		self.lr    = 5e-4

	def preprocess(self):
		self.raw_data = np.genfromtxt('data/train.csv', dtype='float32', delimiter=',')
		self.raw_data = np.delete(self.raw_data, 0, 0)
		
		size = self.raw_data.shape[0]
		day  = size/18
		self.row = self.raw_data[:18, 3:27]
		self.x = self.raw_data[:18, 3:27]
		for i in range(day):
			row    = self.raw_data[i*18:(i+1)*18, 3:27]
			self.row = np.concatenate((self.row, row), axis=1)
			self.x = np.concatenate((self.x, row), axis=1)

		# The first data will repeat, so we delete it here
		self.x = np.delete(self.x, np.s_[0:24], 1)
		self.row = np.delete(self.row, np.s_[0:24], 1)

		for i in range(self.x.shape[0]):
			for j in range(self.x.shape[1]):
				if math.isnan(self.x[i][j]):
					self.x[i][j] = 0

		self.x_para = np.reshape(self.x[:, 0:9], (self.hour * self.dim, 1))
		self.y    = self.raw_data[9, 12]

		for j in range(12):
			for i in xrange(9, 480, 1):
				data_x = np.reshape(self.x[:, j*480+i-9:j*480+i], (self.hour * self.dim, 1))
				self.x_para = np.concatenate((self.x_para, data_x), axis=1)

				self.y      = np.append(self.y, self.row[9, j*480+i]) 
		

		self.x_para = np.delete(self.x_para, [0], 1)
		self.y      = np.delete(self.y, [0], 0)
		self.x_para = np.transpose(self.x_para)
		self.y      = np.reshape(self.y, (self.y.shape[0],1))
		self.x_para = self.x_para.astype(np.float)

		#print self.x_para.dtype
		#exit()


	def input(self):
		self.nn_x = tf.placeholder(tf.float32, [None, self.hour*self.dim])
		self.nn_y = tf.placeholder(tf.float32, [None, 1])
		self.W1   = tf.Variable(tf.random_uniform([self.hour*self.dim, 300], -1, 1))
		self.b1   = tf.Variable(tf.zeros([1, 300]))
		self.W2   = tf.Variable(tf.random_uniform([300, 1], -1, 1))
		self.b2   = tf.Variable(tf.zeros([1, 1]))
		self.W3   = tf.Variable(tf.random_uniform([300, 1], -1, 1))
		self.b3   = tf.Variable(tf.zeros([1,1]))
		#self.Y    = tf.matmul(tf.matmul(tf.matmul(self.nn_x, self.W1) + self.b1, self.W2)+ self.b2, self.W3) + self.b3
		Y         = tf.matmul(self.nn_x, self.W1) + self.b1
		Y         = tf.nn.dropout(Y, 0.7)
		self.Y    = tf.matmul(Y, self.W2)+ self.b2
		self.loss = tf.reduce_mean(tf.square(self.nn_y - self.Y))
		optimizer = tf.train.AdamOptimizer(self.lr)
		self.train     = optimizer.minimize(self.loss)
		self.initial   = tf.initialize_all_variables()

	def session(self):
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		self.sess.run(self.initial)
		for i in range(8000):
			loss, _ = self.sess.run([self.loss, self.train], feed_dict={self.nn_x: self.x_para,
										self.nn_y: self.y})

			#if i % 100 == 0:
			#	plt.plot(i, loss, 'ro')

			sys.stdout.write("\r training loss: %.3f" % loss)
			sys.stdout.flush()

		#plt.show()
		#self.sess.close()

	def predict(self):
		fout = open("result.csv", "w")
		fout.write("id,value\n")
		test_data = np.genfromtxt('data/test_X.csv',delimiter=',')
		test_x    = test_data[:18, 2:11]
		for i in range(test_x.shape[0]):
			for j in range(test_x.shape[1]):
				if math.isnan(test_x[i][j]):
					test_x[i][j] = 0


		test_x_para = np.reshape(test_x, (1, self.dim * self.hour))
		Y_hat = self.sess.run(self.Y, feed_dict={
									self.nn_x: test_x_para
										})

		fout.write("id_0,%.4f\n" % Y_hat[0][0])

		day       = test_data.shape[0]/18
		for i in xrange(1, day, 1):
			test_x = test_data[i*18:(i+1)*18, 2:11]
			for l in range(test_x.shape[0]):
				for n in range(test_x.shape[1]):
					if math.isnan(test_x[l][n]):
						test_x[l][n] = 0
			
			test_x_para = np.reshape(test_x, (1, self.dim * self.hour))
			Y_hat = self.sess.run(self.Y, feed_dict={
									self.nn_x: test_x_para
										})

			fout.write("id_%d,%.4f\n" % (i, Y_hat[0][0]))

		

def main():
	model = PM()
	model.preprocess()
	model.input()
	model.session()
	model.predict()


if __name__ == "__main__":
    main()

















