import tensorflow as tf
#import matplotlib.pyplot as plt
x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(-5.0, name='weight')
X = [1,2,3]
Y = [1,2,3]

hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())


for i in range(300):
	sess.run(W)
	print(i,  sess.run(W), sess.run(train))
