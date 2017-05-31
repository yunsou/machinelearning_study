import tensorflow as tf
#import matplotlib.pyplot as plt
x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(5.)
X = [1,2,3]
Y = [1,2,3]

hypothesis = X * W

gradient = tf.reduce_mean((W * X - Y) * X) *2

cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

gvs = optimizer.compute_gradients(cost)
apply_gradients = optimizer.apply_gradients(gvs)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(100):
	print(i, sess.run([gradient, W, gvs]))
	sess.run(apply_gradients)
