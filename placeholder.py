import tensorflow as tf


#hello = tf.constant("Hello, Tensorflow")

#sess = tf.Session()

#print(sess.run(hello))

node1 = tf.placeholder(tf.float32)
node2 = tf.placeholder(tf.float32)

adder_node = node1 + node2
sess = tf.Session()


print(sess.run(adder_node, feed_dict={node1: 3, node2:4.5}))
print(sess.run(adder_node, feed_dict={node1: [1,3], node2:[2,4]}))
