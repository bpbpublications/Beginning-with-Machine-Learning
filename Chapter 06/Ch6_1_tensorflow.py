import tensorflow as tf

Hey = tf.constant('Hello World')
print(type(Hey))
sess = tf.Session() 
sess.run(Hey) 
print(type(sess.run(Hey)))

x = tf.constant(6) 
y = tf.constant(7)
with tf.Session() as sess:
    print('Operations with Constants') 
    print('Addition', sess.run(x + y))
    print('Subtraction', sess.run(x - y)) 
    print('Multiplication', sess.run(x*y)) 
    print('Division', sess.run(x/y))

x = tf.placeholder(tf.int32) 
y = tf.placeholder(tf.int32)
#Now that we have created the placeholder, let's define some operations for these:
add = tf.add(x, y)
sub = tf.subtract(x, y) 
mul = tf.multiply(x, y) 
div = tf.divide(x, y)
d = {x : 90, y : 100}

with tf.Session() as sess:
    print('Operations with Constants')
    print('Addition', sess.run(add,feed_dict = d)) 
    print('Subtraction', sess.run(sub, feed_dict = d))
    print('Multiplication', sess.run(mul, feed_dict = d)) 
    print('Division',sess.run(div, feed_dict = d))
