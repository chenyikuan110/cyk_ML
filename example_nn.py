import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True);

import tensorflow as tf


# Set Hyper Params
learning_rate = 0.01;
training_iteration = 30;
batch_size = 100;
display_step = 2;

# Tensorflow graph input
x = tf.placeholder("float",[None,784])
y = tf.placeholder("float",[None,10])

# Create a model

# Set model weights
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scope:
	# build a linear model
	model = tf.nn.softmax(tf.matmul(x,W) + b)
	
# Add summary operators to collect data
w_h = tf.summary.histogram("weights", W)
b_h = tf.summary.histogram("biases", b) 

# more name scope will clean up graph representation
with tf.name_scope("cost_function") as scope:
	# minimize error using cross entropy
	
	#cross entropy
	cost_function = -tf.reduce_sum(y*tf.log(model));
	
	#create a summary to monitor the cost function
	tf.summary.scalar("cost_function", cost_function)
	
with tf.name_scope("train") as scope:
	# GD
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
	
# Initializing variables

init = tf.global_variables_initializer()

merged_summary_op = tf.summary.merge_all()


# Launch the graph

with tf.Session() as sess:
	sess.run(init)

	# set the logs writer to the folder /tmp/tensorflow_logs
	summary_writer = tf.summary.FileWriter('./tensorflow_nn/logs', graph_def=sess.graph_def)
	
	# Training cycle
	for iteration in range(training_iteration):
		avg_cost = 0.;
		total_batch = int(mnist.train.num_examples/batch_size)
		
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			
			# Fit training using batch data
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			
			# compute the avg loss
			avg_cost += sess.run(cost_function, feed_dict={x:batch_xs,y:batch_ys})/total_batch;
			
			# write logs for each iteration
			summary_str = sess.run(merged_summary_op, feed_dict= {x:batch_xs,y:batch_ys})
			summary_writer.add_summary(summary_str,iteration*total_batch + i)
			
		# display logs per iteration step
		if iteration % display_step == 0:
			print ("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))
			
		print("training complete!");
		predictions = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(predictions,"float"))
		print("the accuracy is:", accuracy.eval({x: mnist.test.images,y: mnist.test.labels}))
		
	print("All training complete!");
	# Test the model
	predictions = tf.equal(tf.argmax(model,1), tf.argmax(y,1))
	
	# get the accuracy
	accuracy = tf.reduce_mean(tf.cast(predictions,"float"))
	print("the accuracy is:", accuracy.eval({x: mnist.test.images,y: mnist.test.labels}))