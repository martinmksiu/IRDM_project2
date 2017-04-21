
import pandas as pd
import tensorflow as tf
import numpy as np

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
plt.close("all")

tf.reset_default_graph()


# computational graph for Tensorflow:

learning_rate=0.00001
X = tf.placeholder("float32", [None, 136])  # is it 135??
Y = tf.placeholder("float32", [None, 5])  # the relevance
keep_prob = tf.placeholder(tf.float32)  # for dropout


w = tf.Variable(tf.random_normal([136, 5])*0.01)
b = tf.Variable(tf.constant(0.01), [1, 5])

# linear layer
output_lin = tf.matmul(X,w)+b

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets = Y, logits=output_lin))

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


preds = tf.argmax(tf.sigmoid(output_lin),1)
preds_sig = tf.sigmoid(output_lin)

# sigmoid on output
correct_pred = tf.equal(tf.argmax(tf.sigmoid(output_lin), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


def graphdata(data, name):
    """
    data is a 1-d array
    name is the y-axis label

    returns a plot
    """
    print("plot {}".format(name))
    fig = plt.figure()
    plt.plot(data)
    plt.xlabel("Epoch")
    plt.ylabel(name)
    fig.suptitle(name, fontsize=20)



def learn(x, y, id, learning_rate=0.00001, epoch=2, batch_size=256, training=True, testing=False):
    """
    inputs:
        x is the array of featurs
        y is the array of relevance (= labesl)
        id is the query id
        epochs: number of epochs to train for
        batch size is the size of a minibatch
        training    if True, trains the neural net, if false, proceeds to test the current parameters


    returns y_out, id_out, output_out
    where:
            y_out is the ground truth of the relevance
            id_out is a document id
            output_out is the output of the neural network (aka the probability that a document belongs to each class)
    """
    # open


    lossvector = []
    accvector = []
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.allow_growth = True

    array_output = []
    y_out = []
    id_out = []
    output_out = []

    # start a session and run computations
    with tf.Session(config=session_conf) as sess:
        tf.global_variables_initializer().run()

        for i in range(epoch):
            # separate the data into mini batches and loop through all minibatches at each epoch
            for start, end in zip(range(0,x.shape[0], batch_size), range(batch_size, x.shape[0] +1 ,batch_size)):

                if training:
                    sess.run(train_op, feed_dict={X: x[start:end],Y:y[start:end].reshape(-1,5)})  # do this beforehand

                output_temp = sess.run(preds, feed_dict={X: x[start:end]})
                output_temp_sig = sess.run(preds_sig, feed_dict={X: x[start:end]})

                if i == epoch-1:
                    y_out.append(y[start:end])
                    id_out.append(id[start:end])
                    output_out.append(output_temp_sig)
                loss = sess.run(cost, feed_dict={X: x[start:end], Y: y[start:end].reshape(-1,5)})  # batch loss
                acc = sess.run(accuracy, feed_dict={X: x[start:end], Y:y[start:end].reshape(-1,5)})

            # report performance
            if training:
                print("Iter " + str(i) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
                lossvector.append(loss)
                accvector.append(acc)
            if testing:
                print("Iter " + str(i) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Testing Accuracy= " + \
                              "{:.5f}".format(acc))
    # make plots

    graphdata(lossvector, "loss")
    graphdata(accvector, "accuracy")

    return y_out, id_out, output_out







def runq(learn_class=False, testing=False,epoch=1):
    """
     training = True    trains the algorithm using the neural net
     testing = True     evaluates the performance on the test data. saves the outputs
     epoch = number epochs to train for
    """
    print("Opening data")
    X, Y = load_svmlight_file("train.txt")  # X.shape (723 412, 136)
    # id was taken out
    enc = OneHotEncoder()
    a = pd.get_dummies(Y).as_matrix()
    b = load_svmlight_file("train.txt", query_id=True)
    query_id = b[2]


    ###################################### learning command
    if learn_class:
        y_out, id_out, output = learn(x=X.toarray(), y=a, id=query_id, epoch=epoch)

    if testing:
        print("testing")
        X,Y = load_svmlight_file("test.txt")

        a = pd.get_dummies(Y).as_matrix()
        b = load_svmlight_file("test.txt",query_id=True)
        query_id = b[2]
        y_out, id_out, output = learn(x=X.toarray(), y=a, id=query_id, epoch=epoch, training=False)

        first_temp = np.reshape(y_out, (-1,5))
        second_temp = np.reshape(id_out,(-1,1))
        third_temp = np.reshape(output, (-1,5))

        outputs = [first_temp, second_temp, third_temp]

        print("saving")

        # y_out, id_out, output
        max_index_score = np.zeros(len(third_temp))
        get_rank = np.zeros((len(third_temp),2))
        for i in range(0, len(third_temp)):

            current = np.argmax(third_temp[i],0)
            max_index_score[i]= current

            get_rank[i,0]=third_temp[i][current]
            get_rank[i,1] = current

        # save outputs.
        # because the data is large, they were separated into multiple files.
        pdlist = pd.DataFrame(first_temp)
        pdlist.to_csv("first.csv")
        pdlist2 = pd.DataFrame(second_temp)
        pdlist2.to_csv("second.csv")
        pdlist3 = pd.DataFrame(get_rank)
        pdlist3.to_csv("third.csv")
        pdlist4 = pd.DataFrame(third_temp)
        pdlist4.to_csv("fourth")
    return get_rank


get_rank = runq(learn_class=True,testing=True,epoch=40)
