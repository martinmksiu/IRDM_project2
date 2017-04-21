
import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
import csv
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
plt.close("all")

tf.reset_default_graph()

learning_rate=0.00001
X = tf.placeholder("float32", [None, 136])  # is it 135??
Y = tf.placeholder("float32", [None, 5])  # the relevance
keep_prob = tf.placeholder(tf.float32)  # for dropout
# here the y is a value 0,1, or 2. Should that be converted to [0,1,2] vector?

#sparse_place = tf.sparse_placeholder("float32", [None, 136])

w = tf.Variable(tf.random_normal([136, 5])*0.01) # needs to be [136,3] <- one-hot encoding?
b = tf.Variable(tf.constant(0.01), [1, 5])

output_lin = tf.matmul(X,w)+b
#dropout = tf.nn.dropout(output_lin, 0.5)
# add dropout???

cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(targets = Y, logits=output_lin))

train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# # we need a 0/1 labeling for this
preds = tf.argmax(tf.sigmoid(output_lin),1)
preds_sig = tf.sigmoid(output_lin)
#preds = (tf.sigmoid(output_lin),1)
#predict_op = tf.argmax(pred, 1)
correct_pred = tf.equal(tf.argmax(tf.sigmoid(output_lin), 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def yencode(y):
    ynew = 1
    return ynew


def graphdata(data, name):
    """
    plot
    """
    print("plot {}".format(name))
    fig = plt.figure()
    plt.plot(data)
    fig.suptitle(name, fontsize=20)


def dcg_score(y_true, y_score, k=10, gains="exponential"):
    """Discounted cumulative gain (DCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    DCG @k : float
    """
    #order = np.argsort(y_score)[::-1]
    #y_true = np.take(y_true, order[:k])

    if gains == "exponential":
        gains = 2 ** y_true - 1
    elif gains == "linear":
        gains = y_true
    else:
        raise ValueError("Invalid gains option.")

    # highest rank is 1 so +2 instead of +1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)

def ndcg_score(y_true, y_score, k=10, gains="exponential"):
    """Normalized discounted cumulative gain (NDCG) at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    gains : str
        Whether gains should be "exponential" (default) or "linear".
    Returns
    -------
    NDCG @k : float
    """
    best = dcg_score(y_true, y_true, k, gains)
    actual = dcg_score(y_true, y_score, k, gains)
    return actual / best



def metrics(ndcg=True):
    if ndcg:
        # to finish
        ndcg_score()
    return 1


def learn(x, y, id, learning_rate=0.00001, epoch=2, batch_size=256, training=True, testing=False):
    """

    returns
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
    with tf.Session(config=session_conf) as sess:
        tf.global_variables_initializer().run()

        for i in range(epoch):
            #for start, end in zip(range(0, len(x), batch_size), range(batch_size, len(x)+1, batch_size)):
            for start, end in zip(range(0,x.shape[0], batch_size), range(batch_size, x.shape[0] +1 ,batch_size)):
                #sess.run(train_op, feed_dict={X: x[start:end], Y: y[start:end]})

                #sess.run(train_op, feed_dict={sparse_place: x[start:end].toarray(),Y:y[start:end].toarray()})  # do this beforehand
                if training:
                    sess.run(train_op, feed_dict={X: x[start:end],Y:y[start:end].reshape(-1,5)})  # do this beforehand

                output_temp = sess.run(preds, feed_dict={X: x[start:end]})
                output_temp_sig = sess.run(preds_sig, feed_dict={X: x[start:end]})
                #output = sess.run(correct_pred, feed_dict={X:x[start:end]})
                print(output_temp)
                #temp = [y[start:end], id[start:end], output_temp_sig]
                #array_output.append(temp)
                y_out.append(y[start:end])
                id_out.append(id[start:end])
                output_out.append(output_temp_sig)
                #print(outputs)
                #print(Y)

                loss = sess.run(cost, feed_dict={X: x[start:end], Y: y[start:end].reshape(-1,5)})  # batch loss
                acc = sess.run(accuracy, feed_dict={X: x[start:end], Y:y[start:end].reshape(-1,5)})
            print("Iter " + str(i) + ", Minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{:.5f}".format(acc))
            lossvector.append(loss)
            accvector.append(acc)



    graphdata(lossvector, "loss")
    graphdata(accvector, "accuracy")
        # open test data
        #test_acc = sess.run(accuracy, feed_dict={X: teX[test_indices], Y:teY[test_indices]})
        #test_loss = sess.run(cost, feed_dict={X: teX[test_indices], Y: teY[test_indices]})  # batch loss

        #print("TEST: {}: accuracy: {}, loss: {}".format(i, test_acc, test_loss))
    # session end
    return y_out, id_out, output_out

# def testing():
#                 if i== (epoch-1):
#                 print("test data")
#                 X, Y = load_svmlight_file("test.txt")  # X.shape (723 412, 136)
#                 # id was taken out
#                 enc = OneHotEncoder()
#                 a = pd.get_dummies(Y).as_matrix()
#                 b = load_svmlight_file("train.txt", query_id=True)
#                 query_id = b[2]
#                 y_out, id_out, output = learn(x=X.toarray(), y=a, id=query_id, epoch=1)

def rank_query(atts, query_array, query_number = 1):
    """query_array is the array of all query number
    query number is the current query ID
    atts is the matrix of attributes (X)

    returns an array of documents for the current query"""

    docs = []
    length = np.shape(X)[0]
    #counter = 0
    for i in range(0,length):
        if query_array[i] == query_number:
            docs.append(atts[i])
    return docs



def runq(learn_class=False, testing=False,epoch=1):
    print("Opening data")
    X, Y = load_svmlight_file("train.txt")  # X.shape (723 412, 136)
    # id was taken out
    enc = OneHotEncoder()
    a = pd.get_dummies(Y).as_matrix()
    b = load_svmlight_file("train.txt", query_id=True)
    query_id = b[2]


    ###################################### learning command
    if learn_class:
        y_out, id_out, output = learn(x=X.toarray(), y=a, id=query_id, epoch=1)

    if testing:
        print("testing")
        X,Y = load_svmlight_file("test.txt")
        #enc = oneHotEncoder()
        a = pd.get_dummies(Y).as_matrix()
        b = load_svmlight_file("test.txt",query_id=True)
        query_id = b[2]
        y_out, id_out, output = learn(x=X.toarray(), y=a, id=query_id, epoch=epoch, training=False)

        first_temp = np.reshape(y_out, (-1,5))
        second_temp = np.reshape(id_out,(-1,1))
        third_temp = np.reshape(output, (-1,5))
        outputs = [first_temp, second_temp, third_temp]

        print("saving")

        pdlist = pd.DataFrame(first_temp)
        pdlist.to_csv("first.csv")
        pdlist2 = pd.DataFrame(second_temp)
        pdlist2.to_csv("second.csv")
        pdlist3 = pd.DataFrame(third_temp)
        pdlist3.to_csv("third.csv")
    # else:
    #     #np.savetxt("outputs.csv", outputs, delimiter=",")
    #     temp =  X.todense()
    #     #doc_id =
    #     # ranking
    #     # def rank_query(atts, query_array, query_number = 1):
    #     ranking = np.genfromtxt('outputs.csv',delimiter=',')
    #     print(np.shape(ranking)) #(256,)
    #     docs = rank_query(X, query_id, query_number = 1)
    #     print(np.shape(docs))  # (86,)
    #     # need to get id out of learn()

runq(learn_class=True,testing=True,epoch=5)
