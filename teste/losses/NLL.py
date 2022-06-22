import tensorflow as tf

def NLL(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def kl_approx(q, p, q_tensor):
    return tf.reduce_mean(q.log_prob(q_tensor) - p.log_prob(q_tensor))
