import torch
import numpy as np
import torch.nn as nn
def pad_same(in_dim, ks, stride, dilation=1):
    """
    Refernces:
          https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.h
          https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/common_shape_fns.cc#L21
    """
    assert stride > 0
    assert dilation >= 1
    effective_ks = (ks - 1) * dilation + 1
    out_dim = (in_dim + stride - 1) // stride
    p = max(0, (out_dim - 1) * stride + effective_ks - in_dim)

    padding_before = p // 2
    padding_after = p - padding_before
    return padding_before, padding_after
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
def conv2d_samepad(in_dim, in_ch, out_ch, ks, stride, dilation=1, bias=True):
    pad_before, pad_after = pad_same(in_dim, ks, stride, dilation)
    if pad_before == pad_after:
        return [nn.Conv2d(in_ch, out_ch, ks, stride, pad_after, dilation, bias=bias)]
    else:
        return [nn.ZeroPad2d((pad_before, pad_after, pad_before, pad_after)),
                nn.Conv2d(in_ch, out_ch, ks, stride, 0, dilation, bias=bias)]
def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.
    Args:
        outputs: (Variable) dimension batch_size x 11 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5,..., 10]
    Returns:
        loss (Variable): cross entropy loss for all images in the batch
    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """
    num_examples = outputs.size()[0]
    # print(labels)
    return -torch.sum(outputs[range(num_examples), labels]) / num_examples


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5,... 10]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels) / float(labels.size)


# adding F1 Score, obtained from:
# https://codereview.stackexchange.com/questions/36096/implementing-f1-score
def f1_score(outputs, labels):
    outputs = np.argmax(outputs, axis=1)
    labels = set(labels)
    outputs = set(outputs)

    tp = len(labels.intersection(outputs))
    fp = len(outputs.difference(labels))
    fn = len(labels.difference(outputs))

    if tp > 0:
        precision = float(tp) / (tp + fp)
        recall = float(tp) / (tp + fn)

        return 2 * ((precision * recall) / (precision + recall))
    else:
        return 0


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy, 'f1_score': f1_score
    # could add more metrics such as accuracy for each token type
}