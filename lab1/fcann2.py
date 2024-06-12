import numpy as np

np.random.seed(100)

def affine_forward(x, w, b):
    return x @ w + b

def ReLU_forward(x):
    return np.where(x < 0, 0, 1) * x

def softmax_forward(x):
    x -= np.max(x, axis=1, keepdims=True)
    numerator = np.exp(x)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    return numerator / denominator

def cross_entropy_loss(softmax, y):
    N = y.shape[0]
    correct_indexes = np.argmax(y, axis=1)
    return -np.mean(np.log(softmax[np.arange(0, N), correct_indexes] + 1e-10))

def cross_entropy_loss_backward(softmax, y):
    N = y.shape[0]
    correct_indexes = np.argmax(y, axis=1)
    softmax[np.arange(0, N), correct_indexes] -= 1
    dx = softmax / N
    return dx

def affine_backward(dout, x, w, b):
    """
    Arguments:
        x: output from layer before (input data) Nx5, Nx2
        w: Weights, np.array Dx5, 5xC
        b: biases, np.array D, 5
        dout: derivative w.r.t layer output, np.array Dx5, 5xC
    """
    dx = dout @ w.T
    dw = x.T @ dout
    db = np.sum(dout, axis=0)
    return dx, dw, db

def ReLU_backward(dout, x):
    return np.where(x < 0, 0, 1) * dout

def fcann2_train(X, Y, param_niter=int(1e5), param_delta=0.05, param_lambda=1e-3):
    """
    FC layer (Affine) -> ReLU -> FC layer (Affine) -> softmax
    Arguments:
        X: data, np.array NxD
        Y: classes, np.array Nx1
    Return:
        W: Weights, np.array Dx5, 5xC
        b: biases, np.array D, 5
    """

    N, D = X.shape
    C = Y.shape[0]
    input_dim = D
    hidden_dim = 5
    output_dim = 2

    W1 = np.random.normal(0, 0.25, size=(input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    W2 = np.random.normal(0, 0.25, size=(hidden_dim, output_dim))
    b2 = np.zeros(output_dim)

    loss = None

    for epoch in range(param_niter):
        # forward prop
        # 1st layer
        out_affine1 = affine_forward(X, W1, b1) # NxD * D*5 = Nx5
        out_relu1 = ReLU_forward(out_affine1)
        # 2nd layer
        out_affine2 = affine_forward(out_relu1, W2, b2) # Nx5 * 5x2 = Nx2
        out_softmax2 = softmax_forward(out_affine2)

        loss = cross_entropy_loss(out_softmax2, Y) + param_lambda * (np.linalg.norm(W1) + np.linalg.norm(W2))

        # backward prop
        # 2nd layer
        softmax_dout2 = cross_entropy_loss_backward(out_softmax2, Y)

        affine_dout2, dW2, db2 = affine_backward(softmax_dout2, out_relu1, W2, b2)

        # 1st layer
        relu_dout1 = ReLU_backward(affine_dout2, out_affine1)
        dout, dW1, db1 = affine_backward(relu_dout1, X, W1, b1)

        # update weights
        W2 = W2 - param_delta * dW2
        b2 = b2 - param_delta * db2
        W1 = W1 - param_delta * dW1
        b1 = b1 - param_delta * db1

        if epoch % 10000 == 0:
            print("Epoch:", epoch, " , loss=", loss)

    return W1, b1, W2, b2

def fcann2_classify(X, W1, b1, W2, b2):
    """
    Arguments:
        X: data
        W1, b1, W2, b2: weight and bias parameters

    Return:
        probs: probability of classes
    """
    out_affine1 = affine_forward(X, W1, b1) # NxD * D*5 = Nx5
    out_relu1 = ReLU_forward(out_affine1)
    out_affine2 = affine_forward(out_relu1, W2, b2) # Nx5 * 5x2 = Nx2
    out_softmax2 = softmax_forward(out_affine2)

    y_pred = np.argmax(out_softmax2, axis=1)
    return y_pred


def fcann2_decfun(W1, b1, W2, b2):
    def classify(X):
        return fcann2_classify(X, W1, b1, W2, b2)
    return classify
    
    