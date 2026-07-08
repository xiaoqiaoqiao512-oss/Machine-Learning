import numpy as np

x = np.array([[1,2],[3,4],[5,6]], dtype=np.float32)
target = np.array([[5],[11],[17]], dtype=np.float32)
w = np.random.randn(2, 1)
b = np.random.randn()

def forward(x, w, b):
    predict = x@w+b;
    return predict

def compute_loss(pred,target):
    loss = np.mean((pred-target)**2)
    return loss

def backward(predict,target):
    grad_pred = 2*(predict-target)/target.size
    
    grad_w = x.T@grad_pred
    grad_b = np.sum(grad_pred)
    return grad_w, grad_b

def update(grad_w, grad_b, w, b, lr):
    w -= lr * grad_w
    b -= lr * grad_b
    return w, b

def train(lr):
    pred = forward(x, w, b)
    loss = compute_loss(pred, target)
    grad_w, grad_b = backward(pred, target)
    w1, b1 = update(grad_w, grad_b, w, b, lr)
    return w1, b1, loss

if __name__ == "__main__":
    w1, b1, loss = train(lr=0.01)
    print(w1,b1,loss)