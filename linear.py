import numpy as np

class Linear:
    def __init__(self, in_feature, out_feature):
        self.w = np.random.randn(in_feature, out_feature)
        self.b = np.random.randn(out_feature)
        self.x = None

    def forward(self,x):
        self.x = x
        predict = x@self.w + self.b
        return predict
    
    def backward(self,grad_output):
        grad_w = self.x.T@grad_output
        grad_b = np.sum(grad_output,axis=0)
        grad_x = grad_output@self.w.T
        return grad_w, grad_b, grad_x
    
    def update(self,grad_w, grad_b, lr):
        self.w -= lr * grad_w
        self.b -= lr * grad_b

class RELU:
    def __init__(self):
        self.x = None
        
    def forward(self,x):
        self.x = x
        predict = (x>0) * x
        return predict
    
    def backward(self,grad_pred):
        grad_x = (self.x>0) * grad_pred
        return grad_x

if __name__ == "__main__":
    x = np.array([[1,2],[3,4],[5,6]], dtype = np.float32)
    target = np.array([[5],[11],[17]], dtype = np.float32)
    linear1 = Linear(2,4)
    linear2 = Linear(4,1)
    relu = RELU()
    for i in range(1000):
        pred = linear1.forward(x)
        pred = relu.forward(pred)
        pred = linear2.forward(pred)
        loss = np.mean((pred-target)**2)
        grad_output = 2*(pred-target)/target.size
        grad_w, grad_b, grad_x = linear2.backward(grad_output)
        linear2.update(grad_w,grad_b,lr=0.01)
        relu_x = relu.backward(grad_x)
        grad_w, grad_b, grad_x = linear1.backward(relu_x)
        linear1.update(grad_w,grad_b,lr=0.01)
    
    print(linear1.w,linear1.b,linear2.w,linear2.b,loss,pred)