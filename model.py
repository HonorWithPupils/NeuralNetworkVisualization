import numpy as np

def resample(x, n): 
    
    x1, x2 = x[:,0], x[:,1]
        
    t = np.linspace(-1, 1, len(x))
    t_val = np.linspace(-1, 1, n)
    
    x1 = np.interp(t_val, t, x1)
    x2 = np.interp(t_val, t, x2)
    
    x = np.column_stack((x1, x2))
    
    return x

class Linear(object):
    def __init__(self, dim_input, dim_output):
        self.dim_input = dim_input
        self.dim_output = dim_output
        
        self.__init_param()
        
    def __init_param(self): 
        self.weight = np.random.randn(self.dim_input, self.dim_output) * 2
        self.bias = np.zeros([1, self.dim_output])
        
    def forward(self, input):
        
        self.input = input
        self.output = np.dot(input, self.weight) + self.bias
        
        return self.output
    
    def backward(self, top_diff):
        
        self.d_weight = np.dot(self.input.T, top_diff)
        self.d_bias = np.sum(top_diff, axis=0)
        bottom_diff = np.dot(top_diff, self.weight.T)
        
        return bottom_diff
    
    def update_param(self, lr):
        self.weight = self.weight - lr*self.d_weight
        self.bias = self.bias - lr*self.d_bias
        
    def reverse(self, output):
        
        assert self.dim_input == self.dim_output, 'Linear reverse only works for dim_input == num_output'
        
        return (output - self.bias) @ np.linalg.pinv(self.weight)
    
class Tanh(object):
    def __init__(self):
        ...
        
    def forward(self, input):
        
        self.input = input
        self.output = np.tanh(input)
        
        return self.output
    
    def backward(self, top_diff):
        
        bottom_diff = top_diff * (1 - self.output**2)
        
        return bottom_diff
    
    def reverse(self, output):
        
        length = len(output)
        
        output = output[(np.abs(output) < 1).all(axis=1)]
        
        x = resample(output, length)
        
        return np.arctanh(x)

class Sigmod(object):
    def __init__(self):
        ...
        
    def forward(self, input):
        
        self.input = input
        self.output = 1 / (1 + np.exp(-input))
        
        return self.output
    
    def backward(self, top_diff):
        
        bottom_diff = top_diff * self.output * (1 - self.output)
        
        return bottom_diff
    
    def reverse(self, output):
        
        return np.log(output / (1 - output))
    
class Model(object):
    def __init__(self, n_layers:int = 2):
        
        self.bone = [(Linear(2, 2), Tanh()) for i in range(n_layers)]
        
        self.fc = Linear(2, 1)
        
        self.sigmod = Sigmod()
        
    def forward(self, x):
        
        for i in range(len(self.bone)):
            x = self.bone[i][0].forward(x)
            x = self.bone[i][1].forward(x)
        
        x = self.fc.forward(x)
        x = self.sigmod.forward(x)

        return x
    
    def backward(self, top_diff):
        
        diff = self.sigmod.backward(top_diff)
        diff = self.fc.backward(diff)
        
        for i in range(len(self.bone) - 1, -1, -1):
            diff = self.bone[i][1].backward(diff)
            diff = self.bone[i][0].backward(diff)
        
        return diff
    
    def update_param(self, lr):
        
        for i in range(len(self.bone)):
            self.bone[i][0].update_param(lr)
        
        self.fc.update_param(lr)
        
    def opt(self, x, y, lr):
        
        yp = self.forward(x)
        
        loss = - (y * np.log(yp) + (1 - y) * np.log(1 - yp)).mean()
        # loss = (0.5 * (yp - y) ** 2).mean()
        
        diff = - (y / yp - (1 - y) / (1 - yp)) / x.shape[0]
        # diff = (yp - y) / x.shape[0]
        
        self.backward(diff)
        
        self.update_param(lr)
        
        return loss
    
    def accuracy(self, x, y):
        yp = self.forward(x)
        yp = (yp > 0.5).astype(int)
        acc = (yp == y).mean()
        return acc
    
    def spaces(self, x): 
        
        xs = [x]
        
        for i in range(len(self.bone)):
            x = self.bone[i][0].forward(x)
            x = self.bone[i][1].forward(x)
            
            xs.append(x)
        
        return xs, self.devideLines(), self.grids()
    
    def devideLines(self): 
        
        w = self.fc.weight
        b = self.fc.bias
        
        x1 = np.linspace(-1.1, 1.1, int(1e7))
        x2 = -(w[0] * x1 + b[0]) / w[1]
        
        dl = np.concatenate((x1[:,None], x2[:,None]), axis=1)
        
        dls = [resample(dl[(np.abs(dl) < 1.1).all(axis=1)], 100)]
        
        for i in range(len(self.bone)-1, -1, -1):
            dl = self.bone[i][1].reverse(dl)
            dl = self.bone[i][0].reverse(dl)
            
            dls.append(resample(dl[(np.abs(dl) < 1.1).all(axis=1)], 100))
            
        dls = dls[::-1]
        
        return dls
    
    def grids(self, n:int = 5):
        
        c = np.linspace(-1, 1, n)
        
        grid = []
        
        n_sample = 20
        
        for i in range(n):
            x = np.linspace(-1, 1, n_sample)
            x = np.concatenate((x[:,None], c[i] * np.ones((n_sample, 1))), axis=1)
            
            grid.append(x)
            grid.append(x[:,::-1])
            
        grid = np.concatenate(grid, axis=0)
        
        grids = [grid]
        
        for i in range(len(self.bone)):
            grid = self.bone[i][0].forward(grid)
            grid = self.bone[i][1].forward(grid)
            
            grids.append(grid)
            
        return [g.reshape(-1, n_sample, 2) for g in grids]