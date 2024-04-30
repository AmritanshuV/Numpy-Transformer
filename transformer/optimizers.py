import numpy as np

try: 
    import cupy as cp
    is_cupy_available = True
except:
    is_cupy_available = False

from numba import njit
from numba import cuda

'''@cuda.jit
def _update_cupy(alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t):
    i = cuda.grid(1)  # Calculate the unique thread index
    if i < weights.size:  # Ensure not to go out of bounds
        m[i] = beta * m[i] + (1 - beta) * gradient[i]
        v[i] = beta2 * v[i] + (1 - beta2) * np.power(gradient[i], 2)
        m_hat[i] = m[i] / (1 - np.power(beta, t))
        v_hat[i] = v[i] / (1 - np.power(beta2, t))
        weights[i] -= alpha * m_hat[i] / (np.sqrt(v_hat[i]) + epsilon)'''

#threadsperblock=1024
#blockspergrid = 1
#_update_cupy[blockspergrid, threadsperblock](alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t)



class SGD():
    
    def __init__(self, alpha = 0.001):
        self.alpha = alpha


    def update(self, gradient, weights, v, m, v_hat, m_hat, _):
        if is_cupy_available:
            self._update = self._update_cupy
        else:
            self._update = self._update_numpy

        return self._update(self.alpha, gradient, weights, v, m, v_hat, m_hat)

    @staticmethod
    @njit
    def _update_numpy(alpha, gradient, weights, v, m, v_hat, m_hat):
        weights -= gradient * alpha

        return weights, v, m, v_hat, m_hat

    @staticmethod
    def _update_cupy(alpha, gradient, weights, v, m, v_hat, m_hat):
        weights -= gradient * alpha

        return weights, v, m, v_hat, m_hat


    


class Momentum():
    
    def __init__(self, alpha = 0.01, beta = 0.9):
        self.alpha = alpha
        self.beta = beta

    def update(self, gradient, weights, v, m, v_hat, m_hat, _):
        if is_cupy_available:
            self._update = self._update_cupy
        else:
            self._update = self._update_numpy

        return self._update(self.alpha, self.beta, gradient, weights, v, m, v_hat, m_hat)

    @staticmethod
    @njit
    def _update_numpy(alpha, beta, gradient, weights, v, m, v_hat, m_hat):
        v = beta * v + (1 - beta) * gradient
        weights -= v * alpha

        return weights, v, m, v_hat, m_hat

    @staticmethod
    def _update_cupy(alpha, beta, gradient, weights, v, m, v_hat, m_hat):
        v = beta * v + (1 - beta) * gradient
        weights -= v * alpha

        return weights, v, m, v_hat, m_hat
        

class RMSProp():
    
    def __init__(self, alpha = 0.01, beta = 0.9, epsilon = 0.000000001):

        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def update(self, gradient, weights, v, m, v_hat, m_hat, _):
        if is_cupy_available:
            self._update = self._update_cupy
        else:
            self._update = self._update_numpy

        return self._update(self.alpha, self.beta, self.epsilon, gradient, weights, v, m, v_hat, m_hat)

    @staticmethod
    @njit
    def _update_numpy(alpha, beta, epsilon, gradient, weights, v, m, v_hat, m_hat):
        v = beta * v + (1 - beta) * np.power(gradient, 2)
        weights -= alpha * gradient / (np.sqrt(v) + epsilon)

        return weights, v, m, v_hat, m_hat

    @staticmethod
    def _update_cupy(alpha, beta, epsilon, gradient, weights, v, m, v_hat, m_hat):
        v = beta * v + (1 - beta) * np.power(gradient, 2)
        weights -= alpha * gradient / (np.sqrt(v) + epsilon)

        return weights, v, m, v_hat, m_hat


class Adam():
    def __init__(self, alpha=0.001, beta=0.9, beta2=0.999, epsilon=1e-9):
        alpha = np.float32(alpha)
        beta = np.float32(beta)
        beta2 = np.float32(beta2)
        epsilon = np.float32(epsilon)
        #gradient = gradient.astype(np.float32)
        #weights = weights.astype(np.float32)
        #v = v.astype(np.float32)
        #m = m.astype(np.float32)
        #v_hat = v_hat.astype(np.float32)
        #m_hat = m_hat.astype(np.float32)

        self.alpha = alpha
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon


    def update(self, gradient, weights, v, m, v_hat, m_hat, t):
        if is_cupy_available:
            # Transfer data to GPU if not already done
            d_gradient = cuda.to_device(gradient)
            d_weights = cuda.to_device(weights)
            d_v = cuda.to_device(v)
            d_m = cuda.to_device(m)
            d_v_hat = cuda.to_device(v_hat)
            d_m_hat = cuda.to_device(m_hat)

            # Kernel launch configuration
            n = d_weights.size
            threadsperblock = 1024
            blockspergrid = (n + threadsperblock - 1) // threadsperblock

            # Call the CUDA kernel
            #_update_cupy[blockspergrid, threadsperblock](self.alpha, self.beta, self.beta2, self.epsilon, d_gradient, d_weights, d_v, d_m, d_v_hat, d_m_hat, t)

            # Optionally, copy updated arrays back to host
            #weights = d_weights.copy_to_host()
            # Is it required for v, m, v_hat, m_hat if needed, they are just constant
        else:
            # Fallback to CPU-based update (using _update_numpy)
            pass

        return weights, v, m, v_hat, m_hat
    @staticmethod
    @njit
    def _update_numpy(alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t):
        m = beta * m + (1 - beta) * gradient
        v = beta2 * v + (1 - beta2) * np.power(gradient, 2)

        m_hat = m / (1 - np.power(beta, t))
        v_hat = v / (1 - np.power(beta2, t))

        weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        return weights, v, m, v_hat, m_hat

    @cuda.jit
    def _update_cupy(alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t):
        i = cuda.grid(1)  # Calculate the unique thread index
        if i < weights.size:  # Ensure not go out of bounds
            m[i] = beta * m[i] + (1 - beta) * gradient[i]
            v[i] = beta2 * v[i] + (1 - beta2) * np.power(gradient[i], 2)

            m_hat[i] = m[i] / (1 - np.power(beta, t))
            v_hat[i] = v[i] / (1 - np.power(beta2, t))

            weights[i] -= alpha * m_hat[i] / (np.sqrt(v_hat[i]) + epsilon)

    '''@staticmethod
    def _update_cupy(alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t):
        m = beta * m + (1 - beta) * gradient
        v = beta2 * v + (1 - beta2) * np.power(gradient, 2)

        m_hat = m / (1 - np.power(beta, t))
        v_hat = v / (1 - np.power(beta2, t))

        weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        return weights, v, m, v_hat, m_hat'''

class Nadam():
    
    def __init__(self, alpha = 0.001, beta = 0.9, beta2 = 0.999, epsilon = 0.000000001):
        self.alpha = alpha
        self.beta = beta
        self.beta2 = beta2
        self.epsilon = epsilon
        

    def update(self, gradient, weights, v, m, v_hat, m_hat, t):
        if is_cupy_available:
            self._update = self._update_cupy
        else:
            self._update = self._update_numpy
     
        return self._update(self.alpha, self.beta, self.beta2, self.epsilon, gradient, weights, v, m, v_hat, m_hat, t)

    @staticmethod
    @njit
    def _update_numpy(alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t):
        m = beta * m + (1 - beta) * gradient
        v = beta2 * v + (1 - beta2) * np.power(gradient, 2)

        m_hat = m / (1 - np.power(beta, t)) + (1 - beta) * gradient / (
            1 - np.power(beta, t)
        )
        v_hat = v / (1 - np.power(beta2, t))

        weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        return weights, v, m, v_hat, m_hat


    @cuda.jit
    def update_cupy_cuda(alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t):
        i = cuda.grid(1)
        if i < weights.size:  # Assuming weights is a 1D array and all arrays are of the same size
            m[i] = beta * m[i] + (1 - beta) * gradient[i]
            v[i] = beta2 * v[i] + (1 - beta2) * (gradient[i] ** 2)

            m_hat[i] = m[i] / (1 - beta ** t) + (1 - beta) * gradient[i] / (1 - beta ** t)
            v_hat[i] = v[i] / (1 - beta2 ** t)

            weights[i] -= alpha * m_hat[i] / (math.sqrt(v_hat[i]) + epsilon)
        '''
    @staticmethod
    def _update_cupy(alpha, beta, beta2, epsilon, gradient, weights, v, m, v_hat, m_hat, t):
        m = beta * m + (1 - beta) * gradient
        v = beta2 * v + (1 - beta2) * np.power(gradient, 2)

        m_hat = m / (1 - np.power(beta, t)) + (1 - beta) * gradient / (
            1 - np.power(beta, t)
        )
        v_hat = v / (1 - np.power(beta2, t))

        weights -= alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        return weights, v, m, v_hat, m_hat'''

class Noam():
    """Learning rate scheduler for optimizers"""

    def __init__(self, optimizer, model_dim, scale_factor = 1, warmup_steps = 4000) -> None:
        self.optimizer = optimizer
        self.model_dim = model_dim
        self.scale_factor = scale_factor
        self.warmup_steps = warmup_steps
        self.steps_num = 0

    @staticmethod
    @njit
    def compute_learning_rate(scale_factor, model_dim, steps_num, warmup_steps):
        return scale_factor * (
            model_dim ** (-0.5)
            * min(steps_num ** (-0.5), steps_num * warmup_steps ** (-1.5))
        )

    def update(self, gradient, weights, v, m, v_hat, m_hat, t):
        self.steps_num += 1

        self.optimizer.alpha = self.compute_learning_rate(self.scale_factor, self.model_dim, self.steps_num, self.warmup_steps)
        return self.optimizer.update(gradient, weights, v, m, v_hat, m_hat, t)
    


optimizers = {
    
    "sgd": SGD(),
    "momentum": Momentum(),
    "rmsprop": RMSProp(),
    "adam": Adam(),
    "nadam": Nadam(),
    
}
