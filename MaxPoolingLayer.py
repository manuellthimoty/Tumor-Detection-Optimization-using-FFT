import numpy as np
class MaxPoolingLayer:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.max_indices = None  # Simpan posisi max value
    
    def forward(self, input):
        """
        Input shape: (batch, channels, height, width)
        """
        self.input = input
        batch_size, channels, h, w = input.shape
        out_h = h // self.stride
        out_w = w // self.stride
        
        output = np.zeros((batch_size, channels, out_h, out_w))
        self.max_indices = np.zeros((batch_size, channels, out_h, out_w, 2), dtype=np.int32)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        
                        patch = input[b, c, h_start:h_end, w_start:w_end]
                        max_val = np.max(patch)
                        output[b, c, i, j] = max_val
                        
                        # Simpan indeks dari max value
                        max_idx = np.unravel_index(np.argmax(patch), patch.shape)
                        self.max_indices[b, c, i, j, 0] = h_start + max_idx[0]
                        self.max_indices[b, c, i, j, 1] = w_start + max_idx[1]
        
        return output
    
    def backward(self, output_gradient):
        """Backwardprop untuk pooling """
        batch_size, channels, out_h, out_w = output_gradient.shape
        input_gradient = np.zeros(self.input.shape)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        # Ambil koordinat dari max value yang disimpan dari forwrd
                        h_idx = self.max_indices[b, c, i, j, 0]
                        w_idx = self.max_indices[b, c, i, j, 1]
                        
                        # Gradient hanya diberikan ke posisi indeksj max value
                        input_gradient[b, c, h_idx, w_idx] = output_gradient[b, c, i, j]
        
        return input_gradient
