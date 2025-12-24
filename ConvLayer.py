import numpy as np
from FastFourierTransform import multiplypolinoms
class ConvLayer():
    def __init__(self, num_filters, input_channel, kernel_size, padding=0, stride=1):
        self.num_filters = num_filters
        self.input_channel = input_channel
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.weights = np.random.randn(num_filters, input_channel, kernel_size, kernel_size) #  random init for weights
        self.bias = np.zeros(num_filters)
        self.input_padded = None
    
    def forward(self, input):
        """Spatial convolution using im2col trick"""
        self.input = input
        batch_size, input_c, input_h, input_w = input.shape
        
        if self.padding > 0:
            self.input_padded = np.pad(input,
                                      ((0, 0), (0, 0),
                                       (self.padding, self.padding),
                                       (self.padding, self.padding)),
                                      mode='constant', constant_values=0)
        else:
            self.input_padded = input
        
        pad_h, pad_w = self.input_padded.shape[2], self.input_padded.shape[3]
        out_h = (pad_h - self.kernel_size) // self.stride + 1
        out_w = (pad_w - self.kernel_size) // self.stride + 1
        K = self.kernel_size
        
        output = np.zeros((batch_size, self.num_filters, out_h, out_w))
        
        for b in range(batch_size):
            # Convert to kolom: (input_c * K * K, out_h * out_w)
            im2col_matrix = np.zeros((input_c * K * K, out_h * out_w))
            
            idx = 0
            for i in range(out_h):
                for j in range(out_w):
                    h_start = i * self.stride
                    h_end = h_start + K
                    w_start = j * self.stride
                    w_end = w_start + K
                    
                    patch = self.input_padded[b, :, h_start:h_end, w_start:w_end]
                    im2col_matrix[:, idx] = patch.reshape(-1)
                    idx += 1
            
            weights_reshaped = self.weights.reshape(self.num_filters, -1)
            
            # Single matrix multiplication!
            result = np.dot(weights_reshaped, im2col_matrix) 
            result += self.bias.reshape(-1, 1)
            
            output[b] = result.reshape(self.num_filters, out_h, out_w)
        
        return output
    def fft_forward(self, input):
        """
        FFT-based Forward Propagation
        Complexity reduced from O(N^2 * K^2) to O(N^2 * log N).
        """
        self.input = input
        batch_size, input_c, input_h, input_w = input.shape
        
        # PADDING 
        if self.padding > 0:
            self.input_padded = np.pad(input,
                                    ((0, 0), (0, 0),
                                    (self.padding, self.padding),
                                    (self.padding, self.padding)),
                                    mode='constant', constant_values=0)
        else:
            self.input_padded = input
        
        # Hitung dimensi output
        pad_h, pad_w = self.input_padded.shape[2], self.input_padded.shape[3]
        out_h = (pad_h - self.kernel_size) // self.stride + 1
        out_w = (pad_w - self.kernel_size) // self.stride + 1
        
        # PERSIAPAN FFT 
        fft_h = pad_h + self.kernel_size - 1
        fft_w = pad_w + self.kernel_size - 1
        
        # FLIP KERNEL (KRUSIAL!) 
        weights_flipped = np.flip(np.flip(self.weights, axis=2), axis=3)
        
        # PRE-COMPUTE KERNEL FFT 
        kernel_ffts = np.fft.fft2(weights_flipped, s=(fft_h, fft_w))
        
        # Simpan smmpah output
        output = np.zeros((batch_size, self.num_filters, out_h, out_w))

        for b in range(batch_size):
            # prekomputrasi input fft
            input_ffts = np.fft.fft2(self.input_padded[b], s=(fft_h, fft_w))
            
            for k in range(self.num_filters):
                
                # operasi perkalian fft
                spectral_product = input_ffts * kernel_ffts[k] 
                spectral_sum = np.sum(spectral_product, axis=0)
                
                # INVERSE FFT 
                spatial_result = np.real(np.fft.ifft2(spectral_sum))
                
                # Tambahkan Bias
                spatial_result += self.bias[k]
                
                offset_h = self.kernel_size - 1
                offset_w = self.kernel_size - 1
                
                for i in range(out_h):
                    for j in range(out_w):
                        h_idx = offset_h + i * self.stride
                        w_idx = offset_w + j * self.stride
                        output[b, k, i, j] = spatial_result[h_idx, w_idx]
                    
        return output
    
    def backward(self, output_gradient, learning_rate):
        batch_size, num_filters, grad_h, grad_w = output_gradient.shape
        
        # Ambil ukuran Input Padded 
        in_h, in_w = self.input_padded.shape[2], self.input_padded.shape[3]
        
        # Rumus: floor((Input - Kernel) / Stride) + 1
        valid_h = (in_h - self.kernel_size) // self.stride + 1
        valid_w = (in_w - self.kernel_size) // self.stride + 1
        
        # Tentukan batas loop yang aman
        run_h = min(grad_h, valid_h)
        run_w = min(grad_w, valid_w)
        
        kernels_gradient = np.zeros_like(self.weights)
        bias_gradient = np.zeros_like(self.bias)
        input_padded_gradient = np.zeros_like(self.input_padded)

        for b in range(batch_size):
            for k in range(num_filters):
                
                bias_gradient[k] += np.sum(output_gradient[b, k, :run_h, :run_w])
                                
                for i in range(run_h):
                    for j in range(run_w):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        
                        patch = self.input_padded[b, :, h_start:h_end, w_start:w_end]
                        grad = output_gradient[b, k, i, j]
                        
                        kernels_gradient[k] += patch * grad
                        input_padded_gradient[b, :, h_start:h_end, w_start:w_end] += self.weights[k] * grad

        kernels_gradient /= batch_size
        bias_gradient /= batch_size
        
        self.weights -= learning_rate * kernels_gradient
        self.bias -= learning_rate * bias_gradient
        
        # Unpad
        if self.padding > 0:
            input_gradient = input_padded_gradient[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            input_gradient = input_padded_gradient
            
        return input_gradient

    def fft_backward(self, output_gradient, learning_rate):
        """
        FFT-based backward propagation.
        """
        batch_size, num_filters, grad_h, grad_w = output_gradient.shape
        
        # Input padded shape
        in_h, in_w = self.input_padded.shape[2], self.input_padded.shape[3]
        
        valid_h = (in_h - self.kernel_size) // self.stride + 1
        valid_w = (in_w - self.kernel_size) // self.stride + 1
        
        run_h = min(grad_h, valid_h)
        run_w = min(grad_w, valid_w)
        
        kernels_gradient = np.zeros_like(self.weights)
        bias_gradient = np.zeros_like(self.bias)
        input_padded_gradient = np.zeros_like(self.input_padded)
        
        for b in range(batch_size):
            for k in range(num_filters):
                # BIAS GRADIENT 
                bias_gradient[k] += np.sum(output_gradient[b, k, :run_h, :run_w])
                
                # WEIGHT GRADIENT
                for i in range(run_h):
                    for j in range(run_w):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        
                        patch = self.input_padded[b, :, h_start:h_end, w_start:w_end]
                        grad = output_gradient[b, k, i, j]
                        
                        kernels_gradient[k] += patch * grad
                        input_padded_gradient[b, :, h_start:h_end, w_start:w_end] += self.weights[k] * grad
        
        # Normalize
        kernels_gradient /= batch_size
        bias_gradient /= batch_size
        
        # Update weights and bias
        self.weights -= learning_rate * kernels_gradient
        self.bias -= learning_rate * bias_gradient
        
        # Unpad input gradient
        if self.padding > 0:
            input_gradient = input_padded_gradient[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            input_gradient = input_padded_gradient
        
        return input_gradient