import numpy as np

class Linear:
    def __init__(self, in_features, out_features, debug=False):
        self.debug = debug
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))
        
        # Cache for backward pass
        self.A_flat = None
        self.ones = None
        self.original_input_shape = None
        self.in_features = None

    def init_weights(self, W, b):
        self.W = W
        self.b = b

    def forward(self, A):
        # --- 1. Store shape and flatten input ---
        self.original_input_shape = A.shape
        self.in_features = A.shape[-1]
        
        # Calculate batch size from all preceding dimensions
        batch_size = np.prod(A.shape[:-1])

        # Flatten A to (batch_size, in_features)
        self.A_flat = A.reshape(batch_size, self.in_features)

        # --- 2. Cache 'ones' for bias gradient ---
        self.ones = np.ones((batch_size, 1))

        # --- 3. Perform linear transformation ---
        # Z_flat = A_flat @ W.T + b
        Z_flat = self.A_flat @ self.W.T + self.b.T 
        
        # --- 4. Un-flatten output ---
        # Get out_features from weights
        out_features = self.W.shape[0]
        # Reshape to (*original_batch_dims, out_features)
        Z = Z_flat.reshape(*self.original_input_shape[:-1], out_features)
        
        return Z

    def backward(self, dLdZ):
        # --- 1. Get dimensions from cache ---
        out_features = self.W.shape[0]
        batch_size = self.A_flat.shape[0] # Use cached batch size

        # --- 2. Flatten dLdZ ---
        # dLdZ has shape (*original_batch_dims, out_features)
        # Flatten it to (batch_size, out_features)
        dLdZ_flat = dLdZ.reshape(batch_size, out_features)

        # --- 3. Calculate Gradients ---
        
        # Gradient w.r.t. input A
        # (batch_size, out_features) @ (out_features, in_features) -> (batch_size, in_features)
        dLdA_flat = dLdZ_flat @ self.W

        # Gradient w.r.t. weights W
        # (out_features, batch_size) @ (batch_size, in_features) -> (out_features, in_features)
        self.dLdW = dLdZ_flat.T @ self.A_flat

        # Gradient w.r.t. bias b
        # (out_features, batch_size) @ (batch_size, 1) -> (out_features, 1)
        self.dLdb = dLdZ_flat.T @ self.ones

        # --- 4. Un-flatten dLdA ---
        # Reshape dLdA_flat back to the original shape of A
        # (batch_size, in_features) -> (*original_batch_dims, in_features)
        dLdA = dLdA_flat.reshape(*self.original_input_shape)

        if self.debug:
            self.dLdA = dLdA

        return dLdA