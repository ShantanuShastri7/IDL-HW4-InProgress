import numpy as np
from .activation import Softmax

class ScaledDotProductAttention:
    """
    Scaled Dot Product Attention
    """ 
    def __init__(self):
        '''
        Initialize the ScaledDotProductAttention class.
        '''
        # Initialize your softmax layer
        # What dimension should you pass to the softmax constructor?
        self.eps = 1e10 # DO NOT MODIFY
        self.softmax = Softmax(dim=-1) # TODO: Set correct dimension
        
    
    def forward(self, Q, K, V, mask=None):
        """
        :param Q: Query matrix of shape (N, ..., H, L, E)
        :param K: Key matrix of shape (N, ..., H, S, E)
        :param V: Value matrix of shape (N, ..., H, S, Ev)
        :param mask: Boolean mask matrix of shape (N, ..., H, L, S)
        :return: Output matrix of shape (N, ..., H, L, Ev)
        """
        
        # 1. Get embedding dimension for scaling
        self.Q = Q
        self.K = K
        self.V = V
        d_k = Q.shape[-1]
        
        # 2. Transpose K for dot product
        K_T = np.swapaxes(K, -1, -2)
        
        # 3. Calculate scaled dot-product scores
        #    (N, ..., H, L, E) @ (N, ..., H, E, S) -> (N, ..., H, L, S)
        scores = (Q @ K_T) / np.sqrt(d_k)
        
        # 4. Apply mask (if provided) BEFORE softmax
        if mask is not None:
            # Add a large negative number where mask is True (or 1)
            scores = scores + (mask * -self.eps)

        # 5. Compute attention scores (apply softmax)
        self.attention_scores = self.softmax.forward(scores)

        # 6. Calculate output
        #    (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        output = self.attention_scores @ V

        return output
    
    def backward(self, d_output):
        """
        :param d_output: Gradient of loss wrt output of shape (N, ..., H, L, Ev)
        :return: Gradient of loss wrt input Q, K, V
        """
        # TODO: Implement backward pass

        # Calculate gradients for V: (N, ..., H, S, Ev)
        # (N, ..., H, L, S) @ (N, ..., H, S, Ev) -> (N, ..., H, L, Ev) 
        # Use the transpose of stored softmax output to swap last two dimensions   
        d_V = np.swapaxes(self.attention_scores, -1, -2) @ d_output
        
        # Calculate gradients for attention scores
        # (N, ..., H, L, Ev) @ (N, ..., H, Ev, S) -> (N, ..., H, L, S)
        d_attention_scores = d_output @ np.swapaxes(self.V, -1, -2)
        d_scaled_dot_product =  self.softmax.backward(d_attention_scores)
        
        # Scale gradients by sqrt(d_k)
        d_scaled_dot_product = d_scaled_dot_product / np.sqrt(self.Q.shape[-1])
        
        # Calculate gradients for Q and K
        # (N, ..., H, L, S) @ (N, ..., H, S, E) -> (N, ..., H, L, E)   
        d_Q = d_scaled_dot_product @ self.K
        # (N, ..., H, L, S) @ (N, ..., H, L, E) -> (N, ..., H, S, E)
        d_K = np.swapaxes(d_scaled_dot_product, -1, -2) @ self.Q
        
        return d_Q, d_K, d_V
