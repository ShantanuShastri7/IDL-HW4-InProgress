import numpy as np

class Softmax:
    """
    A generic Softmax activation function that can be used for any dimension.
    """
    def __init__(self, dim=-1):
        """
        :param dim: Dimension along which to compute softmax (default: -1, last dimension)
        DO NOT MODIFY
        """
        self.dim = dim
        self.A = None # Cache for backward pass

    def forward(self, Z):
        """
        :param Z: Data Z (*) to apply activation function to input Z.
        :return: Output returns the computed output A (*).
        """
        if self.dim >= len(Z.shape) or self.dim < -len(Z.shape):
            raise ValueError("Dimension to apply softmax to is out of bounds for Z's shape")
        
        # We don't save Z, only the output A
        # Compute the softmax in a numerically stable way
        # Apply it to the dimension specified by the `dim` parameter
        exp_Z = np.exp(Z - np.max(Z, axis=self.dim, keepdims=True))
        self.A = exp_Z / np.sum(exp_Z, axis=self.dim, keepdims=True)

        return self.A

    def backward(self, dLdA):
        """
        :param dLdA: Gradient of loss wrt output
        :return: Gradient of loss with respect to activation input
        """
        # Get the shape of the output (which is the same as the input Z)
        original_shape = self.A.shape
        # Find the dimension (number of classes) along which softmax was applied
        C = original_shape[self.dim]
           
        # Move the softmax axis to the end
        A_moved = np.moveaxis(self.A, self.dim, -1)
        dLdA_moved = np.moveaxis(dLdA, self.dim, -1)
        
        # Save the "moved" shape for un-flattening later
        moved_shape = A_moved.shape 
        
        # Flatten to 2D (batch_size, C) where batch_size is product of all other dims
        A_flat = A_moved.reshape(-1, C)
        dLdA_flat = dLdA_moved.reshape(-1, C)

        # Get the calculated batch_size
        batch_size = A_flat.shape[0]

        # Initialize the 2D gradient for Z
        dLdZ_flat = np.zeros_like(A_flat)

        for i in range(batch_size):
            J = np.zeros((C, C))
            for m in range(C):
                for n in range(C):
                    if m == n:
                      
                        J[m, n] = A_flat[i, m] * (1 - A_flat[i, m])
                    else:
                       
                        J[m, n] = -A_flat[i, m] * A_flat[i, n]
            
            dLdZ_flat[i, :] = J.dot(dLdA_flat[i, :])

        # Reshape back to the "moved" dimensions
        dLdZ_moved = dLdZ_flat.reshape(moved_shape)
        # Move the axis from the end back to its original position
        dLdZ = np.moveaxis(dLdZ_moved, -1, self.dim)

        return dLdZ