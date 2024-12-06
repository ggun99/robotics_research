import numpy as np

class Fastron:
    def __init__(self, data, alpha, gram_computed, G, F, y, N, d, g, max_updates, max_support_points, beta, allowance, kNS, sigma):
        self.data = data
        self.alpha = alpha
        self.gram_computed = gram_computed
        self.G = G
        self.F = F
        self.y = y
        self.N = N#data.shape[0]
        self.d = d#data.shape[1]
        self.g = g 
        self.numberSupportPoints = np.count_nonzero(alpha)
        self.max_updates = max_updates
        self.max_support_points = max_support_points
        self.beta = beta
        self.allowance = allowance
        self.kNS = kNS
        self.sigma = sigma
        self.eval = self.eval

    def compute_gram_matrix_col(self, idx, start_idx = 0):
        """
        Compute a specific column of the Gram matrix.
        """
        r2 = np.ones(self.N - start_idx)
        for j in range(self.d):
            r2 += self.g / 2 * (self.data[start_idx:self.N, j] - self.data[idx, j])**2
        self.G[start_idx:self.N, idx] = 1 / (r2 * r2)
        self.gram_computed[idx] = 1

    def calculate_margin_removed(self, idx):
        """
        Calculate the margin if a point is removed.
        
        Returns:
        (float, int): The maximum margin and the corresponding index.
        """
        max_margin = 0
        for i in range(self.N):
            if self.alpha[i] != 0:
                removed = self.y[i] * (self.F[i] - self.alpha[i])
                if removed > max_margin:
                    max_margin = removed
                    idx = i
        return max_margin, idx

    def keep_select_rows(self, mat, rows_to_retain):
        """
        Keeps only the specified rows in the matrix and removes the rest.

        Parameters:
        mat (numpy.ndarray): The original matrix.
        rows_to_retain (numpy.ndarray): Array of row indices to retain.

        Returns:
        numpy.ndarray: A new matrix with only the specified rows.
        """
        # Ensure rows_to_retain is a numpy array
        rows_to_retain = np.array(rows_to_retain)
        
        # Create a new matrix with only the retained rows
        new_mat = mat[rows_to_retain, :]
        
        return new_mat


    def keep_select_cols(self, mat, cols_to_retain):
        """
        Keeps only the specified columns in the matrix and removes the rest.

        Parameters:
        mat (numpy.ndarray): The original matrix.
        cols_to_retain (numpy.ndarray): Array of column indices to retain.

        Returns:
        numpy.ndarray: A new matrix with only the specified columns.
        """
        # Ensure cols_to_retain is a numpy array
        cols_to_retain = np.array(cols_to_retain)
        
        # Create a new matrix with only the retained columns
        new_mat = mat[:, cols_to_retain]
        
        return new_mat

    def keep_select_rows_cols(self, mat, rows_to_retain, cols_to_retain, shift_only):
        """
        Keeps only the specified rows and columns in the matrix and removes the rest.

        Parameters:
        mat (numpy.ndarray): The original matrix.
        rows_to_retain (numpy.ndarray): Array of row indices to retain.
        cols_to_retain (numpy.ndarray): Array of column indices to retain.
        shift_only (bool): If True, only shift elements without resizing.

        Returns:
        numpy.ndarray: A new matrix with only the specified rows and columns.
        """
        # Ensure rows_to_retain and cols_to_retain are numpy arrays
        rows_to_retain = np.array(rows_to_retain)
        cols_to_retain = np.array(cols_to_retain)
        
        # Create a new matrix with only the retained rows and columns
        new_mat = mat[np.ix_(rows_to_retain, cols_to_retain)]
        
        # If shiftOnly is True, just shift elements without resizing
        if shift_only:
            for j in range(len(cols_to_retain)):
                for i in range(len(rows_to_retain)):
                    mat[i, j] = mat[rows_to_retain[i], cols_to_retain[j]]
            return mat[:len(rows_to_retain), :len(cols_to_retain)]
        
        # If shiftOnly is False, return the resized matrix
        return new_mat

    def sparsify(self):
        retain_idx = np.nonzero(self.alpha)[0]

        self.N = retain_idx.size
        self.numberSupportPoints = self.N

        # Sparsify model
        self.data = self.keep_select_rows(self.data, retain_idx)
        self.alpha = self.keep_select_rows(self.alpha, retain_idx)
        self.gram_computed = self.keep_select_rows(self.gram_computed, retain_idx)
        self.G = self.keep_select_rows_cols(self.G, retain_idx, retain_idx, True)

        # Sparsify arrays needed for updating
        self.F = self.keep_select_rows(self.F, retain_idx)
        self.y = self.keep_select_rows(self.y, retain_idx)


    def update_model(self):
        margin = self.y * self.F

        for i in range(self.max_updates):
            margin = self.y * self.F

            idx = np.argmin(margin)
            if margin[idx] <= 0:
                if not self.gram_computed[idx]:
                    self.compute_gram_matrix_col(idx)
                delta = (-1.0 if self.y[idx] < 0 else self.beta) - self.F[idx]
                if self.alpha[idx] != 0 :  # already a support point, doesn't hurt to modify it
                    self.alpha[idx] += delta
                    
                    self.F += self.G[:, idx:idx+1] * delta
                    continue
                elif self.numberSupportPoints < self.max_support_points:  # adding new support point?
                    self.alpha[idx] = delta
                    self.F += self.G[:, idx:idx+1] * delta
                    self.numberSupportPoints += 1
                    continue
                # else: If you reach this point, there is a need to correct a point but you can't

            # Remove redundant points
            max_margin, idx = self.calculate_margin_removed(idx)
            if max_margin > 0:
                self.F -= self.G[:, idx:idx+1] * self.alpha[idx]
                self.alpha[idx] = 0
                self.numberSupportPoints -= 1
                continue

            if self.numberSupportPoints == self.max_support_points:
                print(f"Fail: Hit support point limit in {i:4} iterations!")
                self.sparsify()
                return self.data, self.y
            else:
                print(f"Success: Model update complete in {i:4} iterations!")
                self.sparsify()
                return self.data, self.y

        print(f"Failed to converge after {self.max_updates} iterations!")
        self.sparsify()
        return self.data, self.y
    
    def active_learning(self):
        N_prev = self.N
        self.N += self.allowance
        self.y = np.resize(self.y, (self.N,1))
        self.y[N_prev:] = 0

        # Make room for new data
        self.data = np.resize(self.data, (self.N, self.data.shape[1]))
        # START ACTIVE LEARNING
        k = 0
        if self.allowance // N_prev > 0:
            # Exploitation
            for k in range(min(self.kNS, self.allowance // N_prev)):
                self.data[(k + 1) * N_prev:(k + 2) * N_prev, :] = (self.data[:N_prev, :] + self.sigma * np.random.randn(N_prev, self.data.shape[1])).clip(-1.0, 1.0)
            # Exploration
            
            self.data[N_prev + k * N_prev:] = np.random.uniform(-1, 1, (self.allowance - k * N_prev, self.data.shape[1]))       
        else:
            idx = list(range(N_prev))
            np.random.shuffle(idx)

            for i in range(self.allowance):
                self.data[N_prev + i, :] = (self.data[idx[i], :] + self.sigma * np.random.randn(1, self.data.shape[1])).clip(-1.0, 1.0)
        # END ACTIVE LEARNING

        # Update Gram matrix
        if self.G.shape[0] < self.N:
            self.G = np.resize(self.G, (self.N, self.N))
        self.gram_computed = np.resize(self.gram_computed, (self.N,1))
        self.gram_computed[N_prev:] = 0

        # Update hypothesis vector and Gram matrix
        self.F = np.resize(self.F, (self.N,1))
        idx = np.nonzero(self.alpha)[0]
        for i in idx:
            self.compute_gram_matrix_col(i, N_prev)
        self.F[N_prev:] = np.dot(self.G[N_prev:, :N_prev], self.alpha[:N_prev])

        self.alpha = np.resize(self.alpha, (self.N,1))
        self.alpha[N_prev:] = 0

        return self.data, self.alpha, self.gram_computed, self.G, self.F
    
    def eval(self,query_points):
        
        """
        This function checks for collisions between query points and data points.

        Args:
            query_points: A numpy array of shape (n_query_points, d) representing query points.
            data: A numpy array of shape (n_data_points, d) representing data points.
            g: A scalar representing a constant value.
            alpha: A scalar representing another constant value.
            d: The dimensionality of the space (number of coordinates).

        Returns:
            A numpy array of shape (n_query_points,) where each element is -1.0 if there's a collision,
            otherwise 1.0.
        """

        n_query_points, _ = query_points.shape
        acc = np.zeros(n_query_points)

        for i in range(n_query_points):

            # Efficient case handling using numpy's where function
            temp = 2.0 / self.g + np.square(self.data[:,0] - query_points[i,0]) + np.square(self.data[:,1] - query_points[i,1]) #+ np.square(self.data[:,2] - query_points[i,2])

            # Collision check and sign assignment
            acc[i] = np.sign(np.sum(self.alpha / (temp**2)))

        return acc