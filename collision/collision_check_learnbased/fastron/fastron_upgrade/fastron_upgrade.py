import numpy as np


class Fastron:
    def __init__(self, data, y) -> None:
        self.data = data              # dataset of configuration
        self.y = y.reshape(y.shape[0],) # true label, let it be zeros first before we add it later
        self.N = self.data.shape[0]   # number of datapoint = number of row the dataset has
        self.d = self.data.shape[1]   # number of dimensionality = number of columns the dataset has (x1, x2, ..., xn)
        self.g = 10                   # kernel width
        self.beta = 100               # conditional bias
        self.maxUpdate = 10000        # max update iteration
        self.maxSupportPoints = 1500  # max support points
        self.G = np.zeros((self.N, self.N))  # gram matrix of dataset
        self.alpha = np.zeros(self.N) # weight , let it be zeros first before we add it later
        self.F = np.zeros(self.N)     # hypothesis , let it be zeros first before we add it later
        
        # active learning parameters
        self.allowance = 800          # number of new samples
        self.kNS = 4                  # number of points near supports
        self.sigma = 0.5              # Gaussian sampling std
        self.exploitP = 0.5           # proportion of exploitation samples

        self.numberSupportPoints = 0  # count of points with nonzero weights

        # misc
        self.gramComputed = np.zeros(self.N) # status of gram matrix compute, if 0 = not computed , 1 = computed
        
    # functions and variables for model update
    def update_model(self):
        for iter in range(self.maxUpdate):
            margin = self.y * self.F
            indx = np.argmin(margin)
            if margin[indx] <= 0:
                indx = np.argmin(self.y * self.F)

                # if self.gramComputed(indx) != 1.0 : self.compute_gram_matrix_col(indx) # optimization by calculate indivisual G column when call. but I dont understand how to implement in code
                self.G = np.dot(self.data.T, self.data) # calculate all instead
                print(f"==>> self.G: \n{self.G}")

                delta = (-1.0 if self.y[indx] < 0 else self.beta) - self.F[indx]

                if self.alpha[indx] != 0.0: # already a support point, doesn't hurt to modify it
                    self.alpha[indx] += delta
                    self.F += self.G[:, indx] * delta
                    continue

                elif self.numberSupportPoints < self.maxSupportPoints: # adding new support point?
                    self.alpha[indx] = delta
                    self.F += self.G[:, indx] * delta
                    self.numberSupportPoints += 1
                    continue

            # else If you reach this point, there is a need to correct a point but you can't
            # remove redundant points
            if self.calculate_margin_removed(index=indx) > 0:
                self.F -= self.G[:, indx] * self.alpha[indx]
                self.alpha[indx] = 0
                margin = self.y * self.F
                self.numberSupportPoints -= 1
                continue

            if self.numberSupportPoints == self.numberSupportPoints:
                self.sparsify()
            else:
                self.sparsify()

        self.sparsify()

    def calculate_margin_removed(self, index):
        max_margin_removed = 0
        max_margin_removed_idx = None
        
        for i in range(len(self.alpha)):
            if self.alpha[i] != 0:
                margin_removed = self.y[i] * (self.F[i] - self.alpha[i])
                if margin_removed > max_margin_removed:
                    max_margin_removed = margin_removed
                    max_margin_removed_idx = i
        
        return max_margin_removed, max_margin_removed_idx

    def gram_computed(self):
        return None # return array
    
    def compute_gram_matrix_col(self, index, startIndex):
        pass # no return

    def sparsify(self):
        retainIndx = self.find(self.alpha)

        N = len(retainIndx)
        self.numberSupportPoints = N

        # sparsify model
        self.data = self.keepSelectRows(self.data, retainIndx)
        self.alpha = self.keepSelectRows(self.alpha, retainIndx)
        self.gramComputed = self.keepSelectRows(self.gramComputed, retainIndx)

        self.G = self.keepSelectRowsCols(self.G, retainIndx, retainIndx, shiftOnly=True)

        # sparsify arrays needed for updating
        self.F = self.keepSelectRows(self.F, retainIndx)
        self.y = self.keepSelectRows(self.y, retainIndx)

    # perform proxy check
    def eval(self, queryPoints):
        return None # return collision status -1 or 1
    
    # active learning function
    def active_learning(self):
        pass # no return

    # kinematic collision detector, use to query collision from real oracle
    def kcd(self, queryPoint, colDetector):
        return None # return collision status -1 or 1
    
    # update all label
    def update_labels(self, colDetector):
        pass # no return




    # support function 
    def find(self, vec): 
        """return the indices of nonzero elements""" 
        idx = [] # list for storing index
        for indx, elem in enumerate(vec):
            if elem != 0.0:
                idx.append(indx) 
        return idx
    
    def keepSelectRows(self, matPtr, rowsToRetain):
        """remove specific rows"""
        matPtr = matPtr[rowsToRetain, :]
        return matPtr

    def keepSelectCols(self, matPtr, colsToRetain):
        """remove specific columns"""
        matPtr = matPtr[:, colsToRetain]
        return matPtr
    
    def keepSelectRowsCols(self, matPtr, rowsToRetain, colsToRetain, shiftOnly = False):
        """remove specific rows and columns; if shiftOnly, do not resize matrix"""
        if shiftOnly:
            matPtr = matPtr[rowsToRetain, :][:, colsToRetain] # not correct yet
        else:
            matPtr = matPtr[rowsToRetain, colsToRetain]
        return matPtr

    def randn(self, m, n):
        """create m x n matrix of 0-mean normal distribution samples"""
        mat = np.random.randn(m, n)
        return mat






if __name__ == "__main__":
    # init model 
    # ft = Fastron(data, y) # shape (N,d), shape (N,)

    # model trainning
    # ft.active_learning() # active learning
    # ft.update_labels() # update label
    # ft.update_model() # train model

    # query proxy collision
    # config = np.array([0,0]).reshape(2,1)
    # pred = ft.eval(config)
    # print(f"==>> pred: \n{pred}")