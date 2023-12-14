from LogisticRegression import gradTotalLoss, totalLoss,logisticLoss
from helpers import estimateGrad


from SparseVector import SparseVector as SV
import numpy as np

def create_data(d,N):
    data = []
    keys = [f"{int(i)}" for i in range(d)]

    for n in range(N):
        x = dict(zip(keys,np.random.randn(d,)))
        y = np.random.choice([-1,1])
        data.append((SV(x),y))

    return data

if __name__ == "__main__":
    d = 10
    N = 20
    lam = 0
    tests = 50
    delta = 1e-8

    keys = [f"{int(i)}" for i in range(d)]

    for i in range(tests):


        data = create_data(d,N)

        beta = SV(dict(zip(keys,np.random.rand(d,))))


        analytical_grad = gradTotalLoss(data,beta, lam)

        partial_fun = lambda beta: totalLoss(data,beta,lam)
        numerical_grad = estimateGrad(partial_fun, beta, delta)


        print(f"Test {i}")
        print("A grad: ",analytical_grad)
        print("N grad: ",numerical_grad)

        assert (numerical_grad-analytical_grad).norm(np.inf) <= 1e-5
