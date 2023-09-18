import numpy 
import torch
from tqdm import tqdm


class GLM(torch.nn.Module):    
    def __init__(self, n_c, n_t, n_r):
        super(GLM, self).__init__()
        self.U = torch.nn.Parameter(torch.randn(n_c, n_r))
        self.V = torch.nn.Parameter(torch.randn(n_r, n_t))
        self.b = torch.nn.Parameter(torch.randn(1, n_c, 1))
        
    def forward(self, y):
        self.beta = torch.einsum("cr,rt->ct", self.U, self.V)
        x_pred = self.beta[None,:,:] * y + self.b
        return x_pred
    
def train_glm(
    X, 
    Y, 
    train,
    test,
    n_r = 2,
    learning_rate=1e-2,
    n_epochs=1000,
    ):
    
    _, n_c, n_t = X.shape
    glm = GLM(n_c, n_t, n_r)
    optimizer = torch.optim.Adam(glm.parameters(), lr=learning_rate)
    criterion = torch.nn.PoissonNLLLoss()
    
    X = torch.tensor(X)
    Y = torch.tensor(Y)[:,None,:]
    train_x, test_x = X[train], X[test]
    train_y, test_y = Y[train], Y[test]
    
    losses = []
    for epoch in tqdm(range(n_epochs), desc="Train GLM:"):
        optimizer.zero_grad()
        x_pred = glm(train_y)
        loss = criterion(x_pred, train_x)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return glm, losses

