import torch

class CustomModel(torch.nn.Module):
    
    def __init__(self, aemodel, n_clusters, hidden_features, alpha):
        super().__init__()
        self.aemodel = aemodel
        self.alpha = alpha        
        self.custom_layer = torch.nn.Parameter(torch.Tensor(n_clusters, hidden_features))
        torch.nn.init.xavier_normal_(self.custom_layer.data)
        
    def forward(self, X):

        X_out, hidden_out = self.aemodel(X)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(hidden_out.unsqueeze(1) - self.custom_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return X_out, q
