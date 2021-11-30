import math
import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self,yhat,y):
        return torch.sqrt(self.mse(yhat,y))


class MatrixFactorization(nn.Module):
    def __init__(self, field_dims, embed_dim=4):
        super(MatrixFactorization, self).__init__()
        n_users, n_movies = field_dims[0], field_dims[1]
        self.u = nn.Embedding(n_users, embed_dim)
        self.m = nn.Embedding(n_movies, embed_dim)
        self.u.weight.data.uniform_(0, 0.05)
        self.m.weight.data.uniform_(0, 0.05)

        
    def forward(self, x):
        users, movies = x[:,0], x[:,1]
        u, m = self.u(users), self.m(movies)
        return (u*m).sum(1).view(-1, 1)


class EE_Loss:
    def __init__(self, model, reg_biase=0.005, reg_lambda=0.005):
        self.reg_biase = reg_biase
        self.reg_lambda = reg_lambda
        self.model = model

    def __call__(self, pred, target):
        loss = (1-self.reg_biase - self.reg_lambda) * nn.MSELoss()(pred, target)
        loss += self.reg_biase * self.model.loss[0]
        loss += self.reg_lambda * self.model.loss[1]
        return torch.sqrt(loss)


class EuclideanEmbedding(nn.Module):

    def __init__(self, field_dims, embed_dim=4):
        super(EuclideanEmbedding, self).__init__()
        n_users, n_movies = field_dims[0], field_dims[1]
        self.Bu = nn.Parameter(torch.randn(n_users), requires_grad=True)
        self.Bm = nn.Parameter(torch.randn(n_movies), requires_grad=True)
        self.u = nn.Embedding(n_users, embed_dim)
        self.m = nn.Embedding(n_movies, embed_dim)
        
        self.u.weight.data.uniform_(0, 0.05)
        self.m.weight.data.uniform_(0, 0.05)
        self.loss = [0,0]

        
    def forward(self, x, global_mean=0):
        users, movies = x[:,0], x[:,1]
        u, m = self.u(users), self.m(movies)
        Bu, Bm = self.Bu[users], self.Bm[movies]
        difference = u-m
        output = global_mean + Bu + Bm - torch.linalg.norm(torch.mul(difference, difference))
        
        self.loss[0] = torch.norm(Bu) + torch.norm(Bm)
        self.loss[1] = torch.norm(u) + torch.norm(m)
        return output


class Beta_Loss:
    def __init__(self, model, reg_biase=0.005, reg_lambda=0.005):
        self.reg_biase = reg_biase
        self.reg_lambda = reg_lambda
        self.model = model

    def __call__(self, pred, target):
        loss = (1-self.reg_biase) * torch.sqrt(nn.MSELoss()(pred.view(-1,1), target))
        loss += self.reg_biase * self.model.loss[0]
        # loss += self.reg_lambda * self.model.loss[1]
        # loss = nn.MSELoss(reduction='sum')(pred.view(-1,1), target)
        loss = torch.nan_to_num(loss)
        return torch.sqrt(loss)

class Regularizer():
    def __init__(self, base_add, min_val, max_val):
        self.base_add = base_add
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, entity_embedding):
        return torch.clamp(entity_embedding + self.base_add, self.min_val, self.max_val)


class BetaRecommendation(nn.Module):

    SUPPORTED_DISTANCE = ['JS', 'KL']

    def __init__(self, field_dims, embed_dim=4, dist_metric='JS', **kwargs):
        super(BetaRecommendation, self).__init__()
        n_users, n_movies = field_dims[0], field_dims[1]

        if dist_metric not in BetaRecommendation.SUPPORTED_DISTANCE:
          raise ValueError("Only {} distance metrics are supported now.".format(BetaRecommendation.SUPPORTED_DISTANCE))
        self.dist_metric = dist_metric

        self.dist_func = {
            'JS': self.JS_divergence,
            'KL': self.KL_divergence
        }[self.dist_metric]        

        self.gamma = nn.Parameter(
            torch.Tensor([kwargs.get('gamma', 12)]), 
            requires_grad=False)
        self.lb = kwargs.get('lb', 1)
        self.ub = kwargs.get('ub', 100)

        self.Bu = nn.Parameter(torch.randn(n_users), requires_grad=True)
        self.Bm = nn.Parameter(torch.randn(n_movies), requires_grad=True)

        self.u = nn.Embedding(n_users, embed_dim * 2)
        # self.u = nn.Parameter(torch.zeros(n_users, embed_dim * 2))
        self.m = nn.Embedding(n_movies, embed_dim * 2)
        # self.m = nn.Parameter(torch.zeros(n_movies, embed_dim * 2))
        
        self.u.weight.data.uniform_(self.lb, self.ub)
        self.m.weight.data.uniform_(self.lb, self.ub)
        
        self.regularizer = Regularizer(1, self.lb, self.ub)
        self.loss = [0,0]

        self.sample_loss = SamplesLoss(loss="sinkhorn", scaling=1e-1000)

        
    def forward(self, x, global_mean=0):
      # Predict rating
        users, movies = x[:,0], x[:,1]
        u, m = self.u(users), self.m(movies)
        Bu, Bm = self.Bu[users], self.Bm[movies]
        u[torch.isnan(u)] = 0.05
        m[torch.isnan(m)] = 0.05

        alpha_u, beta_u = torch.chunk(self.regularizer(u).unsqueeze(-1), 2, dim=1)
        alpha_m, beta_m = torch.chunk(self.regularizer(m).unsqueeze(-1), 2, dim=1)

        u_dist = torch.distributions.beta.Beta(alpha_u, beta_u)
        m_dist = torch.distributions.beta.Beta(alpha_m, beta_m)

        distance = self.dist_func(u_dist, m_dist)

        # distance = self.KL_divergence(u_dist, m_dist)
        # distance = self.JS_divergence(u_dist, m_dist)
        # distance = self.Wasserstein_distance(u_dist, m_dist)
    
        output = Bu + Bm - distance
        
        self.loss[0] = torch.norm(Bu) + torch.norm(Bm)
        return output
      
    def KL_divergence(self, u_dist, m_dist):
      # return torch.norm(torch.distributions.kl.kl_divergence(u_dist, m_dist), p=1, dim=-1)
      return torch.norm(2.0/torch.pi * torch.atan(torch.distributions.kl.kl_divergence(u_dist, m_dist).squeeze()), p=1, dim=-1)
    
    def JS_divergence(self, u_dist, m_dist):
      mean_dist = torch.distributions.beta.Beta(0.5 * (u_dist.concentration0 + m_dist.concentration0), 0.5 * (u_dist.concentration1 + m_dist.concentration1))  
      # 0.5 * (u_dist + m_dist)
      kl_1 = torch.distributions.kl.kl_divergence(u_dist, mean_dist).squeeze()
      kl_2 = torch.distributions.kl.kl_divergence(m_dist, mean_dist).squeeze()
      return torch.norm(2.0/torch.pi * torch.atan((kl_1 + kl_2) * 0.5), p=1, dim=-1)

    def Wasserstein_distance(self, u_dist, m_dist):
      # Generate reference points
      x = torch.linspace(0.001, 0.999, 16).view(1, -1)
      u_ref = torch.exp(u_dist.log_prob(x))
      m_ref = torch.exp(m_dist.log_prob(x))
      return self.sample_loss.forward(u_ref, m_ref)