import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np

class TripletDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df["user_id"].values, dtype=torch.long)
        self.pos = torch.tensor(df["pos_item"].values, dtype=torch.long)
        # explode negative lists to a 2D tensor [N, K]
        self.neg = torch.tensor(df["neg_items"].tolist(), dtype=torch.long)
    def __len__(self): return len(self.users)
    def __getitem__(self, idx):
        return self.users[idx], self.pos[idx], self.neg[idx]

class BPRMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, u, i):
        uvec = self.user_emb(u)
        ivec = self.item_emb(i)
        return (uvec * ivec).sum(dim=-1)

    def score_all(self, u):
        uvec = self.user_emb(u)              # [B, D]
        ivec = self.item_emb.weight          # [I, D]
        return torch.matmul(uvec, ivec.t())  # [B, I]

def bpr_loss(pos_scores, neg_scores):
    # pos: [B], neg: [B, K]
    diff = pos_scores.unsqueeze(1) - neg_scores
    return -F.logsigmoid(diff).mean()

def train_bpr(model, triples_df, epochs=5, batch_size=2048, lr=0.01, weight_decay=0.0, device="cpu"):
    ds = TripletDataset(triples_df)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)
    for e in range(epochs):
        model.train()
        total = 0.0
        for u, p, n in loader:
            u, p, n = u.to(device), p.to(device), n.to(device)
            pos = model(u, p)
            neg = model(u.unsqueeze(1).expand_as(n), n)  # broadcast user over negatives
            loss = bpr_loss(pos, neg)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item())
        print(f"Epoch {e+1}/{epochs} - loss={total/len(loader):.4f}")
    model.cpu()
    return model
