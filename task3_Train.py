#step1:(import libraries)
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

#load X and y from pickle files:
with open("X.pkl","rb") as f:
    X_numeric=pickle.load(f)    
with open("y.pkl","rb") as f:
    y_numeric=pickle.load(f)
print("Loaded X and y from pickle files.")    
X=torch.tensor(X_numeric,dtype=torch.long)
y=torch.tensor(y_numeric,dtype=torch.long)

#step2:(vocabulary size)
vocab_size=max(torch.max(X).item(), torch.max(y).item()) + 1
print("Vocabulary Size:", vocab_size)

#step3:(set hyperparameters)
window_size=X.shape[1]
embedding_dim=64

#step4:
#Token Embedding Layer:
token_embedding=nn.Embedding(vocab_size,embedding_dim)
#Positional Embedding Layer:
positional_embedding=nn.Embedding(window_size,embedding_dim)

#Create position Id's:
positions=torch.arange(0,window_size).unsqueeze(0)
#Get token and positional embeddings:
token_embed=token_embedding(X)
pos_embed=positional_embedding(positions)

#Combine token and positional embeddings:
combined_embed=token_embed+pos_embed

#step5:
print("X shape:", X.shape)
print("Token Embeddings shape:", token_embed.shape)
print("Positional Embeddings shape:", pos_embed.shape)
print("Combined Embeddings shape:", combined_embed.shape)
print("Combined_embed:", combined_embed)

    