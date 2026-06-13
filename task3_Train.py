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
with open("vocab.pkl","rb") as f:
    vocab=pickle.load(f)    
print("Loaded X and y from pickle files.")    
X=torch.tensor(X_numeric,dtype=torch.long)
y=torch.tensor(y_numeric,dtype=torch.long)

#step2:(vocabulary size)
vocab_size=max(torch.max(X).item(), torch.max(y).item()) + 1
print("Vocabulary Size:", vocab_size)

#step3:(set hyperparameters)
window_size=X.shape[1]
embedding_dim=64
hidden_dim=128

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

#step6:Layers:
flattened=combined_embed.reshape(combined_embed.size(0), -1)

#Hidden Layer:
hidden_layer=nn.Linear(window_size*embedding_dim,hidden_dim)
hidden_output=hidden_layer(flattened)

#Relu Activation:
relu=nn.ReLU()
hidden_output=relu(hidden_output)

#Output Layer:
output_layer=nn.Linear(hidden_dim,vocab_size)

#Get logits and probabilities:
logits=output_layer(hidden_output)
softmax=nn.Softmax(dim=1)
probabilities=softmax(logits)

#Loss and Optimizer:
criterion=nn.CrossEntropyLoss()
optimizer = optim.Adam(
    list(token_embedding.parameters()) +list(positional_embedding.parameters()) +list(hidden_layer.parameters()) +list(output_layer.parameters()),lr=0.001)

#step7:loss and training loop:
epochs=150
for epoch in range(epochs):
    optimizer.zero_grad()
    
    #Forward pass:
    token_embed=token_embedding(X)
    pos_embed=positional_embedding(positions)
    combined_embed=token_embed+pos_embed
    flattened=combined_embed.reshape(combined_embed.size(0), -1)
    hidden_output=hidden_layer(flattened)
    hidden_output=relu(hidden_output)
    logits=output_layer(hidden_output)
    
    #Calculate loss and backpropagate:
    loss=criterion(logits,y)
    loss.backward()
    optimizer.step()
    if(epoch+1)%5==0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

print("flattened shape:", flattened.shape)
print("hidden_output shape:", hidden_output.shape)
print("logits shape:", logits.shape)
print("probabilities shape:", probabilities.shape)  

#step8:Save the model:
torch.save({
    'token_embedding': token_embedding.state_dict(),
    'positional_embedding': positional_embedding.state_dict(),
    'hidden_layer': hidden_layer.state_dict(),
    'output_layer': output_layer.state_dict()
}, "model.pth")
print("Model saved to model.pth")

#step9:Save the vocabulary:
with open("vocab.pkl","wb") as f:
    pickle.dump(vocab,f)
print("Vocabulary saved to vocab.pkl")
