#step1:(import libraries)
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

with open("vocab.pkl","rb") as f:
    vocab=pickle.load(f)
vocab_size=len(vocab)        
print("Loaded X and y from pickle files.")

#Hyperparameters:
embedding_dim=64
hidden_dim1=256
hidden_dim2=128
hidden_dim3=64
epochs=100    

for window_size in [2,3,4]:
    with open(f"X_{window_size}.pkl","rb") as f:
        X=pickle.load(f)
    with open(f"y_{window_size}.pkl","rb") as f:
        y=pickle.load(f)        
    print(f"Loaded X_{window_size} and y_{window_size} from pickle files.")
    

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

#step6:Layers:
    flattened=combined_embed.reshape(combined_embed.size(0), -1)

#Hidden Layer:
    relu=nn.ReLU()
    hidden_layer1=nn.Linear(window_size*embedding_dim,hidden_dim1)
    hidden_layer2=nn.Linear(hidden_dim1,hidden_dim2)
    hidden_layer3=nn.Linear(hidden_dim2,hidden_dim3)
    hidden_output1=relu(hidden_layer1(flattened))
    hidden_output2=relu(hidden_layer2(hidden_output1))
    hidden_output3=relu(hidden_layer3(hidden_output2))

#Relu Activation:
    relu=nn.ReLU()
    hidden_output1=relu(hidden_output1)
    hidden_output2=relu(hidden_output2)
    hidden_output3=relu(hidden_output3)

#Output Layer:
    output_layer=nn.Linear(hidden_dim3,vocab_size)

#Get logits and probabilities:
    logits=output_layer(hidden_output3)
    softmax=nn.Softmax(dim=1)
    probabilities=softmax(logits)

#Loss and Optimizer:
    criterion=nn.CrossEntropyLoss()
    optimizer = optim.Adam(
    list(token_embedding.parameters()) +list(positional_embedding.parameters()) +list(hidden_layer1.parameters()) +list(hidden_layer2.parameters()) +list(hidden_layer3.parameters()) +list(output_layer.parameters()),lr=0.005)

#step7:loss and training loop:
    epochs=150
    for epoch in range(epochs):
        optimizer.zero_grad()
    
    #Forward pass:
        token_embed=token_embedding(X)
        pos_embed=positional_embedding(positions)
        combined_embed=token_embed+pos_embed
        flattened=combined_embed.reshape(combined_embed.size(0), -1)
        dropout=nn.Dropout(0.1)
        hidden_output1=relu(hidden_layer1(flattened))
        hidden_output1=dropout(hidden_output1)
        hidden_output2=relu(hidden_layer2(hidden_output1))
        hidden_output2=dropout(hidden_output2)
        hidden_output3=relu(hidden_layer3(hidden_output2))
        hidden_output3=dropout(hidden_output3)
        logits=output_layer(hidden_output3)

    #Calculate loss and backpropagate:
        loss=criterion(logits,y)
        loss.backward()
        optimizer.step()
        if(epoch+1)%20==0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    print("="*40)
    print("flattened shape:", flattened.shape)
    print("hidden_output1 shape:", hidden_output1.shape)
    print("hidden_output2 shape:", hidden_output2.shape)
    print("hidden_output3 shape:", hidden_output3.shape)
    print("logits shape:", logits.shape)
    print("probabilities shape:", probabilities.shape)  

#step8:Save the model:
    torch.save({
    'token_embedding': token_embedding.state_dict(), 
    'positional_embedding': positional_embedding.state_dict(),
    'hidden_layer1': hidden_layer1.state_dict(),
    'hidden_layer2': hidden_layer2.state_dict(),
    'hidden_layer3': hidden_layer3.state_dict(),
    'output_layer': output_layer.state_dict()
    }, f"model_w{window_size}.pth")
    print(f"Model saved to model_w{window_size}.pth")

#step9:Save the vocabulary:
    with open("vocab.pkl","wb") as f:
        pickle.dump(vocab,f)
    print("Vocabulary saved to vocab.pkl")
    print("="*40)