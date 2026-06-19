#step1:(import libraries)
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, TensorDataset

with open("vocab.pkl","rb") as f:
    vocab=pickle.load(f)
vocab_size=len(vocab)        
print("Loaded X and y from pickle files.")

#Hyperparameters:
embedding_dim=64
hidden_dim1=256
hidden_dim2=128
hidden_dim3=64
batch_size=516
epochs=300

for window_size in [2,3,4]:
    with open(f"X_{window_size}.pkl","rb") as f:
        X_numeric=pickle.load(f)
    with open(f"y_{window_size}.pkl","rb") as f:
        y_numeric=pickle.load(f)        
    print(f"Loaded X_{window_size} and y_{window_size} from pickle files.")
    if torch.is_tensor(X_numeric):
        X_tensor = X_numeric.detach().clone().long()
    else:
        X_tensor = torch.tensor(X_numeric, dtype=torch.long)

    if torch.is_tensor(y_numeric):
        y_tensor = y_numeric.detach().clone().long()
    else:
        y_tensor = torch.tensor(y_numeric, dtype=torch.long)
    dataset=TensorDataset(X_tensor, y_tensor)
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Token Embedding Layer:
    token_embedding=nn.Embedding(vocab_size,embedding_dim)
#Positional Embedding Layer:
    positional_embedding=nn.Embedding(window_size,embedding_dim)

#Create position Id's:
    positions=torch.arange(0,window_size).unsqueeze(0)

#step6:Layers:
#Hidden Layer:
    hidden_layer1=nn.Linear(window_size*embedding_dim,hidden_dim1)
    hidden_layer2=nn.Linear(hidden_dim1,hidden_dim2)
    hidden_layer3=nn.Linear(hidden_dim2,hidden_dim3)

#Relu Activation:
    relu=nn.ReLU()

#Output Layer:
    output_layer=nn.Linear(hidden_dim3,vocab_size)

#Get logits and probabilities:
    #logits=output_layer(hidden_output3)
    #softmax=nn.Softmax(dim=1)
    #probabilities=softmax(logits)

#Loss and Optimizer:
    criterion=nn.CrossEntropyLoss()
    optimizer = optim.Adam(
    list(token_embedding.parameters()) +list(positional_embedding.parameters()) +list(hidden_layer1.parameters()) +list(hidden_layer2.parameters()) +list(hidden_layer3.parameters()) +list(output_layer.parameters()),lr=0.005)

#step7:loss and training loop:
    for epoch in range(epochs):
        epoch_loss=0.0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
    
    #Forward pass:
            token_embed=token_embedding(batch_X)
            pos_embed=positional_embedding(positions)
            combined_embed=token_embed+pos_embed
            flattened=combined_embed.reshape(combined_embed.size(0), -1)
            dropout=nn.Dropout(0.1)
            hidden_output1=dropout(relu(hidden_layer1(flattened)))
            hidden_output2=dropout(relu(hidden_layer2(hidden_output1)))
            hidden_output3=dropout(relu(hidden_layer3(hidden_output2)))
            logits=output_layer(hidden_output3)

    #Calculate loss and backpropagate:
            loss=criterion(logits,batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
        if(epoch+1)%20==0:
            avg_loss=epoch_loss/len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], avg_loss: {avg_loss:.4f}")
    print("="*40)
    print("flattened shape:", flattened.shape)
    print("hidden_output1 shape:", hidden_output1.shape)
    print("hidden_output2 shape:", hidden_output2.shape)
    print("hidden_output3 shape:", hidden_output3.shape)
    print("logits shape:", logits.shape)
    #print("probabilities shape:", probabilities.shape)  

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