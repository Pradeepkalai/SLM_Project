import torch
import torch.nn as nn
import pickle
import os

#step1:(Load Vocabulary)
with open("vocab.pkl","rb") as f:
    vocab=pickle.load(f)

#Reverse Vocabulary:
reverse_vocab={idx:word for word,idx in vocab.items()}

#step2:(Hyperparameters):
vocab_size=len(vocab)
embedding_dim=64
hidden_dim1=256
hidden_dim2=128
hidden_dim3=64

#step5:(User Input):
text=input("Enter a sequence of 2 words: ")
words=text.strip().split()
if len(words)!=2:
    print(f"Please enter exactly 2 words.")
    exit()
print(f"\nStarting Sequence:{' '.join(words)}")
for step in range(3):
    current_window_size=len(words)
    if current_window_size == 2:
        window_size = 2
        input_words = words
    elif current_window_size == 3:
        window_size = 3
        input_words = words
    else:
        window_size = 4
        input_words = words[-4:]    
    model_path = f"model_w{window_size}.pth"
    if not os.path.exists(model_path):
        print(f"Error: Missing trained weights file '{model_path}'. Run training first.")
        exit()
    
#step3:(Create layers):
    token_embedding=nn.Embedding(vocab_size,embedding_dim)
    positional_embedding=nn.Embedding(window_size,embedding_dim)
    hidden_layer1=nn.Linear(window_size*embedding_dim,hidden_dim1)
    hidden_layer2=nn.Linear(hidden_dim1,hidden_dim2)
    hidden_layer3=nn.Linear(hidden_dim2,hidden_dim3)
    relu=nn.ReLU()
    output_layer=nn.Linear(hidden_dim3,vocab_size)

#step4:(Load Trained weights):
    checkpoint=torch.load(f"model_w{window_size}.pth")
    token_embedding.load_state_dict(checkpoint["token_embedding"])
    positional_embedding.load_state_dict(checkpoint["positional_embedding"])
    hidden_layer1.load_state_dict(checkpoint["hidden_layer1"])
    hidden_layer2.load_state_dict(checkpoint["hidden_layer2"])
    hidden_layer3.load_state_dict(checkpoint["hidden_layer3"])
    output_layer.load_state_dict(checkpoint["output_layer"])
    token_embedding.eval()
    positional_embedding.eval()
    hidden_layer1.eval()
    hidden_layer2.eval()
    hidden_layer3.eval()
    output_layer.eval()   

#convert words to ids:
    input_ids=[]
    for word in input_words:
       if word in vocab:
          input_ids.append(vocab[word])
       else:
          print(f"Word '{word}' not in vocabulary.")
          exit()
    print("Input IDs:", input_ids)

#Tensor:
    X=torch.tensor([input_ids],dtype=torch.long)

#step6:(Forward Pass):
    with torch.no_grad():
          token_embed=token_embedding(X)
          positions=torch.arange(0,window_size).unsqueeze(0)
          pos_embed=positional_embedding(positions)
          combined_embed=token_embed+pos_embed
          flattened=combined_embed.reshape(combined_embed.size(0), -1)
          hidden_output1=hidden_layer1(flattened)
          hidden_output1=relu(hidden_output1)
          hidden_output2=hidden_layer2(hidden_output1)
          hidden_output2=relu(hidden_output2)
          hidden_output3=hidden_layer3(hidden_output2)
          hidden_output3=relu(hidden_output3)
          logits=output_layer(hidden_output3)
          temperature=0.7
          scaled_logits=logits/temperature

#Softmax:
          softmax=nn.Softmax(dim=1)
          probabilities=torch.softmax(scaled_logits,dim=-1)

#top-k sampling:
          top_k = 5
          values, indices = torch.topk(scaled_logits, top_k)
          top_k_probs = torch.softmax(values, dim=-1)
          sampled_index = torch.multinomial(top_k_probs[0], num_samples=1).item()
          predicted_id = indices[0][sampled_index].item()

#step7:(Get predicted word):
    predicted_word=reverse_vocab[predicted_id]
    words.append(predicted_word)
print("Predicted Sequence:", " ".join(words))
print("Predicted Id:",predicted_id)