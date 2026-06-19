#step1:(import libraries)
import pickle
import torch
with open("full_text.pkl","rb") as f:
    full_text=pickle.load(f)
corpus_tokens=full_text.split()

#Create Vocabulary:
all_words = sorted(list(set(corpus_tokens)))
vocab = {word: idx for idx, word in enumerate(all_words)}
print("Length of vocabulary:", len(vocab))
with open("vocab.pkl","wb") as f:
    pickle.dump(vocab,f)

for window_size in [2,3,4]:
    X_numeric=[]
    y_numeric=[]

#step2:(create sliding windows)
    for i in range(len(corpus_tokens)-window_size):
        X_numeric.append(
        [vocab[word] for word in corpus_tokens[i:i+window_size]]
    )   
        y_numeric.append(
         vocab[corpus_tokens[i+window_size]]
    )
    print("="*40)    
    print("Train:\n",X_numeric[:15])    
    print("Test:\n",y_numeric[:15])
    print("="*40)

    X_tensor=torch.tensor(X_numeric,dtype=torch.long)
    y_tensor=torch.tensor(y_numeric,dtype=torch.long)

#save to pickle files:
    with open(f"X_{window_size}.pkl","wb") as f:
           pickle.dump(X_tensor,f)
    with open(f"y_{window_size}.pkl","wb") as f:
           pickle.dump(y_tensor,f)            
print("Length of X:",len(X_numeric))
print("Length of y:",len(y_numeric))    
