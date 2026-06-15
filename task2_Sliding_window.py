#step1:(import libraries)
import pickle
with open("full_text.pkl","rb") as f:
    full_text=pickle.load(f)
corpus_tokens=full_text.split()
X=[]
y=[]
window_size=2

#step2:(create sliding windows)
for i in range(len(corpus_tokens)-window_size):
    X.append(corpus_tokens[i:i+window_size])
    y.append(corpus_tokens[i+window_size])
    
#step3:(create vocabulary)
all_words = set()
for sample in X:
    for word in sample:
        all_words.add(word)
for word in y:
    all_words.add(word)
vocab = {}
for idx, word in enumerate(sorted(all_words)):
    vocab[word] = idx
print(vocab)

#step4:(convert to numeric)
#convert X to numeric:    
X_numeric = []
for sample in X:
    X_numeric.append([vocab[word] for word in sample])
print("Train:\n",X_numeric)

#convert y to numeric:
y_numeric = [vocab[word] for word in y]
print("Test:\n",y_numeric)

#save to pickle files:
with open("X.pkl","wb") as f:
    pickle.dump(X_numeric,f)
with open("y.pkl","wb") as f:
    pickle.dump(y_numeric,f)
with open("vocab.pkl","wb") as f:
    pickle.dump(vocab,f)            
print("Length of X:",len(X_numeric))
print("Length of y:",len(y_numeric))    
