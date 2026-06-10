from task1_Tokenization import full_text
corpus_tokens=full_text.split()
X=[]
y=[]
window_size=2
for i in range(len(corpus_tokens)-window_size):
    X.append(corpus_tokens[i:i+window_size])
    y.append(corpus_tokens[i+window_size])
print("Train:\n",X)
print("Test:\n",y)
print(len(X))
print(len(y))    
