#step1:
import torch
import torch.nn as nn
import torch.optim as optim
from task2_Sliding_window import X,y
X=torch.tensor(X,dtype=torch.long)
y=torch.tensor(y,dtype=torch.long)

#step2:
vocab_size=max(torch.max(X).item(), torch.max(y).item()) + 1
print("Vocabulary Size:", vocab_size)

#step3:
window_size=X.shape[1]
embedding_dim=64
hidden_dim=128
epochs=10
learning_rate=0.001

#step4:

    