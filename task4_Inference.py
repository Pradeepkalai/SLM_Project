import torch
import torch.nn as nn
import pickle
import os
import streamlit as st

#step1:(Set config):
st.set_page_config(page_title="Small Language Model",layout="centered")
@st.cache_resource
def load_resources():
    if not os.path.exists("vocab.pkl"):
        st.error("Error:'vocab.pkl' not found")
        st.stop()
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    return vocab, reverse_vocab
vocab, reverse_vocab = load_resources()

#step2:Hyperparameters:
vocab_size = len(vocab)
embedding_dim = 64
hidden_dim1 = 256
hidden_dim2 = 128
hidden_dim3 = 64
relu = nn.ReLU()

def predict_next_word(input_words, window_size):
    model_path = f"model_w{window_size}.pth"
    if not os.path.exists(model_path):
        return f"(Error: {model_path} not trained yet)", 0, []
        
    token_embedding = nn.Embedding(vocab_size, embedding_dim)
    positional_embedding = nn.Embedding(window_size,embedding_dim)
    hidden_layer1 = nn.Linear(window_size * embedding_dim, hidden_dim1)
    hidden_layer2 = nn.Linear(hidden_dim1, hidden_dim2)
    hidden_layer3 = nn.Linear(hidden_dim2, hidden_dim3)
    output_layer = nn.Linear(hidden_dim3, vocab_size)

    # Load Weights
    checkpoint = torch.load(model_path)
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

    # Convert words to IDs
    input_ids = [vocab[word] if word in vocab else 0 for word in input_words]
    X = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        token_embed = token_embedding(X)
        positions = torch.arange(0, window_size).unsqueeze(0)
        pos_embed = positional_embedding(positions)
        combined_embed = token_embed + pos_embed
        flattened = combined_embed.reshape(combined_embed.size(0), -1)
        
        hidden_output1 = relu(hidden_layer1(flattened))
        hidden_output2 = relu(hidden_layer2(hidden_output1))
        hidden_output3 = relu(hidden_layer3(hidden_output2))
        logits = output_layer(hidden_output3)
        
        #top-k sampling:
        scaled_logits = logits / 0.7
        top_k = min(5, vocab_size)
        values, indices = torch.topk(scaled_logits, top_k)
        top_k_probs = torch.softmax(values, dim=-1)
        sampled_index = torch.multinomial(top_k_probs[0], num_samples=1).item()
        predicted_id = indices[0][sampled_index].item()

    return reverse_vocab[predicted_id], predicted_id, input_ids

st.title("Small Language Model")

# Single Text Input Element for the User
user_input = st.text_input("Type your input:")

if user_input:
    tokens = user_input.strip().split()
    word_count = len(tokens)
    
    if word_count < 2:
        st.info("Please type at least 2 words")
    else:
        if word_count == 2:
            window_size = 2
            context_words = tokens
        elif word_count == 3:
            window_size = 3
            context_words = tokens
        elif word_count == 4:
            window_size = 4
            context_words = tokens
        else:
            st.stop()
            
        #Predicted:
        predicted_word, predicted_id, input_ids = predict_next_word(context_words, window_size)
        completed_sequence = tokens + [predicted_word]
        
        #Layout:
        st.code(
            f"Input Words: {context_words}\n"
            f"Input IDs:    {input_ids}\n"
            f"Predicted Word: '{predicted_word}'",
            language="text"
        )
        st.success(f"{' '.join(completed_sequence)}")
        
        # Auto termination
        if word_count >= 4:
            st.warning("Terminated.")
            st.stop()         