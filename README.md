# SLM_Project

# 🧠 Custom Small Language Model (SLM)

An end-to-end, custom-built Small Language Model (SLM) pipeline designed to process raw literature, handle dynamic contextual sequences, train deep learning architectures using PyTorch, and serve an interactive text-generation web interface via Streamlit.


## 🚀 Project Overview

Unlike massive commercial models, this project explores the core fundamentals of Natural Language Processing (NLP) and Deep Learning. It builds a next-word prediction engine trained on custom text by treating context lengths as mathematical sliding windows ($W = 2, 3, 4$). 

### Key Features
* **Custom Data Pipeline:** Extracts, cleans, and structures raw text directly from PDF inputs.
* **Token Embeddings + Positional Coding:** Tracks both word meaning and sequential position.
* **Multi-Architecture Checkpointing:** Trains and loads independent PyTorch neural weights tailored specifically to the length of your input phrase.
* **Inference Streamlit UI:** Uses **Top-K Sampling** ($k=5$) and Temperature Scaling ($\tau = 0.7$) to ensure creative yet coherent word generation.

🛠️ The 4-Stage Technical Pipeline

[ Raw PDF ] ➔ (1. Tokenize & Clean) ➔ (2. Sliding Window Matrix) ➔ (3. PyTorch Training) ➔ (4. Streamlit UI Client)

1. Full Corpus Tokenization (task1_Tokenization.py):
   
     Extracts text strings from raw book PDFs using PyPDF2.Standardizes strings using Regular Expressions (re) to strip erratic whitespace.Converts words to unique structural tokens using the microsoft/Phi-3-mini-4k-instruct tokenizer.

2. Dataset Engineering (task2_Sliding_window.py):
   
     Implements a custom sliding window configuration over the integer text token corpus.Dynamically chunks datasets into input groups (X) and target tokens (y) across three distinct window contexts (W=2, W=3, and W=4).

3. Deep Learning Network Architecture (task3_Train.py):
   
     Constructs an architecture containing:nn.Embedding for vocabulary features.Positional Embedding layers to track word sequencing.3 Hidden Dense Layers combined with nn.ReLU activations.Optimizes cross-entropy loss over multiple epochs, checkpointing structural files neatly inside models/checkpoints/.

4. Live Text Generation Interface (task4_Inference.py):
   
     Serves a lightweight web interface via Streamlit.Reads the user's sentence, checks the token count, routes the input to the exact corresponding model size checkpoint, and samples predictions via Top-K multinomial probabilities.

💻 Quick Start & Execution

1. Installation
   
Clone the repository and install all environmental framework dependencies:

pip install -r requirements.txt

2. Run the Full Pipeline Sequentially
# Phase 1: Extract and clean raw data
python src/task1_Tokenization.py

# Phase 2: Build dataset feature matrices
python src/task2_Sliding_window.py

# Phase 3: Train your deep learning weights
python src/task3_Train.py

3. Launch the Interactive Web App

streamlit run src/task4_Inference.py


## 📁 Repository Structure & Alignment

The project follows a standard, industry-grade machine learning directory hierarchy:

```text
SLM_Project/
│
├── data/                       # Data Management Layer
│   ├── raw/                    # Source inputs (e.g., Days at the Morisaki Bookshop.pdf)
│   └── processed/              # Tokenized and serialized tensors (.pkl files)
│
├── models/                     # Weights & Vocabulary
│   ├── vocab.pkl               # Word-to-index mapping dictionary
│   └── checkpoints/            # Trained weights per window configuration
│       ├── model_w2.pth        # Model for 2-word input contexts
│       ├── model_w3.pth        # Model for 3-word input contexts
│       └── model_w4.pth        # Model for 4-word input contexts
│
├── src/                        # Core Application Source Code
│   ├── task1_Tokenization.py   # PDF text extraction & Token embedding setup
│   ├── task2_Sliding_window.py # Feature matrix and target array slicing
│   ├── task3_Train.py          # PyTorch Multi-Layer Perceptron (MLP) training loop
│   └── task4_Inference.py      # Interactive Streamlit Web UI engine
│
├── requirements.txt            # Third-party dependencies
└── README.md                   # Technical documentation
