# 🧠 Explainable ADNET using Graph Attention Autoencoders (X-ADNET)

> A PyTorch Geometric implementation of an **Explainable Anomaly Detection Network (ADNET)** that integrates **Graph Attention Networks (GATs)** and **Autoencoders** for interpretable graph-based anomaly detection.

---

## 🚀 Overview

This project implements **X-ADNET (Explainable ADNET)** — a graph neural network model that combines **autoencoder-based anomaly detection** with **attention-based explainability**.  
It enables **node-level anomaly detection** on graph datasets like **Cora** while also providing **attention-based visual explanations** for why certain nodes are flagged as anomalous.

---

### 🌟 Key Highlights
- 🧩 Built on **Graph Attention Networks (GATConv)** and **Graph Convolutional Networks (GCNConv)**  
- 🕵️‍♂️ Performs **structure and attribute reconstruction** for anomaly detection  
- 🔍 Provides **explainability via attention weights**  
- 📈 Visualizes anomalous nodes and influential neighbors using **NetworkX** and **PyVis**  
- 💡 Supports **interactive visual explanations** for each anomaly node  

---

## 🧬 Model Architecture

nput Graph → GAT Encoder → Latent Embeddings (H)
↓
┌───────────────┴──────────────┐
│ │
Structure Decoder (Â) Attribute Decoder (X̂)
│ │
└───────────────┬──────────────┘
↓
Combined Loss (L)



### 🔹 Components

- **Encoder:** GAT layer with attention weights (`GATConv`)
- **Structure Decoder:** Reconstructs adjacency matrix (Â)
- **Attribute Decoder:** Reconstructs node features (X̂)
- **Loss Function:**  
  \[
  L = (1 - \alpha) \cdot E_S + \alpha \cdot E_A
  \]  
  where  
  \( E_S \) = Structure loss (BCE)  
  \( E_A \) = Attribute loss (MSE)

---

## 🛠️ Installation

### 🔧 Prerequisites
Make sure you have **Python 3.10+** and **pip** installed.

### ⚙️ Setup Instructions

# Clone this repository
git clone https://github.com/yourusername/X-ADNET.git
cd X-ADNET

# Install required dependencies
pip install torch torchvision torchaudio
pip install torch-geometric
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__)")
pip install networkx matplotlib pyvis


📘 Usage

You can run the notebook directly in Google Colab ⤵️


🧠 Training Workflow

1. Load the dataset (Cora from PyTorch Geometric)

2. Train the X-ADNET model for 100 epochs

3. Compute anomaly scores for each node

4. Visualize explainable attention graphs for top anomalies

📊 Sample Training Output
Epoch	Total Loss	Structure Loss	Attribute Loss
10	0.3527	0.6931	0.0123
50	0.3526	0.6931	0.0120
100	0.3526	0.6931	0.0120


--- Top 5 Most Anomalous Nodes ---
Rank 1: Node ID 677 (Normalized Score: 1.0000)
Rank 2: Node ID 442 (Normalized Score: 0.8966)
Rank 3: Node ID 921 (Normalized Score: 0.8754)
Rank 4: Node ID 1794 (Normalized Score: 0.8723)
Rank 5: Node ID 2308 (Normalized Score: 0.8701)



🎯 Explainability Module

The attention mechanism in GATConv provides direct insight into which neighbors influenced the model’s decision
Node 677 has 2 neighbors (incl. self-loop).
This node's embedding was built by paying attention to:
- 0.509 → neighbor Node 954
- 0.491 → itself (self-loop)

🕸️ Visual Attention Graph

Use the built-in visualization function to generate interactive explanations:

visualize_explanation(target_node_id=677, file_name="explanation.html")


Output:

🔴 Red node = detected anomaly

🟢 Green nodes = influential neighbors

Edge color/thickness = attention weight

🧩 Technologies Used
Technology	Description
PyTorch Geometric	Graph neural network framework
GATConv & GCNConv	Core layers for graph learning
NetworkX	Graph structure visualization
PyVis	Interactive, web-based explainability
Matplotlib	Static plots and heatmaps
Cora Dataset	Benchmark dataset for node classification


References

Ding, K., et al. “Deep Anomaly Detection on Attributed Networks.” IJCAI 2019.

Velickovic, P., et al. “Graph Attention Networks.” ICLR 2018.

Kipf, T. N., & Welling, M. “Semi-Supervised Classification with Graph Convolutional Networks.” ICLR 2017.

PyTorch Geometric Documentation — https://pytorch-geometric.readthedocs.io



🧑‍💻 Author

Daksh Jain
📧 Email: dakshjain650dj@.com


🪄 Future Enhancements

🔹 Multi-head attention for richer context

🔹 Integration with heterogeneous graph datasets

🔹 Streamlit-based interactive explainability dashboard



🏁 License

This project is licensed under the MIT License — feel free to use and modify it for research or educational purposes.
