#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantum-Enhanced Transformer (Q-Transformer) for Language Modeling

This script implements a hybrid quantum-classical transformer model that integrates
quantum circuits into the attention mechanism using PennyLane and PyTorch.
"""

import os
import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pennylane as qml
from typing import List, Tuple, Dict, Optional

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
n_qubits = 4  # Number of qubits in quantum circuit
embedding_dim = 64  # Embedding dimension
hidden_dim = 256  # Hidden dimension of feedforward network
num_heads = 4  # Number of attention heads (only 1 quantum head, rest are classical)
num_layers = 2  # Number of transformer layers
max_seq_length = 64  # Maximum sequence length
batch_size = 8  # Batch size for training
vocab_size = 5000  # Vocabulary size (will be updated after tokenization)
learning_rate = 0.0005  # Learning rate for optimizer
num_epochs = 3  # Number of training epochs
grad_clip = 1.0  # Gradient clipping value

###########################################
#        QUANTUM CIRCUIT DEFINITION       #
###########################################

class QuantumCircuit:
    """
    Defines a parameterized quantum circuit using PennyLane.
    
    The circuit encodes classical input vectors using angle encoding
    and contains trainable parameters for rotations and entangling gates.
    It outputs expectation values which are used to compute attention scores.
    """
    
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        """
        Initialize the quantum circuit.
        
        Args:
            n_qubits: Number of qubits in the circuit
            n_layers: Number of variational layers
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Define the device
        # default.qubit is a CPU-based simulator.
        # For GPU-accelerated quantum simulation, PennyLane-Lightning (e.g., "lightning.gpu")
        # would be required, which is an additional dependency.
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Create quantum node
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch", diff_method="backprop")
        
        # Define parameter shape for the quantum circuit
        weight_shapes = {
            "weights_rx": (n_layers, n_qubits),
            "weights_ry": (n_layers, n_qubits),
            "weights_rz": (n_layers, n_qubits),
            "weights_final": (n_qubits,)
        }
        
        # Create QNN layer
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)
        
    def circuit(self, inputs, weights_rx, weights_ry, weights_rz, weights_final):
        """
        Define the quantum circuit structure.
        
        Args:
            inputs: Input data for encoding (expected to be in range [-1, 1] for each element)
            weights_rx: Trainable parameters for RX gates
            weights_ry: Trainable parameters for RY gates
            weights_rz: Trainable parameters for RZ gates
            weights_final: Trainable parameters for the final rotation
            
        Returns:
            List of expectation values for each qubit
        """
        # Inputs are assumed to be normalized (e.g., via tanh) to [-1, 1]
        # Scale inputs for angle encoding to [-π, π]
        inputs_scaled = inputs * math.pi
        
        # Encode the classical input vector
        # qml.AngleEmbedding(inputs_scaled, wires=range(self.n_qubits), rotation='Y') # Alternative
        for i in range(self.n_qubits):
            qml.RY(inputs_scaled[..., i], wires=i) # Use ... to handle batch dimensions
        
        # Variational quantum circuit layers
        for layer in range(self.n_layers):
            # Apply rotation gates with trainable parameters
            for qubit in range(self.n_qubits):
                qml.RX(weights_rx[layer, qubit], wires=qubit)
                qml.RY(weights_ry[layer, qubit], wires=qubit)
                qml.RZ(weights_rz[layer, qubit], wires=qubit)
            
            # Apply entangling gates (CNOT)
            for qubit in range(self.n_qubits - 1):
                qml.CNOT(wires=[qubit, qubit + 1])
            if self.n_qubits > 1: # Avoid CNOT on single qubit
                qml.CNOT(wires=[self.n_qubits - 1, 0])
        
        # Apply a final parameterized rotation before measurement
        for qubit in range(self.n_qubits):
            qml.RY(weights_final[qubit], wires=qubit)
            
        # Return expectation values for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input tensor through quantum circuit.
        
        Args:
            x: Input tensor of shape (batch_dims..., input_dim)
            
        Returns:
            Quantum circuit output tensor of shape (batch_dims..., n_qubits)
        """
        input_size = x.shape[-1]
        
        if input_size != self.n_qubits: # Only pad/truncate if necessary
            if input_size < self.n_qubits:
                # Pad if input is smaller than n_qubits
                padding_shape = (*x.shape[:-1], self.n_qubits - input_size)
                padding = torch.zeros(padding_shape, device=x.device, dtype=x.dtype)
                x_processed = torch.cat([x, padding], dim=-1)
            else: # input_size > self.n_qubits
                # Truncate if input is larger than n_qubits
                x_processed = x[..., :self.n_qubits]
        else:
            x_processed = x

        # Process input through quantum layer
        return self.qlayer(x_processed)

###########################################
#    QUANTUM ATTENTION MECHANISM          #
###########################################

class QuantumAttentionHead(nn.Module):
    """
    Quantum Attention Head module that uses a quantum circuit
    to compute attention scores between query and key vectors.
    """
    
    def __init__(self, embed_dim: int, q_dim: int, n_qubits: int = 4, n_layers: int = 2):
        """
        Initialize the quantum attention head.
        
        Args:
            embed_dim: Embedding dimension
            q_dim: Dimension after the query/key projections (before n_qubits projection)
            n_qubits: Number of qubits in quantum circuit
            n_layers: Number of variational layers in quantum circuit
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.q_dim = q_dim
        self.n_qubits = n_qubits
        
        # Linear transformations for query, key, and value
        self.query_proj = nn.Linear(embed_dim, q_dim)
        self.key_proj = nn.Linear(embed_dim, q_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim) # Value can keep full embed_dim
        
        # Projection for quantum encoding (reduce dimension to match qubits)
        self.q_to_qubits = nn.Linear(q_dim, n_qubits)
        self.k_to_qubits = nn.Linear(q_dim, n_qubits)
        
        # Initialize quantum circuit
        self.quantum_circuit = QuantumCircuit(n_qubits=n_qubits, n_layers=n_layers)
        
        # Output projection for attention scores from quantum output
        self.score_projection = nn.Linear(n_qubits, 1)
        
        # Final attention output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim) # Projects concatenated/summed head output
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the quantum attention head.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Mask tensor of shape (batch_size, seq_len, seq_len) or (1, seq_len, seq_len)
            
        Returns:
            Attention output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project query, key, and value
        q = self.query_proj(x)  # (batch_size, seq_len, q_dim)
        k = self.key_proj(x)    # (batch_size, seq_len, q_dim)
        v = self.value_proj(x)  # (batch_size, seq_len, embed_dim)
        
        # Reduce q and k to n_qubits dimension
        q_reduced = self.q_to_qubits(q) # (batch_size, seq_len, n_qubits)
        k_reduced = self.k_to_qubits(k) # (batch_size, seq_len, n_qubits)

        # Prepare for batched query-key interaction
        # q_expanded: (batch_size, seq_len_q, 1, n_qubits)
        # k_expanded: (batch_size, 1, seq_len_k, n_qubits)
        q_expanded = q_reduced.unsqueeze(2) 
        k_expanded = k_reduced.unsqueeze(1)

        # Combine query and key using element-wise multiplication, then normalize with tanh
        # qk_combined_raw: (batch_size, seq_len_q, seq_len_k, n_qubits)
        qk_combined_raw = q_expanded * k_expanded 
        qk_combined_normalized = torch.tanh(qk_combined_raw) # Normalize to [-1, 1] for angle encoding

        # Reshape for batch processing by the quantum circuit
        # Original shape: (batch_size, seq_len_q, seq_len_k, n_qubits)
        # Reshaped: (batch_size * seq_len_q * seq_len_k, n_qubits)
        current_shape = qk_combined_normalized.shape
        qk_flat = qk_combined_normalized.reshape(-1, self.n_qubits)
        
        # Process through quantum circuit
        # qc_output_flat: (batch_size * seq_len_q * seq_len_k, n_qubits)
        qc_output_flat = self.quantum_circuit.process_input(qk_flat)
        
        # Project quantum output to a scalar attention score
        # attention_scores_flat: (batch_size * seq_len_q * seq_len_k)
        attention_scores_flat = self.score_projection(qc_output_flat).squeeze(-1)
        
        # Reshape attention scores to (batch_size, seq_len_q, seq_len_k)
        attention_scores = attention_scores_flat.view(batch_size, seq_len, seq_len)
        
        # Apply mask if provided
        if attention_mask is not None:
            # Ensure mask is broadcastable: (batch_size, seq_len, seq_len) or (1, seq_len, seq_len)
            # If mask is (seq_len, seq_len), unsqueeze it.
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(0)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Apply attention weights to values
        # v is (batch_size, seq_len_k, embed_dim)
        # attention_weights is (batch_size, seq_len_q, seq_len_k)
        # context is (batch_size, seq_len_q, embed_dim)
        context = torch.bmm(attention_weights, v)
        
        # Final projection
        output = self.out_proj(context)  # (batch_size, seq_len, embed_dim)
        
        return output

class ClassicalAttentionHead(nn.Module):
    """
    Classical attention head used alongside quantum heads in the multi-head attention.
    Implements the standard scaled dot-product attention mechanism.
    """
    
    def __init__(self, embed_dim: int, head_dim: int):
        """
        Initialize the classical attention head.
        
        Args:
            embed_dim: Embedding dimension
            head_dim: Dimension of each attention head's Q, K, V projections
        """
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, head_dim)
        self.k_proj = nn.Linear(embed_dim, head_dim)
        self.v_proj = nn.Linear(embed_dim, head_dim)
        # The output projection takes the head_dim context vector and projects it back to embed_dim.
        # This is crucial for the HybridMultiHeadAttention's sum() combination strategy,
        # ensuring each head contributes an embed_dim sized tensor.
        self.out_proj = nn.Linear(head_dim, embed_dim) 
        self.scale = head_dim ** -0.5
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for classical attention head.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Mask tensor of shape (batch_size, seq_len, seq_len) or (1, seq_len, seq_len)
            
        Returns:
            Attention output tensor of shape (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project query, key, and value
        q = self.q_proj(x)  # (batch_size, seq_len, head_dim)
        k = self.k_proj(x)  # (batch_size, seq_len, head_dim)
        v = self.v_proj(x)  # (batch_size, seq_len, head_dim)
        
        # Compute attention scores
        attn_scores = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (batch_size, seq_len, seq_len)
        
        # Apply mask if provided
        if attention_mask is not None:
            if attention_mask.dim() == 2: # (seq_len, seq_len)
                attention_mask = attention_mask.unsqueeze(0) # (1, seq_len, seq_len) for broadcasting
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # Apply attention weights to values
        context = torch.bmm(attn_weights, v)  # (batch_size, seq_len, head_dim)
        
        # Final projection to embed_dim
        output = self.out_proj(context)  # (batch_size, seq_len, embed_dim)
        
        return output

class HybridMultiHeadAttention(nn.Module):
    """
    Multi-head attention with a mix of quantum and classical attention heads.
    Combines one quantum head with several classical heads.
    Each head's output (already projected to embed_dim) is summed.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, use_quantum: bool = True, n_qubits: int = 4):
        """
        Initialize the hybrid multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Total number of attention heads
            use_quantum: Whether to use quantum head(s) or all classical heads
            n_qubits: Number of qubits in quantum circuit
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_quantum = use_quantum
        
        # head_dim is the internal dimension for Q,K,V projections within each classical head,
        # and for Q,K projections (before n_qubits reduction) in the quantum head.
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Create attention heads
        self.heads = nn.ModuleList()
        
        if use_quantum:
            if num_heads < 1:
                raise ValueError("num_heads must be at least 1 if use_quantum is True.")
            self.heads.append(QuantumAttentionHead(
                embed_dim=embed_dim,
                q_dim=self.head_dim, # Intermediate dimension for Q/K before n_qubit projection
                n_qubits=n_qubits
            ))
            
            # Add remaining classical heads
            for _ in range(num_heads - 1): # if num_heads is 1, this loop doesn't run
                self.heads.append(ClassicalAttentionHead(embed_dim, self.head_dim))
        else:
            # All classical heads for baseline comparison
            if num_heads < 1:
                 raise ValueError("num_heads must be at least 1.")
            for _ in range(num_heads):
                self.heads.append(ClassicalAttentionHead(embed_dim, self.head_dim))
        
        # Output projection. Since each head already outputs embed_dim and we sum them,
        # this final projection processes the summed embed_dim representations.
        self.out_proj = nn.Linear(embed_dim, embed_dim) 
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for hybrid multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Mask tensor of shape (batch_size, seq_len, seq_len) or (1, seq_len, seq_len)
            
        Returns:
            Attention output tensor of shape (batch_size, seq_len, embed_dim)
        """
        head_outputs = [head(x, attention_mask) for head in self.heads] # Each is (B, S, D)
        
        # Stack outputs to (B, S, num_heads, D) then sum over num_heads dimension
        # This effectively averages or sums the representations from each head.
        # An alternative (more standard) would be to have each head output head_dim,
        # then concatenate to (B, S, num_heads*head_dim = D) and then project.
        # The current method sums D-dimensional outputs.
        if len(head_outputs) == 1:
            multi_head_output = head_outputs[0]
        else:
            multi_head_output = torch.stack(head_outputs, dim=0).sum(dim=0)

        # Apply final projection
        output = self.out_proj(multi_head_output) # (batch_size, seq_len, embed_dim)
        
        return output

###########################################
#         TRANSFORMER COMPONENTS          #
###########################################

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    Uses the standard sinusoidal positional encoding from the Transformer paper.
    """
    
    def __init__(self, embed_dim: int, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerBlock(nn.Module):
    """
    A single transformer block with hybrid attention mechanism.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, use_quantum: bool = True, n_qubits: int = 4, dropout_rate: float = 0.1):
        """
        Initialize the transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Hidden dimension of the feedforward network
            use_quantum: Whether to use quantum attention heads
            n_qubits: Number of qubits in quantum circuit
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        self.attention = HybridMultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_quantum=use_quantum,
            n_qubits=n_qubits
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout_rate)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            attention_mask: Mask tensor of shape (batch_size, seq_len, seq_len)
            
        Returns:
            Transformer block output tensor
        """
        # Pre-LayerNorm: Apply norm before passing to sub-layer
        x_norm1 = self.norm1(x)
        attn_output = self.attention(x_norm1, attention_mask)
        x = x + self.dropout(attn_output) # Residual connection
        
        x_norm2 = self.norm2(x)
        ff_output = self.ff(x_norm2)
        x = x + self.dropout(ff_output) # Residual connection
        
        return x

###########################################
#      QUANTUM TRANSFORMER MODEL          #
###########################################

class QuantumTransformer(nn.Module):
    """
    A full transformer model with quantum-enhanced attention mechanism.
    """
    
    def __init__(self, 
                 vocab_size: int, 
                 embed_dim: int, 
                 num_heads: int, 
                 ff_dim: int, 
                 num_layers: int, 
                 max_seq_len: int, 
                 use_quantum: bool = True,
                 n_qubits: int = 4,
                 dropout_rate: float = 0.1):
        """
        Initialize the quantum transformer model.
        
        Args:
            vocab_size: Size of the vocabulary
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Hidden dimension of the feedforward network
            num_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            use_quantum: Whether to use quantum attention heads
            n_qubits: Number of qubits in quantum circuit
            dropout_rate: Dropout rate for embeddings and transformer blocks
        """
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_seq_len)
        self.embed_dropout = nn.Dropout(dropout_rate)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                use_quantum=use_quantum,
                n_qubits=n_qubits,
                dropout_rate=dropout_rate
            ) for _ in range(num_layers)
        ])
        
        # Final layer normalization (applied after all blocks, before output projection)
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the quantum transformer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len)
            attention_mask: Mask tensor of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len)
            
        Returns:
            Output logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        if attention_mask is None:
            # Create a causal mask for language modeling if not provided
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).bool()
        
        # Ensure mask is correctly shaped for broadcasting if needed
        if attention_mask.dim() == 2: # (seq_len, seq_len)
            attention_mask = attention_mask.unsqueeze(0) # (1, seq_len, seq_len)

        x = self.token_embedding(x)  # (batch_size, seq_len, embed_dim)
        x = self.pos_encoding(x)    # (batch_size, seq_len, embed_dim)
        x = self.embed_dropout(x)
        
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
        
        x = self.norm(x) # Final normalization
        x = self.output(x)  # (batch_size, seq_len, vocab_size)
        
        return x

###########################################
#         DATASET PREPARATION            #
###########################################

class SimpleTokenizer:
    """
    A basic tokenizer that splits text into characters or words.
    """
    
    def __init__(self, text: str, tokenize_chars: bool = True):
        self.tokenize_chars = tokenize_chars
        # Build vocab first
        raw_tokens = self._get_raw_tokens(text)
        self.vocab = self._build_vocab(raw_tokens)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.pad_id = self.token_to_id["<PAD>"]
        self.unk_id = self.token_to_id["<UNK>"]
        self.bos_id = self.token_to_id.get("<BOS>") # May not exist if not in special_tokens
        self.eos_id = self.token_to_id.get("<EOS>") # May not exist if not in special_tokens

    def _get_raw_tokens(self, text: str) -> List[str]:
        if self.tokenize_chars:
            return list(text)
        else:
            return text.split()

    def _build_vocab(self, raw_tokens: List[str]) -> List[str]:
        # Special tokens first, to ensure consistent IDs (e.g. PAD=0)
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        # Unique tokens from text, excluding any that might clash with special tokens
        unique_data_tokens = sorted(list(set(t for t in raw_tokens if t not in special_tokens)))
        return special_tokens + unique_data_tokens
    
    def encode(self, text: str) -> List[int]:
        tokens = self._get_raw_tokens(text)
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        tokens = [self.id_to_token.get(id, "<UNK>") for id in ids if id != self.pad_id] # Exclude PAD
        
        if self.tokenize_chars:
            return "".join(tokens)
        else:
            return " ".join(tokens)

class TextDataset(Dataset):
    """
    Dataset for language modeling.
    Uses overlapping windows to create more data.
    """
    
    def __init__(self, text: str, tokenizer: SimpleTokenizer, seq_length: int):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        
        # Tokenize the entire text, add BOS/EOS if desired (not strictly necessary for char LM)
        # For char LM, often raw text is fine.
        self.input_ids = self.tokenizer.encode(text)
        
        self.data = []
        # Using a step of 1 for maximum data, can be increased for less overlap
        step = 1 
        for i in range(0, len(self.input_ids) - seq_length -1, step): # -1 because target is shifted
            input_seq = self.input_ids[i : i + seq_length]
            target_seq = self.input_ids[i + 1 : i + seq_length + 1]
            
            # No explicit padding here, assuming text is long enough or handled by collate_fn if sequences vary
            # The current fixed-length slicing avoids variable lengths.
            # If len(self.input_ids) is small, this loop might not produce much data.
            self.data.append((input_seq, target_seq))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

###########################################
#         TRAINING FUNCTIONS              #
###########################################

def create_mini_shakespeare():
    text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause—there's the respect
    That makes calamity of so long life.
    For who would bear the whips and scorns of time,
    The oppressor's wrong, the proud man's contumely,
    The pangs of despised love, the law's delay,
    The insolence of office, and the spurns
    That patient merit of the unworthy takes,
    When he himself might his quietus make
    With a bare bodkin? Who would fardels bear,
    To grunt and sweat under a weary life,
    But that the dread of something after death,
    The undiscover'd country, from whose bourn
    No traveller returns, puzzles the will,
    And makes us rather bear those ills we have
    Than fly to others that we know not of?
    Thus conscience does make cowards of us all;
    And thus the native hue of resolution
    Is sicklied o'er with the pale cast of thought,
    And enterprises of great pith and moment
    With this regard their currents turn awry
    And lose the name of action.
    """
    # Clean up excessive newlines and leading/trailing whitespace
    return ' '.join(text.strip().split())


def generate_text(model: nn.Module, 
                  tokenizer: SimpleTokenizer, 
                  seed_text: str, 
                  max_length: int = 100, 
                  temperature: float = 0.8):
    model.eval()
    input_ids = tokenizer.encode(seed_text)
    # If seed_text is shorter than model's max_seq_length, pad it for first prediction
    # Or, ensure model can handle shorter sequences. Current model expects fixed length or masking.
    # For generation, we typically don't pad the initial input, but feed what we have.
    
    generated_ids = list(input_ids) # Store all generated IDs

    with torch.no_grad():
        for _ in range(max_length):
            # Prepare current sequence for model input
            # Take last max_seq_length tokens if current sequence is too long
            current_input_ids = generated_ids[-max_seq_length:]
            input_tensor = torch.tensor([current_input_ids], dtype=torch.long).to(device)
            
            outputs = model(input_tensor) # No explicit mask needed if model handles causal internally for generation
            next_token_logits = outputs[:, -1, :] # Logits for the last token in the sequence
            
            next_token_probs = F.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1).squeeze().item()
            
            if tokenizer.eos_id is not None and next_token_id == tokenizer.eos_id:
                break
            
            generated_ids.append(next_token_id)
            
    return tokenizer.decode(generated_ids)


def train_model(model: nn.Module, 
                dataloader: DataLoader, 
                optimizer: torch.optim.Optimizer, 
                num_epochs: int,
                tokenizer: SimpleTokenizer, # Must be provided for PAD ID
                grad_clip_value: float = 1.0): # Renamed grad_clip to avoid conflict
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs) # Causal mask is created internally by default
            
            # Reshape for loss: (batch_size * seq_len, vocab_size) and (batch_size * seq_len)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % (len(dataloader) // 5 + 1) == 0 : # Print ~5 times per epoch
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        losses.append(avg_epoch_loss)
        elapsed_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} completed in {elapsed_time:.2f}s. "
              f"Avg Loss: {avg_epoch_loss:.4f}")
        
        # Generate sample text
        # Ensure seed text is tokenizable and not empty
        sample_seed = "To be" if "To be" in tokenizer.decode(tokenizer.encode("To be")) else tokenizer.decode([tokenizer.vocab.index('T')])
        if not sample_seed: sample_seed = "A" # Fallback if T isn't in vocab

        sample_text = generate_text(model, tokenizer, sample_seed, max_length=50)
        print(f"Sample: {sample_text}\n")
            
    return losses

###########################################
#            MAIN FUNCTION                #
###########################################

def main():
    global vocab_size # Allow modification of global vocab_size based on tokenizer

    print("Initializing Quantum-Enhanced Transformer for Language Modeling...")
    
    # --- Hyperparameter adjustments for potentially faster testing if needed ---
    # global max_seq_length, batch_size, num_epochs, embedding_dim, hidden_dim, num_heads, n_qubits
    # max_seq_length = 32
    # batch_size = 4
    # num_epochs = 1
    # embedding_dim = 32
    # hidden_dim = 128
    # num_heads = 2 # Ensure embed_dim is divisible by num_heads
    # n_qubits = 2
    # print("Using reduced hyperparameters for faster testing.")
    # --- End adjustments ---


    text_corpus = create_mini_shakespeare()
    print(f"Dataset size: {len(text_corpus)} characters, approx {len(text_corpus.split())} words.")
    
    tokenizer = SimpleTokenizer(text_corpus, tokenize_chars=True)
    vocab_size = tokenizer.vocab_size # Update global vocab_size
    print(f"Vocabulary size: {vocab_size} (character-level)")
    
    dataset = TextDataset(text_corpus, tokenizer, max_seq_length)
    if len(dataset) == 0:
        print("ERROR: Dataset is empty. Text corpus might be too short for the given max_seq_length.")
        print(f"Text length: {len(tokenizer.input_ids)}, Required: > {max_seq_length + 1}")
        return
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True) # drop_last if batch size varies
    print(f"Created dataset with {len(dataset)} sequences. Dataloader provides {len(dataloader)} batches.")
    
    dropout_rate = 0.1 # Standard dropout

    print(f"\n--- Quantum Model Training (1 QHead + {num_heads-1} CHead(s) if num_heads > 1) ---")
    q_model = QuantumTransformer(
        vocab_size=vocab_size, embed_dim=embedding_dim, num_heads=num_heads,
        ff_dim=hidden_dim, num_layers=num_layers, max_seq_len=max_seq_length,
        use_quantum=True, n_qubits=n_qubits, dropout_rate=dropout_rate
    ).to(device)
    print(f"Quantum model parameters: {sum(p.numel() for p in q_model.parameters() if p.requires_grad):,}")
    
    optimizer_q = torch.optim.Adam(q_model.parameters(), lr=learning_rate)
    q_losses = train_model(q_model, dataloader, optimizer_q, num_epochs, tokenizer, grad_clip)
    
    print("\nGenerating text with trained Quantum Transformer...")
    q_generated = generate_text(q_model, tokenizer, "To be", max_length=150)
    print(f"Quantum Generated:\n{q_generated}\n")

    print(f"\n--- Classical Model Training ({num_heads} CHead(s)) ---")
    c_model = QuantumTransformer(
        vocab_size=vocab_size, embed_dim=embedding_dim, num_heads=num_heads,
        ff_dim=hidden_dim, num_layers=num_layers, max_seq_len=max_seq_length,
        use_quantum=False, n_qubits=n_qubits, dropout_rate=dropout_rate # use_quantum=False
    ).to(device)
    print(f"Classical model parameters: {sum(p.numel() for p in c_model.parameters() if p.requires_grad):,}")
    
    optimizer_c = torch.optim.Adam(c_model.parameters(), lr=learning_rate)
    c_losses = train_model(c_model, dataloader, optimizer_c, num_epochs, tokenizer, grad_clip)

    print("\nGenerating text with trained Classical Transformer...")
    c_generated = generate_text(c_model, tokenizer, "To be", max_length=150)
    print(f"Classical Generated:\n{c_generated}\n")

    if q_losses and c_losses:
        print("\nComparison of final average training losses:")
        print(f"Quantum Transformer: {q_losses[-1]:.4f}")
        print(f"Classical Transformer: {c_losses[-1]:.4f}")
    
    print("\nScript finished.")

if __name__ == "__main__":
    main()