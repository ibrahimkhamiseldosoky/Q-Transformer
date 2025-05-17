
# 🧠 Q-Transformer: Quantum-Classical Hybrid Transformer for NLP

🚀 A working prototype of a Transformer language model with **quantum attention**, developed entirely from scratch by **Ibrahim (13 years old)**.

This project integrates a **parameterized quantum circuit (PQC)** into the self-attention mechanism of a Transformer model using **PennyLane + PyTorch**, making it one of the first real implementations of a **quantum-enhanced Transformer** for natural language tasks.

---

## ⚛️ Highlights

- 🔁 **Quantum Attention Head**: Replaces classical dot-product attention with a trainable variational quantum circuit.
- 🧠 **Hybrid Multi-Head Attention**: Mixes 1 quantum head with classical attention heads.
- 💡 **End-to-End Training**: Fully differentiable model using PyTorch and PennyLane.
- 📊 **Real Benchmarking**: Compared against classical Transformer trained on Shakespearean character-level dataset.
- 🧑‍🔬 **Built by a 13-year-old independent researcher.**

---

## 🧪 Results

| Model                | Final Training Loss | Sample Output                            |
|---------------------|---------------------|-------------------------------------------|
| **Quantum Transformer** | 2.217                | `To be, Wond thans hous ato ay wavend thepr rpathee of i`             |
| **Classical Transformer** | 2.178                | `To bean, anscMrncoo uf heaknd we Dof ille An the rat  l`           |

Both models demonstrate learning and sequence generation. Quantum attention performs **competitively** and creatively, even with minimal parameter tuning.

---

## 📦 Requirements

- Python 3.8+
- `torch`
- `pennylane`
- `numpy`

Install everything with:

```bash
pip install torch pennylane numpy
````

---

## 🛠️ How to Run

```bash
python quantum_transformer.py
```

This will:

* Train the quantum transformer (3 epochs by default)
* Generate sample text
* Train a classical baseline for comparison
* Output loss metrics and final results

---

## 🧠 Architecture Overview

* `QuantumAttentionHead`: Uses PQC instead of dot-product.
* `HybridMultiHeadAttention`: Combines quantum + classical heads.
* `QuantumTransformer`: Full Transformer architecture with token embedding, positional encoding, transformer blocks, and output projection.
* `PennyLane + PyTorch`: Seamless hybrid model via `qml.qnn.TorchLayer`.

---

## 📁 Project Structure

```
quantum_transformer.py     # Main training script (everything in one file)
```

---

## 👦 About the Author

**Ibrahim**, 13 years old, from Egypt 🇪🇬
Self-taught researcher in quantum computing and AI.
Building at the edge of what’s possible.

---

## 💬 Contact & Collaboration

If you're a professor, researcher, or developer interested in this work, I’d love to connect.
Feel free to email or open an issue in this repo.

---

## 📜 License

MIT License – use, modify, and share freely.

---
