# 🧠 Micrograd — A Tiny Autograd Engine in Python

A minimal, educational implementation of **reverse-mode automatic differentiation** (backpropagation) built from scratch in Python. Inspired by pytorch library, this project builds a scalar-valued autograd engine and uses it to train a small Multi-Layer Perceptron (MLP).

---

## 📌 Overview

This project demonstrates how modern deep learning frameworks like PyTorch compute gradients under the hood. Everything is built on top of a single `Value` class that wraps a scalar number and keeps track of its computational history, enabling automatic gradient computation through the chain rule.

The notebook (`engine.ipynb`) walks through:

1. Building the `Value` autograd engine
2. Visualizing computation graphs with Graphviz
3. Verifying gradients against PyTorch
4. Implementing `Neuron`, `Layer`, and `MLP` classes
5. Training an MLP with gradient descent

---

## 🗂️ Project Structure

```
micrograd/
├── engine.ipynb    # Main notebook: autograd engine + neural network demo
└── README.md       # Project documentation
```

---

## ⚙️ Core Concepts

### `Value` — The Autograd Engine

The `Value` class is the heart of this project. It wraps a scalar and records every mathematical operation applied to it, forming a **dynamic computation graph**. Calling `.backward()` on the output node propagates gradients all the way back to the inputs via **topological sort + chain rule**.

**Supported operations:**

| Operation | Method/Operator |
|-----------|----------------|
| Addition | `+`, `__add__`, `__radd__` |
| Multiplication | `*`, `__mul__`, `__rmul__` |
| Power | `**`, `__pow__` |
| Division | `/`, `__truediv__` |
| Negation | `-`, `__neg__` |
| Subtraction | `-`, `__sub__` |
| Exponential | `.exp()` |
| Tanh activation | `.tanh()` |
| Backpropagation | `.backward()` |

```python
from engine import Value

x1 = Value(2.0, label='x1')
w1 = Value(-3.0, label='w1')
b  = Value(6.881, label='b')

o = (x1 * w1 + b).tanh()
o.backward()

print(x1.grad)  # ∂o/∂x1
print(w1.grad)  # ∂o/∂w1
```

---

### Computation Graph Visualization

Using [Graphviz](https://graphviz.org/), the notebook renders the full computation graph of a neuron, showing each node's **data value** and its **computed gradient** after backpropagation.

```python
draw_dot(o)  # renders an SVG computation graph
```

---

### Neural Network Classes

Built entirely on top of `Value`, these classes implement a fully functional MLP:

| Class | Description |
|-------|-------------|
| `Neuron(nin)` | A single neuron with `nin` inputs, random weights & bias, `tanh` activation |
| `Layer(nin, nout)` | A layer of `nout` neurons |
| `MLP(nin, nouts)` | A multi-layer perceptron: stacks layers defined by `nouts` |

```python
model = MLP(3, [4, 4, 1])   # 3 inputs → [4, 4] hidden → 1 output
print(len(model.parameters()))  # 41 trainable parameters
```

---

### Training Loop

The MLP is trained on a small binary classification dataset using **mean squared error loss** and manual gradient descent:

```python
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # target labels

for k in range(30):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # Backward pass
    for p in model.parameters():
        p.grad = 0.0       # zero gradients before accumulation
    loss.backward()

    # Gradient descent update
    for p in model.parameters():
        p.data += -0.05 * p.grad

    print(k, loss.data)
```

After 30 iterations the loss drops from **~0.071** to **~0.015**, and predictions approach the target values (`±1.0`).

---

## ✅ Gradient Verification with PyTorch

The notebook includes a side-by-side check confirming that the custom `Value` engine produces **identical gradients** to PyTorch's autograd:

```
x2  grad: 0.5000   (PyTorch: 0.5000 ✓)
w2  grad: 0.0000   (PyTorch: 0.0000 ✓)
x1  grad: -1.5000  (PyTorch: -1.5000 ✓)
w1  grad: 1.0000   (PyTorch: 1.0000 ✓)
```

---

## 🛠️ Requirements

| Package | Purpose |
|---------|---------|
| `numpy` | Numerical utilities |
| `matplotlib` | Plotting |
| `graphviz` | Computation graph visualization |
| `torch` | Gradient verification only |

Install dependencies:

```bash
pip install numpy matplotlib graphviz torch
```

> **Note:** You also need the [Graphviz system binary](https://graphviz.org/download/) installed and added to your `PATH` for graph rendering to work.

---

## 🚀 Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rabobahago/micrograd.git
   cd micrograd
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy matplotlib graphviz torch
   ```

3. **Open the notebook:**
   ```bash
   jupyter notebook engine.ipynb
   ```

4. **Run all cells** to follow the full implementation from the autograd engine to a trained neural network.

---

## 🎓 Learning Goals

This project is ideal for understanding:

- How **backpropagation** works at the scalar level
- How PyTorch-style autograd graphs are built dynamically
- How `grad`, `_backward`, and `_prev` tie together in reverse-mode AD
- How simple building blocks (`Value`) compose into full neural networks

---

## 📖 References

- [Andrej Karpathy — Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html)
- [Andrej Karpathy — micrograd (GitHub)](https://github.com/karpathy/micrograd)
- [The spelled-out intro to neural networks and backpropagation (YouTube)](https://www.youtube.com/watch?v=VMj-3S1tku0)

---

## 📄 License

This project is for educational purposes. Feel free to use, modify, and share it.
