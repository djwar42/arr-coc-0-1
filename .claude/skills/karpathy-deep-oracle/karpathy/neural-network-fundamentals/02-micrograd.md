# micrograd - A Tiny Automatic Differentiation Engine

**Building Backpropagation from Scratch in 150 Lines**

---

## What is micrograd?

micrograd is Karpathy's minimal automatic differentiation engine - the core technology behind PyTorch and TensorFlow, implemented in pure Python.

**What it does:**
- Tracks operations on values
- Automatically computes gradients
- Enables backpropagation
- ~150 lines of beautiful, hackable code

**What it doesn't do:**
- GPU acceleration
- Tensor operations
- Pre-built optimizers
- Complex abstractions

*Perfect for learning. Useless for production.*

---

## The Core Idea: Value Objects

Every number becomes an object that remembers:
1. Its value (data)
2. Its gradient (grad)
3. How it was created (_prev)
4. How to backpropagate (_backward)

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
```

---

## Example: Addition

```python
a = Value(2.0)
b = Value(3.0)
c = a + b  # c.data = 5.0

# Forward pass: easy
print(c.data)  # 5.0

# Backward pass: automatic
c.grad = 1.0  # Start with gradient of 1
c._backward()  # Propagate to children

print(a.grad)  # 1.0  (dc/da = 1)
print(b.grad)  # 1.0  (dc/db = 1)
```

**Key insight:** Each operation stores how to compute its gradient.

---

## Implementing Operations

### Addition

```python
def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
        # Gradient of addition distributes equally
        self.grad += 1.0 * out.grad
        other.grad += 1.0 * out.grad

    out._backward = _backward
    return out
```

### Multiplication

```python
def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
        # Gradient of multiplication: d(xy)/dx = y
        self.grad += other.data * out.grad
        other.grad += self.data * out.grad

    out._backward = _backward
    return out
```

### ReLU (Activation)

```python
def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

    def _backward():
        # Gradient is 1 if input > 0, else 0
        self.grad += (out.data > 0) * out.grad

    out._backward = _backward
    return out
```

---

## Building a Neuron

```python
class Neuron:
    def __init__(self, num_inputs):
        self.weights = [Value(random.uniform(-1, 1))
                        for _ in range(num_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, inputs):
        # Weighted sum: w1*x1 + w2*x2 + ... + bias
        activation = sum((wi * xi for wi, xi in zip(self.weights, inputs)),
                         self.bias)
        return activation.relu()
```

---

## Building a Layer

```python
class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, inputs):
        outputs = [neuron(inputs) for neuron in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs
```

---

## Building a Network

```python
class MLP:  # Multi-Layer Perceptron
    def __init__(self, num_inputs, layer_sizes):
        sizes = [num_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i+1])
                       for i in range(len(layer_sizes))]

    def __call__(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def parameters(self):
        return [p for layer in self.layers
                for neuron in layer.neurons
                for p in neuron.weights + [neuron.bias]]
```

---

## Training Example

```python
# Create network: 3 inputs → 4 hidden → 4 hidden → 1 output
model = MLP(3, [4, 4, 1])

# Training data
xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]]
ys = [1.0, -1.0, -1.0, 1.0]  # Targets

# Training loop
learning_rate = 0.01
for epoch in range(100):
    # Forward pass
    ypred = [model(x) for x in xs]

    # Calculate loss (mean squared error)
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # Backward pass
    for p in model.parameters():
        p.grad = 0.0  # Zero gradients

    loss.backward()  # Compute gradients

    # Update weights
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
```

---

## The Backward Pass Implementation

**Key challenge:** How to call `_backward()` on all operations in the right order?

**Solution:** Topological sort!

```python
def backward(self):
    # Build topological order (children before parents)
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    build_topo(self)

    # Initialize gradient at output
    self.grad = 1.0

    # Propagate gradients backwards
    for node in reversed(topo):
        node._backward()
```

**Why topological sort?**
- Ensures parents are processed before children
- Gradients flow backwards correctly
- Each node accumulates gradients from all paths

---

## Visualization: Computation Graphs

micrograd can draw its computation graphs:

```
         x1
          \
  w1 ━━━━━ * ━━┐
          /      \
        x2        + ━━━ ReLU ━━━ output
                 /
              bias
```

Each operation becomes a node. Gradients flow backwards through edges.

---

## Why micrograd Matters

**Educational value:**
- See exactly how automatic differentiation works
- Understand backpropagation deeply
- No magic, no hidden complexity

**Practical value:**
- Debug gradient issues with confidence
- Understand PyTorch/TensorFlow internals
- Implement custom operations

> "The goal here is to demystify this process and make it as transparent as possible. It's not magic - it's just calculus and clever bookkeeping."
>
> — *Philosophy behind micrograd*

---

## From micrograd to PyTorch

**micrograd:**
```python
x = Value(2.0)
y = Value(3.0)
z = x * y + x
z.backward()
```

**PyTorch:**
```python
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = x * y + x
z.backward()
```

*Same API. Same concepts. Just faster and with more features.*

---

## Common Mistakes

**1. Forgetting to zero gradients**
```python
# ❌ WRONG
loss.backward()
optimizer.step()

# ✅ CORRECT
optimizer.zero_grad()  # Or manually: p.grad = 0.0
loss.backward()
optimizer.step()
```

**2. Not initializing output gradient**
```python
# ❌ WRONG
loss.backward()  # Who sets loss.grad?

# ✅ CORRECT (micrograd does this)
loss.grad = 1.0
loss.backward()
```

**3. Wrong gradient accumulation**
```python
# Gradients ACCUMULATE (+=), they don't replace (=)
self.grad += out.grad  # ✅ Correct
self.grad = out.grad   # ❌ Wrong (loses previous gradients)
```

---

## Extending micrograd

**Add more operations:**
- Division: `d(x/y)/dx = 1/y`
- Power: `d(x^n)/dx = n*x^(n-1)`
- Exp: `d(e^x)/dx = e^x`
- Log: `d(log(x))/dx = 1/x`

**Add more activations:**
- Sigmoid: `σ(x) = 1/(1+e^(-x))`
- Tanh: `tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))`
- Softmax: Requires tensor operations

**Add optimizers:**
- SGD with momentum
- Adam
- RMSProp

---

## The Full Picture

```
micrograd (~150 lines)
    ↓
Build neural networks (neurons, layers, MLPs)
    ↓
Train on simple datasets (XOR, circles)
    ↓
Understand PyTorch/TensorFlow internals
    ↓
Build production systems
```

**Start simple. Build intuition. Scale gradually.**

---

## Next Steps

**Practice:**
1. Implement micrograd yourself (don't copy-paste!)
2. Train on XOR dataset
3. Add new operations (division, power)
4. Visualize computation graphs

**Move on:**
- [03-gradient-descent.md](03-gradient-descent.md) - Optimization algorithms
- [../gpt-architecture/](../gpt-architecture/) - Scale to transformers

**Primary sources:**
- `source-documents/34-The spelled-out intro to neural networks.md` - Complete walkthrough
- GitHub: karpathy/micrograd - Full implementation

---

## Karpathy's Philosophy

> "I want you to see all the gears turning. No black boxes. No magic. Just math and code."

micrograd is minimal by design:
- ✅ Easy to understand
- ✅ Easy to modify
- ✅ Teaches fundamentals
- ❌ Not for production
- ❌ Not optimized
- ❌ Not feature-complete

*And that's exactly the point.*
