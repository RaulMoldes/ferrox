# Ferrox

**Ferrox** is a lightweight, CPU-based automatic differentiation engine written in **Rust**, inspired by [PyTorch's autograd](https://pytorch.org/docs/stable/autograd.html) and the [CMU Deep Learning Systems course](https://dlsyscourse.org/). It supports **reverse-mode automatic differentiation (backpropagation)** and uses [`ndarray`](https://crates.io/crates/ndarray) as its tensor computation backend.

---

## Features

- Reverse-mode automatic differentiation
- Dynamic computation graph construction
- Scalar and tensor support via `ndarray`
- Operator overloading for intuitive expressions
- Gradient accumulation
- Graph visualization (requires GraphViz installed).
- Written 100% in safe Rust

Note: as of now, I am extending the project with a custom nn library on top of the engine.

---

## Prerequisites.

First, ensure you have Rust and Cargo installed on your system.
You can install them using rustup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
To run the project:

```bash
cargo run
```
To run unit tests:

```bash
cargo test
```

For visualizations, you will need Graphviz.

**Install Graphviz with**:

On Ubuntu/Debian:

```bash
sudo apt-get install graphviz
```

On macOS (using Homebrew):

```shell
brew install graphviz
```
On Windows (using Chocolatey):

```shell
choco install graphviz
```
You can verify the installation with:

```bash
dot -V
```

By default, graph visualizations are saved on the `imgs/` folder.

## Example usage:
Examples are in the `examples/` folder. To run them:

```bash
cargo run --example ferrox_example
```

## Future improvements.
Currently investigating on how to integrate this with a CUDA based backend for hardware acceleration.