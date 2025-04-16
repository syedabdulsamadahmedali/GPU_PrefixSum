# ⚡ GPU Prefix Sums in Julia: Hillis & Steele vs. Blelloch

This project presents a comparative analysis of the **Hillis & Steele** and **Blelloch** parallel prefix sum (scan) algorithms. Implemented in **Julia** using **CUDA.jl**, it explores the performance of both algorithms on CPU and GPU across diverse datasets such as synthetic arrays, MNIST images, and genomic sequences.

---

## 📌 Key Features

- 📈 Performance benchmarking on CPU vs. GPU
- 🧬 Application to genomic data (rank table construction)
- 🧪 Diverse test cases (sparse, sorted, random, skewed)
- 📊 Visualizations of speedup and execution time
- 💡 Implemented in Julia using CUDA.jl and BioSequences.jl

---

## 🧠 Algorithms

### Hillis & Steele
Inclusive prefix sum using a step-wise doubling approach. High simplicity and CPU/GPU speedup.

### Blelloch
Work-efficient exclusive prefix sum using an upsweep-downsweep tree strategy. Highly scalable.

---

## 🧪 Data Types

- Random, Sparse, Skewed, Sorted, Alternating
- Real-world hybrid synthetic datasets
- Flattened MNIST images
- Synthetic genomic sequences

---

## 📁 Repository Structure

```
gpu-prefix-sum-julia/
├── algorithms/
│   ├── hillis_steele.jl
│   ├── blelloch.jl
│   ├── kernels.jl
├── applications/
│   └── genomic/
│       ├── genomic_main.jl
│       └── genomic_utils.jl
├── benchmarks/
│   ├── main.jl
│   ├── utils.jl
│   └── common.jl
├── plots/   # Auto-generated performance plots
└── README.md
```

---

## 🚀 How to Run

### 1. Install Requirements

- Julia
- NVIDIA GPU (8 GB+ VRAM recommended)
- Packages: CUDA.jl, Plots.jl, BioSequences.jl, MLDatasets.jl, Random.jl, Distributions.jl

```julia
] activate .
] add CUDA Plots BioSequences MLDatasets Random Distributions
```

### 2. Run Benchmarks

```julia
julia> include("benchmarks/main.jl")
```

Generates performance plots for all dataset types and implementations.

### 3. Run Genomic Demo

```julia
julia> include("applications/genomic/genomic_main.jl")
```

Runs DNA sequence simulations, builds rank tables using prefix sums, and visualizes execution time and speedups.

---

## 📊 Performance Summary

| Dataset        | Hillis-Steele Speedup (CPU vs GPU) | Blelloch Speedup | GPU-to-GPU Comparison |
|----------------|-------------------------------------|-------------------|------------------------|
| Random         | 10x                                | 6x                | Blelloch 1.8x faster   |
| Sparse         | 16x                                | 6x                | Blelloch 1.6x faster   |
| Skewed         | 33x                                | 6x                | Blelloch 1.7x faster   |
| Sorted         | 25x                                | 6x                | Blelloch 1.6x faster   |
| Real-world     | 18x                                | 6x                | Blelloch 1.7x faster   |
| MNIST          | 11x                                | 6x                | Blelloch 1.8x faster   |

> Blelloch GPU has better raw GPU performance, while Hillis & Steele shows higher CPU vs GPU speedups.

---

## 🧬 Genomic Application

- **Goal**: Build rank tables from DNA sequences.
- **Steps**:
  1. Generate DNA sequence using `BioSequences.jl`
  2. Encode each base (A, C, G, T) as binary arrays
  3. Apply prefix scan to generate cumulative counts
  4. Benchmark all four algorithm versions (CPU/GPU)

---

## 📈 Example Visualization

![Example Plot](https://raw.githubusercontent.com/yourusername/gpu-prefix-sum-julia/main/plots/genomic_speedup.png)

---

## 📄 License

MIT License. See `LICENSE` file.

---

## 🙋‍♂️ Author

**Your Name**  
GitHub: [@yourusername](https://github.com/yourusername)

---

> For optimal performance at scale, hybrid strategies (e.g., Hillis & Steele for small subarrays + Blelloch for tree-level scans) are worth exploring.
