# UniVQ: Unified Time Series Forecasting with Vector Quantization

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)

> **UniVQ: Vector Quantization is Effective for Unified Time Series Forecasting**  
> Xiaoyuan Zhang*, Ziwen Chen*, Wei Liu*, Ming Zhu*  
> School of Electronic Information and Communications, Huazhong University of Science and Technology  
> *Equal Contribution | *Corresponding Authors

## 📖 Abstract

Time series forecasting faces significant challenges due to highly heterogeneous distributions across domains and limited data coverage of real-world scenarios. UniVQ addresses these challenges through a novel unified vector quantization framework that maps time series data into a finite discrete latent space.

**Key Innovations:**
- **Finite State Quantization (FSQ)** unifies heterogeneous distributions while preserving critical temporal patterns
- **Encoder-Decoder architecture** with unified codebook for cross-domain representation learning
- **One-stage multi-horizon forecasting** in discrete latent space
- **State-of-the-art performance** on both in-domain and zero-shot forecasting tasks

## 🚀 Key Features

- **🌐 Cross-Domain Generalization**: Pretrained on multiple domains (energy, economics, climate, transportation)
- **🔢 Discrete Latent Space**: Finite State Quantization with uniform codebook (Q=3, C=7, 2187 codes)
- **⚡ Parameter Efficient**: Only 37M parameters vs 200M+ in comparable models
- **📈 Multi-Horizon Support**: Unified prediction for horizons {96, 192, 336, 720}
- **🎯 Robust Performance**: Superior results on 6 benchmarks including Weather, Electricity, and ETT datasets

## 🏗️ Architecture
