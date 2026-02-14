# rust-gpu-app

A Rust application that demonstrates running compute workloads on the GPU using **Rust-GPU** and **wgpu**.

## Overview

This project showcases how to execute GPU compute operations using pure Rust for both the kernel and the host code.

## Architecture

- **Rust-GPU (`shader/` crate)**: Compute kernels written in Rust and compiled to SPIR-V.
- **wgpu (Host crate)**: GPU initialization, memory management, and shader execution.

## Getting Started

### Prerequisites

- Rust (nightly required for Rust-GPU)
- GPU with Vulkan, Metal, or DX12 support

### Run

```bash
cargo run --release
```

## How It Works

1. **Shared Logic**: The host and shader share struct definitions via `#[repr(C)]`.
2. **Automatic Compilation**: `build.rs` compiles the shader crate to SPIR-V using `spirv-builder`.
3. **GPU Dispatch**: The host loads the SPIR-V module and dispatches a compute pipeline.

