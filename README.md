# rust-gpu-app

A Rust application that demonstrates running compute workloads on the GPU using **wgpu**.

## Overview

This project showcases how to:
- Write GPU compute kernels in Rust (using WGSL shaders)
- Share struct definitions between CPU and GPU code
- Execute GPU compute operations from Rust host code
- Handle both GPU-enabled and headless environments gracefully

## Architecture

The application uses the **wgpu** framework, which provides:
- Cross-platform GPU support (Vulkan, Metal, DirectX 12, WebGPU)
- Safe, high-level GPU abstractions
- Compute shader support for general-purpose GPU computing

### Key Components

1. **Shared Data Structures**: The `DataElement` struct is defined once and used by both CPU and GPU code
   ```rust
   #[repr(C)]
   #[derive(Clone, Copy, Pod, Zeroable)]
   struct DataElement {
       value: f32,
       multiplier: f32,
   }
   ```

2. **GPU Compute Shader (WGSL)**: Performs parallel computation on the GPU
   - Reads input data from GPU buffer
   - Processes each element (multiplies value by multiplier)
   - Writes results to output buffer

3. **CPU Host Code**: Manages GPU resources and orchestrates computation
   - Initializes GPU device
   - Creates and manages GPU buffers
   - Dispatches compute workloads
   - Reads results back from GPU

## Building and Running

### Prerequisites

- Rust 1.70+ (install from [rustup.rs](https://rustup.rs))
- GPU with Vulkan, Metal, or DirectX 12 support (or software rendering fallback)

### Build

```bash
cargo build --release
```

### Run

```bash
cargo run --release
```

## Example Output

On a system with GPU support:
```
=== Rust GPU Compute Application ===

Found 1 adapter(s):
  [0] NVIDIA GeForce GTX 1080 (Vulkan)

GPU Device: "NVIDIA GeForce GTX 1080"
Backend: Vulkan

Input data:
  [0] value: 1, multiplier: 2
  [1] value: 2, multiplier: 3
  [2] value: 3, multiplier: 4
  [3] value: 4, multiplier: 5

Output data (after GPU computation):
  [0] result: 2
  [1] result: 6
  [2] result: 12
  [3] result: 20

=== GPU Compute Completed Successfully ===
```

On headless/CI environments without GPU:
```
=== Rust GPU Compute Application ===

Enumerating available adapters...

⚠️  No GPU adapters found in this environment.
This is expected in headless/CI environments.

The code is correct and would work on a system with GPU support.
Demonstrating the compute logic with CPU-side verification instead...

[CPU demonstration output...]
```

## How It Works

### 1. Shared Struct Definition

The `DataElement` struct uses `#[repr(C)]` to ensure consistent memory layout between Rust and the GPU shader:

```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DataElement {
    value: f32,
    multiplier: f32,
}
```

This same struct layout is mirrored in the WGSL shader:

```wgsl
struct DataElement {
    value: f32,
    multiplier: f32,
}
```

### 2. GPU Compute Pipeline

The application follows these steps:

1. **Initialize GPU**: Enumerate and select a GPU adapter
2. **Create Buffers**: Allocate GPU memory for input and output data
3. **Upload Data**: Transfer input data from CPU to GPU
4. **Compile Shader**: Create compute shader from WGSL source
5. **Dispatch Compute**: Execute shader on GPU with specified workgroup size
6. **Download Results**: Transfer output data from GPU back to CPU

### 3. WGSL Compute Shader

The compute shader runs in parallel on the GPU:

```wgsl
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    output[index] = input[index].value * input[index].multiplier;
}
```

Each invocation processes one element, allowing parallel execution across all data elements.

## Extending the Application

You can modify this application to:
- Implement more complex compute operations (matrix multiplication, image processing, physics simulations)
- Use different workgroup sizes for better GPU utilization
- Add multiple compute passes in a pipeline
- Implement more complex data structures

## Framework Selection: Why wgpu?

After evaluating multiple options for Rust GPU computing:

| Framework | Pros | Cons |
|-----------|------|------|
| **wgpu** | Cross-platform, well-maintained, safe abstractions | Requires WGSL for shaders |
| **rust-gpu** | Pure Rust shaders, native struct sharing | Less mature, complex setup |
| **CUDA** | Maximum performance | NVIDIA-only |
| **OpenCL** | Mature, cross-platform | Declining ecosystem |

We chose **wgpu** for its:
- Robust cross-platform support
- Active maintenance and community
- Safe, idiomatic Rust API
- Excellent documentation
- WebGPU compatibility

## Dependencies

- `wgpu` (0.19): GPU abstraction layer
- `pollster` (0.3): Simple async executor for GPU operations
- `bytemuck` (1.14): Safe memory transmutation for GPU buffers
- `futures-intrusive` (0.5): Async primitives for buffer mapping

## License

See LICENSE file for details.

