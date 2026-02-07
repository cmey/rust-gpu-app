// GPU Compute Shader (WGSL)
// This shader uses the same struct layout as the CPU DataElement

// Shared struct definition - matches the CPU DataElement struct
struct DataElement {
    value: f32,
    multiplier: f32,
}

@group(0) @binding(0) var<storage, read> input: array<DataElement>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // Perform computation: multiply value by multiplier
    output[index] = input[index].value * input[index].multiplier;
}
