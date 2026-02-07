// GPU Compute Shader (WGSL)
// This shader uses the same struct layout as the CPU DataElement

// Shared struct definition - matches the CPU DataElement struct
struct DataElement {
    value: f32,
    multiplier: f32,
}

@group(0) @binding(0) var<storage, read> input: array<DataElement>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

// Shared memory (workgroup storage)
// This memory is shared between all threads in the same workgroup.
// It is much faster than storage memory (like a L1 cache).
// The size must be a constant known at compile time.
var<workgroup> cached_multipliers: array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let index = global_id.x;
    let local_index = local_id.x;
    
    // Safety check for array bounds
    let num_elements = arrayLength(&input);
    if (index >= num_elements) {
        return;
    }

    // Exploit Shared Memory:
    // 1. Cooperative loading: threads load data from global memory into shared memory
    cached_multipliers[local_index] = input[index].multiplier;
    
    // 2. Synchronize: Ensure all threads in the workgroup have finished their writes
    // to shared memory before any thread proceeds to read from it.
    workgroupBarrier();
    
    // 3. Compute: Use the data from shared memory
    // In this simple example, we're just reading our own value back, 
    // but in complex algorithms like blur or prefix-sum, threads would read 
    // each other's shared data.
    output[index] = input[index].value * cached_multipliers[local_index];
}
