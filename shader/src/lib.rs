#![no_std]

use spirv_std::spirv;
use spirv_std::glam::UVec3;

#[repr(C)]
pub struct BeamformingConfig {
    pub speed_of_sound: f32,
}

#[spirv(workgroup)]
static mut SHARED_SAMPLES: [f32; 64] = [0.0; 64];

#[spirv(compute(threads(64)))]
pub fn main_shader(
    #[spirv(global_invocation_id)] global_id: UVec3,
    #[spirv(local_invocation_id)] local_id: UVec3,
    #[spirv(workgroup_id)] group_id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [f32],
    #[spirv(uniform, descriptor_set = 0, binding = 2)] config: &BeamformingConfig,
) {
    let thread_id = local_id.x as usize;
    let sample_idx = group_id.x as usize;
    const NUM_CHANNELS: usize = 64;

    // 1. Each thread loads its channel's data for this specific time/location
    let global_idx = sample_idx * NUM_CHANNELS + thread_id;
    if global_idx < input.len() {
        unsafe {
            SHARED_SAMPLES[thread_id] = input[global_idx];
        }
    }

    // 2. Synchronize: Ensure all threads have finished writing to shared memory
    unsafe {
        spirv_std::arch::workgroup_barrier();
    }

    // 3. Summation scaled by Speed of Sound
    if thread_id == 0 {
        let mut sum = 0.0;
        for i in 0..NUM_CHANNELS {
            unsafe {
                sum += SHARED_SAMPLES[i];
            }
        }
        output[sample_idx] = sum * config.speed_of_sound;
    }
}
