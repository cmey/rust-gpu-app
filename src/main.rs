use bytemuck::{Pod, Zeroable};

const NUM_CHANNELS: usize = 64;
const NUM_SAMPLES: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BeamformingConfig {
    speed_of_sound: f32,
}

fn main() {
    pollster::block_on(run());
}

async fn run() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .expect("Failed to find GPU adapter");

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor::default(),
            None,
        )
        .await
        .expect("Failed to create device");

    println!("Using GPU: {:?}", adapter.get_info().name);

    let config = BeamformingConfig { speed_of_sound: 1540.0 };
    let mut input_data = vec![0.0f32; NUM_CHANNELS * NUM_SAMPLES];
    for s in 0..NUM_SAMPLES {
        for c in 0..NUM_CHANNELS {
            if s == 8 { input_data[s * NUM_CHANNELS + c] = 1.0; }
        }
    }

    let results = execute_gpu_compute(&device, &queue, &input_data, config).await;

    println!("\nBeamformed Output (Sample 8 pulse):");
    for (i, result) in results.iter().enumerate() {
        if *result > 0.0 {
            println!("  Point [{:2}]: sum = {:8.1}", i, result);
        }
    }
}

async fn execute_gpu_compute(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_data: &[f32],
    config: BeamformingConfig,
) -> Vec<f32> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::util::make_spirv(include_bytes!(env!("SHADER_PATH"))),
    });

    let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (input_data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (NUM_SAMPLES * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let config_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 4,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (NUM_SAMPLES * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(input_data));
    queue.write_buffer(&config_buffer, 0, bytemuck::bytes_of(&config));

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: config_buffer.as_entire_binding() },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main_shader",
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(NUM_SAMPLES as u32, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, (NUM_SAMPLES * 4) as u64);
    queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |res| tx.send(res).unwrap());

    device.poll(wgpu::Maintain::Wait);
    rx.receive().await.unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    result
}

