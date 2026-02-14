use spirv_builder::{SpirvBuilder, SpirvMetadata, ModuleResult};

fn main() {
    let result = SpirvBuilder::new("shader", "spirv-unknown-vulkan1.1")
        .spirv_metadata(SpirvMetadata::Full)
        .build()
        .unwrap();
    
    // We can use the module path in our main code
    let path = match &result.module {
        ModuleResult::SingleModule(path) => path,
        ModuleResult::MultiModule(_) => panic!("Expected single module"),
    };
    println!("cargo:rustc-env=SHADER_PATH={}", path.display());
}
