fn main() {
    // Check if Python environment with onnxruntime and numpy is available for the ONNX test
    println!("cargo::rustc-check-cfg=cfg(no_python_env)");

    let has_python_deps = std::process::Command::new("python3")
        .args(&["-c", "import onnxruntime, numpy"]) 
        .status()
        .map(|status| status.success())
        .unwrap_or(false);

    if !has_python_deps {
        println!("cargo:rustc-cfg=no_python_env");
    }
}
