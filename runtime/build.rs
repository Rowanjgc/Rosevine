//! With the `hip` feature, compile `hip/rvhip.hip` into a shared library with the system `hipcc`
//! (ROCm, gfx1101 by default; override with `ROSEVINE_HIP_ARCH`). Without it, nothing is built.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=hip/rvhip.hip");
    println!("cargo:rerun-if-changed=hip/rvhip.h");
    println!("cargo:rerun-if-env-changed=ROSEVINE_HIP_ARCH");
    if env::var_os("CARGO_FEATURE_HIP").is_none() {
        return;
    }
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let arch = env::var("ROSEVINE_HIP_ARCH").unwrap_or_else(|_| "gfx1101".to_string());
    let lib = out.join("librvhip.so");
    let status = Command::new("hipcc")
        // No FMA contraction: elementwise kernels then follow the CPU executor's operation order.
        .args([
            &format!("--offload-arch={arch}"),
            "-O3",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            "-Wall",
            "-Werror",
            "-o",
        ])
        .arg(&lib)
        .arg("hip/rvhip.hip")
        .status()
        .expect("hipcc not found (the hip feature needs ROCm)");
    assert!(status.success(), "hipcc failed");
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=rvhip");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out.display());
}
