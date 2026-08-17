# Instructions for how to enable PyTorch D2D copying of tensors

## 1. Get the source at the exact tag
 
```bash
git clone --branch v27.0.2.0 --recurse-submodules https://github.com/gfx-rs/wgpu-native.git
cd wgpu-native
```
 
(`--recurse-submodules` matters: `ffi/webgpu-headers` is a submodule needed by build.rs.)
 
## 2. Apply the patch
 
1. Copy `exportable.rs` into `src/exportable.rs`.
2. In `src/lib.rs`, next to the other module declarations near the top
   (`pub mod conv;` etc.), add: 
```rust
mod exportable;
```
3. In `Cargo.toml`, under the existing [dependencies] section (where log, parking_lot, etc. live), add:
```bash
ash = "0.38"
```
 
## 3. Build
 
Requires Rust >= 1.82 (`rustup update stable`) and clang/llvm for bindgen.

### Installing `Rust` 

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# accept defaults, then:
source ~/.cargo/env
rustc --version   # should be well past 1.82
```

### Installing `libclang.so` on Fedora

```bash
sudo dnf install clang-devel clang-libs llvm-libs
ls /usr/lib64/libclang*        
export LIBCLANG_PATH=/usr/lib64
```
 
```bash
cargo build --release
# artifact: target/release/libwgpu_native.so
```
 
Sanity: `nm -D target/release/libwgpu_native.so | grep Exportable` should show
both new symbols.
 
## 4. Point wgpu-py at it
 
```bash
export WGPU_LIB_PATH=/path/to/wgpu-native/target/release/libwgpu_native.so
python -c "import wgpu; a=wgpu.gpu.request_adapter_sync(power_preference='high-performance'); print(a.info)"
```


