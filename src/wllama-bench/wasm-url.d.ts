// Vite `?url` asset import (wllama's llama.cpp WASM build). Mirrors the
// `*.wgsl?raw` declaration in src/compiler/shaders.d.ts — the repo's tsconfig
// sets `types: ["@webgpu/types"]`, so vite/client's ambient module decls
// aren't in scope.
declare module '*.wasm?url' {
  const url: string
  export default url
}
