struct Params {
  M: u32,
  N: u32,
  K: u32,

  alpha: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// MxN matrix
@group(1) @binding(0) var<storage,read> a: array<f32>;
// NxK matrix
@group(1) @binding(1) var<storage,read> b: array<f32>;
// MxK matrix
@group(2) @binding(0) var<storage,read_write> out: array<f32>;


@compute
@workgroup_size(8,8,1)
fn main( @builtin(global_invocation_id) global_id: vec3<u32>) {
  var M: u32 = params.M;
  var N: u32 = params.N;
  var K: u32 = params.K;
  var x: u32 = global_id.x;
  var y: u32 = global_id.y;
  if (x >= N || y >= M) {
    return;
  }

  var sum: f32 = 0.0;
  for(var k: u32 = 0u; k < K; k = k + 1u) {
    sum = a[y * K + k] * b[k * N + x] + sum;
  }
  out[x + y * N] = sum * params.alpha;
}