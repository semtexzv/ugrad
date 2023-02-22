struct Params {
  M: u32,
  N: u32,

  SM: u32,
  SN: u32,
}

//@group(0) @binding(0) var<uniform> gsz: vec3<f32>;
@group(0) @binding(0) var<uniform> params: Params;

// MxN matrix
@group(1) @binding(0) var<storage,read> a: array<f32>;

// NxM matrix
@group(2) @binding(0) var<storage,read_write> out: array<f32>;

@compute
@workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  var M: u32 = params.M;
  var N: u32 = params.N;
  var x: u32 = gid.x;
  var y: u32 = gid.y;
  if (x >= N || y >= M) {
    return;
  }

  for(var k: u32 = 0u; k < K; k = k + 1u) {
    sum = a[y * K + k] * b[k * N + x] + sum;
  }
  out[x + y * N] = sum * params.alpha;
}