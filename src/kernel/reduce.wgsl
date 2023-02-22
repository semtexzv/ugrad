struct Params {
    // If we have higher dimensions, repeat allows us to
    repeat: u32,
    // If we have lower dimensions, we can use stride to work across them
    stride: u32,
    // How many operations we have to do
    size: u32,
    init: f32,
}

@group(0) @binding(0) var<uniform> gsz: vec3<u32>;

@group(0) @binding(1) var<uniform> params: Params;

@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(2) @binding(0) var<storage, read_write> out: array<f32>;

// Strided indexing
fn idx(pos: vec3<u32>, size: vec3<u32>, repeat: u32, stride: u32) -> u32 {
    let d4 = repeat * size.z;
    let d3 = (pos.z + d4) * size.y;
    let d2 = (pos.y + d3) * size.x;
    let d1 = (pos.x + d2) * stride;
    return d1;
}

fn add(a: f32, b: f32) -> f32 {
    return a + b;
}

// Pow is builtin

@compute
@workgroup_size(8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var r: u32 = 0u; r < params.repeat; r++) {
        if (idx(gid, gsz, r, params.stride) >= gsz.x * gsz.y * gsz.z * params.repeat) {
            return;
        }

        var tmp: f32 = params.init;
        for (var i: u32 = 0u; i < params.size; i++) {
            tmp = __OP(tmp, a[idx(gid, gsz, r, params.stride)]);
        }

        out[idx(gid, gsz, r, 1u)] = tmp;
    }
}
