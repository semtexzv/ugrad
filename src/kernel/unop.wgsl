@group(0) @binding(0) var<uniform> gsz: vec3<u32>;

// [.., gsz.z, gsz.y, gsz.x] shaped tensors
@group(1) @binding(0) var<storage, read> a: array<f32>;
@group(2) @binding(0) var<storage, read_write> _out: array<f32>;

const REPEAT: u32 = 1u;
const STRIDE: u32 = 1u;

fn idx(pos: vec3<u32>, size: vec3<u32>, repeat: u32) -> u32 {
    let d4 = repeat * size.z;
    let d3 = (pos.z + d4) * size.y;
    let d2 = (pos.y + d3) * size.x;
    let d1 = (pos.x + d2) * STRIDE;
    return d1;
}

fn neg(value: f32) -> f32 {
    return 0.0 - value;
}

fn relu(value: f32) -> f32 {
    return max(0.0, value);
}

fn gtz(value: f32) -> f32 {
    return sign(max(0.0, value));
}

fn reci(value: f32) -> f32 {
    return 1.0 / value;
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (idx(gid, gsz, 0u) >= gsz.x * gsz.y * gsz.z) {
        return;
    }


    _out[idx(gid, gsz, 0u)] = __OP(a[idx(gid, gsz, 0u)]);
}
