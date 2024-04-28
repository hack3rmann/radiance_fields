use glam::*;



/// Calculates near and far distances from ray origin to ray-box intersection points.
/// 
/// src: <https://github.com/msu-graphics-group/LiteRF/blob/main/example_tracer/example_tracer.cpp>
pub fn intersect_ray_box(ro: Vec3, rd: Vec3, box_lo: Vec3, box_hi: Vec3) -> Option<(f32, f32)> {
    let lo = (box_lo - ro) / rd;
    let hi = (box_hi - ro) / rd;

    let max3 = |a, b, c| f32::max(a, b).max(c);
    let min3 = |a, b, c| f32::min(a, b).min(c);

    let near = max3(lo.x.min(hi.x), lo.y.min(hi.y), lo.z.min(hi.z));
    let far = min3(lo.x.max(hi.x), lo.y.max(hi.y), lo.z.max(hi.z));

    (near <= far).then_some((near, far))
}