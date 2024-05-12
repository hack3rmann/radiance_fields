use bytemuck::{Pod, Zeroable};
use glam::*;
use serde::{Deserialize, Serialize};



/// Calculates near and far distances from ray origin to ray-box intersection points.
/// 
/// src: <https://github.com/msu-graphics-group/LiteRF/blob/main/example_tracer/example_tracer.cpp>
pub fn intersect_ray_box(ro: Vec3, rd: Vec3, box_lo: Vec3, box_hi: Vec3) -> Option<(f32, f32)> {
    let inverse_rd = 1.0 / rd;

    let lo = inverse_rd * (box_lo - ro);
    let hi = inverse_rd * (box_hi - ro);

    let max3 = |a, b, c| f32::max(a, b).max(c);
    let min3 = |a, b, c| f32::min(a, b).min(c);

    let near = max3(lo.x.min(hi.x), lo.y.min(hi.y), lo.z.min(hi.z));
    let far = min3(lo.x.max(hi.x), lo.y.max(hi.y), lo.z.max(hi.z));

    (near <= far).then_some((near, far))
}



#[repr(C)]
#[derive(Clone, Debug, PartialEq, Copy)]
#[derive(Serialize, Deserialize)]
#[derive(Pod, Zeroable)]
pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub const fn new(origin: Vec3, direction: Vec3) -> Self {
        Self { origin, direction }
    }
}

impl Default for Ray {
    fn default() -> Self {
        Self { origin: Vec3::ZERO, direction: Vec3::X }
    }
}



#[repr(C)]
#[derive(Clone, Debug, PartialEq, Copy)]
#[derive(Serialize, Deserialize)]
#[derive(Pod, Zeroable)]
pub struct Aabb {
    pub lo: Vec3,
    pub hi: Vec3,
}

impl Aabb {
    pub const fn new(lo: Vec3, hi: Vec3) -> Self {
        Self { lo, hi }
    }

    pub fn with_translation(self, offset: Vec3) -> Self {
        Self { lo: self.lo + offset, hi: self.hi + offset }
    }
}

impl Default for Aabb {
    fn default() -> Self {
        Self { lo: Vec3::ZERO, hi: Vec3::ONE }
    }
}

impl Intersect<Ray> for Aabb {
    type Output = Option<(f32, f32)>;

    fn intersect(&self, ray: &Ray) -> Self::Output {
        intersect_ray_box(ray.origin, ray.direction, self.lo, self.hi)
    }
}

impl Intersect<Aabb> for Ray {
    type Output = Option<(f32, f32)>;

    fn intersect(&self, other: &Aabb) -> Self::Output {
        other.intersect(self)
    }
}



pub trait Intersect<T> {
    type Output;

    fn intersect(&self, other: &T) -> Self::Output;
}