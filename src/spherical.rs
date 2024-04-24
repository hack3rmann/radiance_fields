use array_init::array_init;
use serde::{Serialize, Deserialize};
use bytemuck::{Pod, Zeroable};
use glam::*;



pub const SPHERICAL_HARMONIC_WIDTH: usize = 9;



#[repr(C)]
#[derive(Clone, Debug, PartialEq, Copy)]
#[derive(Serialize, Deserialize)]
#[derive(Pod, Zeroable)]
pub struct Cell {
    pub density: f32,
    pub sh_r: [f32; SPHERICAL_HARMONIC_WIDTH],
    pub sh_g: [f32; SPHERICAL_HARMONIC_WIDTH],
    pub sh_b: [f32; SPHERICAL_HARMONIC_WIDTH],
}

impl Cell {
    pub fn eval_sh(direction: Vec3, sh: &[f32; SPHERICAL_HARMONIC_WIDTH]) -> f32 {
        let mut sh_coeffs = {
            let [x, y, z] = direction.to_array();

            [
                0.28209479,
                -0.48860251 * y,
                0.48860251 * z,
                -0.48860251 * x,
                1.0925484 * x * y,
                -1.0925484 * y * z,
                0.31539157 * (2.0 * z * z - x * x - y * y),
                -1.0925484 * x * z,
                0.5462742 * (x * x - y * y),
            ]
        };

        let maybe_sum = sh.iter().zip(&sh_coeffs)
            .map(|(&l, &r)| l * r)
            .reduce(std::ops::Add::add);

        // # Safety
        // 
        // `reduce` returns `None` only if iterator is empty,
        // but it contains exactly `SPHERICAL_HARMONIC_WIDTH` != 0 elements.
        unsafe { maybe_sum.unwrap_unchecked() }
    }

    pub fn eval(&self, direction: Vec3) -> CellValue {
        CellValue::new(vec3(
            Self::eval_sh(direction, &self.sh_r),
            Self::eval_sh(direction, &self.sh_g),
            Self::eval_sh(direction, &self.sh_b),
        ), self.density)
    }

    pub fn trilerp(values: [&Self; 8], [x, y, z]: [f32; 3]) -> Self {
        let (nx, ny, nz) = (1.0 - x, 1.0 - y, 1.0 - z);

        *values[0b000] * nx * ny * nz
            + *values[0b001] * nx * ny * z
            + *values[0b010] * nx * y * nz
            + *values[0b011] * nx * y * z
            + *values[0b100] * x * ny * nz
            + *values[0b101] * x * ny * z
            + *values[0b110] * x * y * nz
            + *values[0b111] * x * y * z
    }
}

impl std::ops::Add for Cell {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            density: self.density + rhs.density,
            sh_r: array_init(|i| self.sh_r[i] + rhs.sh_r[i]),
            sh_g: array_init(|i| self.sh_g[i] + rhs.sh_g[i]),
            sh_b: array_init(|i| self.sh_b[i] + rhs.sh_b[i]),
        }
    }
}

impl std::ops::Mul<f32> for Cell {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            density: rhs * self.density,
            sh_r: self.sh_r.map(|x| rhs * x),
            sh_g: self.sh_g.map(|x| rhs * x),
            sh_b: self.sh_b.map(|x| rhs * x),
        }
    }
}



#[repr(C)]
#[derive(Clone, Debug, Default, PartialEq, Copy)]
#[derive(Serialize, Deserialize)]
#[derive(Pod, Zeroable)]
pub struct CellValue {
    pub color: Vec3,
    pub density: f32,
}

impl CellValue {
    pub const fn new(color: Vec3, density: f32) -> Self {
        Self { color, density }
    }
}



#[derive(Clone, Debug, Default, PartialEq, Copy, Eq, PartialOrd, Ord, Hash)]
#[derive(Serialize, Deserialize)]
pub enum Filtering {
    Nearest,
    #[default]
    Trilinear,
}



#[derive(Clone, Default, Debug, PartialEq)]
#[derive(Serialize, Deserialize)]
pub struct RadianceField {
    size: usize,
    pub cells: Vec<Cell>,
}

impl RadianceField {
    pub const fn size(&self) -> usize {
        self.size
    }

    /// Evaluates spherical harmonic by 3D index
    pub fn eval_by_index(&self, index: [usize; 3], direction: Vec3) -> Option<CellValue> {
        Some(self.get(index)?.eval(direction))
    }

    /// Evaluates spherical harmonic by a position in the [0, 1]^3 cube
    pub fn eval_near(&self, pos: Vec3, direction: Vec3) -> Option<CellValue> {
        let index = (self.size as f32 * pos)
            .to_array()
            .map(|f| f as usize);

        self.eval_by_index(index, direction)
    }

    pub fn eval_trilinear(&self, mut pos: Vec3, direction: Vec3) -> Option<CellValue> {
        const EPS: f32 = 0.01;

        if pos.x < EPS || pos.y < EPS || pos.z < EPS
            || pos.x > 1.0 - EPS || pos.y > 1.0 - EPS || pos.z > 1.0 - EPS
        {
            return None;
        }

        pos *= self.size() as f32;
        
        let lo_index = [
            pos.x.floor() as usize,
            pos.y.floor() as usize,
            pos.z.floor() as usize,
        ];

        if lo_index.iter().any(|&i| i + 1 >= self.size()) {
            return None;
        }

        let indices = [
            [lo_index[0], lo_index[1], lo_index[2]],
            [lo_index[0], lo_index[1], lo_index[2] + 1],
            [lo_index[0], lo_index[1] + 1, lo_index[2]],
            [lo_index[0], lo_index[1] + 1, lo_index[2] + 1],
            [lo_index[0] + 1, lo_index[1], lo_index[2]],
            [lo_index[0] + 1, lo_index[1], lo_index[2] + 1],
            [lo_index[0] + 1, lo_index[1] + 1, lo_index[2]],
            [lo_index[0] + 1, lo_index[1] + 1, lo_index[2] + 1],
        ];

        // # Safety
        // 
        // We manually checked that index in bounds,
        // so it is safe to get without checks
        let values = indices.map(|i| unsafe { self.get_unchecked(i) });
        let coeffs = pos.fract_gl().to_array();
        let cell = Cell::trilerp(values, coeffs);

        Some(cell.eval(direction))
    }

    /// Evaluates spherical harmonic on the 3D cube [0, 1]^3
    pub fn eval(&self, pos: Vec3, direction: Vec3, filtering: Filtering) -> Option<CellValue> {
        match filtering {
            Filtering::Nearest => self.eval_near(pos, direction),
            Filtering::Trilinear => self.eval_trilinear(pos, direction),
        }
    }

    /// Calculates index in 3D array
    pub const fn index_of(size: usize, [x, y, z]: [usize; 3]) -> usize {
        x + size * y + size * size * z
    }

    pub fn get(&self, index: [usize; 3]) -> Option<&Cell> {
        self.cells.get(Self::index_of(self.size(), index))
    }

    /// # Safety
    /// 
    /// See `std::slice::get_unchecked`.
    pub unsafe fn get_unchecked(&self, index: [usize; 3]) -> &Cell {
        self.cells.get_unchecked(Self::index_of(self.size(), index))
    }

    pub fn get_mut(&mut self, index: [usize; 3]) -> Option<&mut Cell> {
        self.cells.get_mut(Self::index_of(self.size, index))
    }

    /// # Safety
    /// 
    /// See `std::slice::get_unchecked_mut`
    pub unsafe fn get_unchecked_mut(&mut self, index: [usize; 3]) -> &mut Cell {
        self.cells.get_unchecked_mut(Self::index_of(self.size, index))
    }
}

impl std::ops::Index<[usize; 3]> for RadianceField {
    type Output = Cell;

    fn index(&self, index: [usize; 3]) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl std::ops::IndexMut<[usize; 3]> for RadianceField {
    fn index_mut(&mut self, index: [usize; 3]) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}