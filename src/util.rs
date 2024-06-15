use std::borrow::Borrow;

use bevy::math::{Vec2, Vec3};

// Converts a 2D Vec3 to a Vec2
pub fn vec2(v: impl Borrow<Vec3>) -> Vec2 {
    let v = v.borrow();
    Vec2 { x: v.x, y: v.y }
}
