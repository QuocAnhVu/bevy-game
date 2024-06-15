use std::array;

use crate::{util::vec2, Agent, Next, Prev};
use bevy::prelude::*;
use itertools::{Either, Itertools};

pub struct ChunkPlugin;
impl Plugin for ChunkPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, init_index)
            .add_systems(PostUpdate, transfer_agent);
        app.add_systems(Update, draw_chunks);
    }
}

fn draw_chunks(mut gizmos: Gizmos) {
    for (x, y) in (0..ChunkIndex::X_RADIUS * 2).cartesian_product(0..ChunkIndex::Y_RADIUS * 2) {
        let x = x as f32;
        let y = y as f32;
        let pos = Vec2 {
            x: (x - (ChunkIndex::X_RADIUS as f32) + 0.5) * Chunk::CHUNK_SIZE,
            y: (y - (ChunkIndex::Y_RADIUS as f32) + 0.5) * Chunk::CHUNK_SIZE,
        };
        gizmos.rect_2d(pos, 0.0, Vec2::splat(Chunk::CHUNK_SIZE), Color::DARK_GRAY);
    }
}

fn init_index(mut commands: Commands) {
    let chunk_index = ChunkIndex {
        chunks: array::from_fn(|i| {
            array::from_fn(|j| {
                commands
                    .spawn((Chunk::default(), Name::new(format!("Chunk({i},{j})"))))
                    .id()
            })
        }),
    };
    commands.insert_resource(chunk_index);
}

fn transfer_agent(
    chunk_index: Res<ChunkIndex>,
    q_agents: Query<(&Prev, &Next, Entity), With<Agent>>,
    mut q_chunks: Query<&mut Chunk>,
) {
    for (agent_prev, agent_next, agent_id) in &q_agents {
        let prev = vec2(agent_prev.translation);
        let next = vec2(agent_next.translation);
        let crosses_boundary = (prev.rem_euclid(Vec2::splat(Chunk::CHUNK_SIZE))
            - next.rem_euclid(Vec2::splat(Chunk::CHUNK_SIZE)))
        .abs()
        .max_element()
            > Chunk::CHUNK_SIZE / 2.0;
        if prev != next && (crosses_boundary) {
            if let Some(chunk_id) = chunk_index.get(prev) {
                let mut chunk = q_chunks
                    .get_mut(chunk_id)
                    .expect("Chunk in index has been invalidated: {chunk_idx:?}");
                chunk.rm_agent(agent_id);
            }
            if let Some(chunk_id) = chunk_index.get(next) {
                let mut chunk = q_chunks
                    .get_mut(chunk_id)
                    .expect("Chunk in index has been invalidated: {chunk_idx:?}");
                chunk.add_agent(agent_id);
            }
        }
    }
}

// TODO: Make special OOB chunk.
#[derive(Resource)]
pub struct ChunkIndex {
    pub chunks: [[Entity; ChunkIndex::Y_RADIUS * 2]; ChunkIndex::X_RADIUS * 2],
}
impl ChunkIndex {
    // INVARIANT: Both RADIUS < 2^32
    pub const X_RADIUS: usize = 2;
    pub const Y_RADIUS: usize = 2;
    const BOUND: Vec2 = Vec2 {
        x: ChunkIndex::X_RADIUS as f32 * Chunk::CHUNK_SIZE,
        y: ChunkIndex::Y_RADIUS as f32 * Chunk::CHUNK_SIZE,
    };
    pub fn get(&self, pos: Vec2) -> Option<Entity> {
        let Vec2 { x, y } = ((pos + ChunkIndex::BOUND) / Chunk::CHUNK_SIZE).floor();
        let (x, y) = (x as i32, y as i32);
        if (0..ChunkIndex::X_RADIUS as i32 * 2).contains(&x)
            && (0..ChunkIndex::Y_RADIUS as i32 * 2).contains(&y)
        {
            Some(self.chunks[x as usize][y as usize])
        } else {
            None
        }
    }
    pub fn get_adjacent(&self, pos: Vec2) -> impl Iterator<Item = &Entity> {
        let base_idx = {
            let Vec2 { x, y } = ((pos + ChunkIndex::BOUND) / Chunk::CHUNK_SIZE).floor();
            (x as i32, y as i32)
        };
        (-1..=1)
            .cartesian_product(-1..=1)
            .filter(|(x, y)| (*x != 0) || (*y != 0))
            .map(move |(x_delta, y_delta)| {
                let (x, y) = base_idx;
                (x + x_delta, y + y_delta)
            })
            .filter(|(x, y)| {
                (0..ChunkIndex::X_RADIUS as i32 * 2).contains(x)
                    && (0..ChunkIndex::Y_RADIUS as i32 * 2).contains(y)
            })
            .map(|(x, y)| (x as usize, y as usize))
            .map(|(x, y)| &self.chunks[x][y])
    }
    pub fn get_adjacent_near(&self, pos: Vec2, radius: f32) -> impl Iterator<Item = &Entity> {
        let chunk_pos = pos.rem_euclid(Vec2::splat(Chunk::CHUNK_SIZE));
        let base_idx = {
            let Vec2 { x, y } = ((pos + ChunkIndex::BOUND) / Chunk::CHUNK_SIZE).floor();
            (x as i32, y as i32)
        };
        (-1..=1)
            .cartesian_product(-1..=1)
            .map(|(x, y)| ((x, y), (x as f32, y as f32)))
            .map(move |(idx, (x, y))| {
                (
                    idx,
                    Vec2 {
                        x: chunk_pos.x + x * radius,
                        y: chunk_pos.y + y * radius,
                    },
                )
            })
            .filter(|(_, Vec2 { x, y })| {
                !(0.0..Chunk::CHUNK_SIZE).contains(x) || !(0.0..Chunk::CHUNK_SIZE).contains(y)
            })
            .map(|(idx, _)| idx)
            .map(move |(x, y)| (base_idx.0 + x, base_idx.1 + y))
            .filter(|(x, y)| {
                (0..ChunkIndex::X_RADIUS as i32 * 2).contains(x)
                    && (0..ChunkIndex::Y_RADIUS as i32 * 2).contains(y)
            })
            .map(|(x, y)| (x as usize, y as usize))
            .map(|(x, y)| &self.chunks[x][y])
    }
    pub fn get_aabb(&self, a_pos: Vec2, b_pos: Vec2) -> impl Iterator<Item = &Entity> {
        let a_idx = {
            let Vec2 { x, y } = ((a_pos + ChunkIndex::BOUND) / Chunk::CHUNK_SIZE).floor();
            let (x, y) = (x as i32, y as i32);
            let (x, y) = (
                x.clamp(0, ChunkIndex::X_RADIUS as i32 * 2 - 1),
                y.clamp(0, ChunkIndex::X_RADIUS as i32 * 2 - 1),
            );
            (x as usize, y as usize)
        };
        let b_idx = {
            let Vec2 { x, y } = ((b_pos + ChunkIndex::BOUND) / Chunk::CHUNK_SIZE).floor();
            let (x, y) = (x as i32, y as i32);
            let (x, y) = (
                x.clamp(0, ChunkIndex::X_RADIUS as i32 * 2 - 1),
                y.clamp(0, ChunkIndex::X_RADIUS as i32 * 2 - 1),
            );
            (x as usize, y as usize)
        };
        (a_idx.0..=b_idx.0)
            .cartesian_product(a_idx.1..=b_idx.1)
            .map(|(x, y)| &self.chunks[x][y])
    }
    pub fn get_line(&self, origin_pos: Vec2, target_pos: Vec2) -> impl Iterator<Item = &Entity> {
        #[allow(non_camel_case_types)]
        struct Vec2_i32 {
            x: i32,
            y: i32,
        }
        let scaled_origin_pos = (origin_pos + ChunkIndex::BOUND) / Chunk::CHUNK_SIZE;
        let scaled_target_pos = (target_pos + ChunkIndex::BOUND) / Chunk::CHUNK_SIZE;
        let origin_idx = {
            let Vec2 { x, y } = scaled_origin_pos.floor();
            Vec2_i32 {
                x: x as i32,
                y: y as i32,
            }
        };
        let target_idx = {
            let Vec2 { x, y } = scaled_origin_pos.floor();
            Vec2_i32 {
                x: x as i32,
                y: y as i32,
            }
        };
        let idxs = if origin_idx.y == target_idx.y || origin_idx.x == target_idx.x {
            // Straight (axis-aligned) case
            Either::Left(Either::Left(
                (origin_idx.x..=target_idx.x).cartesian_product(origin_idx.y..=target_idx.y),
            ))
        } else {
            // Diagonal (axis-unaligned) case
            let delta = scaled_target_pos - scaled_origin_pos;
            if delta.x.abs() - delta.y.abs() > f32::EPSILON {
                let slope = delta.y / delta.x;
                let origin_dx = scaled_origin_pos.x.ceil() - scaled_origin_pos.x;
                let origin_dy = origin_dx * slope;
                let origin_finish_y_pos = scaled_origin_pos.y + origin_dy;
                let origin_finish_y_idx = (origin_finish_y_pos + f32::EPSILON).floor() as i32;
                let origin_idxs = (origin_idx.x..=origin_idx.x)
                    .cartesian_product(origin_idx.y..=origin_finish_y_idx);
                let target_dx = scaled_target_pos.x - scaled_target_pos.x.floor();
                let target_dy = target_dx * slope;
                let target_start_y_pos = scaled_target_pos.y - target_dy;
                let target_start_y_idx = (target_start_y_pos + f32::EPSILON).floor() as i32;
                let target_idxs = (target_idx.x..=target_idx.x)
                    .cartesian_product(target_idx.y..=target_start_y_idx);
                if (target_idx.x - origin_idx.x).abs() <= 1 {
                    Either::Right(Either::Left(Either::Left(origin_idxs.chain(target_idxs))))
                } else {
                    let start_x_idx = origin_idx.x + 1;
                    let finish_x_idx = target_idx.x;
                    let middle_idxs = (start_x_idx..=finish_x_idx)
                        .map(|x| (x, x as f32))
                        .flat_map(move |(x, start_x_pos)| {
                            let start_y_pos =
                                scaled_origin_pos.y + (start_x_pos - scaled_origin_pos.x) * slope;
                            let finish_y_pos = start_y_pos + 1.0 * slope;
                            let start_y_idx = start_y_pos.floor() as i32;
                            let finish_y_idx = finish_y_pos.floor() as i32;
                            (start_y_idx..=finish_y_idx).map(move |y| (x, y))
                        });
                    Either::Right(Either::Left(Either::Right(
                        origin_idxs.chain(middle_idxs).chain(target_idxs),
                    )))
                }
            } else if delta.x.abs() - delta.y.abs() < -f32::EPSILON {
                let slope = delta.x / delta.y;
                let origin_dy = scaled_origin_pos.y.ceil() - scaled_origin_pos.y;
                let origin_dx = origin_dy * slope;
                let origin_finish_x_pos = scaled_origin_pos.x + origin_dx;
                let origin_finish_x_idx = (origin_finish_x_pos + f32::EPSILON).floor() as i32;
                let origin_idxs = (origin_idx.x..=origin_finish_x_idx)
                    .cartesian_product(origin_idx.y..=origin_idx.y);
                let target_dy = scaled_target_pos.y - scaled_target_pos.y.floor();
                let target_dx = target_dy * slope;
                let target_start_x_pos = scaled_target_pos.x - target_dx;
                let target_start_x_idx = (target_start_x_pos + f32::EPSILON).floor() as i32;
                let target_idxs = (target_idx.x..=target_start_x_idx)
                    .cartesian_product(target_idx.y..=target_idx.y);
                if (target_idx.y - origin_idx.y).abs() <= 1 {
                    Either::Right(Either::Right(Either::Left(origin_idxs.chain(target_idxs))))
                } else {
                    let start_y_idx = origin_idx.y + 1;
                    let finish_y_idx = target_idx.y;
                    let middle_idxs = (start_y_idx..=finish_y_idx)
                        .map(|y| (y, y as f32))
                        .flat_map(move |(y, start_y_pos)| {
                            let start_x_pos =
                                scaled_origin_pos.x + (start_y_pos - scaled_origin_pos.y) * slope;
                            let finish_x_pos = start_x_pos + 1.0 * slope;
                            let start_x_idx = start_x_pos.floor() as i32;
                            let finish_x_idx = finish_x_pos.floor() as i32;
                            (start_x_idx..=finish_x_idx).map(move |x| (x, y))
                        });
                    Either::Right(Either::Right(Either::Right(
                        origin_idxs.chain(middle_idxs.chain(target_idxs)),
                    )))
                }
            } else {
                // delta.x.abs() == delta.y.abs()
                let reflect_f32 = if delta.x > f32::EPSILON && delta.y > f32::EPSILON {
                    |(x, y): (f32, f32)| (x, y)
                } else if delta.x > f32::EPSILON && delta.y < -f32::EPSILON {
                    |(x, y): (f32, f32)| (x, -y)
                } else if delta.x < -f32::EPSILON && delta.y > f32::EPSILON {
                    |(x, y): (f32, f32)| (-x, y)
                } else if delta.x < -f32::EPSILON && delta.y < -f32::EPSILON {
                    |(x, y): (f32, f32)| (-x, -y)
                } else {
                    unreachable!("If delta.x == 0.0 || delta.y == 0.0, then origin_idx.y == target_idx.y || origin_idx.x == target_idx.x")
                };
                let reflect_i32 = if delta.x > f32::EPSILON && delta.y > f32::EPSILON {
                    |(x, y): (i32, i32)| (x, y)
                } else if delta.x > f32::EPSILON && delta.y < -f32::EPSILON {
                    |(x, y): (i32, i32)| (x, -y)
                } else if delta.x < -f32::EPSILON && delta.y > f32::EPSILON {
                    |(x, y): (i32, i32)| (-x, y)
                } else if delta.x < -f32::EPSILON && delta.y < -f32::EPSILON {
                    |(x, y): (i32, i32)| (-x, -y)
                } else {
                    unreachable!("If delta.x == 0.0 || delta.y == 0.0, then origin_idx.y == target_idx.y || origin_idx.x == target_idx.x")
                };
                let reflectedscaled_origin_pos = {
                    let Vec2 { x, y } = scaled_origin_pos;
                    let (x, y) = reflect_f32((x, y));
                    Vec2 { x, y }
                };
                let reflectedscaled_target_pos = {
                    let Vec2 { x, y } = scaled_target_pos;
                    let (x, y) = reflect_f32((x, y));
                    Vec2 { x, y }
                };
                let origin_pos = reflectedscaled_origin_pos;
                let target_pos = reflectedscaled_target_pos;
                let origin_idx = {
                    let Vec2 { x, y } = origin_pos.floor();
                    Vec2_i32 {
                        x: x as i32,
                        y: y as i32,
                    }
                };
                let target_idx = {
                    let Vec2 { x, y } = target_pos.floor();
                    Vec2_i32 {
                        x: x as i32,
                        y: y as i32,
                    }
                };
                let main_diag = (origin_idx.x..=target_idx.x).zip(origin_idx.y..=target_idx.y);
                let bias = origin_pos % 1.0;
                let idxs = if bias.x - bias.y > f32::EPSILON {
                    let par_diag =
                        ((origin_idx.x + 1)..=target_idx.x).zip(origin_idx.y..target_idx.y);
                    Either::Right(Either::Left(
                        main_diag.interleave(par_diag).map(reflect_i32),
                    ))
                } else if bias.x - bias.y < -f32::EPSILON {
                    let par_diag =
                        (origin_idx.x..target_idx.x).zip((origin_idx.y + 1)..=target_idx.y);
                    Either::Right(Either::Right(
                        main_diag.interleave(par_diag).map(reflect_i32),
                    ))
                } else {
                    // bias.x == bias.y
                    Either::Left(main_diag.map(reflect_i32))
                };
                Either::Left(Either::Right(idxs))
            }
        };
        idxs.filter(|(x, y)| {
            (0..ChunkIndex::X_RADIUS as i32).contains(x)
                && (0..ChunkIndex::Y_RADIUS as i32).contains(y)
        })
        .map(|(x, y)| (x as usize, y as usize))
        .map(|(x, y)| &self.chunks[x][y])
    }
}

#[derive(Component, Default)]
pub struct Chunk {
    // INVARIANT: An agent_id can only exist in at most 1 chunk.
    pub agents: Vec<Entity>,
    pub walls: Vec<Entity>,
}
impl Chunk {
    pub const CHUNK_SIZE: f32 = 8.0;
    pub fn add_agent(&mut self, agent_id: Entity) {
        self.agents.push(agent_id);
    }
    pub fn rm_agent(&mut self, agent_id: Entity) {
        self.agents.retain(|&x| x != agent_id);
    }
    pub fn add_wall(&mut self, wall_id: Entity) {
        self.walls.push(wall_id);
    }
    // pub fn rm_wall(&mut self, wall_id: Entity) {
    //     self.walls.retain(|&x| x != wall_id);
    // }
}

// const LOAD_RADIUS: usize = 8;
// #[derive(Component)]
// struct ChunkCache {
//     chunks: [Chunk; (LOAD_RADIUS * 2 + 1).pow(2)],
// }
