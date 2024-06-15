mod camera;
// mod editor;
mod chunk;
mod fps_counter;
mod input;
mod util;

use std::{collections::VecDeque, ops::BitAnd, time::Duration};

use bevy::{
    ecs::query::{QueryEntityError, QueryFilter},
    prelude::*,
    sprite::{Material2d, MaterialMesh2dBundle, Mesh2dHandle},
    utils::hashbrown::HashSet,
};
// use bevy_inspector_egui::quick::WorldInspectorPlugin;
use camera::CameraPlugin;
use chunk::{Chunk, ChunkIndex, ChunkPlugin};
use fps_counter::FpsCounterPlugin;
use input::{ActivateInput, InputModifier, InputPlugin, InputPosition, MoveInput, SelectInput};
use itertools::Itertools;
use util::vec2;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FpsCounterPlugin)
        // .add_plugins(WorldInspectorPlugin::new())
        .add_plugins(InputPlugin)
        .add_plugins(CameraPlugin)
        .add_plugins(ChunkPlugin)
        .add_systems(PreStartup, setup_colors)
        .add_systems(Startup, (spawn_walls, spawn_agents, setup_bullets))
        .add_systems(PostStartup, (register_agents, register_walls))
        .add_systems(
            Update,
            (
                cache_pos,
                activate_agent,
                select_agent,
                order_agent,
                highlight_agent,
                update_player_agent,
                update_rival_agent,
                fix_agent_collisions
                    .after(update_player_agent)
                    .after(update_rival_agent),
                fix_wall_collisions
                    .after(update_player_agent)
                    .after(update_rival_agent),
                (spawn_bullets, update_bullets).chain(),
            ),
        )
        .add_systems(PostUpdate, (mutate_pos, draw_walls, draw_routes))
        .add_event::<FireBullet>()
        .run();
}

#[allow(non_snake_case)]
#[derive(Resource)]
struct Colors {
    AGENT_RIVAL: Handle<ColorMaterial>,
    AGENT_ACTIVE: Handle<ColorMaterial>,
    AGENT_INACTIVE: Handle<ColorMaterial>,
    GRAY: Handle<ColorMaterial>,
    BULLET: Handle<ColorMaterial>,
}
fn setup_colors(mut commands: Commands, mut materials: ResMut<Assets<ColorMaterial>>) {
    let colors = Colors {
        AGENT_RIVAL: materials.add(Color::rgb(1.0, 0.125, 0.125)),
        AGENT_INACTIVE: materials.add(Color::rgb(0.25, 0.5, 1.0)),
        AGENT_ACTIVE: materials.add(Color::rgb(0.5, 1.0, 1.0)),
        GRAY: materials.add(Color::rgb(0.5, 0.5, 0.5)),
        BULLET: materials.add(Color::rgb(1.0, 1.0, 0.5)),
    };
    commands.insert_resource(colors);
}

#[derive(Component)]
enum Wall {
    Aabb,
    MapBoundary,
}

#[derive(Component)]
struct MapBoundary;

fn spawn_walls(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, colors: Res<Colors>) {
    commands
        .spawn((
            Wall::Aabb,
            SpatialBundle::from_transform(
                Transform::from_xyz(-4.0, 0.0, 0.0).with_scale(Vec3::new(0.5, 2.0, 0.0)),
            ),
        ))
        .with_children(|parent| {
            parent.spawn(MaterialMesh2dBundle {
                mesh: meshes.add(Rectangle::default()).into(),
                material: colors.GRAY.clone(),
                ..default()
            });
        });
}

fn draw_walls(q_walls: Query<(&Transform, &Wall), With<Wall>>, mut gizmos: Gizmos) {
    for (wall, wall_type) in q_walls.iter() {
        if let Wall::MapBoundary = wall_type {
            gizmos.rect_2d(Vec2::ZERO, 0.0, vec2(wall.scale), Color::BLACK);
        }
    }
}

fn register_agents(
    chunk_index: Res<ChunkIndex>,
    q_agents: Query<(&Transform, Entity), With<Agent>>,
    mut q_chunks: Query<&mut Chunk>,
) {
    for (agent, agent_id) in &q_agents {
        let chunk_id = chunk_index
            .get(vec2(agent.translation))
            .expect("Agent initialized outside of chunk.");
        let mut chunk = q_chunks
            .get_mut(chunk_id)
            .expect("Chunk in index has been invalidated: {chunk_idx:?}");
        chunk.add_agent(agent_id);
    }
}

fn register_walls(
    chunk_index: Res<ChunkIndex>,
    q_walls: Query<(&Transform, &Wall, Entity)>,
    mut q_chunks: Query<&mut Chunk>,
) {
    for (wall, wall_type, wall_id) in &q_walls {
        let chunk_ids = match wall_type {
            Wall::Aabb => {
                let min = vec2(wall.translation - wall.scale / 2.0);
                let max = vec2(wall.translation + wall.scale / 2.0);
                chunk_index.get_aabb(min, max)
            }
            Wall::MapBoundary => panic!("Do not manually spawn MapBoundaries."),
        };
        for chunk_id in chunk_ids {
            let mut chunk = q_chunks
                .get_mut(*chunk_id)
                .expect("Chunk in index has been invalidated: {chunk_idx:?}");
            chunk.add_wall(wall_id);
        }
    }
}

#[derive(Component)]
struct Prev {
    translation: Vec3,
}
fn cache_pos(mut q_prev: Query<(&Transform, &mut Prev)>) {
    // for (curr, mut prev) in &mut q_prev {
    //     prev.translation = curr.translation;
    // }
    q_prev
        .par_iter_mut()
        .for_each(|(curr, mut prev)| prev.translation = curr.translation);
}

#[derive(Component)]
struct Next {
    translation: Vec3,
}

fn mutate_pos(mut q_next: Query<(&mut Transform, &Next)>) {
    // for (mut curr, next) in &mut q_next {
    //     if next.translation.is_finite() {
    //         curr.translation = next.translation;
    //     }
    // }
    q_next.par_iter_mut().for_each(|(mut curr, next)| {
        if next.translation.is_finite() {
            curr.translation = next.translation
        }
    });
}

#[derive(Component)]
struct Agent;
#[derive(Bundle)]
struct AgentBundle<M: Material2d> {
    marker: Agent,
    mesh: Mesh2dHandle,
    material: Handle<M>,
    route: Route,
    weapon: Weapon,
    prev: Prev,
    next: Next,
    curr: SpatialBundle,
}
impl<M: Material2d> AgentBundle<M> {
    fn new(transform: Transform, mesh: Mesh2dHandle, material: Handle<M>) -> Self {
        AgentBundle {
            marker: Agent,
            mesh,
            material,
            route: Route::new(),
            weapon: Weapon {
                windup: Timer::new(Duration::from_secs_f32(0.0), TimerMode::Once),
                cooldown: Timer::new(Duration::from_secs_f32(0.1), TimerMode::Once),
            },
            prev: Prev {
                translation: transform.translation,
            },
            next: Next {
                translation: transform.translation,
            },
            curr: SpatialBundle::from_transform(transform),
        }
    }
    fn with_waypoint(mut self, waypoint: Waypoint) -> Self {
        self.route.0.push_back(waypoint);
        self
    }
}
#[derive(Component)]
struct IsActivatedAgent(bool);
#[derive(Component)]
enum PlayerAgent {
    A,
    B,
    // C,
    // D,
}
#[derive(Bundle)]
struct PlayerAgentBundle<M: Material2d> {
    marker: PlayerAgent,
    is_activated: IsActivatedAgent,
    agent_bundle: AgentBundle<M>,
}
impl<M: Material2d> PlayerAgentBundle<M> {
    fn new(
        marker: PlayerAgent,
        transform: Transform,
        mesh: Mesh2dHandle,
        material: Handle<M>,
    ) -> Self {
        PlayerAgentBundle {
            marker,
            is_activated: IsActivatedAgent(false),
            agent_bundle: AgentBundle::new(transform, mesh, material),
        }
    }
}
#[derive(Component, Debug)]
struct Weapon {
    windup: Timer,
    cooldown: Timer,
}

fn spawn_agents(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, colors: Res<Colors>) {
    commands
        .spawn(PlayerAgentBundle::new(
            PlayerAgent::A,
            Transform::from_xyz(0.0, 0.0, 0.0).with_scale(Vec3::splat(1.0)),
            meshes.add(Circle::default()).into(),
            colors.AGENT_INACTIVE.clone(),
        ))
        .with_children(|parent| {
            parent.spawn(Text2dBundle {
                text: Text::from_section(
                    "1",
                    TextStyle {
                        font_size: 32.0,
                        ..default()
                    },
                ),
                transform: Transform {
                    translation: Vec3 {
                        z: 1.0,
                        ..default()
                    },
                    scale: Vec3::splat((-5.0_f32).exp2()),
                    ..default()
                },
                ..default()
            });
        });
    commands
        .spawn(PlayerAgentBundle::new(
            PlayerAgent::B,
            Transform::from_xyz(0.0, 1.0, 0.0).with_scale(Vec3::splat(1.0)),
            meshes.add(Circle::default()).into(),
            colors.AGENT_INACTIVE.clone(),
        ))
        .with_children(|parent| {
            parent.spawn(Text2dBundle {
                text: Text::from_section(
                    "2",
                    TextStyle {
                        font_size: 32.0,
                        ..default()
                    },
                ),
                transform: Transform {
                    translation: Vec3 {
                        z: 1.0,
                        ..default()
                    },
                    scale: Vec3::splat((-5.0_f32).exp2()),
                    ..default()
                },
                ..default()
            });
        });
    commands.spawn(
        AgentBundle::new(
            Transform::from_xyz(4.0, 1.0, 0.0).with_scale(Vec3::splat(1.0)),
            meshes.add(Circle::default()).into(),
            colors.AGENT_RIVAL.clone(),
        )
        .with_waypoint(Waypoint::HugClosest),
    );
    let i_radius = ChunkIndex::X_RADIUS as i32;
    let j_radius = ChunkIndex::Y_RADIUS as i32;
    let circle = meshes.add(Circle::default());
    let crowd: Vec<AgentBundle<ColorMaterial>> = (-i_radius..i_radius)
        .cartesian_product(-j_radius..j_radius)
        .map(|(i, j)| (i as f32, j as f32))
        .flat_map(move |(i, j)| {
            let circle = circle.clone();
            let color = colors.AGENT_RIVAL.clone();
            (0..8)
                .cartesian_product(0..8)
                .map(|(x, y)| (x as f32, y as f32))
                .map(move |(x, y)| {
                    (
                        x * 1.0 + 0.5 + Chunk::CHUNK_SIZE * i,
                        y * 1.0 + 0.5 + Chunk::CHUNK_SIZE * j,
                    )
                })
                .map(move |(x, y)| {
                    AgentBundle::new(
                        Transform::from_xyz(x, y, 0.0).with_scale(Vec3::splat(1.0)),
                        circle.clone().into(),
                        color.clone(),
                    )
                })
        })
        .collect();
    commands.spawn_batch(crowd);
}

#[allow(clippy::type_complexity)]
fn select_agent(
    mut ev_select: EventReader<SelectInput>,
    mut q_agents: Query<(
        &Transform,
        Entity,
        &mut Route,
        Option<&PlayerAgent>,
        Option<&mut IsActivatedAgent>,
    )>,
) {
    let last_select = ev_select.read().fold(None, |_, ev| Some(ev));
    if let Some(select) = last_select {
        let mut players = HashSet::new();
        let mut enemies = Vec::new();
        for (agent, agent_id, _, is_player, _) in &q_agents {
            let activation = match select.position {
                InputPosition::Point(pos) => {
                    pos.distance_squared(vec2(agent.translation)) < (agent.scale.x / 2.0).powi(2)
                }
                InputPosition::Box(b0_pos, b1_pos) => {
                    let agent_pos = vec2(agent.translation);
                    let padding = agent.scale.x / 2.0;
                    let lower_bound = (Vec2::min(b0_pos, b1_pos) - padding).cmple(agent_pos);
                    let upper_bound = (Vec2::max(b0_pos, b1_pos) + padding).cmpge(agent_pos);
                    let bound = BVec2::bitand(lower_bound, upper_bound);
                    bound.x && bound.y
                }
            };
            if activation {
                if is_player.is_some() {
                    players.insert(agent_id);
                } else {
                    enemies.push(agent_id);
                }
            }
        }
        if !players.is_empty() {
            for (_, agent_id, _, is_player, is_activated) in &mut q_agents {
                if is_player.is_none() {
                    continue;
                }
                let mut is_activated = is_activated.expect("IsActivated unexpectedly missing.");
                match select.modifier {
                    InputModifier::Overwrite => is_activated.0 = players.contains(&agent_id),
                    InputModifier::Append => is_activated.0 |= players.contains(&agent_id),
                }
            }
        } else if !enemies.is_empty() {
            let mut routes = Vec::new();
            for (_, _, route, is_player, is_activated) in &mut q_agents {
                if is_player.is_some() && is_activated.expect("IsActivated unexpectedly missing.").0
                {
                    routes.push(route);
                }
            }
            for mut route in routes {
                if let InputModifier::Overwrite = select.modifier {
                    route.0.clear();
                }
                for enemy in &enemies {
                    route.0.push_back(Waypoint::FireOn(*enemy));
                }
            }
        }
    }
}

fn activate_agent(
    mut ev_activate: EventReader<ActivateInput>,
    mut q_agents: Query<(&mut IsActivatedAgent, &PlayerAgent)>,
) {
    let last_activate = ev_activate.read().fold(None, |_, ev| Some(ev));
    if let Some(activation) = last_activate {
        for (mut is_activated, player_agent) in &mut q_agents {
            match player_agent {
                PlayerAgent::A => is_activated.0 = activation.a,
                PlayerAgent::B => is_activated.0 = activation.b,
                // PlayerAgent::C => is_activated.0 = activation.c,
                // PlayerAgent::D => is_activated.0 = activation.d,
            }
        }
    }
}

fn highlight_agent(
    colors: Res<Colors>,
    mut q_agents: Query<(&IsActivatedAgent, &mut Handle<ColorMaterial>), With<PlayerAgent>>,
) {
    for (is_selected, mut material) in &mut q_agents {
        if is_selected.0 {
            // Set to green
            *material = colors.AGENT_ACTIVE.clone();
        } else {
            // Set to purple
            *material = colors.AGENT_INACTIVE.clone();
        }
    }
}

#[derive(Component, Debug)]
struct Route(VecDeque<Waypoint>);
impl Route {
    fn new() -> Route {
        Route(VecDeque::new())
    }
}
#[derive(Debug)]
enum Waypoint {
    Point(Vec2),
    HugClosest,
    FireOn(Entity),
}

fn order_agent(
    q_walls: Query<(&Transform, &Wall), With<Wall>>,
    mut ev_move: EventReader<MoveInput>,
    mut q_agents: Query<(&Transform, &mut Route, &IsActivatedAgent), With<PlayerAgent>>,
) {
    let last_move = ev_move.read().fold(None, |_, ev| Some(ev));
    if let Some(MoveInput {
        position: target,
        modifier,
    }) = last_move
    {
        for (agent, mut route, is_selected) in &mut q_agents {
            if is_selected.0 {
                trace!("WAYPOINT@{target:?}");
                if let InputModifier::Overwrite = modifier {
                    route.0.clear();
                }
                let agent_radius = agent.scale.x / 2.0;
                let target = fix_target(target, agent_radius, &q_walls);
                let waypoint = Waypoint::Point(target);
                route.0.push_back(waypoint);
            }
        }
    }
}
fn fix_target(
    target: &Vec2,
    padding: f32,
    q_walls: &Query<(&Transform, &Wall), With<Wall>>,
) -> Vec2 {
    let mut target = *target;
    for (wall, wall_type) in q_walls {
        match wall_type {
            Wall::Aabb => {
                let wall_min = vec2(wall.translation - (wall.scale / 2.0 + padding));
                let wall_max = vec2(wall.translation + (wall.scale / 2.0 + padding));
                let bound = target.cmpgt(wall_min) ^ target.cmpgt(wall_max);
                if bound.all() {
                    let delta_x = if target.x >= wall.translation.x {
                        wall_max.x - target.x
                    } else {
                        wall_min.x - target.x
                    };
                    let delta_y = if target.y >= wall.translation.y {
                        wall_max.y - target.y
                    } else {
                        wall_min.y - target.y
                    };
                    if delta_x.abs() < delta_y.abs() {
                        target.x += delta_x;
                    } else {
                        target.y += delta_y;
                    }
                }
            }
            Wall::MapBoundary => {
                target = target.clamp(
                    -vec2(wall.scale / 2.0 - padding),
                    vec2(wall.scale / 2.0 - padding),
                );
            }
        }
    }
    target
}

fn draw_routes(q_agents: Query<&Route, With<PlayerAgent>>, mut gizmos: Gizmos) {
    for route in &q_agents {
        let mut intensity = 1.0;
        for waypoint in route.0.iter() {
            if let Waypoint::Point(waypoint_pos) = waypoint {
                gizmos.circle_2d(
                    *waypoint_pos,
                    1.0 / 256.0,
                    Color::rgba(0.0, 1.0, 0.0, intensity),
                );
                intensity *= 0.5;
            }
        }
    }
}

#[derive(Component)]
struct Bullet;
impl Bullet {
    const MOVEMENT_SPEED: f32 = 8.0;
}
#[derive(Bundle)]
struct BulletBundle<M: Material2d> {
    marker: Bullet,
    mesh: Mesh2dHandle,
    material: Handle<M>,
    prev: Prev,
    next: Next,
    curr: SpatialBundle,
}
#[derive(Event)]
struct FireBullet(Vec2, Vec2); // (origin, target)
#[derive(Resource)]
struct BulletFactory<M: Material2d> {
    mesh: Mesh2dHandle,
    material: Handle<M>,
}
impl<M: Material2d> BulletFactory<M> {
    fn shoot(&self, origin: Vec2, target: Vec2) -> BulletBundle<M> {
        let origin = Vec3::from((origin, 0.0));
        let target = Vec3::from((target, 0.0));
        let transform = Transform::from_translation(origin)
            .looking_to(-Vec3::Z, (target - origin).cross(-Vec3::Z))
            .with_scale(Vec3::new(Bullet::MOVEMENT_SPEED / 256.0, 1.0 / 32.0, 0.0));
        BulletBundle {
            marker: Bullet,
            mesh: self.mesh.clone(),
            material: self.material.clone(),
            prev: Prev {
                translation: transform.translation,
            },
            next: Next {
                translation: transform.translation,
            },
            curr: SpatialBundle::from_transform(transform),
        }
    }
}
fn setup_bullets(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, colors: Res<Colors>) {
    commands.insert_resource(BulletFactory {
        mesh: meshes.add(Rectangle::default()).into(),
        material: colors.BULLET.clone(),
    });
}
fn spawn_bullets(
    bullet_factory: Res<BulletFactory<ColorMaterial>>,
    mut commands: Commands,
    mut ev_bullet: EventReader<FireBullet>,
) {
    for bullet in ev_bullet.read() {
        let bullet = bullet_factory.shoot(bullet.0, bullet.1);
        commands.spawn(bullet);
    }
}
fn update_bullets(
    mut commands: Commands,
    time: Res<Time>,
    mut q_bullets: Query<(&Transform, &mut Next, Entity), With<Bullet>>,
) {
    for (bullet, mut bullet_next, bullet_id) in &mut q_bullets {
        if bullet.translation.max_element() > 4096.0 {
            commands.entity(bullet_id).despawn();
            return;
        }
        let forward = bullet.right();
        bullet_next.translation += forward * Bullet::MOVEMENT_SPEED * time.delta_seconds();
    }
}

#[allow(clippy::type_complexity)]
fn update_player_agent(
    time: Res<Time>,
    ev_bullet: EventWriter<FireBullet>,
    q_agents: Query<
        (&Transform, &mut Next, &mut Route, &mut Weapon),
        (With<Agent>, With<PlayerAgent>),
    >,
    q_rivals: Query<&Transform, (With<Agent>, Without<PlayerAgent>)>,
) {
    const MOVEMENT_SPEED: f32 = 8.0;
    update_agent(MOVEMENT_SPEED, time, ev_bullet, q_agents, q_rivals);
}
#[allow(clippy::type_complexity)]
fn update_rival_agent(
    time: Res<Time>,
    ev_bullet: EventWriter<FireBullet>,
    q_rivals: Query<
        (&Transform, &mut Next, &mut Route, &mut Weapon),
        (With<Agent>, Without<PlayerAgent>),
    >,
    q_agents: Query<&Transform, (With<Agent>, With<PlayerAgent>)>,
) {
    const MOVEMENT_SPEED: f32 = 4.0;
    update_agent(MOVEMENT_SPEED, time, ev_bullet, q_rivals, q_agents);
}
fn update_agent<BlueTeamFilter: QueryFilter, RedTeamFilter: QueryFilter>(
    movement_speed: f32,
    time: Res<Time>,
    mut ev_bullet: EventWriter<FireBullet>,
    mut q_blufor: Query<(&Transform, &mut Next, &mut Route, &mut Weapon), BlueTeamFilter>,
    q_redfor: Query<&Transform, RedTeamFilter>,
) {
    for (blue_agent, mut blue_agent_next, mut route, mut weapon) in &mut q_blufor {
        weapon.cooldown.tick(time.delta());
        if route.0.is_empty() {
            continue;
        }
        let waypoint = route.0.front().expect("Route unexpectedly empty.");
        match waypoint {
            Waypoint::Point(waypoint_pos) => {
                weapon.windup.reset();
                let waypoint_pos = Vec3::from((*waypoint_pos, 0.0));
                let delta = (waypoint_pos - blue_agent.translation) * Vec3::new(1.0, 1.0, 0.0);
                let delta_length = delta.length();
                let displacement = movement_speed * time.delta_seconds();
                if delta_length > displacement {
                    let direction = delta / delta_length;
                    blue_agent_next.translation += direction * displacement;
                } else {
                    route.0.pop_front();
                    blue_agent_next.translation = waypoint_pos;
                }
            }
            Waypoint::HugClosest => {
                weapon.windup.reset();
                let closest = &q_redfor
                    .iter()
                    .fold(None, |closest, current| match closest {
                        None => Some(current),
                        Some(closest) => {
                            if (closest.translation - blue_agent.translation).length_squared()
                                > (current.translation - blue_agent.translation).length_squared()
                            {
                                Some(current)
                            } else {
                                Some(closest)
                            }
                        }
                    });
                if let Some(closest) = closest {
                    let delta = closest.translation - blue_agent.translation;
                    let direction = delta / delta.length();
                    let displacement = movement_speed * time.delta_seconds();
                    blue_agent_next.translation += direction * displacement;
                }
            }
            Waypoint::FireOn(red_agent_id) => {
                weapon.windup.tick(time.delta());
                if !weapon.windup.finished() {
                    continue;
                }
                if !weapon.cooldown.finished() {
                    continue;
                }
                let maybe_red_agent = q_redfor.get(*red_agent_id);
                match maybe_red_agent {
                    Ok(red_agent) => {
                        let delta = vec2(red_agent.translation - blue_agent.translation);
                        let blue_agent_radius = blue_agent.scale.x / 2.0;
                        let origin = vec2(blue_agent.translation)
                            + delta / delta.length() * blue_agent_radius;
                        let target = vec2(red_agent.translation);
                        ev_bullet.send(FireBullet(origin, target));
                        weapon.cooldown.reset();
                    }
                    Err(err) => {
                        match err {
                            QueryEntityError::NoSuchEntity(_) => {
                                route.0.pop_front();
                            }
                            QueryEntityError::AliasedMutability(_) => {
                                panic!("While firing, attempted to borrow from mutably borrowed entity.")
                            }
                            QueryEntityError::QueryDoesNotMatch(_) => {
                                panic!("While firing, found no requested component on enemy.")
                            }
                        }
                    }
                }
            }
        }
    }
}

// INVARIANT: An agent_id can only exist in at most 1 chunk.
fn fix_agent_collisions(
    chunk_index: Res<ChunkIndex>,
    q_chunks: Query<&Chunk>,
    q_agents: Query<(&Transform, &mut Next), With<Agent>>,
) {
    q_chunks.par_iter().for_each(|chunk| {
        for &curr_id in &chunk.agents {
            if let Ok((curr, mut curr_next)) = unsafe { q_agents.get_unchecked(curr_id) } {
                let roommates = chunk
                    .agents
                    .iter()
                    .filter(|&&adj_id| adj_id != curr_id)
                    .map(|&adj_id| unsafe { q_agents.get_unchecked(adj_id) })
                    .filter_map(|res| res.ok())
                    .map(|(adj, _)| adj);
                let neighbors = chunk_index
                    .get_adjacent_near(vec2(curr.translation), curr.scale.x)
                    .map(|&chunk_id| q_chunks.get(chunk_id).unwrap())
                    .flat_map(|chunk| chunk.agents.iter())
                    .map(|&adj_id| unsafe { q_agents.get_unchecked(adj_id) })
                    .filter_map(|res| res.ok())
                    .map(|(adj, _)| adj);
                for adj in roommates.chain(neighbors) {
                    let delta = vec2(adj.translation - curr.translation);
                    let curr_radius = curr.scale.x / 2.0;
                    let adj_radius = adj.scale.x / 2.0;
                    if delta.x > curr_radius + adj_radius || delta.y > curr_radius + adj_radius {
                        continue;
                    }
                    if delta.abs().cmple(Vec2::splat(f32::EPSILON)).all() {
                        curr_next.translation.x += 0.1;
                        curr_next.translation.y += 0.1;
                    } else if delta.length_squared() < (curr_radius + adj_radius).powi(2) {
                        let delta_length = delta.length();
                        let direction = Vec3::from(((delta / delta_length), 0.0));
                        let intersection_length = curr_radius + adj_radius - delta_length;
                        curr_next.translation -= direction
                            * intersection_length
                            * (curr_radius / (curr_radius + adj_radius));
                    }
                }
            }
        }
    });
}

// INVARIANT: An agent_id can only exist in at most 1 chunk.
#[allow(clippy::type_complexity)]
fn fix_wall_collisions(
    chunk_index: Res<ChunkIndex>,
    q_chunks: Query<&Chunk>,
    q_walls: Query<(&Transform, &Wall), (With<Wall>, Without<Agent>)>,
    q_agents: Query<(&Transform, &mut Next), (Without<Wall>, With<Agent>)>,
) {
    q_chunks.par_iter().for_each(|chunk| {
        for &agent_id in &chunk.agents {
            if let Ok((agent, mut agent_next)) = unsafe { q_agents.get_unchecked(agent_id) } {
                let adj_walls =
                    chunk_index.get_adjacent_near(vec2(agent.translation), agent.scale.x);
                let walls = chunk
                    .walls
                    .iter()
                    .chain(adj_walls)
                    .map(|&wall_id| q_walls.get(wall_id))
                    .filter_map(|res| res.ok());
                for (wall, wall_type) in walls {
                    let agent_radius = agent.scale.x / 2.0;
                    match wall_type {
                        Wall::Aabb => {
                            let wall_min = wall.translation - wall.scale / 2.0;
                            let wall_max = wall.translation + wall.scale / 2.0;
                            let closest_point = agent.translation.clamp(wall_min, wall_max);
                            let delta =
                                (closest_point - agent.translation) * Vec3::new(1.0, 1.0, 0.0);
                            if delta.length_squared() < agent_radius.powi(2) {
                                let delta_length = delta.length();
                                let direction = delta / delta_length;
                                let intersection_length = f32::abs(agent_radius - delta_length);
                                agent_next.translation -= direction * intersection_length;
                            }
                        }
                        Wall::MapBoundary => {
                            let wall = vec2(wall.scale / 2.0) - agent_radius;
                            agent_next.translation.x = agent.translation.x.clamp(-wall.x, wall.x);
                            agent_next.translation.y = agent.translation.y.clamp(-wall.y, wall.y);
                        }
                    }
                }
            }
        }
    });
}
