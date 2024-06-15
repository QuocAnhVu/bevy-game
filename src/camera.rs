use bevy::prelude::*;

use crate::{input::ScrollInput, MapBoundary};

#[derive(Component)]
pub struct MainCamera;

#[derive(Resource)]
struct Zoom(f32);

pub struct CameraPlugin;
impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (spawn_camera, setup_zoom).chain())
            .add_systems(Update, (move_camera, order_zoom, zoom_camera));
    }
}

fn spawn_camera(mut commands: Commands) {
    let mut camera = Camera2dBundle::default();
    camera.transform = camera.transform.with_scale(Vec3::splat(1.0_f32 / 32.0));
    commands.spawn((camera, MainCamera));
}

fn move_camera(
    time: Res<Time>,
    key: Res<ButtonInput<KeyCode>>,
    q_map: Query<&Transform, (With<MapBoundary>, Without<MainCamera>)>,
    mut q_camera: Query<&mut Transform, With<MainCamera>>,
) {
    const MOVEMENT_SPEED: f32 = 32.0;
    const PADDING: f32 = 4.0;
    let mut camera = q_camera.single_mut();
    if key.pressed(KeyCode::KeyW) {
        camera.translation.y += MOVEMENT_SPEED * time.delta_seconds();
    }
    if key.pressed(KeyCode::KeyA) {
        camera.translation.x -= MOVEMENT_SPEED * time.delta_seconds();
    }
    if key.pressed(KeyCode::KeyS) {
        camera.translation.y -= MOVEMENT_SPEED * time.delta_seconds();
    }
    if key.pressed(KeyCode::KeyD) {
        camera.translation.x += MOVEMENT_SPEED * time.delta_seconds();
    }
    let map = q_map.get_single();
    if let Ok(map) = map {
        let wall = map.translation + map.scale / 2.0 - PADDING;
        camera.translation.x = camera.translation.x.clamp(-wall.x, wall.x);
        camera.translation.y = camera.translation.y.clamp(-wall.y, wall.y);
    }
}

fn setup_zoom(mut commands: Commands, q_camera: Query<&OrthographicProjection, With<MainCamera>>) {
    let camera = q_camera.single();
    commands.insert_resource(Zoom(camera.scale.log2()));
}
fn order_zoom(mut ev_scroll: EventReader<ScrollInput>, mut zoom: ResMut<Zoom>) {
    const SCROLL_FACTOR: f32 = -0.5;
    for scroll in ev_scroll.read() {
        zoom.0 = (zoom.0 + scroll.0 * SCROLL_FACTOR).clamp(-1.0, 1.0);
    }
}
fn zoom_camera(
    time: Res<Time>,
    zoom: Res<Zoom>,
    mut q_camera: Query<&mut OrthographicProjection, With<MainCamera>>,
) {
    const ZOOM_SPEED: f32 = 32.0;
    let mut camera = q_camera.single_mut();
    let difference = zoom.0 - camera.scale.log2();
    let sign = difference / difference.abs();
    let delta = sign * ZOOM_SPEED * time.delta_seconds();
    if difference.abs() > delta.abs() {
        camera.scale *= delta.exp2();
    } else {
        camera.scale = zoom.0.exp2();
    }
}
