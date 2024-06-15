/** NOTES:
* - camera.rs reads keyboard state to pan camera
*/
use bevy::{
    input::{
        keyboard::KeyboardInput,
        mouse::{MouseButtonInput, MouseWheel},
        ButtonState,
    },
    prelude::*,
    window::PrimaryWindow,
};

use crate::camera::MainCamera;

#[derive(Event)]
pub struct SelectInput {
    pub position: InputPosition,
    pub modifier: InputModifier,
}

#[derive(Event)]
pub struct MoveInput {
    pub position: Vec2,
    pub modifier: InputModifier,
}

pub enum InputPosition {
    Point(Vec2),
    Box(Vec2, Vec2), // (start pos, end pos)
}
pub enum InputModifier {
    Overwrite,
    Append,
}

#[derive(Event)]
pub struct ScrollInput(pub f32);

#[derive(Component)]
struct DragBox;

#[derive(Resource, Clone, Copy)]
enum DragState {
    Off,
    Debounce(Vec2), // (start pos, start timer)
    On(Vec2),       // (start pos)
}

pub struct InputPlugin;
impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_drag_ui)
            .add_systems(
                Update,
                (
                    mouse_input,
                    scroll_input,
                    drag_mouse_input,
                    display_drag,
                    keyboard_input,
                ),
            )
            .add_event::<SelectInput>()
            .add_event::<MoveInput>()
            .add_event::<ScrollInput>()
            .add_event::<ActivateInput>();
    }
}

fn setup_drag_ui(mut commands: Commands) {
    commands.insert_resource(DragState::Off);
    commands.spawn((
        DragBox,
        NodeBundle {
            z_index: ZIndex::Global(i32::MAX - 1),
            background_color: BackgroundColor(Color::rgba(0.0, 1.0, 0.0, 0.25)),
            border_color: BorderColor(Color::rgba(0.0, 1.0, 0.0, 1.0)),
            style: Style {
                display: Display::None,
                position_type: PositionType::Absolute,
                border: UiRect::all(Val::Px(1.0)),
                ..default()
            },
            ..default()
        },
    ));
}

// fn keyboard_input(mut keyboard: EventReader<KeyboardInput>) {}

fn mouse_input(
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut mouse: EventReader<MouseButtonInput>,
    mut ev_select: EventWriter<SelectInput>,
    mut ev_move: EventWriter<MoveInput>,
    mut drag_state: ResMut<DragState>,
) {
    if let Some(cursor_pos) = cursor_pos(&window, &camera) {
        for ev in mouse.read() {
            let modifier =
                if keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight) {
                    InputModifier::Append
                } else {
                    InputModifier::Overwrite
                };
            match ev.button {
                MouseButton::Left => match ev.state {
                    ButtonState::Pressed => {
                        ev_select.send(SelectInput {
                            position: InputPosition::Point(cursor_pos),
                            modifier,
                        });
                        *drag_state = DragState::Debounce(cursor_pos);
                    }
                    ButtonState::Released => {
                        *drag_state = DragState::Off;
                    }
                },
                MouseButton::Right => match ev.state {
                    ButtonState::Pressed => {
                        ev_move.send(MoveInput {
                            position: cursor_pos,
                            modifier,
                        });
                    }
                    ButtonState::Released => (),
                },
                _ => (),
            }
        }
    }
}

fn scroll_input(mut scroll: EventReader<MouseWheel>, mut ev_scroll: EventWriter<ScrollInput>) {
    for ev in scroll.read() {
        ev_scroll.send(ScrollInput(ev.y));
    }
}

fn drag_mouse_input(
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut drag_state: ResMut<DragState>,
    mut ev_select: EventWriter<SelectInput>,
) {
    match *drag_state {
        DragState::Off => (),
        DragState::Debounce(start_pos) => {
            *drag_state = match cursor_pos(&window, &camera) {
                Some(cursor_pos) => {
                    if cursor_pos.distance_squared(start_pos) > (16.0_f32).powi(2) {
                        DragState::On(start_pos)
                    } else {
                        DragState::Debounce(start_pos)
                    }
                }
                None => DragState::Off,
            }
        }
        DragState::On(start_pos) => {
            if let Some(cursor_pos) = cursor_pos(&window, &camera) {
                let modifier = if keyboard.pressed(KeyCode::ShiftLeft)
                    || keyboard.pressed(KeyCode::ShiftRight)
                {
                    InputModifier::Append
                } else {
                    InputModifier::Overwrite
                };
                ev_select.send(SelectInput {
                    position: InputPosition::Box(start_pos, cursor_pos),
                    modifier,
                });
            }
        }
    }
}

fn cursor_pos(
    window: &Query<&Window, With<PrimaryWindow>>,
    camera: &Query<(&Camera, &GlobalTransform), With<MainCamera>>,
) -> Option<Vec2> {
    let (camera, camera_transform) = camera.single();
    let window = window.single();
    window
        .cursor_position()
        .and_then(|cursor| camera.viewport_to_world(camera_transform, cursor))
        .map(|ray| ray.origin.truncate())
}

fn display_drag(
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    drag_state: Res<DragState>,
    mut drag_box: Query<&mut Style, With<DragBox>>,
) {
    let window = window.single();
    let mut style = drag_box.single_mut();
    if let (DragState::On(start_pos), Some(cursor_pos)) = (*drag_state, window.cursor_position()) {
        let (camera, camera_transform) = camera.single();
        let start_pos = camera.world_to_viewport(camera_transform, Vec3::from((start_pos, 0_f32)));
        if let Some(start_pos) = start_pos {
            style.display = Display::Grid;
            style.left = Val::Px(f32::min(start_pos.x, cursor_pos.x));
            style.top = Val::Px(f32::min(start_pos.y, cursor_pos.y));
            style.width = Val::Px(f32::abs(start_pos.x - cursor_pos.x));
            style.height = Val::Px(f32::abs(start_pos.y - cursor_pos.y));
        }
    } else {
        style.display = Display::None;
    }
}

#[derive(Event)]
pub struct ActivateInput {
    pub a: bool,
    pub b: bool,
    pub c: bool,
    pub d: bool,
}

fn keyboard_input(
    keyboard_state: Res<ButtonInput<KeyCode>>,
    mut keyboard: EventReader<KeyboardInput>,
    mut ev_activate: EventWriter<ActivateInput>,
) {
    for ev in keyboard.read() {
        if let (
            ButtonState::Pressed,
            KeyCode::Digit1 | KeyCode::Digit2 | KeyCode::Digit3 | KeyCode::Digit4,
        ) = (ev.state, ev.key_code)
        {
            let activate_input = ActivateInput {
                a: keyboard_state.pressed(KeyCode::Digit1),
                b: keyboard_state.pressed(KeyCode::Digit2),
                c: keyboard_state.pressed(KeyCode::Digit3),
                d: keyboard_state.pressed(KeyCode::Digit4),
            };
            ev_activate.send(activate_input);
        }
    }
}
