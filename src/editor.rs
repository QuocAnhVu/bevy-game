use bevy::prelude::*;

pub struct EditorPlugin;
impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {}
}

fn setup_toggle_ui(mut commands: Commands) {
    commands.spawn(NodeBundle {
        style: Style { ..default() },
        ..default()
    });
}
