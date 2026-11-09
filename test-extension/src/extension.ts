import { initialize, type ActivationContext } from "@ableton-extensions/sdk";

export function activate(activation: ActivationContext) {
  const context = initialize(activation, "1.0.0");

  const { tempo } = context.application.song;
  console.log(
    `Hello from test-extension! Your Live Set's tempo is: ${tempo} bpm.`,
  );
}
