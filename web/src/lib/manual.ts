export type ManualAction = { steering: number; throttle: number; brake: number };

export function manualActionFromKeys(keys: ReadonlySet<string>): ManualAction {
  return {
    steering: Number(keys.has('KeyD') || keys.has('ArrowRight')) - Number(keys.has('KeyA') || keys.has('ArrowLeft')),
    throttle: Number(keys.has('KeyW') || keys.has('ArrowUp')),
    brake: Number(keys.has('KeyS') || keys.has('ArrowDown') || keys.has('Space')),
  };
}
