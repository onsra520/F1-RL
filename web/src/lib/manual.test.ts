import { describe, expect, it } from 'vitest';
import { manualActionFromKeys } from './manual';

describe('manualActionFromKeys', () => {
  it('maps WASD to bounded steering, throttle, and brake actions', () => {
    expect(manualActionFromKeys(new Set(['KeyA', 'KeyW', 'KeyS'])))
      .toEqual({ steering: -1, throttle: 1, brake: 1 });
    expect(manualActionFromKeys(new Set(['KeyD', 'KeyW'])))
      .toEqual({ steering: 1, throttle: 1, brake: 0 });
  });

  it('maps arrow keys and Space to the same actions', () => {
    expect(manualActionFromKeys(new Set(['ArrowLeft', 'ArrowUp', 'Space'])))
      .toEqual({ steering: -1, throttle: 1, brake: 1 });
    expect(manualActionFromKeys(new Set(['ArrowRight', 'ArrowDown'])))
      .toEqual({ steering: 1, throttle: 0, brake: 1 });
  });

  it('neutralizes opposing steering and released input', () => {
    expect(manualActionFromKeys(new Set(['KeyA', 'ArrowRight'])))
      .toEqual({ steering: 0, throttle: 0, brake: 0 });
    expect(manualActionFromKeys(new Set()))
      .toEqual({ steering: 0, throttle: 0, brake: 0 });
  });
});
