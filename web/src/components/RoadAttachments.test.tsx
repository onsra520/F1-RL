import { describe, expect, it } from 'vitest';
import { createTrack, pointAt, roadLength, sampleRoad, type TrackProp } from '../lib/track';
import { CheckpointGate, makeStrip } from './RoadAttachments';

describe('track-aligned interactive attachments', () => {
  it('booster geometry follows the elevated road rather than intersecting the ramp', () => {
    const base = createTrack();
    const ramp: TrackProp = { id: 'ramp-test', kind: 'ramp', t: 0.50, offset: 0, rotation: 0 };
    const booster: TrackProp = { id: 'booster-test', kind: 'booster', t: 0.555, offset: 0, rotation: 0 };
    const track = { ...base, props: [ramp, booster] };
    const mesh = makeStrip(track, booster, -2.4, 2.4, -3.1, 3.1, 0.22);
    const vertices = mesh.getAttribute('position');
    const length = roadLength(sampleRoad(track, 400));
    const heights = [];
    for (let i = 0; i < vertices.count; i += 2) {
      const step = i / (vertices.count - 2);
      const t = Math.max(0, Math.min(1, booster.t + (-3.1 + 6.2 * step) / length));
      const center = pointAt(track, t);
      expect(vertices.getY(i)).toBeCloseTo(center.y + 0.22, 4);
      expect(vertices.getY(i + 1)).toBeCloseTo(center.y + 0.22, 4);
      heights.push(vertices.getY(i));
    }
    expect(Math.max(...heights) - Math.min(...heights)).toBeGreaterThan(0.2);
    mesh.dispose();
  });

  it('road-aligned checkpoint geometry spans dynamic road width', () => {
    const track = { ...createTrack(), width: 17 };
    const prop: TrackProp = { id: 'check', kind: 'checkpoint', t: 0.3, offset: 4, rotation: 1.2 };
    const gate = CheckpointGate({ track, prop, selected: false });
    const center = pointAt(track, prop.t);
    expect(gate.props.position).toEqual([center.x, center.y + 0.09, center.z]);
    expect(gate.props.rotation).toEqual([0, center.heading, 0]);
    const supports = gate.props.children[0];
    expect(supports[0].props.position[0]).toBeCloseTo(-(17 / 2 + 0.65));
    expect(supports[1].props.position[0]).toBeCloseTo(17 / 2 + 0.65);
  });
});
