import { afterEach, describe, expect, it, vi } from 'vitest';
import { parseRuntimeSocketEvent, RuntimeClient } from './runtime';

const track = {
  track_id: 'straight-sprint-v1', split: 'TRAIN' as const, purpose: 'development' as const,
  world_sha256: 'a'.repeat(64),
  preview: { schema: 'f1rl_racing_sandbox_draft_v1' as const, id: 'straight-sprint-v1', name: 'Straight', width: 12,
    points: [{ x: 0, z: 0 }, { x: 0, z: 20 }], props: [] },
  road_samples: [{ x: 0, y: 0, z: 0, t: 0, heading: 0, nx: 1, nz: 0 }, { x: 0, y: 0, z: 20, t: 1, heading: 0, nx: 1, nz: 0 }],
};

afterEach(() => vi.unstubAllGlobals());

describe('Python runtime HTTP client', () => {
  it('loads the read-only frozen catalog and rejects a non-TRAIN track', async () => {
    const models = ['A', 'B', 'C', 'D'].flatMap((candidate) => [0, 1, 2].map((seed) => ({
      model_id: `${candidate}-seed${seed}`, sha256: 'b'.repeat(64), bytes: 100,
    })));
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ models }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ tracks: [track] }), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const result = await new RuntimeClient().catalog();
    expect(result.models).toHaveLength(12);
    expect(result.tracks[0].track_id).toBe('straight-sprint-v1');
  });

  it('sends only model ID, track ID, and explicit seed when creating a session', async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ session_id: 's1' }), { status: 201 }));
    vi.stubGlobal('fetch', fetchMock);
    await new RuntimeClient().createPpoSession('A-seed0', 'straight-sprint-v1', 412);
    const [url, options] = fetchMock.mock.calls[0] as [URL, RequestInit];
    expect(url.pathname).toBe('/api/v1/sessions');
    expect(JSON.parse(String(options.body))).toEqual({ model_id: 'A-seed0', track_id: 'straight-sprint-v1', seed: 412 });
    expect(JSON.stringify(options.body)).not.toContain('model_path');
  });

  it('validates and starts custom editor maps through explicit development endpoints', async () => {
    const conversion = {
      schema_version: 'f1rl_custom_track_conversion_v1', development_only: true,
      track_id: 'sandbox-123', track_sha256: 'c'.repeat(64), preview: track.preview,
      road_samples: track.road_samples, auto_bridge_count: 0, ppo_compatible: true,
      surface_count: 10, checkpoint_count: 1, obstacle_count: 0, boost_pad_count: 0, warnings: [],
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(conversion), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify({ session_id: 'custom-session' }), { status: 201 }));
    vi.stubGlobal('fetch', fetchMock);
    const client = new RuntimeClient();
    const doc = { schema: 'f1rl_racing_sandbox_draft_v1', id: 'draft', name: 'Draft', width: 12, points: track.preview.points, props: [] };
    expect((await client.validateCustomTrack(doc)).development_only).toBe(true);
    await client.createCustomSession(doc, 'manual', 5);
    expect(JSON.parse(String((fetchMock.mock.calls[1][1] as RequestInit).body))).toEqual({
      mode: 'manual', seed: 5, custom_track: doc,
    });
  });

  it('turns structured backend failures into typed client errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: {
      code: 'invalid_track', message: 'unknown TRAIN track id',
    } }), { status: 400 })));
    await expect(new RuntimeClient().createPpoSession('A-seed0', 'x', 1)).rejects.toMatchObject({
      code: 'invalid_track', status: 400,
    });
  });

  it('validates versioned live events and rejects frames from another session', () => {
    const frame = {
      schema_version: 'f1rl_runtime_frame_v1', tick: 0, time_s: 0, dt_s: 0.1,
      position: [1, 2, 3], velocity: [0, 0, 0], acceleration_mps2: [0, 0, 0],
      orientation_wxyz: [1, 0, 0, 0], heading_rad: 0, speed_mps: 0, grounded: true,
      action: { steering: 0, throttle: 0, brake: 0 }, route_progress_m: 0,
      route_progress_fraction: 0, route_highwater_m: 0, next_checkpoint_index: 0,
      checkpoint_count: 1, completed_checkpoint_ids: [], checkpoint_crossings: [],
      collision_count: 0, collision_ids: [], physical_event_types: [], outcome: null,
    };
    const event = { protocol_version: 'f1rl_runtime_ws_v1', type: 'frame', session_id: 's1', data: frame };
    expect(parseRuntimeSocketEvent(event, 's1').type).toBe('frame');
    expect(() => parseRuntimeSocketEvent(event, 's2')).toThrow();
    expect(() => parseRuntimeSocketEvent({ ...event, protocol_version: 'v0' })).toThrow();
  });
});
