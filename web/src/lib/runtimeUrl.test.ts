import { describe, expect, it } from 'vitest';
import { resolveRuntimeUrl } from './runtimeUrl';

describe('runtime URL', () => {
  it('accepts loopback endpoints and rejects other origins', () => {
    expect(resolveRuntimeUrl('http://localhost:8001')).toBe('http://localhost:8001');
    expect(resolveRuntimeUrl('https://example.com')).toBe('http://127.0.0.1:8000');
    expect(resolveRuntimeUrl('')).toBe('http://127.0.0.1:8000');
  });
});
