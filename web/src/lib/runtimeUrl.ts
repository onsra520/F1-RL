export function resolveRuntimeUrl(configured: string): string {
  return /^http:\/\/(?:127\.0\.0\.1|localhost):\d+$/.test(configured)
    ? configured : 'http://127.0.0.1:8000';
}
