/** Shortest-arc unit quaternion interpolation. Python frames encode [w,x,y,z]. */
export type Wxyz = [number, number, number, number];

export function slerpWxyz(a: Wxyz, b: Wxyz, t: number): Wxyz {
  const normalize = (q: Wxyz): Wxyz => {
    const length = Math.hypot(...q);
    if (length < 1e-10 || !Number.isFinite(length)) throw new Error('Invalid orientation quaternion');
    return q.map(x => x / length) as Wxyz;
  };
  const p = normalize(a);
  let q = normalize(b);
  let dot = p.reduce((sum, v, i) => sum + v * q[i], 0);
  if (dot < 0) { q = q.map(x => -x) as Wxyz; dot = -dot; }
  const u = Math.max(0, Math.min(1, t));
  if (dot > 0.9995) return normalize(p.map((x, i) => x + u * (q[i] - x)) as Wxyz);
  const angle = Math.acos(Math.max(-1, Math.min(1, dot)));
  const s = Math.sin(angle);
  const w0 = Math.sin((1 - u) * angle) / s;
  const w1 = Math.sin(u * angle) / s;
  return normalize(p.map((x, i) => x * w0 + q[i] * w1) as Wxyz);
}
