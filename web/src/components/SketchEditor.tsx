import { useRef, useState } from 'react';
import { MousePointer2, Pencil, RotateCcw, WandSparkles } from 'lucide-react';
import type { Point2 } from '../lib/track';
import { EXAMPLE_POINTS } from '../lib/track';

type Tool = 'draw' | 'nodes';
const distance = (a: Point2, b: Point2) => Math.hypot(a.x - b.x, a.z - b.z);
function toPath(points: Point2[]): string {
  if (points.length === 0) return '';
  if (points.length === 1) return `M ${points[0].x} ${points[0].z}`;
  let path = `M ${points[0].x} ${points[0].z}`;
  for (let i = 0; i < points.length - 1; i++) {
    const a = points[Math.max(0, i - 1)], b = points[i], c = points[i + 1], d = points[Math.min(points.length - 1, i + 2)];
    path += ` C ${b.x + (c.x - a.x) / 6} ${b.z + (c.z - a.z) / 6}, ${c.x - (d.x - b.x) / 6} ${c.z - (d.z - b.z) / 6}, ${c.x} ${c.z}`;
  }
  return path;
}
function simplify(points: Point2[]): Point2[] {
  // Drag sampling is intentionally bounded: route geometry stays editable.
  const result: Point2[] = [];
  for (const p of points) if (!result.length || distance(result[result.length - 1], p) >= 5) result.push(p);
  const last = points[points.length - 1];
  if (last && result.length && distance(result[result.length - 1], last) > 0.1) result.push(last);
  return result.slice(0, 512);
}

export default function SketchEditor({ points, onChange, onConvert }: {
  points: Point2[];
  onChange: (next: Point2[]) => void;
  onConvert: () => void;
}) {
  const [tool, setTool] = useState<Tool>('draw');
  const [drawing, setDrawing] = useState<Point2[] | null>(null);
  const [dragIndex, setDragIndex] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const current = drawing ?? points;
  const locate = (e: React.PointerEvent<SVGSVGElement>): Point2 => {
    const svg = svgRef.current!;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: 0, z: 0 };
    const p = svg.createSVGPoint();
    p.x = e.clientX; p.y = e.clientY;
    const mapped = p.matrixTransform(ctm.inverse());
    return { x: Math.max(-108, Math.min(108, mapped.x)), z: Math.max(-78, Math.min(78, mapped.y)) };
  };
  const down = (e: React.PointerEvent<SVGSVGElement>) => {
    if (e.button !== 0 || dragIndex !== null) return;
    if (tool === 'draw') setDrawing([locate(e)]);
    else if (points.length < 512) onChange([...points, locate(e)]);
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const move = (e: React.PointerEvent<SVGSVGElement>) => {
    if (dragIndex !== null) {
      onChange(points.map((p, i) => i === dragIndex ? locate(e) : p));
    } else if (drawing) {
      const p = locate(e);
      setDrawing((prev) => prev && distance(prev[prev.length - 1], p) > 1.8 ? [...prev, p] : prev);
    }
  };
  const finish = () => {
    if (drawing && drawing.length >= 2) {
      const path = simplify(drawing);
      if (path.length >= 2) onChange(path);
    }
    setDrawing(null);
    setDragIndex(null);
  };
  const startMove = (e: React.PointerEvent<SVGCircleElement>, index: number) => {
    e.stopPropagation();
    setDragIndex(index);
    svgRef.current?.setPointerCapture(e.pointerId);
  };

  return (
    <div className="sketch-stage">
      <div className="sketch-tools" role="toolbar" aria-label="Sketch tools">
        <button className={tool === 'draw' ? 'tool active' : 'tool'} onClick={() => setTool('draw')}><Pencil size={17} /> Freehand</button>
        <button className={tool === 'nodes' ? 'tool active' : 'tool'} onClick={() => setTool('nodes')}><MousePointer2 size={17} /> Points</button>
        <span className="toolbar-gap" />
        <button className="tool" onClick={() => { onChange(EXAMPLE_POINTS.map((p) => ({ ...p }))); setDrawing(null); }} title="Restore sample route"><RotateCcw size={17} /> Sample</button>
        <button className="tool" onClick={() => { onChange([]); setDrawing(null); }}>Clear</button>
      </div>
      <svg
        ref={svgRef}
        className="sketch-canvas"
        viewBox="-110 -82 220 164"
        preserveAspectRatio="xMidYMid meet"
        onPointerDown={down} onPointerMove={move} onPointerUp={finish} onPointerCancel={finish}
        aria-label="Draw a centerline by dragging or clicking control points"
      >
        <defs>
          <pattern id="small-grid" x="0" y="0" width="10" height="10" patternUnits="userSpaceOnUse">
            <path d="M 10 0 H 0 V 10" fill="none" stroke="#71818b" strokeOpacity=".14" strokeWidth=".45" />
          </pattern>
          <pattern id="large-grid" x="0" y="0" width="50" height="50" patternUnits="userSpaceOnUse">
            <rect width="50" height="50" fill="url(#small-grid)" />
            <path d="M 50 0 H 0 V 50" fill="none" stroke="#71818b" strokeOpacity=".18" strokeWidth=".8" />
          </pattern>
        </defs>
        <rect x="-110" y="-82" width="220" height="164" fill="url(#large-grid)" />
        {current.length > 1 && <>
          <path d={toPath(current)} stroke="#000000" strokeWidth="16" fill="none" strokeLinecap="round" strokeLinejoin="round" opacity=".18" transform="translate(1,2)" />
          <path d={toPath(current)} stroke="#c0c9c8" strokeWidth="13" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          <path d={toPath(current)} stroke="#424e51" strokeWidth="11.8" fill="none" strokeLinecap="round" strokeLinejoin="round" />
          <path d={toPath(current)} stroke="#ffe18d" strokeWidth=".7" fill="none" strokeDasharray="3 4" strokeLinecap="round" />
        </>}
        {current.length > 0 && <>
          <circle cx={current[0].x} cy={current[0].z} r="4" fill="#bdf579" stroke="#20352a" strokeWidth="1.8" />
          <circle cx={current[current.length - 1].x} cy={current[current.length - 1].z} r="4" fill="#ff8665" stroke="#423027" strokeWidth="1.8" />
        </>}
        {tool === 'nodes' && current.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.z} r="3.2" fill="#ffffff" stroke="#ffb45e" strokeWidth="1.6"
            className="sketch-node" onPointerDown={(e) => startMove(e, i)} />
        ))}
      </svg>
      <div className="sketch-hint"><span className="keycap">{tool === 'draw' ? 'DRAG' : 'CLICK'}</span> {tool === 'draw' ? 'Draw a path with your mouse' : 'Add points · drag handles to reshape'}</div>
      <div className="sketch-action">
        <div><strong>Your track starts here.</strong><span>Draw one line. We'll make it drivable.</span></div>
        <button className="primary-button" onClick={onConvert} disabled={points.length < 2}><WandSparkles size={18} /> Build in 3D <span>→</span></button>
      </div>
    </div>
  );
}
