import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import { resolveRuntimeUrl } from './src/lib/runtimeUrl';

export default defineConfig(({ mode }) => {
  const runtimeUrl = resolveRuntimeUrl(loadEnv(mode, process.cwd(), 'VITE_').VITE_RUNTIME_URL || '');
  return {
    plugins: [react(), { name: 'runtime-url-metadata', transformIndexHtml: () => [
      { tag: 'meta', attrs: { name: 'f1rl-runtime-url', content: runtimeUrl }, injectTo: 'head' },
    ] }],
    server: { port: 5173, strictPort: false },
    build: { manifest: true },
  };
});
