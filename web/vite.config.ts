import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(__dirname, 'src') } },
  server: {
    // FastAPI (uv run fastapi dev app/main.py) serves the run event logs
    proxy: { '/api': 'http://localhost:8000' },
  },
})
