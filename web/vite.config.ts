import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development the FastAPI app runs on :8000 and the UI on :5173; the proxy
// keeps every fetch relative to /api so the built app works unchanged when
// FastAPI serves web/dist itself.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
