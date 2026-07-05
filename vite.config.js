import { defineConfig } from 'vite'
import * as path from 'path'

export default defineConfig({
  root: '',
  base: './',
  build: {
    outDir: 'dist',
    cssCodeSplit: true,
    // Disable source maps in production to keep dist clean
    sourcemap: false,
    rollupOptions: {
      input: {
        demo: './index.html',
      },
    },
  },
  server: {
    // host: true exposes the dev server on the local network (0.0.0.0),
    // required for testing on mobile devices and the Muse headset BT workflow.
    host: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
