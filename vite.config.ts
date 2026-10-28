import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const protectBackendArtifacts = {
  name: 'protect-backend-artifacts',
  configureServer(server: { middlewares: { use: (handler: (req: { url?: string }, res: { statusCode: number; setHeader: (name: string, value: string) => void; end: (body: string) => void }, next: () => void) => void) => void } }) {
    server.middlewares.use((req, res, next) => {
      let pathname = (req.url || '/').split('?', 1)[0]
      try { pathname = decodeURIComponent(pathname) } catch { /* malformed paths will fail normally */ }
      pathname = pathname.replaceAll(String.fromCharCode(92), '/')
      pathname = new URL(pathname, 'http://vite.local').pathname
      if (/^\/backend(?:\/|$)/i.test(pathname)) {
        res.statusCode = 404
        res.setHeader('Cache-Control', 'no-store')
        res.end('Not Found')
        return
      }
      next()
    })
  },
}

export default defineConfig({
  plugins: [protectBackendArtifacts, react(), tailwindcss()],
  server: { fs: { deny: ['**/backend/**', '**/*.pkl'] } },
})
