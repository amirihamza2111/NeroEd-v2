/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    // Override with NEXT_PUBLIC_API_URL in .env.local to point at another backend.
    NEXT_PUBLIC_API_URL:
      process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  },
}
module.exports = nextConfig
