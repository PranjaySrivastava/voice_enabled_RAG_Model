import type { NextConfig } from "next";

const BACKEND_INTERNAL_URL = process.env.BACKEND_INTERNAL_URL || process.env.NEXT_PUBLIC_HTTP_BACKEND_URL || "https://dhwani-voice-backend.onrender.com";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${BACKEND_INTERNAL_URL}/api/:path*`,
      },
      {
        source: "/ws/:path*",
        destination: `${BACKEND_INTERNAL_URL}/ws/:path*`,
      },
    ];
  },
};

export default nextConfig;
