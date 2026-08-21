export const BACKEND_URL =
  process.env.NEXT_PUBLIC_HTTP_BACKEND_URL ||
  'https://dhwani-voice-backend.onrender.com';

/**
 * Universal backend fetcher that seamlessly proxies through Next.js /api
 * or falls back directly to the live Render cloud backend if running on Vercel edge.
 */
export async function fetchBackend(
  endpoint: string,
  options?: RequestInit
): Promise<Response> {
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;

  // 1. Try relative route first (works in local dev & Next.js rewrites)
  try {
    const res = await fetch(cleanEndpoint, options);
    if (res.ok || res.status === 400 || res.status === 422) {
      return res;
    }
  } catch (e) {
    // Relative fetch failed, fall through to direct backend
  }

  // 2. Direct fallback to live Render backend with 1 auto-retry for cold starts
  const directUrl = `${BACKEND_URL.replace(/\/$/, '')}${cleanEndpoint}`;
  try {
    return await fetch(directUrl, options);
  } catch (err) {
    // Retry once after 2.5 seconds in case Render container was waking up from cold sleep
    await new Promise((resolve) => setTimeout(resolve, 2500));
    return await fetch(directUrl, options);
  }
}
