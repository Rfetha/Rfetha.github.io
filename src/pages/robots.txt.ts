import type { APIRoute } from 'astro';

/* Why: the Sitemap line is an absolute URL, so a static public/robots.txt held
   a second copy of the origin that astro.config.mjs calls "one place to change
   when the site moves to its own domain" — and the copy that goes stale
   silently, because nothing renders it. Generating the file keeps the promise
   that comment makes. */
export const GET: APIRoute = ({ site }) =>
	new Response(`User-agent: *\nAllow: /\n\nSitemap: ${new URL('sitemap-index.xml', site)}\n`, {
		headers: { 'Content-Type': 'text/plain; charset=utf-8' },
	});
