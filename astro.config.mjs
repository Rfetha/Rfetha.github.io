// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import { unified } from '@astrojs/markdown-remark';
import { defineConfig, fontProviders } from 'astro/config';
import { existsSync, readFileSync, readdirSync } from 'node:fs';

// Why: one place to change when the site moves to its own domain. The sitemap
// serializer below slices this prefix off, so it has to be the same string.
const site = 'https://rfetha.github.io';
const lmroman = './src/assets/fonts';
const posts = './src/content/blog';

// Why: the sitemap integration emits a bare <loc> for every route. Two things
// are missing from that and neither is reachable from inside the integration:
// a date per post, and the knowledge that /blog/ is an empty shell until the
// first post exists. Both are read off disk here, once, at config time.

/** Post pathname -> the date the entry itself claims. Empty until a post lands. */
const postDates = new Map(
	// Why: the directory is empty until the first post, and git does not track
	// empty directories — so a fresh CI checkout has no such path at all. Absent
	// and empty mean the same thing here: no posts.
	(existsSync(posts) ? readdirSync(posts, { recursive: true, withFileTypes: true }) : [])
		.filter((e) => e.isFile() && /\.mdx?$/.test(e.name))
		.flatMap((e) => {
			const file = `${e.parentPath}/${e.name}`;
			// Why: updatedDate wins when present — lastmod means last modified, and
			// claiming the original date for a revised post is the inaccuracy Google
			// says makes it stop trusting the signal site-wide.
			const raw = readFileSync(file, 'utf8').split(/^---$/m)[1] ?? '';
			const stamp = (raw.match(/^updatedDate:\s*(.+)$/m) ??
				raw.match(/^pubDate:\s*(.+)$/m))?.[1];
			const date = stamp && new Date(stamp.trim().replace(/^['"]|['"]$/g, ''));
			if (!date || Number.isNaN(date.valueOf())) return [];
			const id = file
				.slice(posts.length + 1)
				.replace(/\.mdx?$/, '')
				.replace(/(^|\/)index$/, '');
			return [[`/blog/${id}/`.replace(/\/+$/, '/'), date]];
		}),
);

// https://astro.build/config
export default defineConfig({
	site,
	integrations: [
		mdx(),
		sitemap({
			// Why: with no posts, /blog/ is a heading over an empty list. Submitting
			// it asks Google to index a page with nothing on it. The check is a
			// directory read, so the route returns to the sitemap on its own the
			// build after the first post is written.
			filter: (page) => postDates.size > 0 || !page.endsWith('/blog/'),
			serialize(item) {
				// Why: sliced rather than parsed with `new URL`. Every entry here was
				// built by the integration from `site`, so the prefix is guaranteed —
				// and slicing has no throwing path to guard against.
				const date = postDates.get(item.url.slice(site.length));
				// Why: no lastmod at all beats a guessed one. The static pages get
				// none because nothing on disk records when they last changed.
				if (date) item.lastmod = date.toISOString();
				return item;
			},
		}),
	],
	markdown: {
		// Why: the flat remarkPlugins/rehypePlugins keys are deprecated and go in
		// a future major. The pipeline is the same, it just hangs off `processor`.
		processor: unified({
			remarkPlugins: [remarkMath],
			rehypePlugins: [rehypeKatex],
		}),
		// Why: `css-variables` hands syntax colour to global.css, so code blocks
		// belong to this site's palette instead of importing a foreign theme.
		shikiConfig: { theme: 'css-variables' },
	},
	fonts: [
		{
			// Why: Latin Modern Roman is the Unicode successor to Computer Modern.
			// KaTeX renders math in KaTeX_Main (Computer Modern) but that face drops
			// accented Latin — Schölkopf, Poincaré, naïve would break mid-word.
			// Same drawing, full coverage, so prose and formulas share one voice.
			provider: fontProviders.local(),
			name: 'Latin Modern Roman',
			cssVariable: '--font-lmroman',
			fallbacks: ['Georgia', 'serif'],
			options: {
				variants: [
					{
						src: [`${lmroman}/lmroman-regular.woff`],
						weight: 400,
						style: 'normal',
						display: 'swap',
					},
					{
						src: [`${lmroman}/lmroman-italic.woff`],
						weight: 400,
						style: 'italic',
						display: 'swap',
					},
					{
						src: [`${lmroman}/lmroman-bold.woff`],
						weight: 700,
						style: 'normal',
						display: 'swap',
					},
					{
						src: [`${lmroman}/lmroman-bolditalic.woff`],
						weight: 700,
						style: 'italic',
						display: 'swap',
					},
				],
			},
		},
		{
			// Why: a drawn small-caps face. Synthetic `font-variant: small-caps`
			// scales down capitals and thins the stems, which reads as a defect
			// next to a real serif.
			provider: fontProviders.local(),
			name: 'Latin Modern Roman Caps',
			cssVariable: '--font-lmcaps',
			fallbacks: ['Georgia', 'serif'],
			options: {
				variants: [
					{
						src: [`${lmroman}/lmroman-caps.woff`],
						weight: 400,
						style: 'normal',
						display: 'swap',
					},
				],
			},
		},
	],
});
