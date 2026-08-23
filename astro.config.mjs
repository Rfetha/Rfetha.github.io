// @ts-check

import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import { unified } from '@astrojs/markdown-remark';
import { defineConfig, fontProviders } from 'astro/config';

const lmroman = './src/assets/fonts';

// https://astro.build/config
export default defineConfig({
	site: 'https://rfetha.github.io',
	integrations: [mdx(), sitemap()],
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
