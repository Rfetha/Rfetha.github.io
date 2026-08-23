import type { APIRoute } from 'astro';
import { getCollection } from 'astro:content';
import fs from 'node:fs/promises';
import path from 'node:path';
import satori from 'satori';
import sharp from 'sharp';
import { AFFILIATION, SITE_DESCRIPTION, SITE_TITLE } from '../../consts';

// Why: satori needs real font bytes and reads ttf/otf/woff only — the same
// woff files the site serves, so the card is set in the site's own face.
// Resolved from cwd, not import.meta.url: this module gets bundled into
// dist/.prerender/chunks at build time and a relative URL would follow it there.
const font = (name: string) =>
	fs.readFile(path.join(process.cwd(), 'src', 'assets', 'fonts', name));

const INK = '#1a1a1a';
const MUTED = '#454545';
const RULE = '#dadada';
const SIG = '#8c1515';

interface Card {
	title: string;
	subtitle: string;
	/** The site card's title is already the name; the footer must not repeat it. */
	showName: boolean;
}

export async function getStaticPaths() {
	const posts = await getCollection('blog');
	return [
		{
			params: { slug: 'default' },
			props: { title: SITE_TITLE, subtitle: SITE_DESCRIPTION, showName: false } satisfies Card,
		},
		...posts.map((post) => ({
			params: { slug: `blog/${post.id}` },
			props: {
				title: post.data.title,
				subtitle: post.data.description,
				showName: true,
			} satisfies Card,
		})),
	];
}

export const GET: APIRoute = async ({ props }) => {
	const { title, subtitle, showName } = props as Card;

	const svg = await satori(
		{
			type: 'div',
			props: {
				style: {
					width: '1200px',
					height: '630px',
					display: 'flex',
					flexDirection: 'column',
					justifyContent: 'space-between',
					backgroundColor: '#ffffff',
					padding: '72px 80px',
					// the masthead rule, at card scale
					borderTop: `10px solid ${INK}`,
					fontFamily: 'LM Roman',
					color: INK,
				},
				children: [
					{
						type: 'div',
						props: {
							style: { display: 'flex', flexDirection: 'column' },
							children: [
								{
									type: 'div',
									props: {
										style: {
											fontSize: '64px',
											lineHeight: 1.18,
											letterSpacing: '-0.01em',
											// three lines at most, then satori clips
											maxHeight: '230px',
											overflow: 'hidden',
										},
										children: title,
									},
								},
								{
									type: 'div',
									props: {
										style: {
											marginTop: '28px',
											fontSize: '28px',
											lineHeight: 1.45,
											color: MUTED,
											fontStyle: 'italic',
											maxHeight: '124px',
											overflow: 'hidden',
										},
										children: subtitle,
									},
								},
							],
						},
					},
					{
						type: 'div',
						props: {
							style: {
								display: 'flex',
								alignItems: 'center',
								borderTop: `2px solid ${RULE}`,
								paddingTop: '26px',
							},
							children: [
								// the single chromatic mark
								{
									type: 'div',
									props: {
										style: {
											width: '14px',
											height: '14px',
											backgroundColor: SIG,
											marginRight: '18px',
										},
										children: '',
									},
								},
								...(showName
									? [
											{
												type: 'div',
												props: { style: { fontSize: '26px' }, children: SITE_TITLE },
											},
										]
									: []),
								{
									type: 'div',
									props: {
										style: {
											fontSize: '22px',
											color: MUTED,
											fontStyle: 'italic',
											marginLeft: showName ? '18px' : '0px',
										},
										children: showName ? AFFILIATION : `${SITE_TITLE} · ${AFFILIATION}`,
									},
								},
							],
						},
					},
				],
			},
		},
		{
			width: 1200,
			height: 630,
			fonts: [
				{ name: 'LM Roman', data: await font('lmroman-regular.woff'), weight: 400, style: 'normal' },
				{ name: 'LM Roman', data: await font('lmroman-italic.woff'), weight: 400, style: 'italic' },
			],
		},
	);

	const png = await sharp(Buffer.from(svg)).png().toBuffer();

	return new Response(new Uint8Array(png), {
		headers: {
			'Content-Type': 'image/png',
			'Cache-Control': 'public, max-age=31536000, immutable',
		},
	});
};
