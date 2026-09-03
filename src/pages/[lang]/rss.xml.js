import { getCollection } from 'astro:content';
import rss from '@astrojs/rss';
import { SITE_DESCRIPTION, SITE_TITLE } from '../../consts';
import { HTML_LANG, LANGS, UI, localePath, splitId } from '../../i18n';

export async function getStaticPaths() {
	return LANGS.map((lang) => ({ params: { lang } }));
}

export async function GET(context) {
	const { lang } = context.params;
	const posts = (await getCollection('blog'))
		.filter((post) => splitId(post.id).lang === lang)
		.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());

	return rss({
		title: `${SITE_TITLE} — ${UI[lang].writing}`,
		description: SITE_DESCRIPTION,
		site: context.site,
		// Why: a reader subscribing to one language should not be handed the
		// other language's posts; the two feeds are disjoint by construction.
		customData: `<language>${HTML_LANG[lang]}</language>`,
		items: posts.map((post) => ({
			title: post.data.title,
			description: post.data.description,
			pubDate: post.data.pubDate,
			link: localePath(lang, `/blog/${splitId(post.id).slug}/`),
		})),
	});
}
