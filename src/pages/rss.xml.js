import { getCollection } from 'astro:content';
import rss from '@astrojs/rss';
import { SITE_DESCRIPTION, SITE_TITLE } from '../consts';
import { DEFAULT_LANG, HTML_LANG, localePath, splitId } from '../i18n';

/* Why: this route stays a real feed rather than becoming a redirect. The
   language split moved every page under a /<lang>/ prefix, but a feed reader
   already subscribed to /rss.xml cannot follow the meta-refresh a static
   redirect emits — it would simply go quiet. So /rss.xml keeps serving, in
   the default language, and /<lang>/rss.xml is the per-language feed. */
export async function GET(context) {
	const posts = (await getCollection('blog'))
		.filter((post) => splitId(post.id).lang === DEFAULT_LANG)
		.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());

	return rss({
		title: SITE_TITLE,
		description: SITE_DESCRIPTION,
		site: context.site,
		customData: `<language>${HTML_LANG[DEFAULT_LANG]}</language>`,
		items: posts.map((post) => ({
			title: post.data.title,
			description: post.data.description,
			pubDate: post.data.pubDate,
			link: localePath(DEFAULT_LANG, `/blog/${splitId(post.id).slug}/`),
		})),
	});
}
