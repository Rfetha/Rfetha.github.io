/* Why: the language lives in the path segment, never in a filename — a slug is
   shared by both languages (PRODUCT.md, "Deferred: the language switch"). This
   module is the single place that knows which languages exist, which one a
   bare URL means, and what the chrome says in each. Routes derive everything
   else from it, so adding a third language is one entry here plus one content
   directory. */

export const LANGS = ['en', 'tr'] as const;
export type Lang = (typeof LANGS)[number];

/** What `/` resolves to, and what a post's canonical is when both exist. */
export const DEFAULT_LANG: Lang = 'en';

/** BCP 47 tags for `<html lang>`, `hreflang`, and schema.org `inLanguage`. */
export const HTML_LANG: Record<Lang, string> = { en: 'en', tr: 'tr' };

/** The label the switch shows for the language it would take you to. */
export const LANG_LABEL: Record<Lang, string> = { en: 'EN', tr: 'TR' };

/** Why: a content id is `<lang>/<slug>` because the loader globs the whole
    blog directory. Splitting it here keeps that shape in one place. */
export function splitId(id: string): { lang: Lang; slug: string } {
	const [head, ...rest] = id.split('/');
	return { lang: head as Lang, slug: rest.join('/') };
}

export function isLang(value: string | undefined): value is Lang {
	return LANGS.includes(value as Lang);
}

/** `/en` + `/blog/x/` → `/en/blog/x/`. Every internal href goes through this. */
export function localePath(lang: Lang, path = '/'): string {
	const rest = path === '/' ? '' : path.startsWith('/') ? path : `/${path}`;
	return `/${lang}${rest}`;
}

export const UI = {
	en: {
		writing: 'Writing',
		writingDesc: 'Everything published here, newest first.',
		about: 'About',
		feed: 'Feed',
		allWriting: 'All writing',
		elsewhere: 'Elsewhere',
		sections: 'Sections',
		entries: (n: number) => `${n} ${n === 1 ? 'entry' : 'entries'}, newest first.`,
		empty: 'Nothing published yet.',
		emptyFeedPre: 'The ',
		emptyFeedLink: 'feed',
		emptyFeedPost: ' is live if you want to be told when that changes.',
		abstract:
			'I write to understand. These are working notes on machine learning systems and the theory underneath them, written while the material is still difficult rather than after it stopped being.',
		moreAbout: 'More about me',
		abstractLabel: 'Abstract.',
		/* Why: the control names the language it switches to, not the one you
		   are in — a button labelled with the current state reads as a status
		   line and gets clicked by accident. */
		switchTo: 'Türkçe',
		switchAria: 'Bu sayfayı Türkçe oku',
	},
	tr: {
		writing: 'Yazılar',
		writingDesc: 'Burada yayımlanan her şey, yeniden eskiye.',
		about: 'Hakkında',
		feed: 'Akış',
		allWriting: 'Bütün yazılar',
		elsewhere: 'Başka yerlerde',
		sections: 'Bölümler',
		entries: (n: number) => `${n} yazı, yeniden eskiye.`,
		empty: 'Henüz bir şey yayımlanmadı.',
		emptyFeedPre: 'Değiştiğinde haberdar olmak isterseniz ',
		emptyFeedLink: 'akış',
		emptyFeedPost: ' yayında.',
		abstract:
			'Anlamak için yazıyorum. Bunlar makine öğrenmesi sistemleri ve altındaki teori üzerine çalışma notları; malzeme hâlâ zorken yazıldılar, zorluğu geçtikten sonra değil.',
		moreAbout: 'Hakkımda',
		abstractLabel: 'Özet.',
		switchTo: 'English',
		switchAria: 'Read this page in English',
	},
} as const;

export type Strings = (typeof UI)[Lang];
