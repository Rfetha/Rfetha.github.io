## Development

When starting the dev server, use background mode:

```
astro dev --background
```

Manage the background server with `astro dev stop`, `astro dev status`, and `astro dev logs`.

## Language

Every page ships in both languages. Turkish is the **source of truth**: a post
is written and verified against primary sources in Turkish, then translated.
English is the **default** — a bare `/` resolves to `/en/`.

`src/i18n.ts` is the only place that knows which languages exist; adding a third
is one entry there plus one content directory. Content lives in
`src/content/blog/<lang>/<slug>.md` — the slug is shared and the language is a
path segment, never a filename suffix. Pages generate under `/[lang]/`, and the
pre-split `/blog/<slug>/` URLs survive as `noindex` redirect stubs pointing at
Turkish, which is the language those URLs actually served.

Two rules when writing:

- **Add both files.** A post with only one language makes the switch land the
  reader on a 404.
- **Quotations are restored, not translated.** The English version of a post
  must carry the *original English sentence* from the cited paper. Translating
  the Turkish rendering back into English produces text that looks like a quote
  and is not one, which breaks the thing these posts are built on.

Rationale and the full constraint list are in `PRODUCT.md` (untracked, local).

## Documentation

Full documentation: https://docs.astro.build

Consult these guides before working on related tasks:

- [Adding pages, dynamic routes, or middleware](https://docs.astro.build/en/guides/routing/)
- [Working with Astro components](https://docs.astro.build/en/basics/astro-components/)
- [Using React, Vue, Svelte, or other framework components](https://docs.astro.build/en/guides/framework-components/)
- [Adding or managing content](https://docs.astro.build/en/guides/content-collections/)
- [Adding styles or using Tailwind](https://docs.astro.build/en/guides/styling/)
- [Supporting multiple languages](https://docs.astro.build/en/guides/internationalization/)
