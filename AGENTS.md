## Development

When starting the dev server, use background mode:

```
astro dev --background
```

Manage the background server with `astro dev stop`, `astro dev status`, and `astro dev logs`.

## Language

Posts are written in **Turkish**; the site chrome (title, nav, dates, `Abstract`,
`Figure n.` labels) is English. The site publishes Turkish first — there is no
English version of a post yet, and none is a prerequisite for shipping one.

An English track is planned but **not built**: a TR/EN switch over parallel
`/tr/blog/<slug>/` and `/en/blog/<slug>/` routes. Until it lands, do not add
`hreflang`, a locale prefix, or a language-suffixed slug — the language belongs
in the path segment, never in the filename, and today's `/blog/<slug>/` URLs
must keep resolving. Rationale and the full constraint list are in `PRODUCT.md`
(untracked, local).

## Documentation

Full documentation: https://docs.astro.build

Consult these guides before working on related tasks:

- [Adding pages, dynamic routes, or middleware](https://docs.astro.build/en/guides/routing/)
- [Working with Astro components](https://docs.astro.build/en/basics/astro-components/)
- [Using React, Vue, Svelte, or other framework components](https://docs.astro.build/en/guides/framework-components/)
- [Adding or managing content](https://docs.astro.build/en/guides/content-collections/)
- [Adding styles or using Tailwind](https://docs.astro.build/en/guides/styling/)
- [Supporting multiple languages](https://docs.astro.build/en/guides/internationalization/)
