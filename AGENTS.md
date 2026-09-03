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

## Writing a post

**Posts are authored directly in `src/content/blog/<lang>/<slug>.md`. There is
no `drafts/` directory** — one existed, and keeping a parallel copy produced a
three-month divergence in which every correction sat in the draft and none of
it reached the published page. Do not reintroduce that pattern.

Unfinished work goes on a branch off `main`, merged when the post is ready:
a push to `main` deploys the live site with no confirmation step. A local-only
`workspace` branch exists for exactly this.

Filename is the URL segment — lowercase, hyphens, ASCII. `kv-cache.md` serves
at `/tr/blog/kv-cache/` and `/en/blog/kv-cache/`.

Frontmatter: `title` and `description` are required (`description` is the list
line and the `<meta name="description">`, so it is never left empty); `pubDate`
orders the index; `updatedDate` is optional and renders a "revised" line.

Two house constructs, both written as raw HTML with **blank lines inside the
tags** or the markdown will not be processed:

```html
<figure>

$$ ... $$

<figcaption><b>Figure 1.</b> Caption.</figcaption>

</figure>
```

```html
<aside class="sidenote">

Source citation, measurement condition, or aside.

</aside>
```

The margin carries citations, "this number was measured under these
conditions" caveats, and asides — never a step of the main argument; a reader
who skips the margin must still get the whole case. `DESIGN.md` is the
authority on both.

Obsidian wikilinks (`[[link]]`, `![[image.png]]`) do **not** render — they
appear literally. Use `[text](/tr/blog/slug/)` and put images in `public/`.

Verification material lives **outside the repo**, in `../blog-research/`: the
evidence files that record each claim's primary source, and the scripts that
produce the numbers printed in the posts. When a post's measured number or
verbatim quote changes, update the record there too.

## Documentation

Full documentation: https://docs.astro.build

Consult these guides before working on related tasks:

- [Adding pages, dynamic routes, or middleware](https://docs.astro.build/en/guides/routing/)
- [Working with Astro components](https://docs.astro.build/en/basics/astro-components/)
- [Using React, Vue, Svelte, or other framework components](https://docs.astro.build/en/guides/framework-components/)
- [Adding or managing content](https://docs.astro.build/en/guides/content-collections/)
- [Adding styles or using Tailwind](https://docs.astro.build/en/guides/styling/)
- [Supporting multiple languages](https://docs.astro.build/en/guides/internationalization/)
