# rfetha.github.io

Working notes on machine learning systems and the theory underneath them, written
while the material is still difficult rather than after it stopped being.

**→ [rfetha.github.io](https://rfetha.github.io)**

Posts are written in English and Turkish. Both languages are first-class: each post
exists as a pair under `src/content/blog/en/` and `src/content/blog/tr/`, sharing a slug.

## Stack

Astro 7 with MDX, static output, deployed to GitHub Pages by `.github/workflows/deploy.yml`
on push to the default branch.

| concern | how |
| :--- | :--- |
| Math | `remark-math` + `rehype-katex` — KaTeX renders at build time, no client-side JS |
| Open Graph images | `satori` + `sharp` — generated per post at build |
| Syndication | `@astrojs/rss` and `@astrojs/sitemap` |
| Types | `astro check` over content-collection schemas |

## Local development

Requires Node >= 22.12.

```sh
npm install
npm run dev      # http://localhost:4321
npm run build    # static output in dist/
npm run preview  # serve the build
```

## Layout

```
src/
  content/blog/{en,tr}/   # posts, paired by slug across the two languages
  components/             # Astro components
  layouts/                # page shells
  pages/                  # routes, RSS and OG-image endpoints
public/                   # static assets served as-is
```

Design and SEO decisions are recorded in [`DESIGN.md`](DESIGN.md) and [`SEO.md`](SEO.md).
