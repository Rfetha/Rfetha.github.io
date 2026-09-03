# Domain and discoverability

Two things live here: the checklist for moving off `rfetha.github.io`, and the
plan for being found — by search, and by the language models that answer
questions from a search index.

## The domain move

Nothing below is worth doing until the domain exists, and the move itself must
happen **before** any link building. Links earned on `rfetha.github.io` and
then redirected lose part of their weight; links earned on the real domain do
not.

The origin appears in exactly one place in the source — `site` in
`astro.config.mjs`. `robots.txt`, the sitemap, every canonical, every
`hreflang`, the OG image URLs and the BibTeX block in each post are all derived
from it, so they follow automatically.

1. `astro.config.mjs` — set `const site` to `https://rfetha.com`.
2. `public/CNAME` — new file, one line: `rfetha.com`. GitHub Pages reads this
   to claim the custom domain; without it the domain is dropped on every
   deploy.
3. GitHub → repo Settings → Pages → Custom domain → `rfetha.com`. Wait for the
   certificate to issue, then tick **Enforce HTTPS**.
4. DNS at the registrar:
   - apex `rfetha.com` → four `A` records at GitHub Pages'
     `185.199.108.153`, `.109.153`, `.110.153`, `.111.153`
     (confirm against GitHub's current Pages docs before entering them — these
     addresses have changed before)
   - `www` → `CNAME` to `rfetha.github.io`
5. Verify `https://rfetha.github.io/en/blog/why-decoder-only/` 301s to the new
   host. GitHub Pages does this on its own once a custom domain is set; it is
   what carries the old URLs, so check it rather than assume it.
6. Re-add the site as a new property in Search Console and Bing Webmaster
   Tools, and submit `https://rfetha.com/sitemap-index.xml` in both.

Old BibTeX entries that readers already copied will point at the `github.io`
URL. Step 5's redirect is what keeps those honest, which is the reason to
verify it.

## Being found

### Done in the source

- `<article>` no longer hard-codes `lang="tr"`. It predated the language split,
  when the chrome was English and the prose always Turkish; since the split it
  told every crawler and screen reader that the English posts were Turkish.
- The `BlogPosting` schema now carries `image`, pointing at the OG card that
  was already being generated. Google lists it as recommended for `Article`.
- `robots.txt` is generated from `site` instead of being a static file with a
  second, silently rotting copy of the origin.

### Done by hand, no code

In this order:

1. **Bing Webmaster Tools**, sitemap submitted. Bing is the index behind
   ChatGPT search and Copilot, and almost nobody registers there — the highest
   return for ten minutes of work on this list.
2. **Google Search Console**, sitemap submitted. It is what Gemini grounds
   against, and the only way to see which queries actually arrive.
3. **Distribution**, after the domain move: Hacker News, r/MachineLearning, X,
   LinkedIn. A single front page resolves the authority problem that no amount
   of on-page work will.
4. **More posts, cross-linked.** Two entries is not a topic. Five or six on one
   subject, linking to each other, is what reads as authority.

### What ranks, honestly

The head term `decoder only` is held by HuggingFace docs, Wikipedia and blogs
with thousands of inbound links. Page one there is not a near-term outcome and
aiming at it wastes the effort.

What is winnable is the question the post actually answers — *why* decoder-only
rather than encoder-decoder, the T5 comparison, the attention-mask difference.
Thin competition, and the post is better than what currently ranks. The Turkish
side has effectively no competition at all.

For the language models specifically: they answer from a live search index far
more than from training data, so indexing is the prerequisite, not a separate
track. The lever that is actually ours is having a specific, sourced,
quotable claim — "T5 (2019) tested this at equal compute and encoder-decoder
won seven of seven" is exactly the kind of sentence a model retrieves and
repeats. Leave such sentences standing on their own rather than buried
mid-paragraph.

### Deliberately not done

- **FAQ schema** — Google withdrew the rich result for ordinary sites in 2023.
- **`llms.txt`** — no provider reads it yet. Free, but expect nothing.
- **`og:locale`** — `hreflang` already carries the bilingual signal to search;
  this would only affect social previews.
- **Keyword density work of any kind.**
