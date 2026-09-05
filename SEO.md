# Domain and discoverability

Two things live here: the checklist for moving off `rfetha.github.io`, and the
plan for being found — by search, and by the language models that answer
questions from a search index.

## The move to rfetha.com

The blog goes on the apex. `rfetha.com` is the handle the site is already known
by, so it is the identity domain, and the property that should collect every
link belongs on it. The other projects take subdomains off the same name:

    rfetha.com             the blog
    alibinoir.rfetha.com   the Shopify game
    <project>.rfetha.com   whatever follows

Google treats a subdomain as a separate site. That is the right shape for
projects that really are separate and the wrong one for the flagship. The
earlier plan put the blog on `ferhatersoy.rfetha.com`, to get the name into the
hostname; a hostname keyword is worth close to nothing next to the `<title>`,
the `h1` and the `Person` schema, which already carry the name, so it would
have bought nothing and moved the blog's authority one level down — leaving the
apex empty, which is dead weight, because the apex is the first address a
person or a crawler tries.

Nothing goes on the apex *besides* the blog either — no hub page listing the
projects. It would compete with the blog's own home page for the one query that
matters, the name, and both would be thinner for it. `/about/` is already that
page, with content behind it.

DNS moves to Cloudflare; hosting stays on GitHub Pages. Cloudflare is worth
having for the DNS, the analytics beacon, and — once the orange cloud is on —
edge redirect rules. None of those need the origin to move. Cloudflare Pages is
in maintenance mode as of 2026, with new investment going to Workers Static
Assets, so moving the origin would buy a platform migration rather than a
feature.

Nothing below is worth doing until the domain exists, and the move itself must
happen **before** any link building. Links earned on `rfetha.github.io` and
then redirected lose part of their weight; links earned on the real domain do
not.

The origin appears in exactly one place in the source — `site` in
`astro.config.mjs`. `robots.txt`, the sitemap, every canonical, every
`hreflang`, the OG image URLs and the BibTeX block in each post are all derived
from it, so they follow automatically.

1. Buy `rfetha.com`. At the registrar, point the nameservers at the two
   Cloudflare assigns. Minutes usually, up to a day.
2. `astro.config.mjs` — set `const site` to `https://rfetha.com`.
3. `public/CNAME` — new file, one line: `rfetha.com`. GitHub Pages reads this
   to claim the custom domain; without it the domain is dropped on every
   deploy.
4. Cloudflare DNS, all records **proxy off, grey cloud**:
   - apex `rfetha.com` → four `A` records at GitHub Pages'
     `185.199.108.153`, `.109.153`, `.110.153`, `.111.153`
     (confirm against GitHub's current Pages docs before entering them — these
     addresses have changed before)
   - `www` → `CNAME` to `rfetha.github.io`

   The grey cloud is not a preference. Proxied, GitHub cannot complete the HTTP
   challenge Let's Encrypt needs and HTTPS never provisions at all. The orange
   cloud can go on afterwards, once the certificate exists, with SSL/TLS mode
   Full (Strict).
5. GitHub → repo Settings → Pages → Custom domain → `rfetha.com`. Wait for the
   certificate to issue, then tick **Enforce HTTPS**.
6. Verify `https://rfetha.github.io/en/blog/why-decoder-only/` 301s to the new
   host. GitHub Pages does this on its own once a custom domain is set; it is
   what carries the old URLs, so check it rather than assume it.
7. Search Console and Bing Webmaster Tools: add `rfetha.com` and submit
   `https://rfetha.com/sitemap-index.xml`. In Search Console take a **Domain**
   property, not a URL-prefix one — verified by a single DNS TXT record, it
   covers the apex and every project subdomain that follows, so no later site
   needs its own verification. `public/google5aecd7a743b2f1a7.html` only ever
   verified the github.io property; keep that property, it is where the old
   URLs are watched as they drain.

Old BibTeX entries that readers already copied will point at the `github.io`
URL. Step 6's redirect is what keeps those honest, which is the reason to
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

### The name query

`Ferhat Ersoy` is an entity query, not a content query. It is won by being a
recognisable entity across several sites that agree with each other, and no
amount of writing inside a post moves it. The source side is already done:
`SITE_TITLE` is the name, every page emits `Person` schema with `sameAs`
pointing at GitHub and LinkedIn, and `/about/` is a real biography rather than
a stub.

The half that is not in this repo is the half that decides it:

- **Close the loop.** GitHub, LinkedIn and X profiles must each link back to
  `rfetha.com`. `sameAs` is a claim the site makes about itself; the
  reciprocal link is what confirms it, and an unconfirmed claim is discounted.
- **One spelling everywhere.** The same name string on every profile. A middle
  initial on one and not another splits the entity into two weaker ones.
- **Expect to share the page.** It is a common enough name that the result set
  will hold several people. The winnable outcome is the name plus a qualifier —
  `Ferhat Ersoy AI`, `Ferhat Ersoy Egeist`, `Ferhat Ersoy Dokuz Eylül` — which
  follows from the About page stating those facts plainly, which it does.
- **Distribution still does the work.** With no inbound links the site ranks
  behind LinkedIn's domain authority whatever the markup says. This is the same
  conclusion as the section above, arrived at from a different direction.

### Deliberately not done

- **FAQ schema** — Google withdrew the rich result for ordinary sites in 2023.
- **`llms.txt`** — no provider reads it yet. Free, but expect nothing.
- **`og:locale`** — `hreflang` already carries the bilingual signal to search;
  this would only affect social previews.
- **Keyword density work of any kind.**

## Analytics

Two tools, no overlap, and neither needs a consent banner.

**Search Console and Bing Webmaster Tools** are the SEO instrument: which query
arrived, at what position, from which country, on which device. That is already
most of what an analytics product would be asked for here, and it costs no code
at all.

**Cloudflare Web Analytics** covers the one thing Search Console cannot see —
arrivals that were not searches. Referrer, page views, country, device. Free,
cookieless (it measures through the Performance API and drops no tracking
cookie, so no banner), and it does not require the site to be proxied through
Cloudflare. One line in `BaseHead.astro`, before `</head>`:

```astro
<script
	is:inline
	defer
	src="https://static.cloudflareinsights.com/beacon.min.js"
	data-cf-beacon='{"token":"..."}'
></script>
```

`is:inline` is load-bearing. Without it Astro processes the script and the
`data-cf-beacon` attribute — which is where the token lives — does not survive
the build. The token is public by construction: on a static site every
client-side key ends up in the built HTML, so `.env` would move it, not hide
it.

### Not measured, deliberately

- **PostHog, GA4, or any product analytics.** There is no funnel here, no
  account, no sequence of steps with a drop-off worth measuring. A post is read
  or it is not, and that is a page view. They also set cookies, which buys a
  consent banner on a site whose entire design argument is that the page is a
  paper.
- **Server logs.** GitHub Pages does not expose them. Proxying through
  Cloudflare would, and that is the one real reason to turn the orange cloud on
  later.
