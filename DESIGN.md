---
name: Preprint
description: A working engineer's notebook set as a preprint — Latin Modern on white, one red, rules instead of boxes.
colors:
  ink: "#1a1a1a"
  paper: "#ffffff"
  muted: "#454545"
  faint: "#6e6e6e"
  rule: "#dadada"
  rule-soft: "#ececec"
  wash: "#f7f7f5"
  sig: "#8c1515"
  sig-wash: "#f0dcdc"
typography:
  display:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "1.55em"
    fontWeight: 400
    lineHeight: 1.24
    letterSpacing: "normal"
  headline:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "1.45em"
    fontWeight: 400
    lineHeight: 1.24
    letterSpacing: "normal"
  title:
    fontFamily: "Latin Modern Roman Caps, Georgia, serif"
    fontSize: "1.03em"
    fontWeight: 400
    lineHeight: 1.24
    letterSpacing: "0.045em"
  subtitle:
    fontFamily: "Latin Modern Roman Caps, Georgia, serif"
    fontSize: "0.95em"
    fontWeight: 400
    lineHeight: 1.24
    letterSpacing: "0.045em"
  body:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "19px"
    fontWeight: 400
    lineHeight: 1.66
    letterSpacing: "normal"
  secondary:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "0.9em"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "normal"
  meta:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "0.79em"
    fontWeight: 400
    lineHeight: 1.48
    letterSpacing: "normal"
    fontStyle: "italic"
    fontFeature: "tabular-nums"
  label:
    fontFamily: "Latin Modern Roman Caps, Georgia, serif"
    fontSize: "0.82em"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0.05em"
  code:
    fontFamily: "ui-monospace, SFMono-Regular, Cascadia Mono, Menlo, monospace"
    fontSize: "0.83em"
    fontWeight: 400
    lineHeight: 1.55
    letterSpacing: "normal"
rounded:
  none: "0"
spacing:
  gutter: "1.5rem"
  gutter-narrow: "1.1rem"
  para: "0.75rem"
  row: "0.68rem"
  block: "1.6rem"
  section: "1.85rem"
  rule: "2.25rem"
  page-top: "2.75rem"
  page-bottom: "4rem"
components:
  link-body:
    textColor: "{colors.sig}"
    rounded: "{rounded.none}"
  link-nav:
    textColor: "{colors.muted}"
    typography: "{typography.meta}"
    rounded: "{rounded.none}"
  link-nav-active:
    textColor: "{colors.ink}"
  entry-row:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    padding: "0.68rem 0"
    rounded: "{rounded.none}"
  abstract-block:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.muted}"
    typography: "{typography.secondary}"
    padding: "0.75rem 0"
    rounded: "{rounded.none}"
  code-block:
    backgroundColor: "{colors.wash}"
    textColor: "{colors.ink}"
    typography: "{typography.code}"
    padding: "0.85rem 1.05rem"
    rounded: "{rounded.none}"
  placeholder-callout:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.sig}"
    padding: "0.6rem 0.85rem"
    rounded: "{rounded.none}"
  figure-caption:
    textColor: "{colors.muted}"
    typography: "{typography.meta}"
    width: "32rem"
---

# Design System: Preprint

## Overview

**Creative North Star: "The Preprint"**

The page is a paper, not a blog post. Everything the reader sees is a device
borrowed from typeset scholarly printing and used literally: a rule-bound
masthead, a centered title block with an italic affiliation, an Abstract set
between two hairlines, numbered section heads in a drawn small-caps face,
bracketed index numerals down the left of the writing list, tabular italic
dates, and figures and equations that break wider than the text measure the
way plates do. There is no card, no shadow, no rounded corner, no gray
metadata chip. Structure is carried entirely by rules, measure, and type.

The material is white paper and black ink, with one chromatic mark: a deep
Cornell red. It appears on links, on the focus ring, on text selection, and on
the placeholder frame. Nothing else on the site is colored — not headings, not
borders, not code. Syntax highlighting is deliberately near-monochrome, bound
to the same ink/muted/faint ramp as the prose, so a listing reads as set type
rather than as an editor screenshot.

Density is high and confident. Paragraphs in a post body are indented rather
than spaced apart, justified with hyphenation above 900px, and held to a
67-character measure at desktop. The type is Latin Modern Roman throughout,
self-hosted, so prose and KaTeX formulas share one drawing. Reading is the
only affordance the design optimizes for.

**Key Characteristics:**
- White paper, near-black ink, exactly one accent color
- Rules and measure instead of cards, shadows, and radii
- Latin Modern Roman plus a drawn small-caps companion, no third face
- Three fixed measures: prose 43rem, break-wide 52rem, chrome 62rem
- Justified, indented, section-numbered post bodies
- Math is first-class: display equations get plate treatment and their own overflow rules

## Colors

A paper-white ground, a three-step gray ink ramp for hierarchy, two hairline
grays for structure, and a single deep red used sparingly enough to stay a
signal.

### Primary
- **Cornell Red** (`{colors.sig}`): the only chromatic mark on the site. It sets link text and the 38%-alpha underline beneath it, the 2px focus-visible outline, the placeholder callout's frame and text, the placeholder tag in the writing index, and the `--sig` code token reserved for links inside listings. It appears nowhere else — not on headings, not on rules, not on hover backgrounds.

### Neutral
- **Ink** (`{colors.ink}`): body text, headings, active navigation, entry titles, code foreground (16.23:1 on the wash). The masthead's bottom border is drawn in ink at 1px — the only full-strength rule on the page.
- **Paper** (`{colors.paper}`): the page ground, unbroken. No surface on the site is tinted except code.
- **Muted** (`{colors.muted}`): second-rank text — bylines, affiliation, abstract copy, entry descriptions, figure captions, blockquote text, `<h4>`, footer links, definition-list bodies.
- **Faint** (`{colors.faint}`): third-rank text — bracketed numerals, entry dates, CV years, the footer's copyright line, code comments and punctuation.
- **Hairline** (`{colors.rule}`): every structural rule that is not the masthead — the abstract's two hairlines, the horizontal rule, code-block borders, blockquote's left stroke, table header underline, the dotted separator under each writing-index row, the footer's top border, and the scrollbar thumb.
- **Hairline Soft** (`{colors.rule-soft}`): table body row separators only, so rows read lighter than the header rule above them.
- **Listing Wash** (`{colors.wash}`): the code-block ground, a barely-warm off-white that reads as a set listing rather than a colored panel.
- **Selection Rose** (`{colors.sig-wash}`): the text-selection background, with ink kept as the selected foreground.

### Named Rules
**The Single Signal Rule.** Cornell red is reserved for interaction and for the placeholder state: links, focus ring, selection, the placeholder frame and tag. If a new surface wants color for emphasis, it does not get color — it gets the caps face, a rule, or a shift down the gray ramp.

**The Three-Ink Rule.** Text hierarchy is exactly three steps: ink, muted, faint. Do not introduce a fourth gray, and do not use a hairline gray (`--rule`, `--rule-soft`) as a text color; those two values are structure, never type.

**The Monochrome Listing Rule.** Code color comes from this palette through Shiki's `css-variables` theme — keywords and functions in ink, strings and constants in muted, comments and punctuation in faint (comments italic), links in red. Never reintroduce a foreign syntax theme.

## Typography

**Display / Body Font:** Latin Modern Roman (fallback Georgia, serif) — self-hosted under the GUST Font License in four variants: regular, italic, bold, bold-italic.
**Label Font:** Latin Modern Roman Caps (fallback Georgia, serif) — a *drawn* small-caps face, regular only.
**Mono Font:** the system UI-monospace stack (`ui-monospace`, SFMono-Regular, Cascadia Mono, Menlo).

**Character:** Latin Modern Roman is the Unicode successor to Computer Modern — the voice of a typeset paper. It was chosen over KaTeX_Main specifically because KaTeX_Main drops accented Latin (10 missing glyphs, among them `À É Î Õ Ü ç ñ`), which would break author names mid-word; here prose and rendered formulas share one drawing. Its x-height is small, so the face reads smaller than its nominal size — the body sizes compensate upward, not downward.

### Hierarchy
- **Display** (400, `1.55em` ≈ 29px, line-height 1.24): the site name in the home and About title blocks, centered above an italic affiliation.
- **Headline** (400, `1.45em` ≈ 28px, 1.24): a post's title, centered above its byline. Section-numbered `<h2>`s live under it.
- **Title** (Caps face, 400, `1.03em`, letter-spacing 0.045em): `<h2>` section heads. Inside `article.paper` they are auto-numbered `1. `, `2. ` via a CSS counter.
- **Subtitle** (Caps face, 400, `0.95em`, 0.045em): `<h3>`. `<h4>` is the same treatment at `0.9em` in muted.
- **Body** (400, 19px / 1.66; 18px at ≤640px): all prose. Measure is 43rem — measured at 67 characters at a 1440px viewport, 40 characters at 390px.
- **Secondary** (400, `0.9`–`0.92em` / 1.55–1.6, muted): abstract copy, definition-list bodies, closing notes.
- **Meta** (400 italic, `0.76`–`0.84em`, tabular-nums, faint or muted): dates, entry counts, bylines, affiliation, CV years, footer, captions. Dates always render as long-form `en-GB` (`14 March 2026`) inside a `<time>` element.
- **Label** (Caps face, 400, `0.05em` letter-spacing): run-in heads (`Abstract.`, `Placeholder.`), figure-caption plate numbers, CV role names, the placeholder tag.
- **Code** (mono, `0.83em` block / `0.86em` inline, 1.55).

### Named Rules
**The Drawn Caps Rule.** Small caps come from the `--caps` family. `font-variant: small-caps` is never used anywhere in this build and must not be introduced — synthetic caps scale capitals down and thin the stems, which reads as a defect beside a real serif.

**The Identifier Exemption.** Identifiers are never set in the caps face. Repository names, file names, and code symbols keep the roman face at full size, because the caps face renders `ORKIKS` full-height and `fin-cli` half-height off the same baseline. Caps are for editorial labels only.

**The 18px Floor.** Body type never drops below 18px. Dropping to 17px at 390px was measured to buy zero additional characters, and Latin Modern's small x-height already reads under its nominal size. Below 400px the *gutter* narrows to 1.1rem instead.

**The One Voice Rule.** Two families, and only two. Prose, headings, captions, and math all resolve to Latin Modern; the mono stack exists only inside `code`.

## Layout

**Three measures, all centered, all fixed in rem.** The prose column is 43rem
(`--measure`, ~67 characters), applied to `<main>`. Chrome — masthead and
footer — runs to 62rem (`--wide`, the `.shell` class). Break-wide content sits
at 52rem (`--fig`). Horizontal padding is a single `--gutter` token, 1.5rem,
narrowing to 1.1rem below 400px. `<main>` pads 2.75rem above and 4rem below.

**Break-wide.** Figures, blockquotes, code blocks, display math, and the
About page's `.cv` and `.work` definition lists escape the prose column with
`width: min(var(--fig), 100vw - 3rem); margin-left: 50%; transform:
translateX(-50%)`. This is the plate device: the argument stays at 67
characters, the evidence gets more room.

**Vertical rhythm.** Paragraph 0.75rem, list 0.85rem, index row 0.68rem,
figure block 1.6rem, section head 1.85rem above / 0.5rem below, horizontal
rule 2.25rem, footer 2.5rem off the content.

**Breakpoints.** Three, all `max-width`: 900px collapses every break-wide
element to full width and turns off body justification and hyphenation; 640px
drops body type to 18px and narrows the index numeral column from 2.1rem to
1.8rem and the CV grid to a single column; 400px narrows the gutter.

### Named Rules
**The Break-Wide Rule.** Anything wider than the measure breaks symmetrically from the center and collapses to 100% at 900px. Never widen the prose measure itself to fit a figure.

**The Justified Body Rule.** Post bodies are justified with `hyphens: auto` above 900px only, and consecutive paragraphs are indented 1.6em rather than separated by space. Below 900px, justification is off — narrow justified columns river.

**The Child Combinator Rule.** Paper typesetting applies to `article.paper > p`, never to a descendant selector. The byline and placeholder callout live inside the article's `<header>`; a descendant selector indents and hyphenates a bordered callout against its own frame.

## Elevation & Depth

**There are no shadows in this system, and none may be added.** There is no
`box-shadow` anywhere in the build, no blur, no overlay, no z-layering. Depth
is entirely typographic and linear: weight of rule, tint of ground, and
position on the gray ink ramp.

The rule vocabulary is the depth vocabulary. A 1px ink border under the
masthead is the heaviest line on the page and marks the site frame. A 1px
hairline (`--rule`) marks structure inside the page — the abstract's two
bounding lines, code-block borders, blockquote's left stroke, the footer's
top edge. A 1px *dotted* hairline separates rows in the writing index, which
is lighter still and reads as a list rather than a table. `--rule-soft` is
the lightest, used only between table body rows. The single tinted surface
on the site is the code wash.

### Named Rules
**The No-Shadow Rule.** Surfaces are flat, always, in every state. Hover and focus are expressed in ink weight, underline, and the red focus ring — never in lift, glow, or scale.

**The Rule Ladder.** Four line weights, in order of authority: ink solid (site frame) → hairline solid (structural block) → hairline dotted (list row) → soft hairline (table row). Pick the lightest one that reads.

## Shapes

Every corner in the build is square. `--radius` does not exist and no element
declares `border-radius`; the frontmatter records a single `rounded.none: 0`
to make that explicit. Form language is rectangular and rule-drawn: blocks are
delimited by borders on one to four edges (the abstract by top and bottom only,
the blockquote by left only, the code block and placeholder callout by all
four), never by a filled rounded card.

Two ornaments carry identity: the bracketed numeral `[7]` generated with a
counter before each writing-index row, counting *down* so the newest entry
holds the highest number; and the auto-incremented section numeral before every
`<h2>` inside a post. Both are content-generated, tabular, and faint.

Scrollbars are part of the drawing rather than an afterthought: code blocks
and display math get an 8px-tall thin scrollbar with a `--rule` thumb on a
transparent track.

## Components

### Navigation
- **Character:** a masthead, not a nav bar. Full-bleed 1px ink rule beneath, content held to 62rem, baseline-aligned.
- **Style:** brand name at `0.9em` in ink on the left; three links (Writing / About / Feed) at `0.79em` in muted, 1rem apart, on the right. Vertical padding 0.8rem.
- **States:** hover lifts a link to ink; the active route is ink with a 3px-offset underline. Neither the brand nor the nav links carry the global red link underline.
- **Mobile:** unchanged — the flex row and the type sizes hold at 390px.

### Links (body)
- **Default:** Cornell red with a 1px bottom border at 38% alpha, no `text-decoration`.
- **Hover:** the border goes to full-strength red. No color or weight change.
- **Focus:** a 2px red `:focus-visible` outline at 3px offset, applied globally to every focusable element.
- **Demoted variants:** navigation, brand, footer links and entry titles set `border-bottom: 0` and take a gray, using an offset underline on hover instead. Red is not the default link color everywhere — it is the *prose* link color.

### Writing Index (signature component)
- **Character:** a numbered reference list, the site's most distinctive surface. Used by both the home page and `/blog`.
- **Structure:** a 2-column baseline-aligned grid — a 2.1rem numeral column (1.8rem ≤640px) and a content column. The numeral is `[n]` from a CSS counter that decrements, seeded via an inline `--start` custom property so the newest post carries the highest number.
- **Rows:** title at `0.98em` in ink, description at `0.84em` in muted, then an italic tabular meta line at `0.76em` in faint carrying the date. 0.68rem vertical padding, 1px dotted hairline beneath.
- **Hover:** the title takes a 3px-offset underline. Nothing moves.

### Title Block
- **Character:** the top of a paper. Centered name at `1.55em`, italic affiliation at `0.82em` muted directly beneath, then the Abstract.
- **Abstract:** left-aligned inside the centered block, `0.9em`/1.6 in muted, bounded by hairlines top and bottom with 0.75rem of padding. It opens with an italic ink run-in head (`Abstract.`) at regular weight, not bold.

### Post Body (`article.paper`)
- **Character:** typeset, not laid out. A CSS counter numbers each `<h2>`; consecutive paragraphs indent 1.6em rather than separating; text justifies with hyphenation above 900px.
- **Byline:** centered italic at `0.82em` muted — site title, middle dot, long-form date; a revision line, when present, one step smaller in faint.

### Figures
- **Character:** plates. 52rem wide, broken out of the measure, 1.6rem of air above and below.
- **Caption:** centered, `0.79em`/1.48 in muted, capped at 32rem so it stays narrower than its figure. A bold run-in inside the caption renders in the caps face at regular weight in ink — that is the plate number.

### Code Blocks
- **Style:** listing wash ground, 1px hairline border on all four edges, square, 0.85rem/1.05rem padding, `0.83em`/1.55, break-wide to 52rem, horizontal scroll with the thin styled scrollbar.
- **Color:** near-monochrome via Shiki `css-variables` bound to the palette; comments faint and italic.
- **Inline code:** mono at `0.86em` on a light literal tint with 1px/4px padding, no border, no radius.

### Math
- **Inline:** KaTeX at `1.03em`, matching the roman face around it.
- **Display:** break-wide to 52rem, 0.9rem margins, `overflow-x: auto` with `overflow-y: hidden`, and **0.9em of vertical padding** — load-bearing, not decorative. `overflow-x: auto` forces `overflow-y` out of `visible`, so the box clips; a `\top` superscript in a fraction numerator overshoots the content box by ~8px at 19px body. The padding is in `em` so it tracks formula size. Verified across all five display equations on the site: zero clipped, tightest slack 8.8px.
- **Inside a figure:** width returns to 100%, the transform is dropped, and bottom padding falls to 0.25em so the caption binds to its equation (11px) rather than to the paragraph below (26px).
- **Box model:** `.katex, .katex * { box-sizing: content-box }` locally undoes the global border-box reset. KaTeX positions vlists with absolute metrics that assume content-box; under the reset the crossbars clip (measured: vlist clientHeight 30px against scrollHeight 55px). This is required, not incidental.

### Definition Lists (CV / Work)
- **CV:** break-wide, a 7.5rem italic tabular-faint date column beside a body column at `0.92em`; role names in the caps face. Collapses to one column at 640px.
- **Work:** break-wide, no date column. Repository names are links in red at `0.95em` in the roman face — deliberately *not* the caps face, per the Identifier Exemption.

### Placeholder Marker
- **Character:** an honest label on unwritten content, driven by a `placeholder: boolean` in the blog collection schema. It is a content-state device, not a decorative badge.
- **On a post:** a four-sided 1px red frame, red text at `0.82em`, 0.6rem/0.85rem padding, with a caps-face run-in `Placeholder.`; it lives in the article `<header>` and is therefore excluded from justification.
- **In the index:** a caps-face red tag reading `placeholder` at the end of the meta line.
- **Lifecycle:** both markers disappear when the flag flips. Nothing else in the system may borrow the red frame.

### Share Card (generated)
- **Character:** the site's face when a link is pasted somewhere else. Generated at build time by `src/pages/og/[...slug].png.ts` — satori composes the SVG, sharp writes the PNG. Nothing is drawn by hand and no post ever needs cover art.
- **Canvas:** 1200×630 on white, 72px/80px padding, content pushed to the top and bottom edges. A 10px solid ink rule runs across the top — the masthead at card scale.
- **Type:** title at 64px in the roman face, line-height 1.18, clipped at 230px so it stops after three lines; description beneath at 28px italic in muted, clipped at 124px. Both faces are the same woff files the site serves, so the card is set in the site's own voice rather than a system stack.
- **Footer band:** a 2px hairline above, then a 14px Cornell red square — the Single Signal, appearing here as a mark rather than a link — followed by the name at 26px and the italic affiliation at 22px in muted.
- **Variants:** the post card carries name and affiliation separately; the site card, whose title is already the name, drops the name from the band and runs `name · affiliation` as one italic line. The card must never print the name twice.
- **Wiring:** `og:image` plus explicit `1200`/`630` dimensions and an alt equal to the page title; `twitter:card` is `summary_large_image`. Non-post pages fall back to `/og/default.png`.

### Empty State
- **Character:** one honest sentence. No illustration, no dashed placeholder box, no call to action pretending the absence is an opportunity.
- **Style:** `0.9em` italic in muted with 0.6rem of top margin, replacing the writing index entirely rather than rendering an empty list.
- **Copy:** it names the absence and offers the feed, so a visitor who arrives early has something to do. It never apologises and never promises a date.
- **Companion rule:** the `/blog` count line is suppressed at zero rather than reading "0 entries" — a count that counts nothing is noise.

### Footer
- **Style:** 1px hairline top border, content to 62rem, `0.79em` faint, copyright left and three links right, wrapping on narrow. Links are muted with a hairline underline that darkens to ink on hover.

## Do's and Don'ts

### Do:
- **Do** carry every new surface with rules, measure, and the three-ink ramp before reaching for anything else.
- **Do** break wide content out symmetrically to 52rem (`margin-left: 50%; transform: translateX(-50%)`) and collapse it to full width at 900px.
- **Do** use the drawn `--caps` family for editorial labels, section heads, and run-in heads, at 0.045–0.05em letter-spacing and weight 400.
- **Do** set every date, numeral column, and counted list with `font-variant-numeric: tabular-nums`.
- **Do** keep display math's `em`-based vertical padding and KaTeX's `content-box` override; both were measured and both prevent visible clipping.
- **Do** keep body type at 18px or larger and narrow the gutter instead when space runs out.
- **Do** flag unfinished content with the placeholder marker rather than shipping it unlabelled.

### Don't:
- **Don't** add a `box-shadow`, a `border-radius`, or a card. There are none in the build and the world does not use them.
- **Don't** use `font-variant: small-caps`. Use the `--caps` face.
- **Don't** set identifiers — repository names, file names, symbols — in the caps face.
- **Don't** give Cornell red any job beyond links, focus, selection, and the placeholder state.
- **Don't** import a syntax theme; code color comes from the palette through `css-variables`.
- **Don't** justify or hyphenate below 900px, and don't apply paper typesetting with a descendant selector — `article.paper > p` is deliberate.
- **Don't** introduce a third font family or a fourth gray.
