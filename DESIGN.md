---
name: Preprint
description: A working engineer's notebook set as a preprint — Latin Modern on white, one red, rules instead of boxes, notes in the margin.
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
  code-surface: "#fbfbfd"
  code-rule: "#d7dae0"
  code-fg: "#24292f"
  code-keyword: "#a626a4"
  code-string: "#1a7f37"
  code-function: "#4078f2"
  code-constant: "#b5540b"
  code-comment: "#6a737d"
  code-punctuation: "#57606a"
typography:
  display:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "1.9em"
    fontWeight: 400
    lineHeight: 1.24
    letterSpacing: "normal"
  headline:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "1.35em"
    fontWeight: 700
    lineHeight: 1.24
    letterSpacing: "normal"
  title:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "1.15em"
    fontWeight: 700
    lineHeight: 1.24
    letterSpacing: "normal"
    fontStyle: "italic"
  subtitle:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "1em"
    fontWeight: 700
    lineHeight: 1.24
    letterSpacing: "normal"
    fontStyle: "italic"
  body:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "18px"
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
  sidenote:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "0.74em"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "normal"
  label:
    fontFamily: "Latin Modern Roman Caps, Georgia, serif"
    fontSize: "0.82em"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "0.05em"
  code-block:
    fontFamily: "ui-monospace, SFMono-Regular, Cascadia Mono, Menlo, monospace"
    fontSize: "0.73em"
    fontWeight: 400
    lineHeight: 1.65
    letterSpacing: "normal"
  code-inline:
    fontFamily: "ui-monospace, SFMono-Regular, Cascadia Mono, Menlo, monospace"
    fontSize: "0.86em"
    fontWeight: 400
    lineHeight: 1.66
    letterSpacing: "normal"
  rail:
    fontFamily: "Latin Modern Roman, Georgia, serif"
    fontSize: "0.72em"
    fontWeight: 400
    lineHeight: 1.42
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
  sidenote-gap: "2.5rem"
  page-top: "1.75rem"
  page-bottom: "3.5rem"
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
    backgroundColor: "{colors.code-surface}"
    textColor: "{colors.code-fg}"
    typography: "{typography.code-block}"
    padding: "1rem 1.15rem"
    rounded: "{rounded.none}"
  code-inline:
    backgroundColor: "{colors.wash}"
    textColor: "{colors.ink}"
    typography: "{typography.code-inline}"
    padding: "1px 4px"
    rounded: "{rounded.none}"
  sidenote:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.muted}"
    typography: "{typography.sidenote}"
    width: "15rem"
    rounded: "{rounded.none}"
  rail-link:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.faint}"
    typography: "{typography.rail}"
    width: "15rem"
    padding: "0 0 0 0.6rem"
    rounded: "{rounded.none}"
  rail-link-current:
    textColor: "{colors.ink}"
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
between two hairlines, small-caps section heads in a drawn caps face,
bracketed index numerals down the left of the writing list, tabular italic
dates, plate-numbered figures, and notes set into the right margin beside the
paragraph that provokes them. There is no card, no shadow, no rounded corner,
no gray metadata chip. Structure is carried entirely by rules, measure, and
type.

The material is white paper and black ink, with one chromatic mark in the
prose: a deep Cornell red on links, the focus ring, text selection, and the
placeholder frame. The one place color is allowed to multiply is inside a code
listing. Listings were set near-monochrome in an earlier build and read as
straw paper rather than as a notebook; they now carry a real six-color syntax
palette on a cool near-white ground, deliberately cooler than the warm paper
around them so a listing reads as a pane of machine output set into the page.

Density is high and confident. The text column is wide — 992px, holding ~107
characters of 18px Latin Modern. The measure is the author's decision, taken
against 656px and 784px columns set side by side with it and rejected as too
narrow; the larger body step
is what keeps a line this long readable. Paragraphs carry both marks of a
break — a 0.5rem space and a 1.6em first-line indent, every paragraph indented
including the first of a section, and — on a screen wide enough to spare it — a 15rem note hanging
in the right-hand whitespace rather than interrupting the run, mirrored by a
15rem section rail in the left-hand strip. Type is
Latin Modern Roman throughout, self-hosted, so prose and KaTeX formulas share one drawing. Reading is the
only affordance the design optimizes for.

**Key Characteristics:**
- White paper, near-black ink, one accent color in the prose; syntax color only inside listings
- Rules and measure instead of cards, shadows, and radii
- Latin Modern Roman plus a drawn small-caps companion, no third face
- A page-centered text column with sidenotes hanging into the right-hand whitespace and a section rail in the left one, both above 1200px; the column itself steps down from 62rem to 50rem below 1520px so the strips still fit
- Ragged-right post bodies, every paragraph indented and spaced, with hand-numbered sections
- Math is first-class: display equations get plate treatment and their own overflow rules
- One body step, no responsive type ramp; the page adapts by narrowing the gutter, not by resizing prose

## Colors

A paper-white ground, a three-step gray ink ramp for hierarchy, two hairline
grays for structure, one deep red used sparingly enough to stay a signal, and
a self-contained syntax palette that lives only inside `pre`.

### Primary
- **Cornell Red** (`{colors.sig}`): the only chromatic mark in the prose. It sets link text and the 38%-alpha underline beneath it, the 2px focus-visible outline, the placeholder callout's frame and text, the placeholder tag in the writing index, the repository links in the About page's work list, and the link token inside listings. It is not used on headings, rules, or hover backgrounds.

### Neutral
- **Ink** (`{colors.ink}`): body text, headings, active navigation, entry titles. The masthead's bottom border is drawn in ink at 1px — the only full-strength rule on the page.
- **Paper** (`{colors.paper}`): the page ground, unbroken. The only tinted surfaces on the site are the two code grounds.
- **Muted** (`{colors.muted}`): second-rank text — bylines, affiliation, abstract copy, entry descriptions, figure captions, blockquote text, sidenote text, `<h4>`, footer links, definition-list bodies.
- **Faint** (`{colors.faint}`): third-rank text — bracketed numerals, entry dates, CV years, the revision line, the footer's copyright.
- **Hairline** (`{colors.rule}`): every structural rule that is not the masthead — the abstract's two hairlines, the horizontal rule, the blockquote's left stroke, the table header underline, the dotted separator under each writing-index row, the sidenote's left stroke in stacked mode, the footer's top border, and the scrollbar thumb.
- **Hairline Soft** (`{colors.rule-soft}`): table body row separators only, so rows read lighter than the header rule above them.
- **Inline Wash** (`{colors.wash}`): the warm off-white behind inline `code` only. It is not the block-listing ground.
- **Selection Rose** (`{colors.sig-wash}`): the text-selection background, with ink kept as the selected foreground.

### Secondary — the listing palette
Six token colors plus a ground and a border, scoped entirely to code. They are
a closed set: nothing outside `pre` and its Shiki tokens may use them.

- **Listing Ground** (`{colors.code-surface}`) and **Listing Rule** (`{colors.code-rule}`): a cool near-white panel behind a 1px cool-gray border. Cool is the point — it separates machine text from the warm paper without a shadow or a radius.
- **Listing Foreground** (`{colors.code-fg}`), **Keyword** (`{colors.code-keyword}`), **String** (`{colors.code-string}`), **Function** (`{colors.code-function}`), **Constant** (`{colors.code-constant}`), **Comment** (`{colors.code-comment}`, italic), **Punctuation** (`{colors.code-punctuation}`): bound through Shiki's `css-variables` theme via `--astro-code-token-*`. String-expression follows string; parameter follows foreground.

### Named Rules
**The Single Signal Rule.** Cornell red is reserved for interaction and for the placeholder state: links, focus ring, selection, the placeholder frame and tag. If a new *prose* surface wants color for emphasis, it does not get color — it gets the caps face, a rule, or a shift down the gray ramp.

**The Three-Ink Rule.** Text hierarchy is exactly three steps: ink, muted, faint. Do not introduce a fourth gray, and do not use a hairline gray (`--rule`, `--rule-soft`) as a text color; those two values are structure, never type.

**The Listing Quarantine Rule.** Syntax color is real and it stays inside the listing. The six token colors, the cool ground, and the cool border may not appear on any other surface, and no color from the prose palette may be used as a syntax token — except the red link token, which is the same red as everywhere else. Color comes from `css-variables` bound to these tokens; never import a foreign syntax theme.

## Typography

**Display / Body Font:** Latin Modern Roman (fallback Georgia, serif) — self-hosted under the GUST Font License in four variants: regular, italic, bold, bold-italic. The served files are subset from the GUST OpenType originals over Latin-1 **plus Latin Extended-A**: an earlier Latin-1-only subset was missing `ğ Ğ ı İ ş Ş`, so every Turkish word carrying one fell back to Georgia mid-word. Any regenerated subset must keep U+0100–U+017F.
**Label Font:** Latin Modern Roman Caps (fallback Georgia, serif) — a *drawn* small-caps face, regular only.
**Mono Font:** the system UI-monospace stack (`ui-monospace`, SFMono-Regular, Cascadia Mono, Menlo).

**Character:** Latin Modern Roman is the Unicode successor to Computer Modern — the voice of a typeset paper. It was chosen over KaTeX_Main specifically because KaTeX_Main drops accented Latin (10 missing glyphs, among them `À É Î Õ Ü ç ñ`), which would break author names mid-word; here prose and rendered formulas share one drawing. Its x-height is small, so the face reads smaller than its nominal size — body sizes compensate upward, not downward.

### Hierarchy
- **Display** (400, `1.9em`, line-height 1.24): the site name in the home title block and the About page's heading, centered above an italic affiliation.
- **Headline** (700, upright, `1.35em`, 1.24): a post's title, centered above its byline, and `<h2>` section heads. Never italic — a whole head in Latin Modern's italic reads as a long quotation.
- **Title** (700 italic, `1.15em`): `<h3>` sub-heads, 1.85rem above / 0.5rem below. Section *numbers* are typed in the markdown, not generated. Italic, not the caps face — see The Turkish Lowercase Rule.
- **Subtitle** (700 italic, `1em`): `<h4>`, in muted.
- **Body** (400, 18px / 1.66): all prose. One *size* step at every width — the column width is tiered, the type is not.
- **Secondary** (400, `0.9`–`0.92em` / 1.55–1.6, muted): abstract copy, definition-list bodies, closing notes.
- **Meta** (400 italic, `0.76`–`0.84em`, tabular-nums, faint or muted): dates, bylines, affiliation, CV years, footer, captions. Dates always render long-form `en-GB` (`14 March 2026`) inside a `<time>` element.
- **Sidenote** (400, `0.82em` / 1.55, muted, `hyphens: none`): notes in the flow behind a 2px rule — the default; `0.74em` / 1.5 when promoted into the margin at ≥1200px.
- **Label** (Caps face, 400, `0.05em` letter-spacing): run-in heads (`Abstract.`, `Placeholder.`), figure-caption plate numbers, CV role names, the placeholder tag.
- **Code, output** (mono, `0.73em` / 1.65, on `--wash`): the result a listing produced, carried under it with the panels touching and a mono `çıktı` run-in. Selected as `pre[data-language='python'] + pre[data-language='plaintext']`, so a standalone unhighlighted block — an ASCII schematic, say — is never mistaken for output.
- **Code, block** (mono, `0.73em` / 1.65): listings. Set one step below inline code on purpose — a listing is a panel of many lines and reads best denser than the sentence around it.
- **Code, inline** (mono, `0.86em`, inheriting the body's 1.66): a word inside a sentence, so it holds close to the prose size.
- **Rail** (400, `0.72em` / 1.42, faint): the section rail's entries — the smallest type in the system, and the only step that never appears inside the column.

### Named Rules
**The Drawn Caps Rule.** Small caps come from the `--caps` family. `font-variant: small-caps` is never used in this build and must not be introduced — synthetic caps scale capitals down and thin the stems, which reads as a defect beside a real serif.

**The Turkish Lowercase Rule.** The caps face is barred from Turkish prose, not only from identifiers. It draws lowercase as small capitals and a small-cap I carries no dot, so `i` and `ı` leave it as the same shape — the face maps them to separate glyph entries (`i` and `dotlessi`) that are drawn identically. Measured by rendering both characters and differencing the bitmaps: zero differing pixels from the caps face, 34 from the roman. In English that is invisible; in Turkish it deletes a letter. It cost 24 of the 29 sub-heads across the two posts before it was caught. The caps face keeps the fixed English labels (`Abstract.`, `Placeholder.`, `Figure 1.`); everything that can hold a Turkish word takes bold italic roman instead.

**The Identifier Exemption.** Identifiers are never set in the caps face. Repository names, file names, and code symbols keep the roman face at full size, because the caps face renders `ORKIKS` full-height and `fin-cli` half-height off the same baseline. Caps are for editorial labels only.

**The Single Body Step Rule.** Body type is 18px at every width; no media query changes it. Latin Modern's x-height is small for its nominal size, so 17px read a step under what it was; the value moved once, globally. When space runs out the *gutter* narrows to 1.1rem below 400px; the prose size does not move.

**The Two-Step Code Rule.** Block and inline code are deliberately different steps: `0.73em` in a listing, `0.86em` in a sentence. Do not collapse them back to one value — the listing is a panel measured against its own line count, the inline span is measured against the words on either side of it.

**The One Voice Rule.** Two families, and only two. Prose, headings, captions, sidenotes, and math all resolve to Latin Modern; the mono stack exists only inside `code`.

## Layout

**One centered text column; the margin is borrowed, not reserved.** `<main>`
is capped at `--measure` (62rem, stepping to 50rem below 1520px) and centered on the page, so the reading
column is the page's center line. No wrapper is widened to hold a second
column — an earlier build sized `<main>` to hold measure plus margin, which
centered the *box* and pushed the *text* into its left half. Chrome — masthead
and footer — runs wider, to 78rem (`--wide`, the `.shell` class). Horizontal padding is a single `--gutter` token, 1.5rem,
narrowing to 1.1rem below 400px. `<main>` pads 1.75rem above and 3.5rem below;
the top pad is deliberately shallow so the title block sits close under the
masthead rule.

**Sidenotes.** In the flow by default. From `min-width: 1200px` a note is
promoted out of the column: it floats right, clears right, takes
`--sidenote-w`, and is pulled past the column edge by a negative right margin
of `-(--sidenote-w + --sidenote-gap)` into the page's own right-hand
whitespace. The threshold is arithmetic, not taste — a full strip on *both*
sides needs `--measure + 2 × (--sidenote-w + --sidenote-gap)`, and that sum is
what the tier is built to satisfy:

| | measure | strip + gap | pair needs |
|---|---|---|---|
| ≥1520px | 62rem / 992px | 15rem + 2.5rem = 280px | 1504px |
| 1200–1520px | 50rem / 800px | 10rem + 1.75rem = 188px | 1176px |

So a promoted note can never cause horizontal scroll. The gap is the tighter
constraint of the two, not the width: `--fig` is pinned to `--measure`, so a
plate overhangs the column by one gutter on each side, and a strip whose inner
edge sits inside that overhang gets a code block laid on top of it. At 1.25rem
that overlap measured 4px on every listing beside a citation.
Authoring is a bare `<aside class="sidenote">` in the markdown, documented in
`drafts/README.md`.

**The section rail.** The left-hand strip is the sidenote strip's mirror. A
post's `<h2>` sections are listed in a fixed rail pinned at `top: 5.5rem`, at
`--sidenote-w` wide, positioned at `50% - --measure/2 + --gutter -
--sidenote-gap - --sidenote-w` — the same width and the same gap as a promoted
note, measured from the other edge. It is `fixed` rather than floated because
it has to survive scrolling. It appears at exactly the sidenote threshold
(`min-width: 1200px`) and is `display: none` below it, because it depends on
the same strip existing; it caps at `100vh - 9rem` and scrolls internally.

**Break-wide.** Figures, blockquotes, code blocks, display math, and the About
page's `.cv` and `.work` definition lists are centered on the text column with
`width: min(var(--fig), 100vw - 3rem); margin-left: 50%; transform:
translateX(-50%)`. `--fig` tracks `--measure` at every tier — so
these elements now sit flush with the text column rather than breaking past
it. That is intentional: at this measure, an element wider than the prose
would run into the whitespace a sidenote hangs in.

**Vertical rhythm.** Paragraph 0.75rem (0.3rem inside `article.paper`), list
0.85rem, index row 0.68rem, figure block 1.6rem, code block 1.5rem/1.6rem,
section head 1.85rem above / 0.5rem below, horizontal rule 2.25rem, footer
2.5rem off the content.

**Breakpoints.** Five in the global sheet, one component-scoped. The only
`min-width` is 1200px: sidenotes are promoted into the right margin and the
section rail appears in the left one — a single threshold governing both
strips. The `max-width` set: **1519.98px** steps `--measure` and `--fig` down
to 50rem and the strips to 10rem / 1.75rem, and it deliberately has no lower
bound — with one, the scale stopped being monotonic and a 900px window drew a
*longer* line (850px) than a 1200px one (784px) with a 24px page margin left
over; **48rem** turns a table into the scrolling box instead of the page, since
the five-model comparison will not compress below ~643px of content; **900px**
collapses every break-wide element to 100%; **400px** narrows the gutter to
1.1rem. **640px** lives in the components — the writing-index numeral column
and the CV definition grid — and touches no type.

### Named Rules
**The Margin Column Rule.** The text column is centered on the page and nothing is widened to make room beside it. The whitespace to its right belongs to sidenotes, which hang out of the column rather than being given a column of their own; no content element may exceed `--measure` to borrow that strip — that is why `--fig` is pinned to `--measure`.

**The Sidenote Promotion Rule.** In the flow, behind a 2px `--rule` stroke, is a sidenote's *default* state; the margin is the exception, granted only at ≥1200px where a full strip exists on both sides. A note is never hidden and never becomes a tooltip, so margin content must always read in sequence — the main argument never lives in the margin.

**The Ragged-Right Rule.** Post bodies are set flush left and ragged right at every width, with `text-wrap: pretty`; consecutive paragraphs are indented 1.6em rather than separated by space. Justification was tried and removed: it is only legible when the renderer can hyphenate, and browsers do not reliably ship Turkish patterns — measured here, `hyphens: auto` broke zero words across 28 lines while the document was still declared `lang="en"`, and the word spaces stretched to 2.69× the narrowest to compensate. The article is now marked `lang="tr"`, but do not restore justification without first confirming that hyphens actually appear in Turkish prose. Sidenotes are never justified and never hyphenated.

**The Child Combinator Rule.** Paper typesetting applies to `article.paper > p`, never to a descendant selector. The byline and the placeholder callout live inside the article's `<header>`; a descendant selector indents and hyphenates a bordered callout against its own frame.

## Elevation & Depth

**There are no shadows in this system, and none may be added.** There is no
`box-shadow` anywhere in the build, no blur, no overlay, no z-layering. Depth
is entirely typographic and linear: weight of rule, tint of ground, and
position on the gray ink ramp.

The rule vocabulary is the depth vocabulary. A 1px ink border under the
masthead is the heaviest line on the page and marks the site frame. A 2px
`--rule` stroke marks a demoted sidenote. A 1px hairline marks structure
inside the page — the abstract's two bounding lines, the blockquote's left
stroke, the footer's top edge. A 1px *dotted* hairline separates rows in the
writing index, which reads as a list rather than a table. `--rule-soft` is the
lightest, used only between table body rows. The code block's 1px `--code-rule`
border is the one line drawn in a color that is not from the gray ramp,
because the panel it encloses is not paper.

### Named Rules
**The No-Shadow Rule.** Surfaces are flat, always, in every state. Hover and focus are expressed in ink weight, underline, and the red focus ring — never in lift, glow, or scale.

**The Rule Ladder.** Line weights in order of authority: ink solid 1px (site frame) → red solid 2px (the section the reader is in, on the rail) → hairline solid 2px (demoted sidenote) → hairline solid 1px (structural block) → hairline dotted 1px (list row) → soft hairline (table row). Pick the lightest one that reads. The 2px left stroke is the system's "this block is set apart" mark, and it carries state as well as structure: in `--rule` it demotes a sidenote, in `--sig` it marks the current section. Those are its only two colors.

## Shapes

Every corner in the build is square. `--radius` does not exist and no element
declares `border-radius`; the frontmatter records a single `rounded.none: 0` to
make that explicit. This survived the listing recolor deliberately: the code
block got real syntax color but no rounded corner, because a radius would make
it a card. Form language is rectangular and rule-drawn — blocks are delimited
by borders on one to four edges (the abstract by top and bottom, the blockquote
and the demoted sidenote by left only, the code block and placeholder callout
by all four), never by a filled rounded card.

One ornament carries identity: the bracketed numeral `[7]` generated with a
CSS counter before each writing-index row, counting *down* so the newest entry
holds the highest number. It is content-generated, tabular, and faint. Section
numerals are typed by the author in the markdown and are ordinary text.

Scrollbars are part of the drawing rather than an afterthought: code blocks and
display math get an 8px-tall thin scrollbar with a `--rule` thumb on a
transparent track.

## Components

### Navigation
- **Character:** a masthead, not a nav bar. Full-bleed 1px ink rule beneath, content held to 78rem, baseline-aligned.
- **Style:** brand name at `0.9em` in ink on the left; three links (Writing / About / Feed) at `0.79em` in muted, 1rem apart, on the right. Vertical padding 0.8rem.
- **States:** hover lifts a link to ink; the active route is ink with a 3px-offset underline. Neither the brand nor the nav links carry the global red link underline.
- **Mobile:** unchanged — the flex row and the type sizes hold at 390px.

### Links (body)
- **Default:** Cornell red with a 1px bottom border at 38% alpha, no `text-decoration`.
- **Hover:** the border goes to full-strength red. No color or weight change.
- **Focus:** a 2px red `:focus-visible` outline at 3px offset, applied globally to every focusable element.
- **Demoted variants:** navigation, brand, footer links and entry titles set `border-bottom: 0` and take a gray, using an offset underline on hover instead. Red is the *prose* link color, not the global one.

### Writing Index (signature component)
- **Character:** a numbered reference list, the site's most distinctive surface. Used by both the home page and `/blog`.
- **Structure:** a 2-column baseline-aligned grid — a 2.1rem numeral column (1.8rem ≤640px) and a content column. The numeral is `[n]` from a CSS counter that decrements, seeded via an inline `--start` custom property so the newest post carries the highest number.
- **Rows:** title at `0.98em` in ink, description at `0.84em` in muted, then an italic tabular meta line at `0.76em` in faint carrying the date. 0.68rem vertical padding, 1px dotted hairline beneath.
- **Hover:** the title takes a 3px-offset underline. Nothing moves.

### Title Block
- **Character:** the top of a paper. Centered name at `1.55em`, italic affiliation at `0.82em` muted directly beneath, then the Abstract.
- **Abstract:** left-aligned inside the centered block, `0.9em`/1.6 in muted, bounded by hairlines top and bottom with 0.75rem of padding. It opens with an italic ink run-in head (`Abstract.`) at regular weight, not bold.

### Post Body (`article.paper`)
- **Character:** typeset, not laid out. Every paragraph indents 1.6em, the first of a section included, and the gap between them is 0.5rem — both marks, by the author's decision. Text is ragged right at every width.
- **Section numbers:** typed in the markdown. A CSS counter was removed because the prose cross-references sections by number and the counter also numbered the Abstract, shifting every reference by one. Those same `<h2>` strings now also feed the section rail, so a hand-typed numeral appears twice — once in the text, once in the rail — and the two agree because they are literally the same string.
- **Byline:** centered italic at `0.82em` muted — site title, middle dot, long-form date; a revision line, when present, one step smaller in faint.

### Sidenote (signature component)
- **Character:** a note in the margin, at the height of the paragraph that provoked it. It carries source citations, measurement conditions, original-language quotations, and asides — never a step of the main argument.
- **Default (any width):** in the flow, full column width, `0.82em`/1.55 in muted, 1rem/1.2rem margins, 0.9rem left padding, and a 2px `--rule` stroke on the left. Left-aligned, hyphenation off. Inline `code` inside it steps down to `0.94em`.
- **Promoted (≥1200px):** floats right at `--sidenote-w`, pulled past the column edge by a negative right margin of `--sidenote-w + --sidenote-gap` — 15rem / 17.5rem at full width, 10rem / 11.75rem in the narrow tier — `0.74em`/1.5, stroke and left padding removed, 0.35rem above and 1.1rem below.
- **Authoring:** `<aside class="sidenote">` with blank lines inside it so markdown still processes; documented in `drafts/README.md`.

### Section Rail (signature component)
- **Character:** the reader's position in the paper, set in the left margin — the sidenote strip's mirror, and the only navigational surface inside a post.
- **Source:** not hand-authored. `render(post)` returns the `headings` array, `[...slug].astro` passes it into `BlogPost.astro` as a prop, and the layout filters `depth === 2`. `h3` is excluded on purpose: it marks sub-parts *inside* a section, so including it would make the rail longer than the prose it indexes.
- **Suppression:** the rail renders only when a post has more than one `<h2>`; a single-section post gets no rail, because a one-item index is not an index.
- **Style:** `0.72em`/1.42, links in faint with no underline (the global red link border is removed), each entry a block with a 2px transparent left border and 0.6rem of left padding. Hover takes a link to ink.
- **Current section:** the entry takes ink text and a 2px `--sig` left border, filling the transparent stroke it already reserved so nothing shifts.
- **Behavior:** a small inline script listens to `scroll` (passive) and marks the last `<h2>` whose top is above 140px. It is the **only client-side JavaScript on the site** — KaTeX still renders at build time and ships no math runtime. The script is strictly additive: without it the rail is a working list of anchor links, which is the state it must always degrade to.
- **Responsive:** hidden below 1200px, full stop. There is no mobile variant, no drawer, no toggle. The threshold moved down from 1520px because a 1080p screen at 150% OS scaling reports 1280 CSS pixels — the author could not see the rail on the machine the site is made on.

### Figures
- **Character:** plates, closed top and bottom by a 1px `--rule` hairline — the same stroke as `hr`, no new device. Full text-column width (`--fig`, capped at `100vw - 3rem`), 1.6rem of air outside the rules and 1.4rem/1.15rem inside them. The rules are load-bearing: a plate is only one gutter wider than the prose, which is not enough on its own to mark where it begins and ends, and without them the caption floated between the figure and the next paragraph instead of belonging to the plate.
- **Caption:** centered, `0.79em`/1.48 in muted, capped at 32rem so it stays narrower than its figure. A bold run-in inside the caption renders in the caps face at regular weight in ink — that is the plate number.

### Tables
- **Character:** ruled, never filled. A caps-free bold roman header row over a 1px `--rule`, body rows on 1px `--rule-soft`, no zebra, no header ground, `tabular-nums` throughout. The header is bold roman rather than the caps face per The Turkish Lowercase Rule.
- **Responsive:** below 48rem the table itself becomes the scrolling box (`display: block; width: max-content; max-width: 100%; overflow-x: auto`). The page must never scroll sideways to reveal a table — that drags the prose off screen with it.

### Code Blocks
- **Character:** a pane of machine output set into the paper, not a quotation of it. This is a corrected decision: the earlier near-monochrome listing read as straw paper and was rejected.
- **Style:** cool listing ground, 1px `--code-rule` border on all four edges, square, 1rem/1.15rem padding, `0.73em`/1.65, 1.5rem above and 1.6rem below, horizontal scroll with the thin styled scrollbar. `pre > code` resets with `all: unset` so only the mono family survives; Shiki's inline background is overridden back to the token ground.
- **Color:** six syntax tokens through Shiki `css-variables` — keyword, string, function, constant, comment (italic), punctuation — plus red for links.
- **Inline code:** mono at `0.86em` on the warm `--wash` with 1px/4px padding, no border, no radius — one step *larger* than a listing. Inline and block differ in both ground temperature and size; inline code is a word in the sentence, a block is a panel.

### Math
- **Inline:** KaTeX at `1.03em`, matching the roman face around it.
- **Display:** centered on the text column, 0.9rem margins, `overflow-x: auto` with `overflow-y: hidden`, and **0.9em of vertical padding** — load-bearing, not decorative. `overflow-x: auto` forces `overflow-y` out of `visible`, so the box clips; a `\top` superscript in a fraction numerator overshoots the content box (measured at ~8.3px when the body was 19px). The padding is in `em`, so it tracked the body down to 17px without remeasurement — that is why it is expressed in `em` and must stay that way.
- **Inside a figure:** width returns to 100%, the transform is dropped, and bottom padding falls to 0.25em so the caption binds to its equation rather than to the paragraph below.
- **Box model:** `.katex, .katex * { box-sizing: content-box }` locally undoes the global border-box reset. KaTeX positions vlists with absolute metrics that assume content-box; under the reset the crossbars clip (measured: vlist clientHeight 30px against scrollHeight 55px). This is required, not incidental.

### Definition Lists (CV / Work)
- **CV:** a 7.5rem italic tabular-faint date column beside a body column at `0.92em`; role names in the caps face. Collapses to one column at 640px.
- **Work:** no date column. Repository names are links in red at `0.95em` in the roman face — deliberately *not* the caps face, per the Identifier Exemption.

### Placeholder Marker
- **Character:** an honest label on unwritten content, driven by a `placeholder: boolean` in the blog collection schema. It is a content-state device, not a decorative badge.
- **On a post:** a four-sided 1px red frame, red text at `0.82em`, 0.6rem/0.85rem padding, with a caps-face run-in `Placeholder.` — an English label, so the caps face is safe here; it lives in the article `<header>` and is therefore outside the paper typesetting selector.
- **In the index:** a caps-face red tag reading `placeholder` at the end of the meta line.
- **Lifecycle:** both markers disappear when the flag flips. Nothing else in the system may borrow the red frame.

### Share Card (generated)
- **Character:** the site's face when a link is pasted somewhere else. Generated at build time by `src/pages/og/[...slug].png.ts` — satori composes the SVG, sharp writes the PNG. Nothing is drawn by hand and no post ever needs cover art.
- **Canvas:** 1200×630 on white, 72px/80px padding, content pushed to the top and bottom edges. A 10px solid ink rule runs across the top — the masthead at card scale.
- **Type:** title at 64px in the roman face, line-height 1.18, clipped at 230px so it stops after three lines; description beneath at 28px italic in muted, clipped at 124px. Both faces are the same woff files the site serves.
- **Footer band:** a 2px hairline above, then a 14px Cornell red square — the Single Signal as a mark rather than a link — followed by the name at 26px and the italic affiliation at 22px in muted.
- **Variants:** the post card carries name and affiliation separately; the site card, whose title is already the name, drops the name from the band and runs `name · affiliation` as one italic line. The card must never print the name twice.

### Empty State
- **Character:** one honest sentence. No illustration, no dashed placeholder box, no call to action pretending the absence is an opportunity.
- **Style:** `0.9em` italic in muted with 0.6rem of top margin, replacing the writing index entirely rather than rendering an empty list.
- **Copy:** it names the absence and offers the feed. It never apologises and never promises a date.
- **Companion rule:** the `/blog` count line is suppressed at zero rather than reading "0 entries" — a count that counts nothing is noise.

### Footer
- **Style:** 1px hairline top border, content to 78rem, `0.79em` faint, copyright left and three links right, wrapping on narrow. Links are muted with a hairline underline that darkens to ink on hover.

## Do's and Don'ts

### Do:
- **Do** carry every new prose surface with rules, measure, and the three-ink ramp before reaching for anything else.
- **Do** keep `--measure` centered and `--fig` equal to it, so a sidenote is the only thing that ever leaves the column, and collapse break-wide elements to 100% at 900px.
- **Do** put citations, measurement conditions, and asides in a `<aside class="sidenote">`, and keep the main argument in the flow.
- **Do** use the drawn `--caps` family for editorial labels, section heads, and run-in heads, at 0.045–0.05em letter-spacing and weight 400.
- **Do** set every date, numeral column, and counted list with `font-variant-numeric: tabular-nums`.
- **Do** keep display math's `em`-based vertical padding and KaTeX's `content-box` override; both were measured and both prevent visible clipping.
- **Do** keep body type at a single 18px step at every width; narrow the gutter instead when space runs out.
- **Do** keep listings at `0.73em` and inline code at `0.86em` — two steps, on purpose.
- **Do** build the section rail from the rendered `headings` array filtered to `depth === 2`, never from a hand-written list, and keep it working as plain anchors with JavaScript off.
- **Do** number sections by hand in the markdown, so in-text cross-references stay correct.

### Don't:
- **Don't** add a `box-shadow`, a `border-radius`, or a card — including around the code block, which earns its separation from the ground temperature and a 1px border.
- **Don't** use `font-variant: small-caps`. Use the `--caps` face.
- **Don't** set identifiers — repository names, file names, symbols — in the caps face.
- **Don't** give Cornell red any job beyond links, focus, selection, and the placeholder state.
- **Don't** let the listing palette leak out of `pre`, and don't import a foreign syntax theme; color comes from the `--astro-code-token-*` bindings.
- **Don't** reintroduce a CSS section counter on `article.paper` — sections are numbered in the source, referenced by number in the prose, and reused verbatim in the section rail.
- **Don't** give the rail a mobile form (drawer, toggle, sticky bar), and don't render it for a single-section post; below 1200px there is no strip to put it in.
- **Don't** add a second client-side script without a hard reason. The rail's scroll listener is the only one, and the page must keep working with it off.
- **Don't** widen `<main>` to hold the margin column — that centers the box and pushes the text left, which is the defect the current centering fixed.
- **Don't** justify or hyphenate below 900px, don't justify a sidenote at any width, and don't apply paper typesetting with a descendant selector — `article.paper > p` is deliberate.
- **Don't** introduce a third font family or a fourth gray.
