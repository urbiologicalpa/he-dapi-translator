# SEO & Discovery Checklist

Goal: make `he-dapi-translator` findable when people search for it or its
keywords — realistic target is the **long tail** (`he-dapi`, `H&E DAPI
translator`, `virtual staining 3.7GB CPU`), not broad head terms.

## Done (technical baseline)

- [x] Pages `index.html`: keyword-rich `<title>`/`description`, `canonical`,
      `robots` meta, Open Graph + Twitter cards, `SoftwareSourceCode` JSON-LD.
- [x] `robots.txt` (allow all + sitemap ref) and `sitemap.xml` on `gh-pages`.
- [x] Repo **description** + **topics** (`digital-pathology`, `image-to-image`,
      `fluorescence-microscopy`, `unet`, `computer-vision`, …).
- [x] PyPI + Hugging Face pages link back to GitHub (free backlinks).

## 1. Get indexed (highest priority — needs your Google account)

Without this a new `github.io` page can stay unindexed for weeks.

- [ ] **Google Search Console** → <https://search.google.com/search-console>
  - Add a **URL-prefix** property: `https://urbiologicalpa.github.io/he-dapi-translator/`
  - Verify with the **HTML tag**: paste the `<meta name="google-site-verification" ...>`
    into the `gh-pages` `index.html` `<head>` (or send me the file and I'll add it),
    **or** the **HTML file** method (add the downloaded `googleXXXX.html` to the
    `gh-pages` root).
  - Sitemaps → submit `https://urbiologicalpa.github.io/he-dapi-translator/sitemap.xml`.
  - URL Inspection → paste the page URL → **Request indexing**.
- [ ] **Bing Webmaster Tools** → <https://www.bing.com/webmasters> (import from GSC).

## 2. Backlinks (the real ranking driver)

- [ ] **Awesome-lists** — search GitHub for and open a PR adding the repo:
  `awesome digital pathology`, `awesome medical imaging`, `awesome computer vision`,
  `awesome low-resource`. Keep the entry short and factual.
- [ ] **Reddit** (link the live in-browser demo, not just the repo):
  - r/MachineLearning (use the bi-weekly "Show" thread), r/computervision,
    r/pathology, r/bioinformatics.
- [ ] **Show HN** → <https://news.ycombinator.com/submit> (title like
  "Show HN: H&E→DAPI virtual staining trained on a 3.7GB CPU laptop").
- [ ] **Blog / dev.to / Medium** write-up linking the repo and HF Space.
- [ ] **LinkedIn** post (good for lab/clinician reach).
- [ ] **Zenodo DOI**: connect the GitHub repo at <https://zenodo.org/account/settings/github/>
  → cut a release → get a DOI → add it to `CITATION.cff`. Papers/Google Scholar
  index DOIs.

## 3. Consistency

- Use the exact phrase **"H&E to DAPI virtual staining"** in README, Pages
  `<title>`, PyPI summary, and the HF model card so Google binds the phrase to us.
- Keep the repo **description**, **topics**, and **links** in sync everywhere.

## 4. Track it

- Search Console → Performance: impressions/clicks per query.
- GitHub repo → Insights → Traffic: views, clones, referrers.
- Revisit after ~2 weeks; double down on whichever free channel drives referrers.

## Expectations

| Horizon | Realistic outcome |
|---------|-------------------|
| Days | Indexed for the exact repo/package name |
| 1–2 weeks | Ranking for `he-dapi`, exact long-tail phrases |
| 1–3 months | Some long-tail keyword traffic, if backlinks land |
| (broader terms) | Not realistic without stars/authority — grow via contributors |
