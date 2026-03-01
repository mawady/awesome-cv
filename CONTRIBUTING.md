# Contributing to awesome-cv

Thank you for taking the time to contribute. This list exists to help people learn computer vision. Every good addition makes it more useful for students and researchers worldwide.

---

## Table of Contents

- [What we're looking for](#what-were-looking-for)
- [What we're NOT looking for](#what-were-not-looking-for)
- [How to contribute](#how-to-contribute)
- [Entry format and style rules](#entry-format-and-style-rules)
- [Specific rules per section](#specific-rules-per-section)
- [Reporting broken or outdated links](#reporting-broken-or-outdated-links)
- [Code of conduct](#code-of-conduct)

---

## What we're looking for

This list has an **educational focus**. A good contribution is one that helps someone genuinely understand computer vision, not just collect links.

Specifically, we welcome:

- **Papers** that introduced a lasting idea, architecture, or benchmark (not every recent preprint)
- **Libraries and repos** that are actively maintained, well-documented, and widely used or uniquely useful
- **Courses** that are freely available (or widely accessible) and taught by recognised instructors
- **Books** that have proven track records in university curricula
- **YouTube channels** with consistent, high-quality educational content
- **Datasets and annotation tools** that are public and practically useful
- **Corrections** to existing entries: wrong links, outdated descriptions, deprecated projects

---

## What we're NOT looking for

To keep the list high-quality and focused, please do **not** submit:

- ❌ Papers that have not yet had meaningful community uptake
- ❌ Personal projects, student coursework, or self-promotional repos
- ❌ Paid or paywalled courses with no free version
- ❌ Libraries that have been abandoned (last commit over 3 years ago) unless they have clear historical or educational value
- ❌ Duplicate entries that already exist in a different section
- ❌ Blog posts, newsletters, or social media accounts (YouTube channels are the exception)
- ❌ Anything that requires payment or registration to access the core content

If you're unsure whether something qualifies, open an **Issue** to discuss it before spending time on a PR.

---

## How to contribute

### Option A: Open an Issue (easiest)

If you're not comfortable with pull requests, just [open an issue](../../issues/new) and describe:

1. What you want to add or change
2. Which section it belongs in
3. Why it meets the inclusion criteria above

We'll take it from there.

### Option B: Submit a Pull Request

1. **Fork** this repository
2. **Create a branch** with a descriptive name:
   ```
   git checkout -b add-depth-estimation-papers
   ```
3. **Make your changes** to `README.md` following the format rules below
4. **Check your links** to make sure every URL works and points to the right place
5. **Submit a PR** with a clear title and a short explanation of what you added and why it belongs

#### PR title format

```
Add [entry name] to [section name]
Fix broken link for [entry name]
Update description for [entry name]
```

#### What to write in the PR description

- What you're adding or changing
- Why it meets the educational criteria
- For papers: approximate citation count and whether open-source code exists

---

## Entry format and style rules

Consistency matters. It makes the list easier to read and maintain.

### Papers (Popular Articles section)

```markdown
[ShortName, YEAR] Lastname, Firstname, et al. "Full paper title." Venue (YEAR).
```

Examples:
```markdown
[ResNet, 2016] He, Kaiming, et al. "Deep residual learning for image recognition." CVPR (2016).
[ViT, 2020] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words." ICLR (2021).
```

Rules:
- Use the short name the community actually uses (ResNet, not DRFIR)
- List the first author's full name, then "et al." for three or more authors
- Include venue and year
- Papers should be ordered chronologically within each subsection

### Libraries (table format)

```markdown
| [LibraryName](https://github.com/org/repo) | Short description of what it does |
```

Rules:
- Link to the GitHub repo, not a homepage or docs page
- The description should say *what it does*, not just repeat the library name
- If a library is archived or unmaintained, add `(last updated: YEAR)` at the end

### Repos (table format)

```markdown
| [repo-name](https://github.com/org/repo) | `[TAG]` | Description |
```

Use the existing tags: `[ObjCls]`, `[ObjDet]`, `[ObjSeg]`, `[GenLib]`, etc.
If your entry needs a new tag, propose it in the PR description.

### Courses (table format)

```markdown
| [Course Title](URL) | YEAR | Instructor Name | Institution |
```

Rules:
- Link directly to the course page or YouTube playlist
- Use the year the course was last taught or recorded
- Free and openly accessible courses only

### YouTube channels

```markdown
* [@handle](https://www.youtube.com/@handle) `[TAG]`, Real Name: one sentence description.
```

Tags: `[Individual]`, `[Conferences]`, `[University]`, `[Talks]`, `[Papers]`

---

## Specific rules per section

| Section | Key inclusion criteria |
|---|---|
| **Popular Articles** | Must have at least 50 citations and be a landmark contribution to the field |
| **Reference Books** | Must be used in at least one university CV curriculum |
| **Courses** | Must be freely available; instructor must be at a recognised institution or have clear expertise |
| **Libraries** | Must have at least 100 GitHub stars and active maintenance (or clear historical value) |
| **Repos** | Must have open-source code; must describe a usable tool, not just a paper implementation |
| **Dataset Collections** | Public datasets only; must include a direct access link |
| **Annotation Tools** | Must be open-source or freely usable |

---

## Reporting broken or outdated links

If you find a broken link, outdated description, or a library that has been archived since it was added:

1. [Open an issue](../../issues/new) with the title: `Broken link: [entry name]`
2. Include the section, the current (broken) URL, and a suggested replacement if you have one

You don't need to submit a PR for link fixes. An issue is enough and we'll patch it quickly.

---

## Code of conduct

This project follows a simple rule: **be kind and constructive**.

Contributions are reviewed by a researcher and lecturer who cares about the field. Feedback on PRs is honest but respectful. We expect the same from contributors.

Behaviour that will get a PR or issue closed immediately:
- Self-promotion without disclosure (if you're the author of something you're suggesting, say so; it's not disqualifying, but it needs to be transparent)
- Spam or off-topic submissions
- Dismissive or hostile responses to review feedback

---

## Questions?

Open an [issue](../../issues/new) and tag it with the `question` label. We're happy to help you figure out whether something is a good fit before you spend time writing a PR.

---

*Thanks for helping make this a better resource for the computer vision community.*