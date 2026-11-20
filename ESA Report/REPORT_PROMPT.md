# Brief: Capstone Phase-3 (ESA) report for PW26_RJ_02

This brief was written after reading every file in `ESA Report/`:
- `Report-  Guidelines.pdf` (5 pages)
- the four templates in `Report initial Pages(Pg 1-10)/`
- `PPT-Template_Phase3- Review3.pptx` (16 slides)
- last semester's `PW_26_RJ_02_Final Report-2.pdf` (27 pages)

It also draws on the project's current results (`STATUS.md`, `results/`) and the GRACE paper draft. The report generator (`build/build_report.py`) follows it.

## Task

Write the UE23CS441A Capstone Project Phase-3 report (August – Dec, 2026) for project PW26_RJ_02.
- **Take only these from last semester's report:** personal details and references. That means title, team names and SRNs, guide, project ID and acknowledgement people.
- **Write all technical content fresh** from the project's latest state, the GRACE system built in Phase 3. That covers results, design and method.
- **Follow the department guidelines and templates exactly.** Where the two disagree, see "Conflicts and how they are resolved" below.

## Fixed details (from last semester's report and the templates)

- **Title:** "Generalized Deepfake Detection Framework with Scalable Multimodal Integration". This is the registered title. The system it describes is GRACE.
- **Project ID:** PW26_RJ_02
- **Course:** UE23CS441A – Capstone Project Phase - 3, seventh semester, August – Dec, 2026
- **Team, in last semester's order:**

  | Name | SRN |
  | --- | --- |
  | Charan L | PES1UG23CS159 |
  | Deepak S N | PES1UG23CS178 |
  | Samrudh N Shindhe | PES1UG23CS508 |
  | Samarth Gowda G M | PES1UG23CS502 |

- **Guide:** Dr. Jyothi R, Associate Professor, Department of CSE, PES University
- **Project Coordinator:** Dr. Chaitra B (Phase-3 template)
- **Chairperson:** Dr. Shylaja S S. **Registrar:** Dr. Sridhar K S.
- **Vice-Chancellor:** Dr. Suryaprasad J. **Chancellor:** Prof. Jawahar Doreswamy.

## Formatting rules (guidelines)

- **Page and text:** A4 (210 × 297 mm), 2.00 cm margins on all sides, one side only. Body is Times New Roman 12, 1.5 line spacing, justified.
- **Headings:**
  - Chapter number and title centred, 18 pt bold.
  - Section ("2.1 Title") left-aligned, 16 pt bold.
  - Subsection ("2.1.1 Title") left-aligned, 14 pt bold.
- **Numbering:** chapters in Arabic numerals. Figures, tables and equations are numbered per chapter (Figure 3.2, Table 3.1, equation (3.1)).
- **References:** IEEE style, numbered in order of first citation, cited in the text as [n].
- **Header and footer** (from Chapter 1 on):
  - Header: project title on the left, institution logo on the right.
  - Footer: "Dept. of CSE" on the left, "August – Dec, 2026" in the middle, page number on the right.
  - The first page of each chapter shows the footer only, no header.
- **Front matter, in order:** outer/inner title page, certificate, declaration, acknowledgement, abstract, table of contents, list of figures, list of tables.
- **Abstract:** at most 150 words, salient features of the work.
- **Acknowledgement order:** guide, project coordinator, chairperson, registrar, Vice-Chancellor, (CEO), Chancellor, parents, family and friends.
- **Chapters, in order:**
  1. Introduction
  2. Problem Definition
  3. Literature Review
  4. Project Requirements Specification
  5. High-Level Design
  6. Low-Level Design
  7. System Design (Detailed)
  8. Proposed Methodology
  9. Implementation and Pseudocode
  10. Experimentation Results and Discussion
  11. Conclusion and Future Work

  These are followed by References, Appendix A (Definitions, Acronyms and Abbreviations) and Appendix B (User Manual).
- **Conclusion:** summarise the work and its usefulness, plus future work.
- **Units and equations:** SI units only. Important equations are numbered.
- **Plagiarism:** no verbatim copying; quote and cite any short excerpt. The plagiarism limit is 15%, so all prose is written fresh, including prose adapted from our own paper draft.

## Conflicts and how they are resolved

| Conflict | Decision |
| --- | --- |
| Templates are US Letter; guidelines say A4 | A4 for every page; the template layouts are kept |
| Guidelines say "Title/Subtitle: Bookman Style, 12 bold underline", but also give 18/16/14 pt sizes | Use the explicit content-organization sizes (18/16/14, bold) in Times New Roman, matching the templates |
| Guidelines' footer says "Aug – Dec 2021" | Use "August – Dec, 2026", as the Phase-3 template does |
| Template abstract says "not exceeding two paragraphs" | Two short paragraphs, at most 150 words in total |
| Guidelines' acknowledgement lists a CEO; the template does not name one | Follow the template wording and add parents; leave the CEO out unless the team supplies a name |
| Template table of contents names "Problem Statement" and a combined "System Design" | Use the guidelines' fuller list (Problem Definition, PRS, HLD, LLD, System Design) |

## Content rules (honesty and accuracy)

- **Every number comes from the project's results files.** These are `results/tables.md`, `results/*.csv`, `splits/counts.md` and `STATUS.md`. The generator reads the tables directly, so re-running it after re-evaluation refreshes the numbers.
- **Celeb-DF:** the official test list arrived on 20 Nov 2026 and re-evaluation is queued. Until it finishes, the report states which Celeb-DF test set its numbers use, interim or official, read from `splits/counts.md`.
- **Report negative results plainly:**
  - DANN on Swin-B hurt.
  - Patch-level DANN (F2) collapsed.
  - Unsupervised Celeb-DF adaptation (D2) lowered Celeb-DF AUC.
  - Distillation did not raise clean accuracy.
  - Deletion tests are weak.
- **Phase-2 result:** last semester reported 98.98% validation accuracy. Phase 3 found that number was inflated by train/validation leakage. Explain this and show the leak-free numbers.
- **Final model:** F1, chosen on FF++ validation only. It is CLIP ViT-L/14 + LoRA, a DCT branch with gated fusion, the concept bottleneck and self-blended images, with no DANN. X1 (trained with DFD) is reported as an extra experiment.
- **Multimodal (audio):** not built. The registered title mentions it, so it is presented as a scalable design point and future work.
- **App:** describe the Phase-3 app ("Simple answer" design) and its tested endpoints.

## Figures to use

- **Paper draft:** architecture overview, training diagram, video-to-verdict diagram.
- **Results:** ROC curves, robustness curves, explanation grid, t-SNE.
- **SBI examples:** `cache/sbi_check.png`.
- **App screenshots:** taken by the browser test.
- **New, drawn for the design chapters:** context and deployment diagram, layered architecture, LLD class diagram, app sequence diagram, and the experiment and queue workflow.

## Deliverables

- `ESA Report/output/PW26_RJ_02_Phase3_Report.docx`: editable Word file. Fields are updated by Word: table of contents, list of figures, list of tables, page numbers.
- `ESA Report/output/PW26_RJ_02_Phase3_Report.pdf`: PDF exported by Word, used to check every page.
- `ESA Report/build/build_report.py`: re-runnable generator, run as `python "ESA Report/build/build_report.py"`.
