# JD Rubric — Rationale

**Authored by Claude, offline, by reading `job_description.docx` in full.** This is a static
committed artifact (`artifacts/jd_rubric.json`). `rank.py` reads it at rank-time; **no LLM or
network call happens during ranking.**

The JD is unusually candid. It does not give a checklist — it tells you what they *mean* and
lists the disqualifiers they *actually apply*. The rubric is a faithful, machine-readable
encoding of that text. Below, each rubric element is tied to the JD line that motivates it.

## What "good" means (the positive signal)

The JD's "ideal candidate" is explicit: **6–8 years total, 4–5 in applied ML/AI at product
companies (not services), having shipped at least one end-to-end ranking/search/recommendation
system to real users.** So the positive signal is not "owns AI keywords" — it is **evidence of
building retrieval/ranking systems in production at a product company.**

- **`facets`** — ten short requirement statements embedded by `precompute.py`. A candidate's
  `semantic_fit` is the aggregate cosine of their profile embedding against these. This is what
  surfaces the **plain-language Tier-5** (JD: *"A Tier 5 candidate may not use the words 'RAG'
  or 'Pinecone' … but if their career history shows they built a recommendation system at a
  product company, they're a fit."*). Facets are written in plain English so semantic match
  fires on "built a recommendation system" even without buzzwords.
- **`role_taxonomy`** — drives `role_coherence`, the **decisive anti-keyword-stuffer signal**.
  JD: *"A candidate who has all the AI keywords listed as skills but whose title is Marketing
  Manager is not a fit, no matter how perfect their skill list looks."* We blend title-taxonomy
  match (current + historical) with the semantic fit of the career text, so a real engineer with
  an unusual title still scores, while an Accountant with 9 AI skills does not.
- **`career_evidence_terms`** — drives `career_evidence`: did they actually *build*
  ranking/search/recsys/retrieval, and at **product** (not services) companies? Parsed from the
  free-text role descriptions, which is where real builders leave fingerprints.
- **`experience`** — soft curve, band 5–9, peak 6–8. The JD insists this is *"a range, not a
  requirement"*, so it is a score component, never a gate.

## What "bad" means (the hard-negatives)

Each is a **multiplicative penalty** (not a hard zero — that's reserved for honeypots) so the
reasoning can name the concern and a strong candidate with one soft flag isn't annihilated.

| Gate | JD basis | Penalty |
|---|---|---|
| `services_only` | "People who have only worked at consulting firms (TCS, Infosys, …) in their entire career." Currently-at-services-with-prior-product is explicitly fine. | ×0.45 |
| `location_visa` | "Outside India: case-by-case, but we don't sponsor work visas." | ×0.30 (won't relocate) / ×0.80 (will, visa risk) |
| `research_only` | "pure research environments … without any production deployment — we will not move forward." | ×0.55 |
| `cv_speech_only` | "primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure." | ×0.60 |
| `langchain_only_recent` | "'AI experience' … primarily recent (<12mo) projects using LangChain to call OpenAI … unless … substantial pre-LLM-era ML production experience." | ×0.65 |
| `title_chaser` | "optimizing for Senior → Staff → Principal titles by switching companies every 1.5 years … we need someone who plans to be here for 3+ years." | ×0.80 |
| `non_technical_role` | The canonical trap: non-engineering current role + career, regardless of listed AI skills. | ×0.25 |

## Behavioral signals are a modifier, never a driver

JD: *"a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter
response rate is, for hiring purposes, not actually available. Down-weight them appropriately."*
The behavioral block multiplies the fit score within **[0.80, 1.12]** — enough to sink an
unreachable candidate, not enough to lift a non-fit into contention.

**Sentinels are neutral.** EDA showed `github_activity_score == -1` on 64% of the pool,
`offer_acceptance_rate == -1` on 60%, and empty `skill_assessment_scores` on 76%. Absence of
these is the norm, so it is never penalized — only *present* signals move the modifier.

## Trust over claims

`trust_skills` weights each skill by `proficiency × duration_months × endorsements ×
assessment_score`. A skill claimed "expert" with 0 months used, no endorsements, and a low
Redrob assessment barely counts. This is what makes keyword-stuffing structurally unprofitable:
the keywords are there, but they carry no weight. (See `CAND_0000001`: "advanced" NLP with a
38.8 assessment, "advanced" Fine-tuning with 41.6 — the trust multiplier discounts these.)

## Weights

`component_weights` put the most mass on `role_coherence` (0.26) and `career_evidence` (0.24),
because those two are what actually separate true fits from the keyword traps. These weights are
tuned on the Claude-authored eval set to maximise NDCG@10; the ablation in `eval/results.md`
quantifies how much `role_coherence` is doing.
