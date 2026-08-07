---
name: metabolic_health_report
version: 1.1
description: Defines structure, tone, and content guidelines for generating human-readable metabolic health reports from CGM and dietary data
injection_points:
  - analyzer
tools:
  - cgm_parser
  - cgm_metrics
  - meal_parser
  - qc_diet_data
---

# Metabolic Health Report

<!--
This is the report skill from the full system, pruned to the four tools in this
repository. Sections whose capability is absent here — How You Compare,
Meal-Glucose Connections, Optimized Meal Suggestions, Looking Ahead: Health
Predictions, Full Risk Factor Analysis, References — have been removed rather
than left in place, because a section with no tool behind it is precisely what
teaches the model to fabricate (see Skill-Tool Calibration below).

The full system injects this skill at the analyzer only; tool-chain guidance
lives in separate workflow skills. This build ships two skills, so the workflow
section below is carried here and the skill is injected at both stages.
-->

## Purpose

Generate a metabolic health report that is:
1. **Human-readable first** — written for a person, not a data scientist
2. **Actionable** — focuses on what users can actually do
3. **Grounded** — every claim tied to specific data, no speculation
4. **Honest about scope** — states what the data can and cannot support
5. **Concise** — main report readable in 5-7 minutes (~900-1,300 words)

---

## Evaluation Contract

When the CGM and diet tools have run successfully, **preserve the report
scaffold below even if the user asks for a short answer.** Do not collapse the
report into a couple of paragraphs, and do not drop the Technical Appendix.

The prompts this system is evaluated against are realistic user requests, and
real users ask for brevity ("half a page", "just the actionable stuff"). Honour
the *tone* of that request — tighter prose, less hedging, lead with what
matters — but not by discarding structure. The structure is what makes the
report auditable, and a grounded claim the reader cannot trace is worth little
more than an ungrounded one.

Where an ask cannot be answered at all, say so briefly in the section it would
have belonged to, and move on. Do not pad, and do not substitute.

---

### Required Extractor-Friendly Anchors

Include these stable labels exactly when the values are available:

```markdown
| Metric | Value |
|--------|-------|
| **Mean Glucose** | X mg/dL |
| **CV** | X% |
| **Time in Range (70-180 mg/dL)** | X% |
| **Time Above 180 mg/dL** | X% |
| **GMI** | X% |
```

If eA1c is not available from CGM metrics, do not invent it. If GMI is
available, state it with the exact label `GMI`.

**Report these values at the precision the tool returned, to two decimal
places — do NOT round them.** This table is the one place in the report where
the rounding rule under "Numbers in Prose" does not apply. Write
`116.49 mg/dL`, not `116 mg/dL`.

The reason: this table is a contract with the eval harness, which locates these
values by pattern and compares them against the tool output. A rounded value
reads as a transcription error to the check even though the report is correct.
Round freely in the surrounding prose, where a human is reading; keep full
precision here, where a machine is.

Renaming a label, or folding several metrics into one cell, has the same effect:
the value becomes invisible to the evals even though the report states it.

### Required Diet Anchors

In `### Your Diet`, include a data-quality line drawn from
`qc_diet_data.severity_counts`:

    **Meal Data Quality:** Quality validation was performed on the diet log.
    Pass: X entries (X%), Check: X (X%), Fail: X (X%).

**There is no daily nutrition anchor in this build.** The full system reports
per-day energy and macronutrient totals, computed by the meal_to_text tool, which is
not shipped here. Do not substitute your own arithmetic over the meal rows: see
Your Diet below.

---

## Report Structure

The report has TWO main parts:

```
MAIN REPORT (Human-Readable, 5-7 minute read)
│
├── Executive Summary
├── What You Can Do (Recommendations)
├── Your Health Profile
│   ├── Your Glucose Patterns
│   └── Your Diet
│
TECHNICAL APPENDIX
├── Data Quality Assessment
├── Complete Metrics
└── Methodology Notes
```

---

## Part 1: Main Report

### Section 1: Executive Summary

**Purpose**: Provide the essential takeaway in 2-3 paragraphs. A reader who only
reads this section should understand their key findings and top opportunity.
**Pull the single most important insight from each major section.**

**REQUIRED elements (synthesize from each capability):**
1. Glucose control — key metric and interpretation from "Your Glucose Patterns" (1-2 sentences)
2. Day-to-day consistency — the most notable variation across the recording (1 sentence)
3. Diet — key factual finding from "Your Diet" if diet data is available (1 sentence)
4. Top actionable opportunity — the single most impactful recommendation (1-2 sentences)

**Word count**: 150-250 words

**Format**: Flowing prose paragraphs. Numbers embedded naturally in sentences.
No bullet points, no headers within this section.

**EXAMPLE — GOOD:**

> Your 13-day glucose monitoring reveals stable metabolic patterns with a mean
> glucose of 98 mg/dL and 97% of readings in the healthy range (70-180 mg/dL).
> Your glucose variability is low, with a coefficient of variation of 11%, well
> within the consensus target of 36% or below.
>
> Day-to-day consistency was strong throughout the monitoring period. Your daily
> means ranged from 92 to 106 mg/dL, and your most variable day, November 13th,
> still held 94% of readings in range. You logged 138 food items across the
> period, with a consistent three-meal pattern.
>
> Your top opportunity: three of your logged entries did not pass data-quality
> validation, which limits what the dietary summary can support. More consistent
> logging would strengthen these insights.

**EXAMPLE — BAD:**

> **Key Metrics:**
> - Mean glucose: 98 mg/dL
> - Time in range: 97%
> - CV: 10.8%
>
> **Findings:**
> - Stable glucose control
> - Low variability
> - Evening elevations noted

*(Problem: Bullet points, no narrative flow, no actionable insight)*

---

### Section 2: What You Can Do

**Purpose**: Consolidate ALL recommendations into one prioritized section. This
appears early because users value actionability.

**REQUIRED:**
- 3-5 concrete recommendations maximum
- Prioritized by expected impact
- Focus on MODIFIABLE factors only
- Each recommendation in 2-3 sentences of prose
- Connect each to specific data from their report

**FORBIDDEN:**
- Listing non-modifiable factors (age, sex, genetics) as action items
- Generic advice not tied to their specific data
- More than 5 recommendations (cognitive overload)
- Medical diagnoses or medication recommendations
- Bullet-point lists

**Format**: Short paragraphs, each focused on one recommendation. Use
transitional language ("Additionally," "You might also consider").

---

### Section 3: Your Health Profile

This section contains the detailed findings, organized by capability. Begin with
a brief transition:

> "The following sections provide detailed findings from your monitoring period,
> including glucose patterns and dietary logging."

#### Your Glucose Patterns

**Capability**: CGM Analysis (cgm_parser, cgm_metrics) — *Complete*

**Purpose**: Present CGM findings in understandable prose with key numbers
embedded.

**REQUIRED:**
- Mean glucose with interpretation
- Time in range with clinical context
- Variability assessment (CV) with plain-language meaning
- Notable patterns (day-to-day consistency)
- Any episodes of concern (hypoglycemia, significant hyperglycemia)
- The critical-metrics table with the exact anchor labels given above

**Format**: 2-3 paragraphs of prose. Embed numbers naturally. One small table
acceptable for daily patterns if helpful.

**Grounding rule**: Every pattern claimed must reference specific data points
(dates, values).

`cgm_metrics` returns these as top-level scalars: `cgm_mean`, `cgm_gmi`,
`cgm_cv`, `cgm_sd`, `cgm_in_range_70_180`, `cgm_below_70`, `cgm_below_54`,
`cgm_above_180`, `cgm_above_250`, `cgm_mage`, `cgm_j_index`, `days_valid`,
`days_total`, `units`.

Consensus reference values you may cite (Battelino et al., *Diabetes Care*
2019), naming them as consensus targets and not as this reader's norms: time in
range above 70%, time below 70 mg/dL under 4%, time below 54 mg/dL under 1%,
time above 180 mg/dL under 25%, CV at or below 36%.

**Day-to-day detail**: use `df_head('<daily_metrics_aid>', n=30)` — one row per
day carrying per-day metrics and coverage. State the actual dates.

**EXAMPLE — GOOD:**

> Over your 13-day monitoring period, your glucose averaged 125 mg/dL with 89%
> of readings falling within the healthy range of 70-180 mg/dL. Your glucose
> variability (CV of 32%) indicates moderate fluctuation around this average —
> within the consensus target below 36% but suggesting some day-to-day variation
> worth noting.
>
> Your most stable day was November 8th (CV 18%, mean 102 mg/dL), while November
> 13th and 15th showed more variation with daily means of 138 and 141 mg/dL
> respectively. Day-to-day consistency was moderate — your daily means ranged
> from 98 to 141 mg/dL across the period. You experienced no readings below
> 70 mg/dL during the monitoring period.

**FORBIDDEN:**
- "Dangerous," "alarming," "concerning" language
- Diagnostic labels ("diabetic patterns," "pre-diabetic")
- Claiming patterns without specific data references
- Time-of-day glucose patterns (e.g., "morning glucose was 95-110 mg/dL from
  6-10 AM") — no tool computes hourly or time-block aggregations from raw CGM
  data
- Per-meal glucose responses — no tool in this build links meals to glucose

**Sub-daily patterns**: not available in this build. The finest granularity any
tool here produces is per-day, from `daily_metrics_aid`. Do not extrapolate to
within-day claims.

**Visualization**: the Reporter appends a daily glucose-profile figure to the
report after you finish. Do not attempt to embed a plot yourself, and do not
describe a figure you have not seen.

---

#### Your Diet

**Capability**: Meal Logging QC (qc_diet_data) — *Complete*, Meal Analysis
(meal_parser) — *Complete*

**Purpose**: Summarize dietary patterns factually. No causal speculation about
glucose effects.

**REQUIRED:**
- Number of items logged with completeness note
- Typical daily pattern (example day, not averages across days)
- Meal timing patterns
- Data quality caveat if validation flagged entries

**CONDITIONAL**: Only include if diet data was provided. If not available, omit
this section entirely (do not write "no diet data available" in the main
report — note in appendix only).

**Format**: 1-2 paragraphs prose, with one example day shown.

**How to ground it**: read the meal table for what it directly contains.

    df_schema('<meal_aid>')        # columns and row count
    df_head('<meal_aid>', n=100)   # the meals themselves

You may describe, from what you read: how many items were logged
(`meal_parser.rows_parsed`) over how many days (`meal_parser.date_range`), when
in the day entries cluster, which food categories recur, and what a
representative day contained by name.

**You may NOT state per-day or per-period energy or macronutrient totals.** No
tool in this build computes them, and adding the rows up yourself is not
grounding — in testing that produced a day whose energy happened to be right
while its carbohydrate was understated by 46%, its protein by 30% and its fat
by 27%, in a sentence that read as authoritative. The full system computes these
in its meal_to_text tool; without it, the claim is out of scope.

**EXAMPLE — GOOD:**

> You logged 138 food items over the 14-day period. A typical day from your
> logging, November 13th, included breakfast at 7:48 AM (porridge with milk and
> coffee), a midday meal at 12:10 PM (wholemeal pasta with chicken and salad),
> and dinner at 7:15 PM. Your entries were typically spaced 4-5 hours apart,
> with an eating window spanning approximately 12 hours.
>
> **2024-01-13:** 1,940 kcal | 210g carbohydrates (43%) | 118g protein (24%) | 71g fat (33%)
>
> **Meal Data Quality:** Quality validation was performed on all logged entries.
> Pass: 123 entries (97.6%), Check: 0 (0.0%), Fail: 3 (2.4%).

**FORBIDDEN:**
- Claiming specific foods "caused" glucose responses
- Any statement linking a meal to a glucose value — no tool computes this
- Any per-day or per-period energy or macronutrient total. No tool computes
  these here, and hand arithmetic over a `df_head` view has been observed to be
  badly wrong while reading as authoritative
- Calculating averages across days (show one representative day by name only)
- Dietary recommendations beyond factual patterns
- Bullet-point meal lists
- Reporting the item count as a meal count: `meal_parser.rows_parsed` counts food
  items, and no tool here groups items into meals
- Dietary quality scores or indices (HEI, DASH, Mediterranean, NOVA,
  Nutri-Score) — the tool that computed these is not part of this build

---

## Part 2: Technical Appendix

**Purpose**: Provide complete data for users who want details, clinicians
reviewing the report, and transparency about methodology.

**Tone shift**: More technical language acceptable. Tables and bullet points
appropriate. Comprehensive rather than curated.

### Appendix Sections

#### Data Quality Assessment

Include:
- CGM coverage statistics (% of expected readings captured), from `cgm_parser`:
  `n_rows`, `n_days`, `gap_count`, `gap_max_minutes`, `gap_total_minutes`,
  `nan_count`, `sampling_interval_seconds`, `original_units`,
  `conversion_factor`; and `days_valid` against `days_total` from `cgm_metrics`
- Meal logging completeness and quality scores, from `qc_diet_data`:
  `severity_counts`, `failed_groups`, `total_rows`, `total_days`,
  `optional_columns_missing`, `warnings`
- Any data gaps or quality concerns
- Statement on data limitations

If rows failed QC, name the rule groups that fired and say what that means for
the dietary section in Part 1. A diet summary computed over rows that failed
validation must be caveated as such.

#### Complete Metrics

Include:
- Full CGM metrics table (mean, SD, CV, TIR at multiple thresholds, MAGE,
  J-index, GMI, etc.). Read beyond the top-level scalars with
  `list_metrics('<global_metrics_aid>')` to see the 55 available, then
  `get_metric('<global_metrics_aid>', '<name>')`. Units are in `metric_units_aid`
- Daily breakdown table, from `df_head('<daily_metrics_aid>', n=30)`

#### Methodology Notes

Include:
- How CGM metrics were calculated, and by which library
- Which tools ran, and their versions
- Limitations and caveats, including the absence of population reference data
  and predictive models in this build
- A statement that reference ranges are published consensus values, not values
  derived from this reader's peers

---

## Writing Style Guidelines

### Tone

- **Warm but professional**: Write as a knowledgeable colleague, not a clinical report
- **Empowering, not alarming**: Focus on opportunities, not problems
- **Honest about uncertainty**: Acknowledge what we don't know
- **Second person**: Use "your" and "you" throughout

### Formatting

| Element | Main Report | Appendix |
|---------|-------------|----------|
| Bullet points | Avoid | Acceptable |
| Tables | Minimal (≤5 rows) | Encouraged |
| Headers | Minimal nesting | Full hierarchy OK |
| Technical terms | Define on first use or avoid | Acceptable without definition |

### Numbers in Prose

Embed numbers naturally in sentences rather than listing them:

**BAD:** "Mean glucose: 98 mg/dL. CV: 11%. TIR: 99%."

**GOOD:** "Your glucose averaged 98 mg/dL with excellent stability (CV of 11%)
and you spent 99% of time in the healthy range."

When rounding:
- Percentages: Round to whole numbers in prose (97.04% → "97%")
- Glucose values: One decimal place max (98.3 mg/dL or "98 mg/dL")

**The critical-metrics anchor table is the exception**: report those values at
full tool precision (two decimal places), never rounded. See Required
Extractor-Friendly Anchors above.

### Anchoring Relative Terms

**FORBIDDEN without anchor:**
- "Higher" → must specify "higher than X"
- "Elevated" → must specify reference range
- "Moderate" → must define the scale
- "Significant" → must clarify statistical vs. clinical significance

**REQUIRED anchoring examples:**
- "Higher than your period mean of 98 mg/dL"
- "Elevated compared to the reference range of 70-100 mg/dL"
- "Moderate variability (CV 28%), within the consensus target below 36%"

Note that in this build the anchor can never be a peer group or a percentile.
Anchor to the reader's own period statistics or to a named consensus target.

### Hedging Language for Associations

When discussing any correlational finding:

| FORBIDDEN (Causal) | ALLOWED (Associative) |
|--------------------|----------------------|
| "caused" | "was associated with" |
| "resulted in" | "coincided with" |
| "due to" | "may be related to" |
| "because of" | "was followed by" |
| "led to" | "tended to occur with" |
| "the X in this meal..." | "meals containing X..." |

---

## Capability Integration

| Capability | Status | Report Section |
|------------|--------|----------------|
| CGM Analysis | Complete | Your Glucose Patterns |
| Meal Analysis | Complete | Your Diet |
| Daily Nutrition Totals | Not in this build | — |
| Meal Logging QC | Complete | Appendix (Data Quality) |
| Population Comparison | Not in this build | — |
| Meal-CGM Response | Not in this build | — |
| Meal Optimization | Not in this build | — |
| Predictions | Not in this build | — |
| Literature Context | Not in this build | — |

**When data is missing for a complete capability**: Omit that section from the
main report. Note the data gap in the Appendix under Data Quality Assessment.

**When a capability is not in this build**: do not mention it, do not
placeholder it, and do not produce its content from your own knowledge.

---

## Quality Checklist

Before finalizing any report, verify:

### Structure
- [ ] Executive summary is 150-250 words, prose only, no bullets
- [ ] Recommendations section has 3-5 items maximum, all modifiable
- [ ] Main report is 900-1,300 words total
- [ ] Technical details are in appendix, not main report
- [ ] Sections for absent capabilities are omitted (not placeholdered)
- [ ] The critical-metrics anchor table appears with the exact labels

### Content
- [ ] Every claim references specific data (dates, values, percentages)
- [ ] No causal language for meal-glucose relationships
- [ ] No percentiles, peer comparisons, risk scores, or predictions anywhere
- [ ] All relative terms have anchors
- [ ] Non-modifiable factors acknowledged but not emphasized
- [ ] No information repeated across multiple sections

### Tone
- [ ] No alarming language ("dangerous," "concerning," "alarming")
- [ ] No diagnostic labels ("diabetic," "pre-diabetic")
- [ ] Empowering and actionable focus throughout
- [ ] Written in second person ("your," "you")

### Format
- [ ] Minimal bullet points in main report
- [ ] Numbers embedded naturally in prose
- [ ] Tables kept small (≤5 rows) in main report
- [ ] Clear transitions between sections

---

## FORBIDDEN Patterns (Summary)

| Pattern | Why Forbidden | Alternative |
|---------|--------------|-------------|
| Bullet-point findings | Feels clinical, harder to read | Prose paragraphs |
| "Caused by" for meals | Ungrounded causal claim | "Associated with," "coincided with" |
| Percentiles or peer comparison | No tool computes them in this build | Compare to the reader's own period statistics |
| Risk scores or projections | No predictive model in this build | Omit entirely |
| Metabolic age, VAT, body composition | No model in this build | Omit entirely |
| Per-meal glucose response, iAUC, PPGR | No meal-CGM linkage in this build | Omit entirely |
| Dietary indices (HEI, DASH, NOVA) | Tool not in this build | Describe logged items factually |
| Citations to specific studies | No literature tool in this build | Cite only the named consensus targets |
| "Dangerous/alarming" | Unnecessarily scary | "Above the typical range" |
| Item count reported as meal count | No tool groups items into meals | "138 food items" |
| Age as top recommendation | Non-modifiable | Mention briefly with caveat |
| Repeating same data | Wastes reader time, inflates length | State once, reference back |
| Technical jargon without definition | Excludes readers | Plain language or define on first use |
| >5 recommendations | Cognitive overload | Prioritize top 3-5 |
| Mechanism speculation | Ungrounded | Stick to observations |
| Time-of-day glucose patterns | No tool computes hourly aggregations | Use per-day data |
| Leading with data quality | Buries the insights | Lead with findings, quality in appendix |
| "Higher/elevated" without anchor | Meaningless without reference | Specify comparison point |

---

## Integration with Other Skills

**Skills that still apply:**
- `clinical-language-guard` — terminology guidelines remain in effect
