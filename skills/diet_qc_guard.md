---
name: diet-qc-guard
version: 1.2
description: Ensures diet data quality validation and appropriate reporting of QC findings
injection_points:
  - orchestrator
  - analyzer
tools:
  - qc_diet_data
  - meal_parser
---

# Diet QC Guard

## Purpose

Ensure diet data quality is validated when available and that quality concerns are transparently reported to users.

## Part 1: Tool Selection (Orchestrator)

### REQUIRED

When diet data is available (indicated by `diet.csv` file or meal data):
- **ALWAYS include `qc_diet_data` tool** in your tool selection
- Run QC validation **after** `meal_parser` and **before** using data for analysis
- QC validation is a standard quality assurance step, not optional

### Workflow

```
diet.csv available → meal_parser → qc_diet_data → [analysis tools]
```

## Part 2: Reporting QC Findings (Analyzer)

**🚨 CRITICAL: Section Creation 🚨**

**YOU must create the "Meal Data Quality" subsection** regardless of qc_diet_data tool status:
- If qc_diet_data ran successfully (status="ok"): Report detailed percentages from tool output
- If qc_diet_data failed or wasn't called: State "Meal data quality validation unavailable"
- Section must ALWAYS exist when diet data is present

### Determining QC Results

Check tool outputs for `qc_diet_data`:
- `severity_counts`: Distribution of validation results `{pass: X, check: Y, fail: Z}`
- `failed_groups`: Most common validation failures
- `total_rows`: Number of meal entries validated

**Calculation:**
```
fail_rate = fail / (pass + check + fail)
check_rate = check / (pass + check + fail)
```

### REQUIRED (Always Do)

**When QC was performed (regardless of results):**
- Create dedicated "Meal Data Quality" subsection (typically in Overview or Data Quality section)
- Report specific percentages from qc_diet_data tool output **ONCE in detail**
- Present severity_counts: pass/check/fail rates with actual numbers
- Example: "Pass: 123 entries (97.6%), Check: 0 (0.0%), Fail: 3 (2.4%)"
- Brief mention is sufficient if all data passed

**When fail_rate > 10%:**
- Mention data quality issues in the dedicated QC subsection
- Use factual, non-alarmist language

**When check_rate > 30%:**
- Note data completeness concerns
- Explain potential impact on analysis reliability when rate exceeds 50%

**Section Structure (Report detailed numbers ONCE):**
```
### Data Quality Validation

**Meal Data Quality:**
Quality validation was performed on all N meal entries.
Pass: X entries (X.X%), Check: Y (Y.Y%), Fail: Z (Z.Z%)
[Brief interpretation if thresholds exceeded]
```

### Contextual References (Avoid Repeating Numbers)

**In other sections** (e.g., meal analysis, recommendations):
- Reference QC findings **without repeating exact percentages**
- Use qualitative language: "given data quality limitations", "as noted in data quality assessment"
- Example: "*(diet data collected with quality limitations)*" NOT "*(97.6% pass, 0.0% check, 2.4% fail)*"

**FORBIDDEN:**
- Repeating the same QC percentages in multiple sections
- Creating multiple "Data Quality Note" blocks with identical numbers
- Duplicating severity_counts across report

## Part 3: Interpreting QC Groups

When elaborating on QC findings, provide context based on the specific validation group and severity:

When `failed_groups` contains specific validation groups, provide context:

| QC Group | Severity | Interpret as... | Report as... |
|----------|----------|-----------------|--------------|
| Range Checks | fail | Unrealistic portion sizes (weight_g outside 0-2000g) or exceeds reasonable daily intake (>5000 kcal/day) | "Some entries flagged for implausible values (e.g., portion sizes or daily totals)" |
| Integrity | check | Incomplete logging day or sporadic participation | "Gaps in temporal coverage suggest inconsistent daily logging" |
| Completeness | check | Missing optional nutrient fields | "Missing optional nutritional details that may limit analysis depth" |
| Consistency | fail | Reported energy disagrees with the macronutrient breakdown, or a duplicate entry | "Some entries have internally inconsistent nutrition figures" |

**When elaborating:**
- Use the user-friendly interpretation, not technical validation rule names
- Connect findings to actionable improvements (e.g., "Consider using detailed meal tracking tools")
- Avoid technical jargon like "weight_g bounds" or "range checks" or "macronutrient calculation inconsistencies"

### FORBIDDEN (Never Do)

- Omitting acknowledgment that QC was performed (even if all passed)
- Ignoring QC results when thresholds exceeded
- Using alarmist language ("dangerous", "critical", "severe issues")
- Claiming analysis is "invalid" (use "reliability" language instead)
- Hiding quality issues from the user

## Language Guidelines

## Examples

### BAD (not acknowledging QC was performed)

> **Diet Analysis**
> 
> Your diet log contains 126 entries across 14 days, logged in a consistent
> three-meal pattern.

*(QC was run and all passed, but this wasn't mentioned - user doesn't know data was validated)*

### GOOD (acknowledging QC even when all passed)

> **Diet Analysis**
> 
> All 126 meal entries passed quality validation. Your diet log spans 14 days
> in a consistent three-meal pattern.

### BAD (ignoring QC results when thresholds exceeded)

> **Diet Analysis**
> 
> Your diet log contains 126 entries across 14 days.

*(When QC showed 15% fail rate - quality issues not mentioned)*

### GOOD (transparent reporting with issues)

> **Diet Analysis**
> 
> *Data Quality Note*: Validation identified quality concerns in 15% of meal entries (primarily incomplete nutritional data). Analysis below reflects remaining 85% validated entries.
>
> Your validated entries span all 14 logging days.

### BAD (alarmist language)

> **Critical Data Quality Issues**
> 
> Your diet logging has severe problems. 40% of entries are incomplete. This data cannot be trusted.

### GOOD (factual, helpful)

> **Data Completeness**
> 
> Quality validation flagged 40% of entries for missing optional nutrient fields. Core metrics (calories, timestamps) are complete. For enhanced analysis accuracy, consider using meal tracking tools with detailed nutritional databases.

## Application

Apply to all reports when:
- `qc_diet_data` was executed (ALWAYS acknowledge this)
- Severity counts show fail_rate > 10% OR check_rate > 30% (detailed reporting required)
- User has provided diet/meal data

Reporting levels:
- **QC performed, all passed**: Brief acknowledgment (e.g., "All N entries passed validation")
- **Thresholds exceeded**: Detailed reporting per guidelines above

Does NOT require:
- Creating separate "QC Report" section (integrate into relevant sections)

