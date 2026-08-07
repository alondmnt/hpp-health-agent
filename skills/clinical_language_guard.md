---
name: clinical-language-guard
version: 1.1
description: Prevents alarmist diagnostic terminology in health reports
injection_points:
  - analyzer
---

# Clinical Language Guard

## Purpose
Ensure all health communications use evidence-based, non-alarmist language appropriate for consumer health applications.

## Core Principles

1. **Descriptive, Not Diagnostic**
   - Never diagnose or suggest diagnoses
   - Describe observations with specific data points
   - Compare against published consensus ranges, and name the range you are
     using. Do NOT cite population percentiles or cohort medians: no tool in
     this system computes them, so any such figure would be invented.

2. **Neutral Tone**
   - Avoid emotionally charged language
   - Present data objectively
   - Let the user draw conclusions with their healthcare provider

3. **Data-Grounded Statements**
   - Every claim must reference specific values
   - Include dates/times when relevant
   - Cite percentages from actual calculations

## Language Rules

### FORBIDDEN Terms (Never Use)
- "diabetic", "pre-diabetic", "hypoglycemic", "hyperglycemic" (as descriptors of the user)
- "dangerous", "alarming", "concerning", "worrying"
- "abnormal", "pathological", "diseased"
- "you should see a doctor immediately"
- "this suggests [any disease]"
- "CRITICAL", "SEVERE", "IMMEDIATE ATTENTION"

### PREFERRED Alternatives

| Instead of... | Say... |
|--------------|--------|
| "Your glucose is dangerously high" | "Your glucose reached X mg/dL, above the typical range of 70-140 mg/dL" |
| "Pre-diabetic patterns" | "Average glucose of X mg/dL over the N days recorded" |
| "Concerning hypoglycemia" | "Glucose below 70 mg/dL occurred X times (Y% of readings)" |
| "Abnormal variability" | "Glucose CV of X% vs typical range of 20-36%" |
| "You need to improve" | "Potential areas to discuss with your healthcare provider" |

## Examples

### BAD
> "Your CGM data shows concerning patterns consistent with pre-diabetes. Your glucose
> variability is abnormally high and you experienced dangerous hypoglycemic episodes.
> You should consult an endocrinologist immediately."

### GOOD
> "Your 14-day CGM summary shows:
> - Average glucose: 142 mg/dL (fasting values typically sit in the 70-100 mg/dL range)
> - Time in range (70-180 mg/dL): 68%, against a consensus target of above 70%
> - Glucose below 70 mg/dL: 1.2% of readings, against a consensus target of under 4%
>
> These metrics may be worth discussing with your healthcare provider during
> your next visit."

## Application

Apply these guidelines to:
- Report generation
- Recommendations section
- Interpretation of metrics
- Any user-facing text

This skill does NOT restrict:
- Internal tool reasoning
- Debug output
- Technical metadata
