# Prompts

These are the three prompts behind the paper's 210-report matrix (14
participants x 3 prompts x 5 conditions, Sections 6.1-6.2), reproduced verbatim.

| File | Matrix prompt | Answerable with four tools |
| --- | --- | --- |
| `metabolic_scorecard.md` | 1 (= paper Prompt A) | partly — the default |
| `meal_grounded_action.md` | 2 (= paper Prompt B) | barely |
| `cardiovascular_risk_scorecard.md` | 3 | no |

Supplementary S5.2 reproduces the first two, because those are also the prompts
of the Section 6.3 case study. The third is named in Section 6.1.

## What to expect

All three ask for things this build cannot ground — peer comparison and
cardiovascular risk in every case, and meal-glucose attribution in the second.
The agent will decline those and say why. That is the FORBIDDEN block and the
grounding contract working, and it is worth seeing once.

`metabolic_scorecard.md` is the default because it asks for the most that four
tools can actually answer: glucose health from the CGM metrics, and two
concrete changes. Its peer-comparison and cardiovascular asks are declined.

`cardiovascular_risk_scorecard.md` will produce almost nothing, since every
grade it asks for needs the SCORE2 tool.
