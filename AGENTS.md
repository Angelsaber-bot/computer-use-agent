# Repository Guardrails

## Architecture

- Put reusable production logic under `src/computer_agent/`.
- Experiments should only compose production components, prepare fixtures, print results, and collect evidence.
- Production modules must never import from `experiments/`.
- Do not reimplement production ranking, safety, or selection algorithms inside experiment acceptance code. Test public behavior instead.
- Prefer table-driven experiment cases and parametrized pytest tests.
- Split production modules by responsibility, not merely to reduce line count.

## Experiments

- Treat completed experiments as frozen evidence unless a confirmed bug requires a focused correction.
- Avoid unrelated refactoring of historical experiments.
- Aim for experiment scripts around 300 lines.
- If an experiment approaches or exceeds 400 lines, pause and determine whether reusable logic, data definitions, or harness utilities belong in `src/` or a shared test/helper module.
- This is a design warning, not a rule to game through formatting.

## Phase 04 Sequence

1. Reusable perception
2. Deterministic UI grounding
3. Action grounding
4. Verification
5. Recovery and re-grounding
6. Structured planning
7. LLM reasoner
8. Agent loop
9. Dynamic UI
10. Cross-application agent

Experiments 04.01 through 04.05 remain deterministic. Do not introduce an LLM before Experiment 04.07. Code owns coordinates, safety, execution, verification, and retry behavior.

## Workflow

- Work on one experiment at a time.
- Keep dry-run behavior as the default.
- Execution must require an explicit option and only be added in experiments whose scope permits execution.
- Run focused tests and the complete test suite.
- Run `pip check`.
- Check import safety and the final diff when Python code changes.
- Update documentation with completed behavior and evidence.
- Keep each commit focused.
- Do not switch branches, commit, push, merge, or delete files unless the user explicitly asks.
