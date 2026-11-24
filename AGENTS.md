# Repository instructions

## Commit discipline

- Make small, focused commits: one logical change per commit.
- Keep code and documentation in separate commits. Do not mix implementation,
  tests, or build/configuration changes with documentation changes.
- Code-only commits may include directly related tests and build/configuration
  changes needed for that implementation.
- Docs-only commits include plans, results reports, README changes, and repository
  instructions such as this file.
- Split independently reviewable changes into separate commits; avoid unrelated
  cleanup or broad formatting changes.
- Stage intended paths or hunks explicitly. Review the staged diff and run
  `git diff --cached --check` before committing.
- Run relevant tests for code changes and report any failures or unavailable
  validation. Do not claim unrun tests passed.
- Do not include generated benchmark artifacts unless explicitly requested.

## Car configuration

- Put configurable real-car controller values in `config/car_config.yaml`. Keep
  ROS parameter declarations and generic defaults in code, but use this YAML file
  for car-specific settings and tuning rather than adding launch arguments for
  each setting.

## Abstraction Levels and Separation of Concerns

### Principle

Every function operates at one level of abstraction. A reader should understand a function by reading its calls in order, without dropping into low-level mechanics or climbing back up to system-level policy. Execution boundaries (compilation units, transactions, process or device boundaries) are separate from architectural boundaries: where helpers run inside the same execution unit, function boundaries are free, so use them for readability instead of flattening code.

### Rules

1. **Orchestrators orchestrate.** A top-level function is a sequence of named domain steps. It contains no low-level data manipulation (construction, indexing, slicing, unpacking).
2. **Helpers own details.** Each helper has one nameable responsibility and hides the mechanics of it.
3. **One level per function.** Every statement in a body sits at the same conceptual level. Interleaving high-level steps with low-level operations is a violation.
4. **Name by intent.** Names say what a step accomplishes in the domain, not how it is implemented.
5. **Don't extract trivia.** No wrappers that hide a single trivial call or forward arguments under an identical signature. If the concern belongs to an existing component, give it to that component.
6. **Ownership follows the consumer.** State consumed by a component is advanced by that component and returned with its outputs, not managed by the orchestrator.
7. **Separate host and compute concerns.** Timing, I/O, transfers, event handling, and publication stay in the host layer; pure computation stays in the compute layer.

### Review Checklist

- [ ] Can the top-level function be read as a list of domain steps?
- [ ] Does each helper have a single, nameable responsibility?
- [ ] Does every statement in a function sit at the same abstraction level?
- [ ] Is low-level manipulation hidden inside helpers rather than the orchestrator?
- [ ] Is state advanced by the component that consumes it?
- [ ] Is the execution boundary only at the entry point?
- [ ] Is every helper more than a pass-through or single-call wrapper?
- [ ] Are host and compute responsibilities cleanly separated?


