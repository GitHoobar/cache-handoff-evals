# Does a model handoff change an executable code edit?

This protocol was written for a prospective experiment, before generating predictions on the coding fixtures. All inputs are original, fictional Python repositories. The study asks whether a receiver that has the same task, tools and visible repository makes a different executable edit after cache translation. Separate measurements on SQuAD motivated the question.

## Task and comparison

Each fixture contains two valid named constants. The task requires changing the value of one constant, while preserving a second behavior. The exposed tool fixes the file and replacement value. The only argument the model generates is the name of the constant to change. Each answer is one greedy free generation. There is no retry or post-generation repair.

For each repository, run all six ordered sender-to-receiver pairs among Qwen 3 14B, Ministral 3 14B and Gemma 3 12B. For each direction, compare two answers from the same receiver. The direct-reading arm prefills the repository natively. The other arm uses the released BF16 translator to construct the receiver cache. The task suffix and target document tokens must agree across arms. All families, counterbalanced variants, directions and parser failures remain in the totals.

The current-only condition presents the authoritative repository. A matched stress condition places a clearly labelled superseded repository snapshot before the same authoritative repository. That snapshot points the request to the other editable symbol. The model must use the current snapshot. Both conditions retain the same requested edit, behavior oracle and tool. Any loss specific to the stress condition may reflect the presence of the archive, its position, or the resulting change in prompt length. This two-condition design cannot separate these factors.

## Executable outcome

The primary outcome is a completed action that passes the requested behavior check and preserves the control behavior. The experiment dispatcher accepts the complete stripped answer only if it parses as exactly one JSON object with one string field named `symbol`. The value must match an editable constant in the specific repository, with case preserved. A duplicate key, unknown argument, code fence, trailing prose or malformed output makes no edit. The runner never executes model-generated Python or shell text. It edits the allowlisted name in the authored fixture, then executes the fixture's two behavior checks. Empty output or an exact plain-text abstention is retained as a no-action outcome.

Before scoring outputs, the evaluator runs each fixture unchanged and with each of its two legal edits. The unchanged fixture must fail the requested check and preserve the control. The intended edit must pass both. The alternative edit must fail the requested check and break the preserved behavior. A discrepancy stops scoring for diagnosis. These controls establish that a valid argument can have an observable unintended effect and that the requested edit is achievable.

Record raw generations and stop reasons, valid actions, requested behavior and preservation checks, gains, losses, and valid edits that introduce a regression. Separately report attempts that would become valid by removing a single matching `json` or unlabelled Markdown fence. The recorded tool leaves such attempts undispatched. This distinguishes response formatting from semantic edit selection.

## Denominators and interpretation

Each panel has 12 scenario families, two paired repository variants per family and six ordered directions, or 144 paired native/translated decisions. Report the two panels separately, by direction and by family. Report the subset where both endpoints passed the task natively, using the matching native reads as witnesses. A receiver's native prediction runs against the same inputs for both senders, so verify their identity for every case and report the number of distinct native decisions alongside route-weighted summaries. Measure how the native-to-translated pass-rate difference changes between the two panels using matched base fixtures and directions. These reused families and directions are dependent observations. Report paired pass-rate differences, resampling whole families for a descriptive interval. A degenerate bootstrap interval when all differences are zero supplies no bound on error rates for new tasks.

The exercise is one tool decision on small controlled repositories. The tests establish the effect of the selected symbol inside these fixtures. Live routers, autonomous tool invocation, multi-step repository changes, production failure frequency and model internals are outside this design. Freeze later dataset or method extensions separately, and retain the original panel.
