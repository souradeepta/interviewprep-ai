# Structured Output Remediation

Use this page after **Q20: “Make JSON output reliable when schemas evolve.”**
Treat model text as untrusted input until it passes a versioned contract.

## Contract-first design

Define a schema with a version, required fields, types, bounds, and explicit
unknown/null behavior. Keep the wire contract separate from the prompt prose.
For a schema change, accept old versions during a migration window, transform
them in a deterministic adapter, and emit only the current version downstream.
Never silently coerce a missing required field into a plausible default.

## Defense in depth

1. Use provider constrained decoding or a grammar when available.
2. Parse the response strictly; reject trailing prose, duplicate keys, and
   invalid UTF-8 or numeric values.
3. Validate schema, business invariants, authorization scope, and safety policy
   separately. A JSON document can be syntactically valid but unsafe.
4. Retry only with bounded, structured feedback for repairable validation
   errors. Preserve the original response and validation errors in the trace.
5. Fall back to a typed refusal or human review when repair fails. Do not pass
   malformed model output directly to a side-effecting tool.

## Evaluation contract

Track parse success, schema-valid rate, business-invariant pass rate, repair
success, refusal precision, duplicate-key rate, p95 latency, and token cost.
Build a versioned fixture set with missing fields, extra fields, wrong types,
long strings, adversarial values, Unicode, and schema-version mixtures. Test
both valid outputs and safe refusals; a system that returns `{}` for every
request is not reliable.

## Common failure modes

- Treating Markdown fences as a parser instead of validating the payload.
- Retrying identical prompts until a malformed response happens to parse.
- Letting a repair model invent values for required fields.
- Validating syntax but not tenant, permission, range, or side-effect rules.
- Deploying a schema change without replaying historical outputs.

## References

- [Prompting concept](../concepts/12-prompting.md)
- [Evaluation concept](../concepts/32-evaluation.md)
- [LLM coding exercises](llm-coding-exercises.md)
