# Contributing

Thanks for considering a contribution.

## Reporting an issue

Open a GitHub issue. If it's about a specific number or claim, please cite the exact
file and JSON key (or `.tex` line) you're looking at: every quantitative claim in the
paper traces to a committed artifact, and pointing at the mismatch directly is the
fastest way to get it looked at.

## Proposing a change

1. Fork the repo and create a branch off `main`.
2. If you're changing a result, regenerate it from the training/eval script rather
   than hand-editing the committed JSON, and include both the script's stdout and the
   new JSON in your PR.
3. If you're changing a figure, regenerate it with the corresponding script in
   `figures/` rather than editing the PNG directly.
4. Keep commits scoped: one logical change per commit, with a message that says what
   changed and why.
5. Open a PR against `main` describing what you changed and, if applicable, which
   claim in the paper it affects.

## Code style

- Python: no strict formatter enforced, but match the surrounding file's style.
- No GPU-only code should silently fall back to CPU; training scripts assert
  `torch.cuda.is_available()` and are meant to fail loudly rather than produce
  results from a different code path than what's documented.

## What especially helps

See the README's "Contributing" section for the specific open problems this project
would benefit most from help on.
