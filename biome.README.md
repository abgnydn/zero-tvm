# Why the biome FORMATTER is off here

`biome.json` disables the formatter and the pre-commit hook runs `biome lint`
with no `--write`. That is deliberate, and it should stay that way.

This tree is hand-aligned. Trailing comments line up in columns, uniform blocks
sit at matched indentation, and shader constants are grouped so a reader can
scan them. That alignment is not incidental in a codebase whose stated point is
that the whole forward pass is readable end-to-end in one sitting.

Biome's formatter collapses it. Run once with `--write` on 2026-08-18, it
rewrote 1,256 lines of `src/compiler/constraints.ts` into tabs and semicolons —
against the style of every file around it — inside a commit that was supposed
to be three documentation corrections. The diff buried the change that mattered.

The linter is a different matter and stays on: it catches real defects and
touches nothing.

## Why it is not in the pre-commit hook either

`biome lint src/ scripts/` currently reports 20 findings — 11 `useTemplate`,
3 `useNodejsImportProtocol`, 3 unused imports, 2 unused bindings, 1
`useHtmlLang`. All pre-existing, none a bug.

A blocking hook over staged files would hand those to whoever next edits one of
those files, forcing them to clean warnings they did not create in order to
commit the change they did. So the linter runs on demand, not in the hook. The
pre-push gates (typecheck, unit tests, and the regression-test rule) are the
ones that block, and they are clean.
