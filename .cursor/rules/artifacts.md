When a task involves running or modifying a simulation, save before/after
screenshots to artifacts/ and commit them on the PR branch. Embed them in
the PR description with absolute raw GitHub URLs so they render on GitHub
(relative paths like `artifacts/before.png` do not work in PR bodies):

![before](https://raw.githubusercontent.com/<owner>/<repo>/<branch>/artifacts/before.png)

Do not rely on chat-only artifacts. Prefer ready-for-review PRs (not draft)
once the work is done unless the user asks for a draft.
