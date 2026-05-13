---
description: End-of-session wrap-up — commit changes, update architecture docs, notify Slack
---
End-of-session wrap-up for the current development branch. Run all three steps in order.

## Step 1: Git Commit

1. Run `git status` and `git diff --stat` to see all changes (staged + unstaged + untracked).
2. Run `git log --oneline -5` to match the repo's commit-message style.
3. Review the changes and draft a concise commit message summarizing what was done in this session. Focus on the "why" not the "what."
4. Stage all relevant changed files (be specific — do not blindly `git add -A`; skip `.env`, credentials, `__pycache__/`, `.tmp/`, and other ephemeral files).
5. Commit with the message. Append `Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>`.
6. Confirm the commit succeeded via `git log --oneline -1`.
7. Push the current branch to GitHub: `git push origin HEAD`. If the branch has no upstream yet, use `git push -u origin HEAD`.

Save the list of changed files and the commit message — you will need them in Steps 2 and 3.

---

## Step 2: Update Architecture Docs

1. Identify the architecture-docs folder. 
2. For each doc, compare its content against the code changes from this session. Ask:
   - Does anything in this doc contradict what the code now does?
   - Is there new behavior, a new component, or a changed data flow that this doc should mention?
   - Are there removed features or deprecated paths that should be cleaned up?
3. Edit docs in-place to reflect the current state. Insert new sections where needed. Do not rewrite docs wholesale — make surgical updates that keep the existing structure.
4. If no docs need updating, say so and skip to Step 3.
5. If docs were updated, stage and commit them separately with a message like: `docs: update architecture docs to reflect [brief description]`.
6. Push again: `git push origin HEAD`.

---

## Step 3: Notify Slack

Compose a concise bullet-point summary of this session's changes. Format:

```
*[Branch name] — Session wrap-up*

• [Change 1 — what and why, one line]
• [Change 2]
• ...

_Docs updated:_ [list of doc files touched, or "none"]
```

Send this to the `#bridgr` Slack channel using `mcp__plugin_slack_slack__slack_send_message`.

---

## Done

Confirm to the user: commit hash, docs updated (if any), and Slack notification sent.
