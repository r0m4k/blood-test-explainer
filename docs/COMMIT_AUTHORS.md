# Commit authorship

## Codex (agent) commits

All agent-generated commits must use:

```text
Author: Codex <chatgpt-codex-connector[bot]@users.noreply.github.com>
```

Example:

```bash
git -c user.name="Codex" \
    -c user.email="chatgpt-codex-connector[bot]@users.noreply.github.com" \
    commit -m "$(cat <<'EOF'
Your message here.

Co-authored-by: Codex <chatgpt-codex-connector[bot]@users.noreply.github.com>
EOF
)"
```

Do **not** use:

- `Cursor <cursoragent@cursor.com>`
- `chatgpt-codex-connector[bot]` as the author name without the `Codex` display name
- `Codex <codex@openai.com>` (legacy; rewritten in history)

Human commits (`ro_d`, `Dimitris`) stay on their own identities. Agent work is attributed to **Codex** only.
