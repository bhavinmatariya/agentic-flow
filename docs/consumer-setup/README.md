# Consumer repo setup

Use this folder as a template when adding **agentic-flow** to any GitHub repository (any account or org).

## What to copy into your app repo

| Copy from here | Paste into your repo |
|----------------|----------------------|
| `agentic-flow.yml` | `.github/workflows/agentic-flow.yml` |
| `repos.json.example` → rename to `repos.json` | `repos.json` (repo root, **optional**) |

Example layout:

```text
your-app-repo/
├── repos.json                    ← optional (cross-repo read context)
├── README.md
└── .github/
    └── workflows/
        └── agentic-flow.yml      ← required
```

## GitHub secrets

Open **Settings → Secrets and variables → Actions** in your app repo and add:

| Secret | Required | Purpose |
|--------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes | Claude API key for investigate / implement / review |
| `AGENT_GITHUB_TOKEN` | Yes | GitHub token with `repo` scope to read/write this repo (and linked repos if used) |

### Which token to use for `AGENT_GITHUB_TOKEN`

- **Same repo, public or private (simple case):** create a fine-grained or classic PAT with **Contents**, **Issues**, and **Pull requests** write access for this repository. You can also use `${{ secrets.GITHUB_TOKEN }}` in the workflow instead of a PAT if you do not need cross-repo access.
- **Private linked repos in another org/account:** use a PAT that can **read** those repos (and **write** to the app repo where the issue lives).

## Optional: linked repositories (`repos.json`)

Put `repos.json` at the **root** of your app repo. Format:

```json
{
  "linked": [
    {
      "name": "backend",
      "repo": "your-org/your-backend-repo"
    }
  ]
}
```

- **Primary repo** (where fixes and PRs are created) is always the repo running the workflow — you do not list it in `repos.json`.
- **Linked repos** are for **read/search context** during investigation (clone, list files, search). The agent does **not** commit code into linked repos today.
- **Note:** Reading `repos.json` from the consumer repo root requires a small agentic-flow update (path wiring). Until that is on `@main`, linked repos are only loaded from the agentic-flow action checkout. Skip `repos.json` if you only use a single repo.

## How the workflow runs

1. **Open an issue** → workflow runs with `event: issue_opened` → agent investigates and posts fix options.
2. **Comment on the issue** (e.g. `approve approach 1`) → workflow runs with `event: issue_comment` → agent implements, reviews, opens a PR.
3. If the run stalls → label `agent:needs-human` → comment **`continue`** or **`retry`** to resume.

Bot comments containing `agentic-flow:auto` are ignored to prevent loops.

## Permissions

The workflow needs:

```yaml
permissions:
  contents: write
  issues: write
  pull-requests: write
```

## Quick checklist

- [ ] Copy `agentic-flow.yml` → `.github/workflows/agentic-flow.yml`
- [ ] Add secret `ANTHROPIC_API_KEY`
- [ ] Add secret `AGENT_GITHUB_TOKEN` (or use `GITHUB_TOKEN` in the YAML if same-repo only)
- [ ] (Optional) Add `repos.json` at repo root for cross-repo context
- [ ] Open a test issue and approve an approach in a comment

## Action reference

This workflow calls:

```yaml
uses: bhavinmatariya/agentic-flow@main
```

Point `uses:` at your own fork if you host a private copy of agentic-flow.
