#!/usr/bin/env bash
# Create an atomic sprint branch off the integration branch.
#
# Usage:   scripts/new-pr.sh <type> <slug>
# Example: scripts/new-pr.sh feat debate-graph
#
# Types follow Conventional Commits: feat|fix|chore|docs|ci|refactor|test|perf
set -euo pipefail

type="${1:?type required (feat|fix|chore|docs|ci|refactor|test|perf)}"
slug="${2:?slug required (kebab-case)}"
base="${BASE_BRANCH:-feat/handbook-tutor}"

git fetch origin "$base"
git switch -c "${type}/${slug}" "origin/${base}"

cat <<EOF
On branch ${type}/${slug} (base: ${base}).
Next:
  git add -A && git commit -m "${type}: <message>"
  git push -u origin ${type}/${slug}
  gh pr create --base ${base} --fill        # if gh is installed
EOF
