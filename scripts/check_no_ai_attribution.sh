#!/bin/bash
# commit-msg hook: reject commits whose message mentions Claude/AI attribution.
msg_file="$1"

if grep -qiE 'claude|anthropic|co-authored-by.*\b(claude|gpt|copilot|gemini|ai)\b|generated with.*ai|🤖' "$msg_file"; then
    echo "Commit message contains AI attribution -- remove it before committing." >&2
    grep -niE 'claude|anthropic|co-authored-by.*\b(claude|gpt|copilot|gemini|ai)\b|generated with.*ai|🤖' "$msg_file" >&2
    exit 1
fi
