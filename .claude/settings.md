# Claude Settings Documentation

## Overview

This file documents the Claude Code settings for the ML Automation Framework.

## Hooks

### PostToolUse Hooks

1. **Python Auto-Format**
   - Triggers after any Python file is written or edited
   - Runs `ruff format` on the file
   - Timeout: 30 seconds

### PreToolUse Hooks

1. **Force Push Protection**
   - Blocks `git push --force` and `git commit --force` commands
   - Provides feedback suggesting regular push

## Environment Variables

- `INSIDE_CLAUDE_CODE`: Set to "1" to indicate Claude Code environment
- `BASH_DEFAULT_TIMEOUT_MS`: Default bash command timeout (5 minutes)
- `BASH_MAX_TIMEOUT_MS`: Maximum bash timeout (10 minutes)

## Skills

Located in `.claude/skills/`:
- `ml-pipeline-patterns/SKILL.md` - ML pipeline design patterns
- `databricks-patterns/SKILL.md` - Databricks-specific patterns
- `feature-engineering/SKILL.md` - Feature engineering best practices

## Commands

Located in `.claude/commands/`:
- `/train` - Train a model from config
- `/evaluate` - Evaluate model performance

## Agents

Located in `.claude/agents/`:
- `ml-reviewer.md` - ML code review checklist
