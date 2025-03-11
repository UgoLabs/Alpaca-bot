# Model Cleanup Guidance

This document describes the procedures for cleaning up and managing model files to maintain system performance and storage efficiency.

## When to cleanup models

- When disk space is becoming limited
- When older models are no longer being used
- After major algorithm changes that make older models obsolete

## Cleanup procedure

1. Use the cleanup.py script with the --models flag
2. Archive important models before deletion
3. Document which models have been removed

## Important considerations

Never delete models that are currently in use by the trading system.
Always backup models before performing a major cleanup operation.