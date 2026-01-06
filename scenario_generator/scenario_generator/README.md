# Automated Scenario Generation Pipeline

## Overview

This pipeline automatically generates diverse, valid driving scenarios for multi-agent coordination testing. It uses an LLM-based approach with:

1. **Creative Generation**: Produces varied scenario descriptions within pipeline capabilities
2. **Constraint Validation**: Ensures scenarios are expressible in the IR format
3. **Generation Loop**: Attempts generation with automatic retry on failure
4. **Diversity Tracking**: Prevents duplicate or overly similar scenarios

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCENARIO META-GENERATOR                       │
├─────────────────────────────────────────────────────────────────┤
│  1. Category Selection (weighted by coverage gaps)              │
│  2. Difficulty Sampling (1-5)                                   │
│  3. Constraint Template Selection                               │
│  4. Creative Variation Injection                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 STRUCTURED PROMPT BUILDER                        │
├─────────────────────────────────────────────────────────────────┤
│  - Pipeline capability constraints                              │
│  - Topology requirements                                        │
│  - Actor type/count limits                                      │
│  - Spatial arrangement vocabulary                               │
│  - Timing vocabulary (spawn-relative)                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM SCENARIO WRITER                           │
├─────────────────────────────────────────────────────────────────┤
│  Generates natural language scenario description                │
│  that is expressive yet parseable                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CONSTRAINT VALIDATOR                             │
├─────────────────────────────────────────────────────────────────┤
│  Stage A: Structural validation (topology, actor types)         │
│  Stage B: Semantic validation (no impossible arrangements)      │
│  Stage C: Parse validation (heuristic extraction checks)        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
               ┌───────────┴───────────┐
               │                       │
          [PASS]                  [FAIL + feedback]
               │                       │
               ▼                       ▼
┌─────────────────────┐    ┌─────────────────────────┐
│  Emit to batch      │    │  Retry with repairs     │
│  (scenarios.json)   │    │  (up to N attempts)     │
└─────────────────────┘    └─────────────────────────┘
```

## Usage

```bash
# Generate 50 new scenarios across all categories
python scenario_builder_api/generate_scenarios.py --count 50 --output-dir new_scenarios

# Generate scenarios for specific categories only
python scenario_builder_api/generate_scenarios.py --categories "Highway On-Ramp Merge" --count 20

# Validate existing scenarios.json
python scenario_builder_api/scenario_generator/validator.py --input scenarios.json

# Generate with specific difficulty range
python scenario_builder_api/generate_scenarios.py --difficulties 3 4 5 --count 30
```

## Supported Categories

See `capabilities.py` for the full mapping of categories to pipeline features.
