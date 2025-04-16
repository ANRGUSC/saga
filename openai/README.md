## Command-Line Interface: `scripts/query_api.py`

This script runs a scheduling experiment between two algorithms with optional verbosity and graph visualization.

### Usage (From root directory)

```bash
python openai/scripts/query_api.py <algorithm_1> <algorithm_2> [-p PROMPT_LEVEL] [-v]
```

### Positional Arguments

| Argument       | Type | Description                                                                 |
|----------------|------|-----------------------------------------------------------------------------|
| `algorithm_1`  | int  | ID of the first scheduling algorithm to be tested                           |
| `algorithm_2`  | int  | ID of the second (baseline) scheduling algorithm for comparison              |

### Available Scheduling Algorithms

- `1`: **BILScheduler**
- `2`: **CpopScheduler**
- `3`: **DuplexScheduler**
- `4`: **ETFScheduler**
- `5`: **FastestNodeScheduler**
- `6`: **FCPScheduler**
- `7`: **FLBScheduler**
- `8`: **GDLScheduler**
- `9`: **HeftScheduler**
- `10`: **MaxMinScheduler**
- `11`: **MCTScheduler**
- `12`: **METScheduler**
- `13`: **MinMinScheduler**
- `14`: **OLBScheduler**
- `15`: **WBAScheduler**
- `16`: **SDBATSScheduler**

---

### Optional Flags

| Flag     | Type    | Default | Description                                                                 |
|----------|---------|---------|-----------------------------------------------------------------------------|
| `-p`     | int     | `0`     | Prompt level (verbosity of explanation, e.g., 0 = minimal)                  |
| `-v`     | boolean | `False` | Enable visualization of the generated task and network graphs               |

> **Note:** `-v` is a flag — if present, visualization is enabled; if omitted, it is disabled.

### Available Prompt Level

- `0`: **Basic description**
- `1`: **Basic + task description**
- `2`: **Basic + task + algorithm description**
- `3`: **Basic + task + algorithm + code description**
- `4`: **Basic + task + algorithm + code + instance description**


---

### Examples

Run with default prompt level and no visualization:
```bash
python openai/scripts/query_api.py 1 2
```

Run with prompt level 2:
```bash
python openai/scripts/query_api.py 1 2 -p 2
```

Run with visualization enabled:
```bash
python openai/scripts/query_api.py 1 2 -v
```

Run with both prompt level and visualization:
```bash
python openai/scripts/query_api.py 1 2 -p 3 -v
```

