# free-model-router

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/elangrr/free-model-router/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/elangrr/free-model-router/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that speaks the OpenAI Chat Completions and Anthropic Messages APIs and answers each request with the best free model across a directory of free-tier providers. It reads every provider's catalog in its own format (OpenAI, Anthropic or Gemini), ranks the free models that meet a request's needs by quality, plans the workload around each provider's hidden quota, and fails over when a provider refuses a call.

## How it works

- `lib/brief.ts` reads the provider directory, the quality table and the workload from the brief.
- `lib/catalog.ts` asks each provider for its model list and builds one catalog, which `--list` prints.
- `lib/router.ts` picks a provider and model for each request and re-plans after every answer or refusal.
- `lib/neutral.ts` and `lib/upstream.ts` translate a request between the three formats.
- `lib/answer.ts` reads a provider's answer and writes it back as JSON or server-sent events.

Node 22, standard library only.

## Run it

```sh
MAKEFUN_ENV_URL=http://localhost:8080 MAKEFUN_PROMPT_FILE=prompt.txt node main.ts
```

`node main.ts --list` prints the catalog and exits.

---

<sub>Entered on [make.fun](https://make.fun/?ref=pncsh537&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
