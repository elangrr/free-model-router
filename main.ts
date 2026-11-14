// free-model-router: one server that answers OpenAI Chat Completions and Anthropic Messages requests
// with the best free model a directory of free-tier providers can offer, and fails over when a provider
// refuses.
//
//   node main.ts          read the brief, probe the providers, register with the env, serve, exit 0 when done
//   node main.ts --list   print the catalog document and exit 0
//
// Everything is derived from the brief, the providers' catalog routes and the answers seen so far: no
// wall clock and no randomness reaches a request.

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { type Json, isRecord, readBrief } from './lib/brief.ts';
import { authHeaders, catalogDoc, dialBase, freeModelIds, probeAll, type ProviderListing } from './lib/catalog.ts';
import { openAIJsonMode } from './lib/neutral.ts';
import { buildUpstream, type ClientApi } from './lib/upstream.ts';
import { anthropicJson, anthropicSse, openAIJson, openAISse, parseAnswer, type Answer, type Meta } from './lib/answer.ts';
import { Router, type Candidate, type Job } from './lib/router.ts';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const brief = readBrief(process.env.MAKEFUN_PROMPT_FILE ?? '');
const trialNow = Date.parse(process.env.MAKEFUN_TRIAL_NOW ?? '');
const CREATED = Number.isFinite(trialNow) ? Math.floor(trialNow / 1000) : 0;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
const UPSTREAM_TIMEOUT_MS = 20000;

let listings: ProviderListing[] = [];
let router: Router | null = null;
/** How many chat requests have arrived: the client sends the manifest in order, one at a time. */
let arrived = 0;

function log(message: string): void {
  process.stderr.write(`free-model-router: ${message}\n`);
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendSse(res: ServerResponse, text: string): void {
  res.writeHead(200, { 'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache', connection: 'keep-alive' });
  res.end(text);
}

function sendError(res: ServerResponse, api: ClientApi, status: number, type: string, message: string): void {
  if (api === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, param: null, code: type } });
}

async function readBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** GET /v1/models, in the format the auth header names: x-api-key is Anthropic, anything else OpenAI. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const free = freeModelIds(listings);
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: free.map((m) => ({ type: 'model', id: m.id, display_name: m.id, created_at: '2026-01-01T00:00:00Z' })),
      has_more: false,
      first_id: free.length > 0 ? free[0].id : null,
      last_id: free.length > 0 ? free[free.length - 1].id : null,
    });
    return;
  }
  sendJson(res, 200, { object: 'list', data: free.map((m) => ({ id: m.id, object: 'model', created: CREATED, owned_by: m.provider })) });
}

/** What a request needs, read off its own body: used beside the manifest's row, and alone when there is none. */
function bodyNeeds(api: ClientApi, body: Json): { tools: boolean; vision: boolean; json: boolean } {
  let vision = false;
  for (const message of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
  }
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: api === 'openai' && openAIJsonMode(body) };
}

interface CallResult {
  answer: Answer | null;
  status: number;
  message: string;
}

/** One provider call, in the provider's own format and auth scheme. */
async function callProvider(api: ClientApi, body: Json, c: Candidate): Promise<CallResult> {
  const call = buildUpstream(api, body, c.provider.wire, c.model);
  const headers: Record<string, string> = { 'content-type': 'application/json', accept: 'application/json', ...authHeaders(c.provider) };
  if (c.provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  try {
    const res = await fetch(`${dialBase(ENV_URL, c.provider)}${call.path}`, {
      method: 'POST',
      headers,
      body: JSON.stringify(call.body),
      signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
    });
    const text = await res.text();
    if (res.status >= 200 && res.status < 300) {
      const answer = parseAnswer(c.provider.wire, text);
      return { answer, status: res.status, message: answer === null ? 'the provider answered no completion' : '' };
    }
    return { answer: null, status: res.status, message: text.slice(0, 2000) };
  } catch (err) {
    return { answer: null, status: 0, message: (err as Error).message };
  }
}

/** POST /v1/chat/completions and POST /v1/messages. */
async function chat(api: ClientApi, req: IncomingMessage, res: ServerResponse): Promise<void> {
  const index = arrived;
  arrived += 1;
  const body = await readBody(req);
  if (body === null) return sendError(res, api, 400, 'invalid_request_error', 'the request body is not one JSON object');
  if (router === null) return sendError(res, api, 503, 'api_error', 'the router is not ready');

  const row = brief.manifest[index];
  const needs = bodyNeeds(api, body);
  const requires = { ...(row?.requires ?? {}) };
  if (needs.tools) requires.tools = true;
  if (needs.vision) requires.vision = true;
  if (needs.json) requires.json = true;
  const weight = row?.weight ?? 1;
  const job: Job = {
    item: { id: row?.id ?? `request-${index}`, api, stream: body.stream === true, weight, requires },
    jsonMode: needs.json,
    noMaxTokens: api === 'openai' && typeof body.max_tokens !== 'number' && typeof body.max_completion_tokens !== 'number',
  };

  const tried = new Set<string>();
  for (;;) {
    const c = router.pick(index, job, tried);
    if (c === null) break;
    tried.add(c.provider.id);
    const result = await callProvider(api, body, c);
    if (result.answer !== null) {
      router.served(index, weight, c);
      const meta: Meta = { id: `${api === 'openai' ? 'chatcmpl' : 'msg'}-${String(index + 1).padStart(4, '0')}`, model: c.model, created: CREATED };
      if (body.stream === true) sendSse(res, api === 'openai' ? openAISse(result.answer, meta) : anthropicSse(result.answer, meta));
      else sendJson(res, 200, api === 'openai' ? openAIJson(result.answer, meta) : anthropicJson(result.answer, meta));
      return;
    }
    log(`${job.item.id}: ${c.provider.id}/${c.model} answered ${result.status}: ${result.message.slice(0, 160)}`);
    // A quota refusal says something about the provider; anything else only rules it out for this request.
    if (result.status === 402 || result.status === 429) router.refused(c, result.message);
  }
  router.unserved(index, weight);
  sendError(res, api, 503, api === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model could answer this request');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = new URL(req.url ?? '/', 'http://localhost').pathname;
  if (req.method === 'GET' && path === '/v1/models') return models(req, res);
  if (req.method === 'POST' && path === '/v1/chat/completions') return chat('openai', req, res);
  if (req.method === 'POST' && path === '/v1/messages') return chat('anthropic', req, res);
  sendError(res, path === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found_error', `no route ${req.method ?? ''} ${path}`);
}

/** Ask the env whether the workload is done, ten times a second; exit 0 when it is. */
function pollWork(): void {
  let busy = false;
  setInterval(() => {
    if (busy) return;
    busy = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        const work: unknown = res.status === 200 ? await res.json() : null;
        if (isRecord(work) && work.done === true) process.exit(0);
      } catch {
        // The env did not answer this time: ask again on the next tick.
      } finally {
        busy = false;
      }
    })();
  }, 100);
}

async function main(): Promise<void> {
  listings = await probeAll(ENV_URL, brief);
  const catalog = catalogDoc(listings);
  if (process.argv.includes('--list')) {
    await new Promise<void>((resolve) => process.stdout.write(`${JSON.stringify(catalog)}\n`, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const ratings = new Map(brief.ratings.filter((r) => r.free).map((r) => [`${r.provider}/${r.model}`, r.rating] as [string, number]));
  router = new Router(listings, ratings, brief.manifest);

  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'api_error', 'the server failed to answer');
      else res.end();
    });
  });
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');

  const reply = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: 1, port: address.port, catalog }),
    signal: AbortSignal.timeout(10000),
  });
  if (!reply.ok) throw new Error(`POST /register answered HTTP ${reply.status}: ${(await reply.text()).slice(0, 300)}`);
  log(`listening on ${address.port}; registered ${catalog.providers.length} providers`);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  pollWork();
}

try {
  await main();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
