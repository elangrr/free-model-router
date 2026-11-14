// The catalog: each provider's own model listing, read in its own wire, merged with the brief's
// quality table. The listing is the truth for which models exist and which are free; the table adds
// ratings. The register body and `--list` print the same object, so it is built exactly once.

import { type Brief, type Json, type Provider, type Rating, isRecord } from './brief.ts';

export interface ListedModel {
  id: string;
  free: boolean;
  contextLength: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

export interface ProviderListing {
  provider: Provider;
  ok: boolean;
  models: ListedModel[];
}

export interface CatalogDoc {
  providers: Array<{
    id: string;
    base_url: string;
    key_env: string;
    wire: string;
    models: Array<{ id: string; free: boolean; context_length: number | null }>;
  }>;
}

/** The headers that carry a provider's key, in its own auth scheme. */
export function authHeaders(provider: Provider): Record<string, string> {
  if (provider.auth === 'bearer') return { authorization: `Bearer ${provider.key}` };
  if (provider.auth === 'x-api-key') return { 'x-api-key': provider.key };
  if (provider.auth === 'x-goog-api-key') return { 'x-goog-api-key': provider.key };
  return {};
}

/** Where to dial a provider: the env origin joined to the provider's own path. */
export function dialBase(envUrl: string, provider: Provider): string {
  if (/^https?:\/\//.test(provider.path)) return provider.path.replace(/\/+$/, '');
  return `${envUrl}${provider.path.startsWith('/') ? '' : '/'}${provider.path}`.replace(/\/+$/, '');
}

const boolOrNull = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
const numOrNull = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function ratingRow(ratings: Rating[], provider: string, model: string): Rating | null {
  return ratings.find((r) => r.provider === provider && r.model === model) ?? null;
}

/** A listing item's model id, with Gemini's `models/` prefix taken off. */
function listedId(item: Json): string | null {
  if (typeof item.id === 'string' && item.id !== '') return item.id;
  if (typeof item.name === 'string' && item.name !== '') return item.name.replace(/^models\//, '');
  return null;
}

function parseListing(provider: Provider, body: unknown, ratings: Rating[]): ListedModel[] {
  if (!isRecord(body)) return [];
  const items = Array.isArray(body.data) ? body.data : Array.isArray(body.models) ? body.models : [];
  const out: ListedModel[] = [];
  for (const item of items) {
    if (!isRecord(item)) continue;
    const id = listedId(item);
    if (id === null) continue;
    const row = ratingRow(ratings, provider.id, id);
    // A listing that states the free flag is the truth; otherwise only a model the free table rates is free.
    const free = typeof item.free === 'boolean' ? item.free : row !== null && row.free;
    out.push({
      id,
      free,
      contextLength: numOrNull(item.context_length) ?? numOrNull(item.inputTokenLimit) ?? numOrNull(item.context_window) ?? row?.contextLength ?? null,
      tools: boolOrNull(item.tools) ?? (row === null ? null : row.tools),
      vision: boolOrNull(item.vision) ?? (row === null ? null : row.vision),
      json: boolOrNull(item.json) ?? (row === null ? null : row.json),
    });
  }
  return out;
}

/** GET one provider's catalog route in its own wire. Catalog calls spend no budget. */
export async function probe(envUrl: string, provider: Provider, ratings: Rating[]): Promise<ProviderListing> {
  const base = dialBase(envUrl, provider);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(url, { headers, signal: AbortSignal.timeout(5000) });
      const text = await res.text();
      if (res.status >= 200 && res.status < 300) {
        return { provider, ok: true, models: parseListing(provider, JSON.parse(text), ratings) };
      }
    } catch {
      // Unreachable or not JSON: try again, then fall back to the brief's table below.
    }
  }
  // No listing: report what the quality table knows, so the catalog still names the provider.
  const models: ListedModel[] = ratings
    .filter((r) => r.provider === provider.id)
    .map((r) => ({ id: r.model, free: r.free, contextLength: r.contextLength, tools: r.tools, vision: r.vision, json: r.json }));
  return { provider, ok: false, models };
}

export async function probeAll(envUrl: string, brief: Brief): Promise<ProviderListing[]> {
  if (envUrl === '') return [];
  return Promise.all(brief.providers.map((p) => probe(envUrl, p, brief.ratings)));
}

export function catalogDoc(listings: ProviderListing[]): CatalogDoc {
  return {
    providers: listings.map((l) => ({
      id: l.provider.id,
      base_url: l.provider.path,
      key_env: l.provider.keyEnv,
      wire: l.provider.wire,
      models: l.models.map((m) => ({ id: m.id, free: m.free, context_length: m.contextLength })),
    })),
  };
}

/** The free model ids across every provider, each once, in catalog order: what GET /v1/models lists. */
export function freeModelIds(listings: ProviderListing[]): Array<{ id: string; provider: string }> {
  const seen = new Set<string>();
  const out: Array<{ id: string; provider: string }> = [];
  for (const l of listings) {
    for (const m of l.models) {
      if (!m.free || seen.has(m.id)) continue;
      seen.add(m.id);
      out.push({ id: m.id, provider: l.provider.id });
    }
  }
  return out;
}
