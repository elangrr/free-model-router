// The trial brief: three JSON documents (provider directory, quality table, workload manifest)
// embedded one per line in the prompt file. Everything the router knows up front comes from here.

import { readFileSync } from 'node:fs';

export type Wire = 'openai' | 'anthropic' | 'gemini';
export type Auth = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
export type Json = Record<string, unknown>;

export interface Provider {
  id: string;
  name: string;
  wire: Wire;
  auth: Auth;
  keyEnv: string;
  key: string;
  path: string;
}

export interface Rating {
  model: string;
  provider: string;
  rating: number;
  contextLength: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

export interface WorkItem {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: number;
  requires: Requirement;
}

export interface Brief {
  providers: Provider[];
  ratings: Rating[];
  manifest: WorkItem[];
}

export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

const str = (v: unknown, fallback = ''): string => (typeof v === 'string' ? v : fallback);
const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

function wireOf(v: unknown): Wire {
  return v === 'anthropic' || v === 'gemini' ? v : 'openai';
}

function authOf(v: unknown): Auth {
  return v === 'bearer' || v === 'x-api-key' || v === 'x-goog-api-key' ? v : 'none';
}

/** Every JSON object the text holds on a line of its own, in order. */
function jsonLines(text: string): Json[] {
  const out: Json[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line.startsWith('{') || !line.endsWith('}')) continue;
    try {
      const value: unknown = JSON.parse(line);
      if (isRecord(value)) out.push(value);
    } catch {
      // Not a document line: prose that happens to start with a brace.
    }
  }
  return out;
}

export function parseBrief(text: string): Brief {
  const brief: Brief = { providers: [], ratings: [], manifest: [] };
  for (const doc of jsonLines(text)) {
    const schema = str(doc.schema);
    if (schema.endsWith('/providers/v1') && Array.isArray(doc.providers)) {
      for (const p of doc.providers) {
        if (!isRecord(p) || typeof p.id !== 'string') continue;
        brief.providers.push({
          id: p.id,
          name: str(p.name, p.id),
          wire: wireOf(p.wire),
          auth: authOf(p.auth),
          keyEnv: str(p.key_env),
          key: str(p.key),
          path: str(p.path, `/p/${p.id}`),
        });
      }
    } else if (schema.endsWith('/ratings/v1') && Array.isArray(doc.ratings)) {
      for (const r of doc.ratings) {
        if (!isRecord(r) || typeof r.model !== 'string' || typeof r.provider !== 'string') continue;
        brief.ratings.push({
          model: r.model,
          provider: r.provider,
          rating: num(r.rating_milli) ?? 0,
          contextLength: num(r.context_length),
          tools: r.tools === true,
          vision: r.vision === true,
          json: r.json === true,
          free: r.free !== false,
        });
      }
    } else if (schema.endsWith('/workload/v1') && Array.isArray(doc.manifest)) {
      for (const w of doc.manifest) {
        if (!isRecord(w) || typeof w.id !== 'string') continue;
        const req = isRecord(w.requires) ? w.requires : {};
        const requires: Requirement = {};
        if (req.tools === true) requires.tools = true;
        if (req.vision === true) requires.vision = true;
        if (req.json === true) requires.json = true;
        const ctx = num(req.min_context);
        if (ctx !== null) requires.min_context = ctx;
        brief.manifest.push({
          id: w.id,
          api: w.api === 'anthropic' ? 'anthropic' : 'openai',
          stream: w.stream === true,
          weight: num(w.weight) ?? 1,
          requires,
        });
      }
    }
  }
  return brief;
}

export function readBrief(file: string): Brief {
  if (file === '') return { providers: [], ratings: [], manifest: [] };
  try {
    return parseBrief(readFileSync(file, 'utf8'));
  } catch {
    return { providers: [], ratings: [], manifest: [] };
  }
}
