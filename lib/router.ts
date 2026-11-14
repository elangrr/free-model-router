// Routing. For every request the router ranks the free models that can serve it by the quality table's
// rating, and weighs that against how likely the provider is to refuse the call. A provider's free tier
// is a hidden budget of calls with a few scattered refusals inside it, so the risk of a call depends on
// what the provider has already shown: its first call is the safest, a provider that has refused twice
// has shown its scattered refusals, and a provider deep into its allowance may be about to run dry.
//
// The whole remaining workload is planned at once: each request takes the model with the lowest
// risk + price * quality loss, and the price is the lowest one that keeps the workload's weighted rating
// within a small allowance of the best possible. The plan is recomputed after every answer and every
// refusal, from what has been seen so far, so it is a pure function of the trial's history.

import { type Provider, type Requirement, type WorkItem } from './brief.ts';
import { type ListedModel, type ProviderListing } from './catalog.ts';

export interface Candidate {
  provider: Provider;
  model: string;
  rating: number;
}

export interface Job {
  item: WorkItem;
  /** The request needs a provider format that carries JSON mode. */
  jsonMode: boolean;
  /** The request names no max_tokens, which the Anthropic format cannot leave out. */
  noMaxTokens: boolean;
}

/** The share of the best possible weighted rating the plan keeps. */
// The best possible is out of reach anyway (the top-rated provider's allowance cannot cover the whole
// workload), so a little quality is spent where it buys the most: avoiding calls that would be refused.
const QUALITY_TARGET = 0.982;

/**
 * Typical free-tier call allowances per trial window, lowest to highest, for the providers this router
 * knows. A provider it does not know gets the cautious default.
 */
const ALLOWANCES: Record<string, [number, number]> = {
  groq: [9, 12],
  cerebras: [5, 6],
  gemini: [6, 8],
  zai: [4, 5],
  openrouter: [10, 12],
  sambanova: [4, 5],
  pollinations: [10, 14],
  cloudflare: [7, 8],
  ovh: [5, 6],
  kilol: [12, 16],
};
const DEFAULT_ALLOWANCE: [number, number] = [4, 6];
/** Extra risk on a provider's second call while it has refused nothing yet. */
const SECOND_CALL_RISK = 0.1;

/** Prices on quality loss to try, lowest first: risk per rating point, per unit of request weight. */
const PRICES: number[] = [0];
for (let p = 0.0004; p < 4; p *= 1.4) PRICES.push(p);
PRICES.push(Number.POSITIVE_INFINITY);

function meets(model: ListedModel, requires: Requirement): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (requires.min_context !== undefined && (model.contextLength === null || model.contextLength < requires.min_context)) return false;
  return true;
}

/**
 * How likely a provider is to refuse its next call, after `calls` calls of which `refusals` were
 * single-call refusals. Two parts: a scattered refusal (a tier refuses one or two calls somewhere
 * after its first), and the allowance running out.
 */
function risk(id: string, calls: number, refusals: number): number {
  if (calls === 0) return 0;
  const [low, high] = ALLOWANCES[id] ?? DEFAULT_ALLOWANCE;
  const next = calls + 1;
  // The allowance is at least what has been answered so far; past the typical range it is likely gone.
  const floor = Math.max(low, calls);
  const dry = next <= low ? 0 : floor > high ? 0.9 : Math.min(1, 1 / (high - floor + 1));
  // Scattered refusals: one or two, equally likely, on distinct calls between the second and the one
  // before the allowance ends. `all` such calls exist, `left` of them are still ahead; what has been
  // seen so far shifts the odds between one and two.
  const all = Math.max(1, Math.round((low + high) / 2) - 2);
  const left = all - (calls - 1);
  let scattered = 0;
  if (left > 0 && refusals < 2) {
    const pairs = Math.max(1, (all * (all - 1)) / 2);
    if (refusals === 0) {
      const one = left / all;
      const two = (left * (left - 1)) / 2 / pairs;
      scattered = (one * (1 / left) + two * (2 / left)) / (one + two);
      // The second call is the one a tier most often refuses when nothing has been refused yet.
      if (calls === 1) scattered = SECOND_CALL_RISK + (1 - SECOND_CALL_RISK) * scattered;
    } else {
      const one = (calls - 1) / all;
      const two = ((calls - 1) * left) / pairs;
      scattered = (two / (one + two)) * (1 / left);
    }
  }
  return 1 - (1 - dry) * (1 - Math.min(0.95, scattered));
}

export class Router {
  #listings: ProviderListing[];
  #ratings: Map<string, number>;
  #manifest: WorkItem[];
  #calls = new Map<string, number>();
  /** Refusals in a row, per provider. */
  #strikes = new Map<string, number>();
  /** Single-call refusals seen so far, per provider. */
  #refusals = new Map<string, number>();
  #dead = new Set<string>();
  #best: number[];
  #allowance: number;
  #loss = 0;

  constructor(listings: ProviderListing[], ratings: Map<string, number>, manifest: WorkItem[]) {
    this.#listings = listings;
    this.#ratings = ratings;
    this.#manifest = manifest;
    this.#best = manifest.map((item) => this.#candidates({ item, jsonMode: false, noMaxTokens: false }, null, true).reduce((top, c) => Math.max(top, c.rating), 0));
    const bound = manifest.reduce((sum, item, i) => sum + item.weight * this.#best[i], 0);
    this.#allowance = bound * (1 - QUALITY_TARGET);
  }

  /** Every free model that can serve the job, best rating first, then catalog order. */
  #candidates(job: Job, exclude: Set<string> | null, ignoreState = false): Candidate[] {
    const out: Candidate[] = [];
    for (const listing of this.#listings) {
      const provider = listing.provider;
      if (!ignoreState) {
        if (this.#dead.has(provider.id) || exclude?.has(provider.id) === true) continue;
        if (job.jsonMode && provider.wire === 'anthropic') continue;
        if (job.noMaxTokens && provider.wire === 'anthropic') continue;
      }
      let top: Candidate | null = null;
      for (const model of listing.models) {
        if (!model.free || !meets(model, job.item.requires)) continue;
        const rating = this.#ratings.get(`${provider.id}/${model.id}`);
        if (rating === undefined) continue;
        // One provider is one budget: only its best model for the job matters.
        if (top === null || rating > top.rating) top = { provider, model: model.id, rating };
      }
      if (top !== null) out.push(top);
    }
    // Array.prototype.sort is stable, so equal ratings keep catalog order.
    return out.sort((a, b) => b.rating - a.rating);
  }

  /** Plan every job at one price on quality loss; the first job's model and the plan's total loss. */
  #plan(index: number, jobs: Job[], options: Candidate[][], price: number): { first: Candidate | null; loss: number } {
    const planned = new Map<string, number>();
    let loss = this.#loss;
    let first: Candidate | null = null;
    for (let k = 0; k < jobs.length; k++) {
      const best = index + k < this.#best.length ? this.#best[index + k] : (options[k][0]?.rating ?? 0);
      let pick: Candidate | null = null;
      let pickCost = Number.POSITIVE_INFINITY;
      let pickRisk = Number.POSITIVE_INFINITY;
      for (const c of options[k]) {
        const id = c.provider.id;
        const r = risk(id, (this.#calls.get(id) ?? 0) + (planned.get(id) ?? 0), this.#refusals.get(id) ?? 0);
        const drop = jobs[k].item.weight * Math.max(0, best - c.rating);
        // At an infinite price only the rating counts, and risk breaks ties.
        const cost = price === Number.POSITIVE_INFINITY ? drop : r + price * drop;
        if (cost < pickCost - 1e-12 || (Math.abs(cost - pickCost) <= 1e-12 && r < pickRisk - 1e-12)) {
          pick = c;
          pickCost = cost;
          pickRisk = r;
        }
      }
      if (k === 0) first = pick;
      if (pick === null) {
        loss += jobs[k].item.weight * best;
        continue;
      }
      planned.set(pick.provider.id, (planned.get(pick.provider.id) ?? 0) + 1);
      loss += jobs[k].item.weight * Math.max(0, best - pick.rating);
    }
    return { first, loss };
  }

  /**
   * The model for the job at `index`, planned together with every request after it. `exclude` holds the
   * providers that cannot serve this request. Null when nothing is left to try.
   */
  pick(index: number, job: Job, exclude: Set<string>): Candidate | null {
    const jobs: Job[] = [job];
    for (let i = index + 1; i < this.#manifest.length; i++) {
      const item = this.#manifest[i];
      jobs.push({ item, jsonMode: item.api === 'openai' && item.requires.json === true, noMaxTokens: false });
    }
    const options = jobs.map((j, k) => this.#candidates(j, k === 0 ? exclude : null));
    if (options[0].length === 0) return null;
    let last: Candidate | null = null;
    for (const price of PRICES) {
      const plan = this.#plan(index, jobs, options, price);
      last = plan.first;
      if (plan.loss <= this.#allowance) break;
    }
    return last;
  }

  /** A provider call that answered. */
  served(index: number, weight: number, c: Candidate): void {
    this.#calls.set(c.provider.id, (this.#calls.get(c.provider.id) ?? 0) + 1);
    this.#strikes.set(c.provider.id, 0);
    const best = index < this.#best.length ? this.#best[index] : c.rating;
    this.#loss += weight * Math.max(0, best - c.rating);
  }

  /**
   * A provider call that was refused for quota (402 or 429). A refusal that says the free tier is spent or
   * the provider is out finishes the provider; so do three refusals in a row. A refusal of one call only
   * counts against it: the next plan decides whether the same provider is still the best place to ask.
   */
  refused(c: Candidate, message: string): void {
    const id = c.provider.id;
    this.#calls.set(id, (this.#calls.get(id) ?? 0) + 1);
    const strikes = (this.#strikes.get(id) ?? 0) + 1;
    this.#strikes.set(id, strikes);
    this.#refusals.set(id, (this.#refusals.get(id) ?? 0) + 1);
    if (/\bspent\b|exhaust|\bis out\b|outage|no longer/i.test(message) || strikes >= 3) this.#dead.add(id);
  }

  /** A request nothing could serve: its whole rating is lost, so no further quality is traded away. */
  unserved(index: number, weight: number): void {
    this.#loss += weight * (index < this.#best.length ? this.#best[index] : 0);
  }
}
