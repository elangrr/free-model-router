// Routing. For every request the router ranks the free models that can serve it by the quality table's
// rating and plans the rest of the workload at once, because a provider's free tier is a shared, hidden
// budget: every call to a provider after its first may be refused. The plan starts from the best-rated
// model for each request, then moves requests off providers that are being leaned on to providers that
// have not been called yet, cheapest quality loss first, while the whole workload stays within a small
// quality allowance. It is recomputed after every answer and every refusal, from what has been seen.

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
const QUALITY_TARGET = 0.992;

function meets(model: ListedModel, requires: Requirement): boolean {
  if (requires.tools === true && model.tools !== true) return false;
  if (requires.vision === true && model.vision !== true) return false;
  if (requires.json === true && model.json !== true) return false;
  if (requires.min_context !== undefined && (model.contextLength === null || model.contextLength < requires.min_context)) return false;
  return true;
}

export class Router {
  #listings: ProviderListing[];
  #ratings: Map<string, number>;
  #manifest: WorkItem[];
  #calls = new Map<string, number>();
  #strikes = new Map<string, number>();
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
      for (const model of listing.models) {
        if (!model.free || !meets(model, job.item.requires)) continue;
        const rating = this.#ratings.get(`${provider.id}/${model.id}`);
        if (rating === undefined) continue;
        out.push({ provider, model: model.id, rating });
      }
    }
    // Array.prototype.sort is stable, so equal ratings keep catalog order.
    return out.sort((a, b) => b.rating - a.rating);
  }

  /**
   * The model for the job at `index`, planned together with every request after it. `exclude` holds the
   * providers that already refused this request. Null when nothing is left to try.
   */
  pick(index: number, job: Job, exclude: Set<string>): Candidate | null {
    const jobs: Job[] = [job];
    for (let i = index + 1; i < this.#manifest.length; i++) {
      const item = this.#manifest[i];
      jobs.push({ item, jsonMode: item.api === 'openai' && item.requires.json === true, noMaxTokens: false });
    }
    const options = jobs.map((j, k) => this.#candidates(j, k === 0 ? exclude : null));
    if (options[0].length === 0) return null;

    // Start from the best-rated model for each request; among equals, the provider leaned on least.
    const planned = new Map<string, number>();
    const usage = (id: string): number => (this.#calls.get(id) ?? 0) + (planned.get(id) ?? 0);
    const chosen: Array<Candidate | null> = [];
    let loss = this.#loss;
    for (let k = 0; k < jobs.length; k++) {
      let pickK: Candidate | null = null;
      for (const c of options[k]) {
        if (pickK === null || (c.rating === pickK.rating && usage(c.provider.id) < usage(pickK.provider.id))) pickK = c;
        if (pickK !== null && c.rating < pickK.rating) break;
      }
      chosen.push(pickK);
      if (pickK === null) continue;
      planned.set(pickK.provider.id, (planned.get(pickK.provider.id) ?? 0) + 1);
      const best = index + k < this.#best.length ? this.#best[index + k] : pickK.rating;
      loss += jobs[k].item.weight * Math.max(0, best - pickK.rating);
    }

    // Move requests from providers called more than once to providers not called yet: a provider's first
    // call is the one least likely to be refused. Cheapest quality loss first, inside the allowance.
    for (;;) {
      let move: { k: number; to: Candidate; cost: number } | null = null;
      for (let k = 0; k < jobs.length; k++) {
        const from = chosen[k];
        if (from === null || usage(from.provider.id) < 2) continue;
        for (const to of options[k]) {
          if (usage(to.provider.id) !== 0) continue;
          const cost = jobs[k].item.weight * (from.rating - to.rating);
          if (loss + cost > this.#allowance) break;
          if (move === null || cost < move.cost) move = { k, to, cost };
          break;
        }
      }
      if (move === null) break;
      const from = chosen[move.k];
      if (from !== null) planned.set(from.provider.id, (planned.get(from.provider.id) ?? 1) - 1);
      planned.set(move.to.provider.id, 1);
      chosen[move.k] = move.to;
      loss += move.cost;
    }
    return chosen[0];
  }

  /** A provider call that answered. */
  served(index: number, weight: number, c: Candidate): void {
    this.#calls.set(c.provider.id, (this.#calls.get(c.provider.id) ?? 0) + 1);
    this.#strikes.set(c.provider.id, 0);
    const best = index < this.#best.length ? this.#best[index] : c.rating;
    this.#loss += weight * Math.max(0, best - c.rating);
  }

  /** A provider call that was refused for quota (402 or 429): decide whether the provider is finished. */
  refused(c: Candidate, message: string): void {
    const id = c.provider.id;
    this.#calls.set(id, (this.#calls.get(id) ?? 0) + 1);
    const strikes = (this.#strikes.get(id) ?? 0) + 1;
    this.#strikes.set(id, strikes);
    const finished = /\bspent\b|exhaust|\bis out\b|outage|no longer/i.test(message);
    const oneCall = /refused this call|try again|retry/i.test(message);
    if (finished || strikes >= (oneCall ? 3 : 2)) this.#dead.add(id);
  }

  /** A request nothing could serve: its whole rating is lost, so no further quality is traded away. */
  unserved(index: number, weight: number): void {
    this.#loss += weight * (index < this.#best.length ? this.#best[index] : 0);
  }
}
