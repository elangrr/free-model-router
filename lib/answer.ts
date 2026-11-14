// A provider's answer, read out of its own format into one shape, and written back in the format the
// client used: one JSON body, or server-sent events when the client asked for a stream. The text is kept
// whole, the usage counts are the provider's own, and the stop reason is mapped between vocabularies.

import { type Json, type Wire, isRecord } from './brief.ts';

export type Stop = 'end' | 'length' | 'tool' | 'filter';

export interface ToolCall {
  id: string;
  name: string;
  /** The arguments as JSON text. */
  args: string;
}

export interface Answer {
  text: string;
  calls: ToolCall[];
  stop: Stop;
  input: number;
  output: number;
}

export interface Meta {
  /** A deterministic id for this response. */
  id: string;
  model: string;
  created: number;
}

const count = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);

function parseOpenAI(body: Json): Answer | null {
  const choice = Array.isArray(body.choices) && isRecord(body.choices[0]) ? body.choices[0] : null;
  if (choice === null || !isRecord(choice.message)) return null;
  const message = choice.message;
  let text = '';
  if (typeof message.content === 'string') text = message.content;
  else if (Array.isArray(message.content)) {
    for (const block of message.content) if (isRecord(block) && typeof block.text === 'string') text += block.text;
  }
  const calls: ToolCall[] = [];
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      const args = call.function.arguments;
      calls.push({ id: String(call.id ?? `call_${calls.length}`), name: String(call.function.name ?? ''), args: typeof args === 'string' ? args : JSON.stringify(args ?? {}) });
    }
  }
  const reason = choice.finish_reason;
  const stop: Stop = reason === 'length' ? 'length' : reason === 'tool_calls' || reason === 'function_call' ? 'tool' : reason === 'content_filter' ? 'filter' : 'end';
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, calls, stop, input: count(usage.prompt_tokens), output: count(usage.completion_tokens) };
}

function parseAnthropic(body: Json): Answer | null {
  if (!Array.isArray(body.content)) return null;
  let text = '';
  const calls: ToolCall[] = [];
  for (const block of body.content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') calls.push({ id: String(block.id ?? `toolu_${calls.length}`), name: String(block.name ?? ''), args: JSON.stringify(block.input ?? {}) });
  }
  const reason = body.stop_reason;
  const stop: Stop = reason === 'max_tokens' ? 'length' : reason === 'tool_use' ? 'tool' : reason === 'refusal' ? 'filter' : 'end';
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, calls, stop, input: count(usage.input_tokens), output: count(usage.output_tokens) };
}

function parseGemini(body: Json): Answer | null {
  const candidate = Array.isArray(body.candidates) && isRecord(body.candidates[0]) ? body.candidates[0] : null;
  if (candidate === null) return null;
  let text = '';
  const calls: ToolCall[] = [];
  const parts = isRecord(candidate.content) && Array.isArray(candidate.content.parts) ? candidate.content.parts : [];
  for (const part of parts) {
    if (!isRecord(part)) continue;
    if (typeof part.text === 'string') text += part.text;
    else if (isRecord(part.functionCall)) {
      calls.push({ id: `call_${calls.length}`, name: String(part.functionCall.name ?? ''), args: JSON.stringify(part.functionCall.args ?? {}) });
    }
  }
  const reason = candidate.finishReason;
  const stop: Stop = calls.length > 0 ? 'tool' : reason === 'MAX_TOKENS' ? 'length' : reason === 'SAFETY' || reason === 'RECITATION' || reason === 'PROHIBITED_CONTENT' ? 'filter' : 'end';
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, calls, stop, input: count(usage.promptTokenCount), output: count(usage.candidatesTokenCount) };
}

/** Read a provider's 2xx body; null when it is not an answer of its format. */
export function parseAnswer(wire: Wire, text: string): Answer | null {
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return null;
  }
  if (!isRecord(body)) return null;
  if (wire === 'openai') return parseOpenAI(body);
  if (wire === 'anthropic') return parseAnthropic(body);
  return parseGemini(body);
}

const OPENAI_STOP: Record<Stop, string> = { end: 'stop', length: 'length', tool: 'tool_calls', filter: 'content_filter' };
const ANTHROPIC_STOP: Record<Stop, string> = { end: 'end_turn', length: 'max_tokens', tool: 'tool_use', filter: 'refusal' };

/** The text in pieces of whole code points, for streaming. */
function pieces(text: string, size = 40): string[] {
  const points = Array.from(text);
  const out: string[] = [];
  for (let at = 0; at < points.length; at += size) out.push(points.slice(at, at + size).join(''));
  return out;
}

function parsedArgs(args: string): unknown {
  try {
    return JSON.parse(args);
  } catch {
    return {};
  }
}

export function openAIJson(a: Answer, meta: Meta): Json {
  const message: Json = { role: 'assistant', content: a.text === '' && a.calls.length > 0 ? null : a.text };
  if (a.calls.length > 0) message.tool_calls = a.calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } }));
  return {
    id: meta.id,
    object: 'chat.completion',
    created: meta.created,
    model: meta.model,
    choices: [{ index: 0, message, finish_reason: OPENAI_STOP[a.stop], logprobs: null }],
    usage: { prompt_tokens: a.input, completion_tokens: a.output, total_tokens: a.input + a.output },
  };
}

export function openAISse(a: Answer, meta: Meta): string {
  const chunk = (delta: Json, finish: string | null): string =>
    `data: ${JSON.stringify({ id: meta.id, object: 'chat.completion.chunk', created: meta.created, model: meta.model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
  const frames: string[] = [chunk({ role: 'assistant', content: '' }, null)];
  for (const piece of pieces(a.text)) frames.push(chunk({ content: piece }, null));
  a.calls.forEach((c, index) => {
    frames.push(chunk({ tool_calls: [{ index, id: c.id, type: 'function', function: { name: c.name, arguments: c.args } }] }, null));
  });
  frames.push(chunk({}, OPENAI_STOP[a.stop]));
  const usage = { prompt_tokens: a.input, completion_tokens: a.output, total_tokens: a.input + a.output };
  frames.push(`data: ${JSON.stringify({ id: meta.id, object: 'chat.completion.chunk', created: meta.created, model: meta.model, choices: [], usage })}\n\n`);
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

export function anthropicJson(a: Answer, meta: Meta): Json {
  const content: Json[] = [];
  if (a.text !== '' || a.calls.length === 0) content.push({ type: 'text', text: a.text });
  for (const c of a.calls) content.push({ type: 'tool_use', id: c.id, name: c.name, input: parsedArgs(c.args) });
  return {
    id: meta.id,
    type: 'message',
    role: 'assistant',
    model: meta.model,
    content,
    stop_reason: ANTHROPIC_STOP[a.stop],
    stop_sequence: null,
    usage: { input_tokens: a.input, output_tokens: a.output },
  };
}

export function anthropicSse(a: Answer, meta: Meta): string {
  const frame = (event: string, data: Json): string => `event: ${event}\ndata: ${JSON.stringify({ type: event, ...data })}\n\n`;
  const frames: string[] = [
    frame('message_start', {
      message: { id: meta.id, type: 'message', role: 'assistant', model: meta.model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: a.input, output_tokens: 0 } },
    }),
  ];
  let index = 0;
  if (a.text !== '' || a.calls.length === 0) {
    frames.push(frame('content_block_start', { index, content_block: { type: 'text', text: '' } }));
    for (const piece of pieces(a.text)) frames.push(frame('content_block_delta', { index, delta: { type: 'text_delta', text: piece } }));
    frames.push(frame('content_block_stop', { index }));
    index += 1;
  }
  for (const c of a.calls) {
    frames.push(frame('content_block_start', { index, content_block: { type: 'tool_use', id: c.id, name: c.name, input: {} } }));
    frames.push(frame('content_block_delta', { index, delta: { type: 'input_json_delta', partial_json: c.args } }));
    frames.push(frame('content_block_stop', { index }));
    index += 1;
  }
  frames.push(frame('message_delta', { delta: { stop_reason: ANTHROPIC_STOP[a.stop], stop_sequence: null }, usage: { output_tokens: a.output } }));
  frames.push(frame('message_stop', {}));
  return frames.join('');
}
