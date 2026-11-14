// Building the request a provider receives. Same format in and out: the client's body goes through as it
// is, with only the model named (and a whole answer asked for: this server streams to its client itself).
// A different format: the body is rebuilt from the neutral form.

import { type Json, type Wire, isRecord } from './brief.ts';
import { type Neutral, type Part, fromAnthropic, fromOpenAI } from './neutral.ts';

export type ClientApi = 'openai' | 'anthropic';

export interface UpstreamCall {
  /** Path under the provider's base URL. */
  path: string;
  body: Json;
}

const DEFAULT_ANTHROPIC_MAX_TOKENS = 4096;

function textOf(parts: Part[]): string {
  return parts.filter((p) => p.kind === 'text').map((p) => (p.kind === 'text' ? p.text : '')).join('\n');
}

function toOpenAIBody(n: Neutral, model: string): Json {
  const messages: Json[] = [];
  if (n.system !== '') messages.push({ role: 'system', content: n.system });
  for (const turn of n.turns) {
    if (turn.role === 'tool') {
      for (const p of turn.parts) if (p.kind === 'tool_result') messages.push({ role: 'tool', tool_call_id: p.id, content: p.content });
      continue;
    }
    if (turn.role === 'assistant') {
      const message: Json = { role: 'assistant', content: textOf(turn.parts) || null };
      const calls = turn.parts.filter((p) => p.kind === 'tool_call');
      if (calls.length > 0) {
        message.tool_calls = calls.map((p) => (p.kind === 'tool_call' ? { id: p.id, type: 'function', function: { name: p.name, arguments: JSON.stringify(p.args ?? {}) } } : null));
      }
      messages.push(message);
      continue;
    }
    const hasImage = turn.parts.some((p) => p.kind === 'image');
    if (!hasImage && (turn.plain || turn.parts.length <= 1)) {
      messages.push({ role: 'user', content: textOf(turn.parts) });
      continue;
    }
    const content: Json[] = [];
    for (const p of turn.parts) {
      if (p.kind === 'text') content.push({ type: 'text', text: p.text });
      else if (p.kind === 'image') content.push({ type: 'image_url', image_url: { url: p.data !== null ? `data:${p.mediaType};base64,${p.data}` : p.url ?? '' } });
    }
    messages.push({ role: 'user', content });
  }
  const body: Json = { model, messages };
  if (n.maxTokens !== null) body.max_tokens = n.maxTokens;
  if (n.temperature !== null) body.temperature = n.temperature;
  if (n.topP !== null) body.top_p = n.topP;
  if (n.stops.length > 0) body.stop = n.stops;
  if (n.tools.length > 0) {
    body.tools = n.tools.map((t) => ({ type: 'function', function: { name: t.name, ...(t.description === null ? {} : { description: t.description }), parameters: t.parameters } }));
    if (n.toolChoice === 'auto' || n.toolChoice === 'none') body.tool_choice = n.toolChoice;
    else if (n.toolChoice === 'any') body.tool_choice = 'required';
    else if (n.toolChoice !== null) body.tool_choice = { type: 'function', function: { name: n.toolChoice.name } };
  }
  return body;
}

function toAnthropicBody(n: Neutral, model: string): Json {
  const messages: Array<{ role: string; content: unknown }> = [];
  const push = (role: string, blocks: Json[], plainText: string | null): void => {
    const last = messages[messages.length - 1];
    // Tool results and the user text after them share one user message, as the Messages API expects.
    if (last !== undefined && last.role === role && Array.isArray(last.content)) {
      last.content.push(...blocks);
      return;
    }
    messages.push({ role, content: plainText !== null ? plainText : blocks });
  };
  for (const turn of n.turns) {
    if (turn.role === 'tool') {
      const blocks: Json[] = [];
      for (const p of turn.parts) if (p.kind === 'tool_result') blocks.push({ type: 'tool_result', tool_use_id: p.id, content: p.content });
      push('user', blocks, null);
      continue;
    }
    const blocks: Json[] = [];
    for (const p of turn.parts) {
      if (p.kind === 'text') blocks.push({ type: 'text', text: p.text });
      else if (p.kind === 'image') {
        blocks.push({ type: 'image', source: p.data !== null ? { type: 'base64', media_type: p.mediaType, data: p.data } : { type: 'url', url: p.url ?? '' } });
      } else if (p.kind === 'tool_call') blocks.push({ type: 'tool_use', id: p.id, name: p.name, input: p.args ?? {} });
    }
    const onlyText = turn.parts.every((p) => p.kind === 'text');
    push(turn.role, blocks, turn.plain && onlyText ? textOf(turn.parts) : null);
  }
  const body: Json = { model, max_tokens: n.maxTokens ?? DEFAULT_ANTHROPIC_MAX_TOKENS, messages };
  if (n.system !== '') body.system = n.system;
  if (n.temperature !== null) body.temperature = n.temperature;
  if (n.topP !== null) body.top_p = n.topP;
  if (n.stops.length > 0) body.stop_sequences = n.stops;
  if (n.tools.length > 0) {
    body.tools = n.tools.map((t) => ({ name: t.name, ...(t.description === null ? {} : { description: t.description }), input_schema: t.parameters }));
    if (n.toolChoice === 'auto' || n.toolChoice === 'any' || n.toolChoice === 'none') body.tool_choice = { type: n.toolChoice };
    else if (n.toolChoice !== null) body.tool_choice = { type: 'tool', name: n.toolChoice.name };
  }
  return body;
}

/** A JSON schema without the keywords Gemini's function declarations refuse. */
function geminiSchema(schema: unknown): unknown {
  if (Array.isArray(schema)) return schema.map(geminiSchema);
  if (!isRecord(schema)) return schema;
  const out: Json = {};
  for (const [key, value] of Object.entries(schema)) {
    if (key === 'additionalProperties' || key === '$schema' || key === '$id' || key === 'strict') continue;
    out[key] = geminiSchema(value);
  }
  return out;
}

function toGeminiBody(n: Neutral): Json {
  const names = new Map<string, string>();
  const contents: Json[] = [];
  for (const turn of n.turns) {
    const parts: Json[] = [];
    for (const p of turn.parts) {
      if (p.kind === 'text') parts.push({ text: p.text });
      else if (p.kind === 'image') {
        parts.push(p.data !== null ? { inlineData: { mimeType: p.mediaType, data: p.data } } : { fileData: { mimeType: p.mediaType, fileUri: p.url ?? '' } });
      } else if (p.kind === 'tool_call') {
        names.set(p.id, p.name);
        parts.push({ functionCall: { name: p.name, args: p.args ?? {} } });
      } else {
        parts.push({ functionResponse: { name: names.get(p.id) ?? p.id, response: { content: p.content } } });
      }
    }
    contents.push({ role: turn.role === 'assistant' ? 'model' : 'user', parts });
  }
  const body: Json = { contents };
  if (n.system !== '') body.systemInstruction = { parts: [{ text: n.system }] };
  const config: Json = {};
  if (n.maxTokens !== null) config.maxOutputTokens = n.maxTokens;
  if (n.temperature !== null) config.temperature = n.temperature;
  if (n.topP !== null) config.topP = n.topP;
  if (n.stops.length > 0) config.stopSequences = n.stops;
  if (n.jsonMode) {
    config.responseMimeType = 'application/json';
    if (n.jsonSchema !== null && n.jsonSchema !== undefined) config.responseSchema = geminiSchema(n.jsonSchema);
  }
  if (Object.keys(config).length > 0) body.generationConfig = config;
  if (n.tools.length > 0) {
    body.tools = [{ functionDeclarations: n.tools.map((t) => ({ name: t.name, ...(t.description === null ? {} : { description: t.description }), parameters: geminiSchema(t.parameters) })) }];
    if (n.toolChoice === 'auto' || n.toolChoice === 'any' || n.toolChoice === 'none') {
      body.toolConfig = { functionCallingConfig: { mode: n.toolChoice.toUpperCase() } };
    } else if (n.toolChoice !== null) {
      body.toolConfig = { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [n.toolChoice.name] } };
    }
  }
  return body;
}

/** The client's own body for a provider of the same format: only the model changes, and no stream is asked. */
function passThrough(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  delete out.stream;
  delete out.stream_options;
  return out;
}

export function buildUpstream(api: ClientApi, body: Json, wire: Wire, model: string): UpstreamCall {
  if (wire === 'openai') {
    return { path: '/v1/chat/completions', body: api === 'openai' ? passThrough(body, model) : toOpenAIBody(fromAnthropic(body), model) };
  }
  if (wire === 'anthropic') {
    return { path: '/v1/messages', body: api === 'anthropic' ? passThrough(body, model) : toAnthropicBody(fromOpenAI(body), model) };
  }
  const neutral = api === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  return { path: `/v1beta/models/${encodeURIComponent(model)}:generateContent`, body: toGeminiBody(neutral) };
}
