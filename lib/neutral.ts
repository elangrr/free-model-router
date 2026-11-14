// A neutral form of a chat request. A client request arrives in the OpenAI or Anthropic format and the
// chosen provider may speak a different one, so a request that crosses formats is first read into this
// shape, which keeps everything a provider parses: system prompt, messages with text and images, tool
// definitions, tool calls and results, max_tokens, stop sequences, sampling and JSON mode.

import { type Json, isRecord } from './brief.ts';

export type Part =
  | { kind: 'text'; text: string }
  | { kind: 'image'; mediaType: string; data: string | null; url: string | null }
  | { kind: 'tool_call'; id: string; name: string; args: unknown }
  | { kind: 'tool_result'; id: string; content: string };

export interface Turn {
  role: 'user' | 'assistant' | 'tool';
  parts: Part[];
  /** The message's content was one plain string: keep it a string where the target format allows. */
  plain: boolean;
}

export interface ToolDef {
  name: string;
  description: string | null;
  parameters: unknown;
}

export type ToolChoice = 'auto' | 'any' | 'none' | { name: string } | null;

export interface Neutral {
  system: string;
  turns: Turn[];
  tools: ToolDef[];
  toolChoice: ToolChoice;
  maxTokens: number | null;
  stops: string[];
  temperature: number | null;
  topP: number | null;
  jsonMode: boolean;
  jsonSchema: unknown;
}

export const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);

export function stopList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  return Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string') : [];
}

/** Text of a content value: a string, or its text blocks joined by newlines. */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  return parts.join('\n');
}

function parseArgs(text: unknown): unknown {
  if (typeof text !== 'string') return isRecord(text) ? text : {};
  try {
    return JSON.parse(text);
  } catch {
    return {};
  }
}

function imageFromUrl(url: string): Part {
  const data = /^data:([^;,]*)[^,]*;base64,(.*)$/s.exec(url);
  if (data !== null) return { kind: 'image', mediaType: data[1] || 'image/png', data: data[2], url: null };
  return { kind: 'image', mediaType: 'image/png', data: null, url };
}

/** Whether an OpenAI-format body asks for JSON mode. */
export function openAIJsonMode(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

export function fromOpenAI(body: Json): Neutral {
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const m of messages) {
    if (m.role === 'system' || m.role === 'developer') {
      const text = blocksText(m.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (m.role === 'tool') {
      turns.push({ role: 'tool', parts: [{ kind: 'tool_result', id: String(m.tool_call_id ?? ''), content: blocksText(m.content) }], plain: false });
      continue;
    }
    const role = m.role === 'assistant' ? 'assistant' : 'user';
    const parts: Part[] = [];
    if (typeof m.content === 'string') {
      if (m.content !== '' || role === 'user') parts.push({ kind: 'text', text: m.content });
    } else if (Array.isArray(m.content)) {
      for (const block of m.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
        else if (block.type === 'image_url') {
          const url = isRecord(block.image_url) ? block.image_url.url : block.image_url;
          if (typeof url === 'string' && url !== '') parts.push(imageFromUrl(url));
        }
      }
    }
    if (role === 'assistant' && Array.isArray(m.tool_calls)) {
      for (const call of m.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ kind: 'tool_call', id: String(call.id ?? ''), name: String(call.function.name ?? ''), args: parseArgs(call.function.arguments) });
      }
    }
    turns.push({ role, parts, plain: typeof m.content === 'string' });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || !isRecord(tool.function) || typeof tool.function.name !== 'string') continue;
      tools.push({
        name: tool.function.name,
        description: typeof tool.function.description === 'string' ? tool.function.description : null,
        parameters: tool.function.parameters ?? { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice = null;
  if (body.tool_choice === 'auto') toolChoice = 'auto';
  else if (body.tool_choice === 'required') toolChoice = 'any';
  else if (body.tool_choice === 'none') toolChoice = 'none';
  else if (isRecord(body.tool_choice) && isRecord(body.tool_choice.function) && typeof body.tool_choice.function.name === 'string') {
    toolChoice = { name: body.tool_choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  return {
    system: systemParts.join('\n'),
    turns,
    tools,
    toolChoice,
    maxTokens: num(body.max_tokens) ?? num(body.max_completion_tokens),
    stops: stopList(body.stop),
    temperature: num(body.temperature),
    topP: num(body.top_p),
    jsonMode: openAIJsonMode(body),
    jsonSchema: format !== null && isRecord(format.json_schema) ? format.json_schema.schema ?? null : null,
  };
}

export function fromAnthropic(body: Json): Neutral {
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages.filter(isRecord) : [];
  for (const m of messages) {
    const role = m.role === 'assistant' ? 'assistant' : 'user';
    if (typeof m.content === 'string') {
      turns.push({ role, parts: [{ kind: 'text', text: m.content }], plain: true });
      continue;
    }
    const parts: Part[] = [];
    const results: Part[] = [];
    for (const block of Array.isArray(m.content) ? m.content : []) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ kind: 'text', text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        const src = block.source;
        if (src.type === 'base64' && typeof src.data === 'string') {
          parts.push({ kind: 'image', mediaType: typeof src.media_type === 'string' ? src.media_type : 'image/png', data: src.data, url: null });
        } else if (src.type === 'url' && typeof src.url === 'string') {
          parts.push({ kind: 'image', mediaType: 'image/png', data: null, url: src.url });
        }
      } else if (block.type === 'tool_use') {
        parts.push({ kind: 'tool_call', id: String(block.id ?? ''), name: String(block.name ?? ''), args: block.input ?? {} });
      } else if (block.type === 'tool_result') {
        results.push({ kind: 'tool_result', id: String(block.tool_use_id ?? ''), content: blocksText(block.content) });
      }
    }
    // Tool results come first in an Anthropic user message; they are turns of their own elsewhere.
    for (const result of results) turns.push({ role: 'tool', parts: [result], plain: false });
    if (parts.length > 0 || results.length === 0) turns.push({ role, parts, plain: false });
  }
  const tools: ToolDef[] = [];
  if (Array.isArray(body.tools)) {
    for (const tool of body.tools) {
      if (!isRecord(tool) || typeof tool.name !== 'string') continue;
      tools.push({
        name: tool.name,
        description: typeof tool.description === 'string' ? tool.description : null,
        parameters: tool.input_schema ?? { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice = null;
  if (isRecord(body.tool_choice)) {
    const type = body.tool_choice.type;
    if (type === 'auto') toolChoice = 'auto';
    else if (type === 'any') toolChoice = 'any';
    else if (type === 'none') toolChoice = 'none';
    else if (type === 'tool' && typeof body.tool_choice.name === 'string') toolChoice = { name: body.tool_choice.name };
  }
  return {
    system: blocksText(body.system),
    turns,
    tools,
    toolChoice,
    maxTokens: num(body.max_tokens),
    stops: stopList(body.stop_sequences),
    temperature: num(body.temperature),
    topP: num(body.top_p),
    jsonMode: false,
    jsonSchema: null,
  };
}
