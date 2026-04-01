/**
 * OpenAI-compatible /v1/chat/completions endpoint.
 *
 * Accepts OpenAI chat-completions format, converts to Anthropic /v1/messages,
 * forwards through ClaudeRequest, and converts the response back to OpenAI format.
 *
 * Reference: fuergaosi233/claude-code-proxy (Python, conversion/ directory)
 */

const { randomUUID } = require('crypto');
const ClaudeRequest = require('./ClaudeRequest');
const Logger = require('./Logger');

// ─── Request conversion: OpenAI → Claude ────────────────────────────────────

function convertOpenAIToClaude(body) {
  const claudeBody = {
    model: body.model || 'claude-sonnet-4-6-20250514',
    max_tokens: body.max_tokens ?? body.max_completion_tokens ?? 8192,
    stream: !!body.stream,
  };

  // ── messages ──
  let systemParts = [];
  const claudeMessages = [];

  for (const msg of body.messages || []) {
    if (msg.role === 'system') {
      // Collect system messages
      const text = typeof msg.content === 'string'
        ? msg.content
        : (msg.content || []).filter(b => b.type === 'text').map(b => b.text).join('\n');
      if (text) systemParts.push(text);

    } else if (msg.role === 'user') {
      claudeMessages.push({
        role: 'user',
        content: convertUserContent(msg.content),
      });

    } else if (msg.role === 'assistant') {
      const content = [];
      if (msg.content) {
        content.push({ type: 'text', text: msg.content });
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          const fn = tc.function || {};
          let args = {};
          try { args = JSON.parse(fn.arguments || '{}'); } catch {}
          content.push({
            type: 'tool_use',
            id: tc.id || `toolu_${randomUUID().replace(/-/g, '').slice(0, 24)}`,
            name: fn.name || '',
            input: args,
          });
        }
      }
      claudeMessages.push({
        role: 'assistant',
        content: content.length ? content : (msg.content ?? ''),
      });

    } else if (msg.role === 'tool') {
      // OpenAI tool results → Claude tool_result content block
      // Must be inside a user message; merge with previous if also tool_result
      const toolResult = {
        type: 'tool_result',
        tool_use_id: msg.tool_call_id || '',
        content: msg.content || '',
      };

      const prev = claudeMessages[claudeMessages.length - 1];
      if (prev && prev.role === 'user' && Array.isArray(prev.content)
          && prev.content.every(b => b.type === 'tool_result')) {
        prev.content.push(toolResult);
      } else {
        claudeMessages.push({ role: 'user', content: [toolResult] });
      }
    }
  }

  if (systemParts.length) {
    claudeBody.system = [{ type: 'text', text: systemParts.join('\n\n') }];
  }
  claudeBody.messages = claudeMessages;

  // ── tools ──
  if (body.tools && body.tools.length) {
    claudeBody.tools = body.tools
      .filter(t => t.type === 'function' && t.function)
      .map(t => ({
        name: t.function.name,
        description: t.function.description || '',
        input_schema: t.function.parameters || { type: 'object', properties: {} },
      }));
  }

  // ── tool_choice ──
  if (body.tool_choice) {
    if (body.tool_choice === 'auto') {
      claudeBody.tool_choice = { type: 'auto' };
    } else if (body.tool_choice === 'none') {
      // Claude doesn't have "none", just omit tool_choice
    } else if (body.tool_choice === 'required') {
      claudeBody.tool_choice = { type: 'any' };
    } else if (typeof body.tool_choice === 'object' && body.tool_choice.function) {
      claudeBody.tool_choice = { type: 'tool', name: body.tool_choice.function.name };
    }
  }

  // ── optional params ──
  if (body.temperature != null) claudeBody.temperature = body.temperature;
  if (body.top_p != null) claudeBody.top_p = body.top_p;
  if (body.stop) claudeBody.stop_sequences = Array.isArray(body.stop) ? body.stop : [body.stop];

  return claudeBody;
}

function convertUserContent(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return String(content ?? '');

  const parts = [];
  for (const block of content) {
    if (block.type === 'text') {
      parts.push({ type: 'text', text: block.text });
    } else if (block.type === 'image_url' && block.image_url?.url) {
      const url = block.image_url.url;
      const m = url.match(/^data:([^;]+);base64,(.+)$/);
      if (m) {
        parts.push({ type: 'image', source: { type: 'base64', media_type: m[1], data: m[2] } });
      }
    }
  }
  if (parts.length === 1 && parts[0].type === 'text') return parts[0].text;
  return parts;
}

// ─── Response conversion: Claude → OpenAI ───────────────────────────────────

function convertClaudeResponseToOpenAI(claudeData, requestModel) {
  const textParts = [];
  const toolCalls = [];

  for (const block of claudeData.content || []) {
    if (block.type === 'text') {
      textParts.push(block.text);
    } else if (block.type === 'tool_use') {
      toolCalls.push({
        id: block.id,
        type: 'function',
        function: {
          name: block.name,
          arguments: JSON.stringify(block.input || {}),
        },
      });
    }
  }

  const stopMap = { end_turn: 'stop', max_tokens: 'length', tool_use: 'tool_calls' };

  const message = { role: 'assistant', content: textParts.join('') || null };
  if (toolCalls.length) message.tool_calls = toolCalls;

  return {
    id: claudeData.id || `chatcmpl-${randomUUID()}`,
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: requestModel,
    choices: [{
      index: 0,
      message,
      finish_reason: stopMap[claudeData.stop_reason] || 'stop',
    }],
    usage: {
      prompt_tokens: claudeData.usage?.input_tokens || 0,
      completion_tokens: claudeData.usage?.output_tokens || 0,
      total_tokens: (claudeData.usage?.input_tokens || 0) + (claudeData.usage?.output_tokens || 0),
    },
  };
}

// ─── Streaming: Claude SSE → OpenAI SSE ─────────────────────────────────────

function createStreamingConverter(res, requestModel) {
  const chatId = `chatcmpl-${randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);
  let buffer = '';
  const toolCalls = {}; // index -> { id, name, args_buffer }

  function sendChunk(delta, finishReason) {
    const chunk = {
      id: chatId,
      object: 'chat.completion.chunk',
      created,
      model: requestModel,
      choices: [{ index: 0, delta, finish_reason: finishReason || null }],
    };
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
  }

  // Send initial role chunk
  sendChunk({ role: 'assistant', content: '' }, null);

  return {
    processChunk(data) {
      buffer += data;
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (!payload || payload === '[DONE]') continue;

        let event;
        try { event = JSON.parse(payload); } catch { continue; }

        switch (event.type) {
          case 'content_block_delta': {
            const delta = event.delta;
            if (delta?.type === 'text_delta' && delta.text) {
              sendChunk({ content: delta.text }, null);
            } else if (delta?.type === 'input_json_delta' && delta.partial_json != null) {
              // Tool call argument streaming
              const idx = event.index - 1; // offset by text block at 0
              if (idx >= 0 && toolCalls[idx]) {
                sendChunk({
                  tool_calls: [{
                    index: idx,
                    function: { arguments: delta.partial_json },
                  }],
                }, null);
              }
            }
            break;
          }

          case 'content_block_start': {
            const block = event.content_block;
            if (block?.type === 'tool_use') {
              const idx = event.index - 1; // text block is 0
              toolCalls[idx] = { id: block.id, name: block.name };
              sendChunk({
                tool_calls: [{
                  index: idx,
                  id: block.id,
                  type: 'function',
                  function: { name: block.name, arguments: '' },
                }],
              }, null);
            }
            break;
          }

          case 'message_delta': {
            const stopReason = event.delta?.stop_reason;
            const stopMap = { end_turn: 'stop', max_tokens: 'length', tool_use: 'tool_calls' };
            sendChunk({}, stopMap[stopReason] || 'stop');

            // Send usage chunk if available
            if (event.usage) {
              const usageChunk = {
                id: chatId,
                object: 'chat.completion.chunk',
                created,
                model: requestModel,
                choices: [],
                usage: {
                  prompt_tokens: event.usage.input_tokens || 0,
                  completion_tokens: event.usage.output_tokens || 0,
                  total_tokens: (event.usage.input_tokens || 0) + (event.usage.output_tokens || 0),
                },
              };
              res.write(`data: ${JSON.stringify(usageChunk)}\n\n`);
            }
            break;
          }

          case 'message_stop':
            break;

          case 'error':
            Logger.error('Claude stream error:', JSON.stringify(event));
            break;
        }
      }
    },

    finish() {
      // Flush any remaining buffer
      if (buffer.trim()) this.processChunk('\n');
      res.write('data: [DONE]\n\n');
      res.end();
    },
  };
}

// ─── Main handler ───────────────────────────────────────────────────────────

async function handleChatCompletions(req, res, body) {
  const requestModel = body.model || 'claude-sonnet-4-6-20250514';
  const isStream = !!body.stream;

  // Convert OpenAI → Claude
  const claudeBody = convertOpenAIToClaude(body);
  Logger.debug('Converted OpenAI→Claude request:', JSON.stringify(claudeBody, null, 2));

  const claudeReq = new ClaudeRequest(req);
  const t0 = Date.now();
  Logger.info(`[openai-compat] → model=${requestModel} stream=${isStream} messages=${(body.messages||[]).length} max_tokens=${claudeBody.max_tokens}`);

  try {
    const claudeResponse = await claudeReq.makeRequest(claudeBody);

    // Handle 401 retry
    if (claudeResponse.statusCode === 401) {
      Logger.info('Got 401 on OpenAI compat, retrying with refreshed token');
      ClaudeRequest.cachedToken = null;
      await claudeReq.loadOrRefreshToken().then(t => { ClaudeRequest.cachedToken = t; });
      const retryResponse = await claudeReq.makeRequest(claudeBody);
      return processClaudeResponse(retryResponse, res, requestModel, isStream);
    }

    return processClaudeResponse(claudeResponse, res, requestModel, isStream);
  } catch (err) {
    Logger.error(`[openai-compat] ← ERROR in ${Date.now() - t0}ms: ${err.message}`);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({
      error: { message: err.message, type: 'server_error', code: 500 },
    }));
  }
}

function processClaudeResponse(claudeResponse, res, requestModel, isStream) {
  if (claudeResponse.statusCode !== 200) {
    // Pass through error
    let errorBody = '';
    claudeResponse.on('data', chunk => { errorBody += chunk; });
    claudeResponse.on('end', () => {
      Logger.error(`[openai-compat] ← Claude ${claudeResponse.statusCode}: ${errorBody.slice(0, 500)}`);
      res.writeHead(claudeResponse.statusCode, { 'Content-Type': 'application/json' });
      try {
        const parsed = JSON.parse(errorBody);
        res.end(JSON.stringify({
          error: { message: parsed.error?.message || errorBody, type: 'api_error', code: claudeResponse.statusCode },
        }));
      } catch {
        res.end(JSON.stringify({
          error: { message: errorBody, type: 'api_error', code: claudeResponse.statusCode },
        }));
      }
    });
    return;
  }

  const contentType = claudeResponse.headers['content-type'] || '';

  if (isStream && contentType.includes('text/event-stream')) {
    // Streaming: Claude SSE → OpenAI SSE
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });

    const converter = createStreamingConverter(res, requestModel);

    claudeResponse.on('data', chunk => {
      converter.processChunk(chunk.toString());
    });

    claudeResponse.on('end', () => {
      converter.finish();
    });

    claudeResponse.on('error', err => {
      Logger.error('Claude stream error:', err.message);
      converter.finish();
    });

    res.on('close', () => {
      if (!claudeResponse.destroyed) claudeResponse.destroy();
    });

  } else {
    // Non-streaming: collect full response and convert
    let responseData = '';
    claudeResponse.on('data', chunk => { responseData += chunk; });
    claudeResponse.on('end', () => {
      try {
        const claudeData = JSON.parse(responseData);
        const openaiResponse = convertClaudeResponseToOpenAI(claudeData, requestModel);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(openaiResponse));
      } catch (err) {
        Logger.error('Failed to convert Claude response:', err.message);
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          error: { message: 'Failed to convert response', type: 'server_error' },
        }));
      }
    });
  }
}

// ─── Models endpoint ────────────────────────────────────────────────────────

function handleModels(req, res) {
  const models = [
    'claude-sonnet-4-20250514',
    'claude-opus-4-20250514',
    'claude-haiku-4-5-20251001',
    'claude-sonnet-3-7-20250219',
  ];
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    object: 'list',
    data: models.map(id => ({
      id,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'anthropic',
    })),
  }));
}

module.exports = { handleChatCompletions, handleModels, convertOpenAIToClaude, convertClaudeResponseToOpenAI };
