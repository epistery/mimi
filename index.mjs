import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { existsSync, mkdirSync, readFileSync, unlinkSync } from 'fs';
import { execFile } from 'child_process';
import { randomBytes } from 'crypto';
import Anthropic from '@anthropic-ai/sdk';
import { Config } from 'epistery';
import { createSTTProvider } from './stt.mjs';
import { installWhisper, uninstallWhisper, checkWhisperInstall } from './install-whisper.mjs';
import { homedir } from 'os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Mimi Agent - Voice portal for epistery host
 *
 * Provides a browser-based voice interface that converts speech to text,
 * sends it to Claude with access to epistery MCP tools, and speaks the
 * response back.
 */

// Claude model the admin can select. All support tool use, web_search and streaming.
const DEFAULT_MODEL = 'claude-sonnet-4-6';
const AVAILABLE_MODELS = [
  { id: 'claude-opus-4-8', label: 'Claude Opus 4.8 — most capable' },
  { id: 'claude-opus-4-7', label: 'Claude Opus 4.7' },
  { id: 'claude-sonnet-4-6', label: 'Claude Sonnet 4.6 — balanced (default)' },
  { id: 'claude-haiku-4-5', label: 'Claude Haiku 4.5 — fastest' }
];

// Media types Claude can read natively as content blocks. A files_read result
// for these comes back as base64; handed to Claude as a `document`/`image`
// block it's read directly (PDF text + page images, image pixels) instead of
// arriving as a useless wall of base64 inside a tool_result string.
const DOC_MIME = new Set(['application/pdf']);
const IMG_MIME = new Set(['image/png', 'image/jpeg', 'image/gif', 'image/webp']);
// PDF input caps at 32MB/100 pages on the API; files_read caps inline reads at
// 25MB. Stay comfortably under and skip anything larger (Claude gets a note).
const ATTACH_MAX_BYTES = 20 * 1024 * 1024;

/**
 * If a tool returned inline base64 file content in a media type Claude reads
 * natively (PDF, image), build the matching content block. Returns null
 * otherwise (caller falls back to the JSON tool_result).
 */
function fileAttachmentBlock(result) {
  if (!result || result.encoding !== 'base64' || typeof result.content !== 'string') return null;
  if (typeof result.size === 'number' && result.size > ATTACH_MAX_BYTES) return null;
  const mt = result.mimetype || 'application/octet-stream';
  if (DOC_MIME.has(mt)) {
    return { type: 'document', source: { type: 'base64', media_type: mt, data: result.content } };
  }
  if (IMG_MIME.has(mt)) {
    return { type: 'image', source: { type: 'base64', media_type: mt, data: result.content } };
  }
  return null;
}

/**
 * Turn a tool call's result into what we feed back to Claude. Normally a single
 * tool_result carrying the JSON result. When the tool returned a PDF or image,
 * the raw base64 is meaningless as text, so the tool_result is reduced to a
 * compact reference (name/type/size) and the bytes ride along as a separate
 * `attachment` content block the caller appends as a user turn.
 */
function packToolResult(toolUse, result) {
  const attachment = fileAttachmentBlock(result);
  if (!attachment) {
    return {
      toolResult: { type: 'tool_result', tool_use_id: toolUse.id, content: JSON.stringify(result) },
      attachment: null
    };
  }
  const ref = {
    id: result.id, name: result.name, mimetype: result.mimetype, size: result.size,
    note: 'Content attached as a document/image in the following message — read it directly.'
  };
  return {
    toolResult: { type: 'tool_result', tool_use_id: toolUse.id, content: JSON.stringify(ref) },
    attachment
  };
}

export default class MimiAgent {
  constructor(config = {}) {
    this.config = config;
    this.conversations = new Map();   // sessionId -> message history
    this.anthropic = null;
    this.sttProvider = null;
    this.internalPort = null;
    this.audioDir = null;
    this.whisperInstalling = false;
    this.whisperProgress = [];
  }

  /**
   * Initialize Anthropic client from epistery Config
   * Looks for anthropic_api_key in the domain config
   */
  getAnthropicClient(domain) {
    if (this.anthropic) return this.anthropic;

    const cfg = new Config();
    cfg.setPath(domain);

    // Look for API key in domain config
    const apiKey = cfg.data?.anthropic_api_key
      || cfg.data?.claude?.anthropicKey
      || process.env.ANTHROPIC_API_KEY;

    if (!apiKey) {
      throw new Error('No Anthropic API key configured. Set anthropic_api_key in domain config.');
    }

    this.anthropic = new Anthropic({ apiKey });
    return this.anthropic;
  }

  /**
   * Resolve the configured Claude model for a domain.
   * Read fresh each turn so an admin change takes effect without a restart.
   * Falls back to DEFAULT_MODEL if unset or no longer offered.
   */
  getModel(domain) {
    const cfg = new Config();
    cfg.setPath(domain);
    const model = cfg.data?.claude?.model;
    return AVAILABLE_MODELS.some(m => m.id === model) ? model : DEFAULT_MODEL;
  }

  /**
   * Detect the internal port epistery-host is listening on
   */
  getInternalPort() {
    if (this.internalPort) return this.internalPort;
    this.internalPort = parseInt(process.env.PORT || 4080);
    return this.internalPort;
  }

  /**
   * Lazy-init STT provider from domain config
   */
  getSTTProvider(domain) {
    if (this.sttProvider) return this.sttProvider;
    this.sttProvider = createSTTProvider(domain);
    return this.sttProvider;
  }

  /**
   * Create and return temp dir for TTS audio files
   */
  getAudioDir() {
    if (this.audioDir) return this.audioDir;
    this.audioDir = '/tmp/mimi-audio';
    if (!existsSync(this.audioDir)) {
      mkdirSync(this.audioDir, { recursive: true });
    }
    return this.audioDir;
  }

  /**
   * Get configured TTS voice for a domain
   */
  getTTSVoice(domain) {
    if (this._ttsVoice) return this._ttsVoice;
    const cfg = new Config();
    cfg.setPath(domain);
    this._ttsVoice = cfg.data?.tts?.voice || null;
    return this._ttsVoice;
  }

  /**
   * Generate TTS audio via espeak-ng, returns audio ID
   * Auto-cleanup after 5 minutes
   */
  generateTTS(text, domain) {
    return new Promise((resolve, reject) => {
      const id = randomBytes(12).toString('hex');
      const dir = this.getAudioDir();
      const filePath = path.join(dir, `${id}.wav`);

      // Strip markdown formatting for cleaner speech
      const clean = text
        .replace(/[*_~`#>\[\]]/g, '')
        .replace(/\n+/g, '. ')
        .substring(0, 2000);

      const args = ['-w', filePath];
      const voice = this.getTTSVoice(domain);
      if (voice) args.push('-v', voice);
      args.push(clean);

      execFile('espeak-ng', args, (err) => {
        if (err) {
          console.error('[mimi] espeak-ng error:', err.message);
          return reject(err);
        }
        // Auto-cleanup after 5 minutes
        setTimeout(() => {
          try { unlinkSync(filePath); } catch (_) {}
        }, 5 * 60 * 1000);
        resolve(id);
      });
    });
  }

  /**
   * Check if transcribed text starts with the wake word "mimi"
   * Returns { matched: boolean, command: string }
   */
  checkWakeWord(text) {
    const lower = text.toLowerCase().trim();
    // Match "mimi", "hey mimi", "hi mimi", "ok mimi", "okay mimi"
    const match = lower.match(/^(?:hey|hi|ok(?:ay)?)\s*[,.]?\s*mimi[,.\s!]*(.*)$/i)
      || lower.match(/^mimi[,.\s!]*(.*)$/i);
    if (match) {
      const command = match[1]?.trim() || '';
      // Check for "clear" command
      if (/^clear$/i.test(command)) {
        return { matched: true, command: '', clear: true };
      }
      return { matched: true, command };
    }
    return { matched: false, command: '' };
  }

  /**
   * Trim and condense conversation history to stay within token limits.
   * Keeps the last 10 exchanges, strips web search content from older turns,
   * and collapses assistant tool-use turns into just their final text.
   */
  /**
   * Strip citations and server_content from a content block array.
   * Citations reference search results by index — once the search results
   * (server_content / web_search_tool_result) are gone, orphaned citations
   * cause 400 errors from the API. Strip them from ALL messages, not just
   * the ones that contained the search.
   */
  _cleanContentBlocks(content) {
    if (typeof content === 'string') return content;
    if (!Array.isArray(content)) return content;

    const cleaned = [];
    for (const block of content) {
      // Drop search result payloads entirely (massive HTML)
      if (block.type === 'server_content' || block.type === 'web_search_tool_result') continue;
      // Drop server-side tool_use blocks (e.g. web_search). These are paired
      // with web_search_tool_result blocks which we strip above; if we keep
      // the server_tool_use the API 400s on replay with "tool use ... was
      // found without a corresponding ... result block".
      if (block.type === 'server_tool_use') continue;

      if (block.type === 'text') {
        // Always strip citations — they're rendering metadata, not context
        if (block.citations) {
          cleaned.push({ type: 'text', text: block.text });
        } else {
          cleaned.push(block);
        }
      } else if (block.type === 'tool_use') {
        // Drop web_search tool_use — its results (server_content /
        // web_search_tool_result) are already stripped above, so keeping
        // the tool_use orphans it and causes a 400 on replay.
        if (block.name === 'web_search') continue;
        // Keep other tool_use but truncate large inputs
        const inputStr = JSON.stringify(block.input);
        if (inputStr.length > 500) {
          cleaned.push({ ...block, input: { summary: inputStr.substring(0, 200) + '...' } });
        } else {
          cleaned.push(block);
        }
      } else if (block.type === 'tool_result') {
        // Truncate large tool results
        if (typeof block.content === 'string' && block.content.length > 1000) {
          cleaned.push({ ...block, content: block.content.substring(0, 500) + '... [truncated]' });
        } else {
          cleaned.push(block);
        }
      } else {
        cleaned.push(block);
      }
    }
    return cleaned;
  }

  /**
   * Repair orphaned tool_use / tool_result pairs in history.
   * Every assistant message containing tool_use blocks MUST be followed
   * immediately by a user message whose tool_result IDs match.  If not,
   * strip the unmatched tool_use blocks (or add synthetic tool_results).
   */
  _repairToolPairs(history) {
    for (let i = 0; i < history.length; i++) {
      const msg = history[i];
      if (msg.role !== 'assistant' || typeof msg.content === 'string') continue;
      if (!Array.isArray(msg.content)) continue;

      const toolUseIds = msg.content
        .filter(b => b.type === 'tool_use')
        .map(b => b.id);
      if (toolUseIds.length === 0) continue;

      // Check the next message for matching tool_results
      const next = history[i + 1];
      const resultIds = new Set();
      if (next && next.role === 'user' && Array.isArray(next.content)) {
        for (const b of next.content) {
          if (b.type === 'tool_result') resultIds.add(b.tool_use_id);
        }
      }

      const orphanIds = toolUseIds.filter(id => !resultIds.has(id));
      if (orphanIds.length === 0) continue;

      // Strip orphaned tool_use blocks from the assistant message
      const orphanSet = new Set(orphanIds);
      const cleaned = msg.content.filter(b =>
        b.type !== 'tool_use' || !orphanSet.has(b.id)
      );
      if (cleaned.length === 0) {
        history[i] = { role: 'assistant', content: [{ type: 'text', text: '(tool results unavailable)' }] };
      } else {
        history[i] = { role: msg.role, content: cleaned };
      }
      console.log(`[mimi] Repaired ${orphanIds.length} orphaned tool_use block(s) at message ${i}`);
    }

    // Also strip orphaned tool_result messages (user message with tool_results
    // whose preceding assistant message has no matching tool_use)
    for (let i = 0; i < history.length; i++) {
      const msg = history[i];
      if (msg.role !== 'user' || typeof msg.content === 'string') continue;
      if (!Array.isArray(msg.content)) continue;
      if (!msg.content.some(b => b.type === 'tool_result')) continue;

      const prev = history[i - 1];
      const prevToolIds = new Set();
      if (prev && prev.role === 'assistant' && Array.isArray(prev.content)) {
        for (const b of prev.content) {
          if (b.type === 'tool_use') prevToolIds.add(b.id);
        }
      }

      // Keep only tool_results that match a preceding tool_use, plus any non-tool_result blocks
      const cleaned = msg.content.filter(b =>
        b.type !== 'tool_result' || prevToolIds.has(b.tool_use_id)
      );
      if (cleaned.length === 0) {
        // Entire message was orphaned tool_results — remove it
        history.splice(i, 1);
        i--;
      } else if (cleaned.length !== msg.content.length) {
        history[i] = { role: msg.role, content: cleaned };
      }
    }
  }

  trimHistory(convKey, history) {
    // Clean every message: strip citations, server_content, truncate large blocks
    for (let i = 0; i < history.length; i++) {
      const msg = history[i];
      if (typeof msg.content === 'string') continue;
      const cleaned = this._cleanContentBlocks(msg.content);
      // If cleaning removed all blocks (e.g. a message was only server_content),
      // replace with a placeholder so the message isn't empty
      if (Array.isArray(cleaned) && cleaned.length === 0) {
        history[i] = { role: msg.role, content: [{ type: 'text', text: '(search results)' }] };
      } else {
        history[i] = { role: msg.role, content: cleaned };
      }
    }

    // Drop bulky base64 attachments from all but the most recent message. Once
    // Claude has read a PDF/image, keeping the raw bytes just bloats every
    // future request (and the saved vault). The last message is spared so an
    // in-flight retry (429/400) still has the attachment it was about to read.
    for (let i = 0; i < history.length - 1; i++) {
      const msg = history[i];
      if (!Array.isArray(msg.content)) continue;
      if (!msg.content.some(b => b.type === 'document' || b.type === 'image')) continue;
      history[i] = {
        role: msg.role,
        content: msg.content.map(b =>
          (b.type === 'document' || b.type === 'image')
            ? { type: 'text', text: `[${b.type} attachment removed from history]` }
            : b
        )
      };
    }

    // Keep max 20 messages
    if (history.length > 20) {
      const trimmed = history.slice(-20);
      // Ensure first message is a plain user message (not a tool_result).
      while (trimmed.length > 0) {
        const first = trimmed[0];
        if (first.role !== 'user') {
          trimmed.shift();
          continue;
        }
        if (Array.isArray(first.content) && first.content.some(b => b.type === 'tool_result')) {
          trimmed.shift();
          continue;
        }
        break;
      }
      history.length = 0;
      history.push(...trimmed);
      this.conversations.set(convKey, history);
    }

    // Repair any broken tool_use / tool_result pairs (can happen after trimming)
    this._repairToolPairs(history);
  }

  /**
   * Snapshot the request headers we need for internal proxying.
   * Must be called synchronously during the request handler.
   */
  captureRequestContext(req) {
    return {
      // Resolved public domain, NOT the raw Host header. On an internal summon
      // (message-board → mimi) undici strips the caller's Host header, so the
      // raw header is "127.0.0.1:PORT"; only X-Forwarded-Host carries the real
      // domain, and req.hostname is where trust proxy='loopback' surfaces it.
      // Forwarding the raw header made every proxied agent (connector, files)
      // resolve to the empty 127.0.0.1 domain store.
      host: req.hostname || req.headers?.host || 'localhost',
      authorization: req.headers?.authorization || null,
      cookie: req.headers?.cookie || null,
      me: req.me,
      domainAcl: req.domainAcl,
      hostname: req.hostname,
      port: this.getInternalPort(req),
      userVault: req.userVault || null
    };
  }

  async proxyToolCall(toolName, args, reqCtx) {
    const port = reqCtx.port;
    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Host': reqCtx.host,
      'X-Forwarded-Host': reqCtx.host
    };
    if (reqCtx.authorization) headers['Authorization'] = reqCtx.authorization;
    if (reqCtx.cookie) headers['Cookie'] = reqCtx.cookie;

    const baseUrl = `http://127.0.0.1:${port}`;

    const api = async (urlPath, opts = {}) => {
      // No ACL pre-check here. Each agent owns its own authorization and enforces
      // it against the forwarded session (req.me reconstructed from the poster's
      // cookie/bearer). Re-deriving it here is a shadow gate: it only sees coarse
      // agent-level ACL under the global default stance, so it wrongly blocks
      // grants the agent itself would honor — per-connector list membership,
      // per-file ownership, an agent's own ACL stance. Forward the call and let
      // the agent decide; whatever the poster could reach directly, mimi reaches
      // on their behalf, and a denial comes back as the agent's own response.
      const url = `${baseUrl}${urlPath}`;
      try {
        const res = await fetch(url, { ...opts, headers: { ...headers, ...opts.headers } });
        return res.json();
      } catch (e) {
        console.error(`[mimi] Proxy error: ${opts.method || 'GET'} ${url} → ${e.message}`);
        throw e;
      }
    };

    try {
      // whoami is handled directly — everything else routes through agent registry
      if (toolName === 'whoami') {
        return {
          wallet: reqCtx.me?.identityAddress || null,
          role: reqCtx.me?.role || null,
          authenticated: !!reqCtx.me?.authenticated,
          domain: reqCtx.hostname
        };
      }

      const agentTool = this._findAgentTool(toolName);
      if (!agentTool) return { error: `Unknown tool: ${toolName}` };

      // Bridged tools route through PeerBridge WebSocket
      if (agentTool.bridged && typeof this.config.callBridgedTool === 'function') {
        return await this.config.callBridgedTool(agentTool.peerId, toolName, args);
      }

      // Substitute {param} placeholders in path with arg values
      const pathParams = new Set(agentTool.path.match(/\{(\w+)\}/g)?.map(p => p.slice(1, -1)) || []);
      const toolPath = agentTool.path.replace(/\{(\w+)\}/g, (_, key) =>
        encodeURIComponent(args[key] || '')
      );
      const fullPath = `${agentTool.basePath}${toolPath}`;

      if (agentTool.method === 'GET') {
        const queryArgs = Object.entries(args).filter(([k]) => !pathParams.has(k) && args[k] != null);
        if (queryArgs.length) {
          const qs = new URLSearchParams(queryArgs).toString();
          return await api(`${fullPath}?${qs}`);
        }
        return await api(fullPath);
      } else {
        return await api(fullPath, {
          method: agentTool.method,
          body: JSON.stringify(args)
        });
      }
    } catch (e) {
      return { error: e.message };
    }
  }

  /**
   * Look up a tool in the agent registry
   */
  _findAgentTool(toolName) {
    const getAgentTools = this.config.getAgentTools;
    if (typeof getAgentTools !== 'function') return null;
    return getAgentTools().find(t => t.name === toolName) || null;
  }

  /**
   * Tool definitions for Claude (matches MCPTools.mjs TOOLS)
   */
  getTools() {
    // whoami is the only tool Mimi handles directly (not an agent)
    const tools = [
      {
        name: 'whoami',
        description: 'Show your current wallet identity, auth method, and permissions.',
        input_schema: { type: 'object', properties: {} }
      }
    ];

    // All other tools come from the agent registry (declared in each agent's epistery.json)
    const getAgentTools = this.config.getAgentTools;
    if (typeof getAgentTools === 'function') {
      for (const tool of getAgentTools()) {
        tools.push({
          name: tool.name,
          description: tool.description,
          input_schema: tool.inputSchema || { type: 'object', properties: {} }
        });
      }
    }

    return tools;
  }

  /**
   * Get tools with dynamic descriptions from agents that support describeTools().
   * Falls back to static descriptions for agents that don't.
   */
  async getToolsForDomain(domain) {
    const tools = this.getTools();
    const getAgentTools = this.config.getAgentTools;
    if (typeof getAgentTools !== 'function') return tools;

    // Find agent tools in the list and try to enrich with dynamic descriptions
    const agentManager = this.config._agentManager;
    if (!agentManager) return tools;

    for (const [, agentData] of agentManager.agents) {
      if (typeof agentData.instance?.describeTools !== 'function') continue;

      try {
        const dynamicTools = await agentData.instance.describeTools(domain);
        if (!Array.isArray(dynamicTools)) continue;

        for (const dt of dynamicTools) {
          const existing = tools.find(t => t.name === dt.name);
          if (existing) {
            existing.description = dt.description;
            if (dt.inputSchema) existing.input_schema = dt.inputSchema;
          }
        }
      } catch (e) {
        console.error(`[mimi] describeTools() failed for ${agentData.manifest.name}:`, e.message);
      }
    }

    return tools;
  }

  /**
   * Call Claude with retry on rate limits (same as pro-research)
   */
  async callClaudeWithRetry(client, params, maxRetries = 3) {
    let attempt = 0;
    while (attempt < maxRetries) {
      try {
        return await client.messages.create(params);
      } catch (error) {
        if (error.status === 429) {
          attempt++;
          const retryAfter = error.headers?.['retry-after']
            ? parseInt(error.headers['retry-after']) * 1000
            : 5000 * attempt;
          if (attempt >= maxRetries) {
            throw new Error(`Rate limit exceeded after ${maxRetries} retries`);
          }
          console.log(`[mimi] Rate limit (attempt ${attempt}/${maxRetries}), waiting ${Math.ceil(retryAfter / 1000)}s...`);
          await new Promise(resolve => setTimeout(resolve, retryAfter + 1000));
        } else {
          throw error;
        }
      }
    }
    throw new Error('Maximum retry attempts exceeded');
  }

  /**
   * Attach the agent to an Express router
   */
  attach(router) {
    // Ensure trailing slash on root so relative URLs resolve correctly
    router.use((req, res, next) => {
      if (req.path === '/' && !req.originalUrl.endsWith('/') && req.method === 'GET') {
        return res.redirect(301, req.originalUrl + '/');
      }
      next();
    });

    // Domain middleware
    router.use(async (req, res, next) => {
      req.domain = req.hostname || 'localhost';
      next();
    });

    // Serve icon
    router.get('/icon.svg', (req, res) => {
      res.set('Content-Type', 'image/svg+xml');
      res.sendFile(path.join(__dirname, 'icon.svg'));
    });

    // Serve client directory
    router.use('/client', express.static(path.join(__dirname, 'client')));

    // Status endpoint
    router.get('/status', (req, res) => {
      res.json({
        agent: 'mimi',
        version: '0.1.0',
        activeSessions: this.conversations.size
      });
    });

    // Admin page
    router.get('/admin', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      res.sendFile(path.join(__dirname, 'client/admin.html'));
    });

    // Admin: check if keys are set
    router.get('/admin/key', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      const cfg = new Config();
      cfg.setPath(req.hostname || 'localhost');
      const hasKey = !!(cfg.data?.anthropic_api_key || cfg.data?.claude?.anthropicKey);
      const hasOpenAIKey = !!(cfg.data?.openai?.apikey);
      res.json({ hasKey, hasOpenAIKey });
    });

    // Admin: save key
    router.post('/admin/key', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      const { key, provider } = req.body;
      if (!key || !key.startsWith('sk-')) {
        return res.status(400).json({ error: 'Invalid API key' });
      }
      const domain = req.hostname || 'localhost';
      const cfg = new Config();
      cfg.setPath(domain);

      if (provider === 'openai') {
        if (!cfg.data.openai) cfg.data.openai = {};
        cfg.data.openai.apikey = key;
        // Reset cached STT provider so next request picks up new key
        this.sttProvider = null;
      } else {
        if (!cfg.data.claude) cfg.data.claude = {};
        cfg.data.claude.anthropicKey = key;
        // Reset cached client so next request picks up new key
        this.anthropic = null;
      }

      cfg.save();
      res.json({ success: true });
    });

    // Admin: get AI notes
    router.get('/admin/notes', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      const cfg = new Config();
      cfg.setPath(req.hostname || 'localhost');
      res.json({ notes: cfg.data?.ai_notes || '' });
    });

    // Admin: save AI notes
    router.post('/admin/notes', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      const { notes } = req.body;
      if (typeof notes !== 'string') {
        return res.status(400).json({ error: 'notes must be a string' });
      }
      const domain = req.hostname || 'localhost';
      const cfg = new Config();
      cfg.setPath(domain);
      cfg.data.ai_notes = notes;
      cfg.save();
      res.json({ success: true });
    });

    // Admin: get available models and the currently selected one
    router.get('/admin/model', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      res.json({ models: AVAILABLE_MODELS, current: this.getModel(req.hostname || 'localhost') });
    });

    // Admin: set the Claude model
    router.post('/admin/model', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      const { model } = req.body;
      if (!AVAILABLE_MODELS.some(m => m.id === model)) {
        return res.status(400).json({ error: 'Unknown model' });
      }
      const domain = req.hostname || 'localhost';
      const cfg = new Config();
      cfg.setPath(domain);
      if (!cfg.data.claude) cfg.data.claude = {};
      cfg.data.claude.model = model;
      cfg.save();
      res.json({ success: true, model });
    });

    // Admin: whisper install status
    router.get('/admin/whisper', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      const whisperDir = path.join(homedir(), '.epistery', 'whisper');
      const status = checkWhisperInstall(whisperDir);
      // Check which STT mode is active
      const cfg = new Config();
      cfg.setPath(req.hostname || 'localhost');
      const hasLocal = status.installed;
      const hasOpenAI = !!(cfg.data?.openai?.apikey || process.env.OPENAI_API_KEY);
      res.json({
        ...status,
        installing: this.whisperInstalling,
        sttMode: hasLocal ? 'local' : (hasOpenAI ? 'openai' : 'none')
      });
    });

    // Admin: install whisper
    router.post('/admin/whisper/install', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      if (this.whisperInstalling) {
        return res.status(409).json({ error: 'Installation already in progress' });
      }

      const domain = req.hostname || 'localhost';
      const whisperDir = path.join(homedir(), '.epistery', 'whisper');
      this.whisperInstalling = true;
      this.whisperProgress = ['Starting installation...'];

      // Kick off install in background
      installWhisper(whisperDir, (msg) => {
        this.whisperProgress.push(msg);
        console.log(`[mimi-whisper] ${msg}`);
      }).then(({ binaryPath, modelPath }) => {
        // Save to domain config
        const cfg = new Config();
        cfg.setPath(domain);
        if (!cfg.data.whisper) cfg.data.whisper = {};
        cfg.data.whisper.binary = binaryPath;
        cfg.data.whisper.model = modelPath;
        cfg.data.whisper.threads = '4';
        cfg.save();

        // Reset STT provider so next request picks up local
        this.sttProvider = null;
        this.whisperProgress.push('Installation complete. Local whisper is now active.');
        this.whisperInstalling = false;
      }).catch((err) => {
        console.error('[mimi-whisper] Install failed:', err);
        this.whisperProgress.push(`ERROR: ${err.message}`);
        this.whisperInstalling = false;
      });

      res.json({ success: true, message: 'Installation started' });
    });

    // Admin: poll install progress
    router.get('/admin/whisper/progress', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      res.json({
        installing: this.whisperInstalling,
        progress: this.whisperProgress
      });
    });

    // Admin: uninstall whisper
    router.post('/admin/whisper/uninstall', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      if (this.whisperInstalling) {
        return res.status(409).json({ error: 'Installation in progress, cannot uninstall' });
      }

      const domain = req.hostname || 'localhost';
      const whisperDir = path.join(homedir(), '.epistery', 'whisper');

      uninstallWhisper(whisperDir);

      // Clear config
      const cfg = new Config();
      cfg.setPath(domain);
      delete cfg.data.whisper;
      cfg.save();

      // Reset STT provider to fall back to OpenAI
      this.sttProvider = null;
      this.whisperProgress = [];

      res.json({ success: true, message: 'Whisper uninstalled. Falling back to OpenAI API.' });
    });

    // Admin: list available TTS voices
    router.get('/admin/voices', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      try {
        const voices = await new Promise((resolve, reject) => {
          execFile('espeak-ng', ['--voices'], { timeout: 5000 }, (err, stdout) => {
            if (err) return reject(err);
            const lines = stdout.trim().split('\n');
            // First line is header: Pty  Language  Age/Gender  VoiceName   File   Other Languages
            const results = [];
            for (let i = 1; i < lines.length; i++) {
              const parts = lines[i].trim().split(/\s+/);
              if (parts.length >= 4) {
                results.push({
                  priority: parts[0],
                  language: parts[1],
                  gender: parts[2],
                  name: parts[3],
                  file: parts[4] || ''
                });
              }
            }
            resolve(results);
          });
        });
        const cfg = new Config();
        cfg.setPath(req.hostname || 'localhost');
        const current = cfg.data?.tts?.voice || null;
        res.json({ voices, current });
      } catch (err) {
        console.error('[mimi] Voice list error:', err.message);
        res.status(500).json({ error: 'Failed to list voices: ' + err.message });
      }
    });

    // Admin: set TTS voice
    router.post('/admin/voices', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      const { voice } = req.body;
      const domain = req.hostname || 'localhost';
      const cfg = new Config();
      cfg.setPath(domain);
      if (voice) {
        if (!cfg.data.tts) cfg.data.tts = {};
        cfg.data.tts.voice = voice;
      } else {
        delete cfg.data.tts?.voice;
      }
      cfg.save();
      this._ttsVoice = null; // reset cache
      res.json({ success: true, voice: voice || 'default' });
    });

    // Admin: preview a voice
    router.post('/admin/voices/preview', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      const { voice } = req.body;
      try {
        const id = randomBytes(12).toString('hex');
        const filePath = path.join(this.getAudioDir(), `${id}.wav`);
        const args = ['-w', filePath];
        if (voice) args.push('-v', voice);
        args.push('Hello, I am Mimi. This is how I sound.');

        await new Promise((resolve, reject) => {
          execFile('espeak-ng', args, { timeout: 10000 }, (err) => {
            if (err) return reject(err);
            resolve();
          });
        });

        setTimeout(() => { try { unlinkSync(filePath); } catch (_) {} }, 60000);
        res.json({ audioUrl: `audio/${id}` });
      } catch (err) {
        res.status(500).json({ error: 'Preview failed: ' + err.message });
      }
    });

    // Main portal page — always serve the SPA so common.js can establish
    // the epistery session; data is still gated by permissions on API endpoints.
    router.get('/', (req, res) => {
      res.sendFile(path.join(__dirname, 'client/portal.html'));
    });

    // Serve TTS audio files
    router.get('/audio/:id', (req, res) => {
      const filePath = path.join(this.getAudioDir(), `${req.params.id}.wav`);
      if (!existsSync(filePath)) {
        return res.status(404).json({ error: 'Audio not found' });
      }
      res.set('Content-Type', 'audio/wav');
      res.sendFile(filePath);
    });

    // History endpoint — restore prior conversation from UserVault
    router.get('/history', async (req, res) => {
      try {
        const permissions = await this.getPermissions(req);
        if (!permissions.read || !req.userVault) {
          return res.json({ history: [] });
        }
        const vault = await req.userVault.get();
        const history = vault.mimi?.history;
        if (!Array.isArray(history) || history.length === 0) {
          return res.json({ history: [] });
        }
        // Extract displayable messages (role + text content only)
        const messages = [];
        for (const msg of history) {
          const role = msg.role;
          let text = '';
          if (typeof msg.content === 'string') {
            text = msg.content;
          } else if (Array.isArray(msg.content)) {
            text = msg.content
              .filter(b => b.type === 'text')
              .map(b => b.text)
              .join('');
          }
          if (text && (role === 'user' || role === 'assistant')) {
            messages.push({ role, text });
          }
        }
        res.json({ history: messages });
      } catch (err) {
        console.error('[mimi] History endpoint error:', err.message);
        res.json({ history: [] });
      }
    });

    // Clear endpoint — wipe conversation history
    router.post('/clear', async (req, res) => {
      try {
        const permissions = await this.getPermissions(req);
        if (!permissions.read) {
          return res.status(403).json({ error: 'Permission required' });
        }
        const { sessionId } = req.body || {};
        // Clear in-memory session
        if (sessionId && this.conversations.has(sessionId)) {
          this.conversations.delete(sessionId);
        }
        // Clear vault history
        if (req.userVault) {
          await req.userVault.merge({ mimi: { history: [], updatedAt: Date.now() } });
        }
        res.json({ success: true });
      } catch (err) {
        console.error('[mimi] Clear error:', err.message);
        res.status(500).json({ error: err.message });
      }
    });

    // Voice audio endpoint — transcribe + wake word check only.
    // Returns the transcribed text; client streams response via /message.
    router.post('/audio', async (req, res) => {
      try {
        const permissions = await this.getPermissions(req);
        if (!permissions.read) {
          return res.status(403).json({ error: 'Permission required' });
        }

        const { audio, attentive } = req.body;
        if (!audio) {
          return res.status(400).json({ status: 'error', message: 'No audio data' });
        }

        // Decode base64 WAV
        const audioBuffer = Buffer.from(audio, 'base64');

        // Transcribe via STT provider
        let text;
        try {
          const stt = this.getSTTProvider(req.domain);
          text = await stt.transcribe(audioBuffer);
        } catch (err) {
          console.error('[mimi] STT error:', err.message);
          return res.json({ status: 'error', message: 'Transcription failed: ' + err.message });
        }

        if (!text || !text.trim()) {
          return res.json({ status: 'ignored', reason: 'empty' });
        }

        // In attentive mode (post-response window), skip wake word check
        let message;
        if (attentive) {
          const wake = this.checkWakeWord(text);
          if (wake.clear) return res.json({ status: 'clear' });
          message = wake.matched ? (wake.command || text) : text;
        } else {
          const wake = this.checkWakeWord(text);
          if (!wake.matched) {
            return res.json({ status: 'ignored', reason: 'no-wake-word', text });
          }
          if (wake.clear) return res.json({ status: 'clear' });
          message = wake.command || text;
        }

        res.json({ status: 'matched', text: message });
      } catch (error) {
        console.error('[mimi] Audio endpoint error:', error);
        res.status(500).json({ status: 'error', message: error.message });
      }
    });

    // Message endpoint — SSE streaming response
    router.post('/message', async (req, res) => {
      try {
        const permissions = await this.getPermissions(req);
        if (!permissions.read) {
          return res.status(403).json({ error: 'Permission required' });
        }

        const { message, sessionId, voice } = req.body;

        if (!message) {
          return res.status(400).json({ status: 'error', message: 'Message is required' });
        }

        // Get or initialize conversation
        const convKey = sessionId || `mimi-${req.me?.identityAddress || 'anon'}-${Date.now()}`;
        if (!this.conversations.has(convKey)) {
          // Try to restore from UserVault so context survives agent switches
          let restored = [];
          if (req.userVault) {
            try {
              const vault = await req.userVault.get();
              if (vault.mimi?.history && Array.isArray(vault.mimi.history)) {
                restored = vault.mimi.history;
                console.log(`[mimi] Restored ${restored.length} messages from vault for ${req.me?.identityAddress}`);
              }
            } catch (err) {
              console.error('[mimi] Vault restore error:', err.message);
            }
          }
          this.conversations.set(convKey, restored);
        }
        const history = this.conversations.get(convKey);

        // Add user message
        history.push({ role: 'user', content: message });

        // Switch to SSE streaming
        res.writeHead(200, {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          'X-Session-Id': convKey
        });

        // Send session ID as first event
        this.sendSSE(res, 'session', { sessionId: convKey });

        // Stream the response
        const reqCtx = this.captureRequestContext(req);
        await this.processMessageStream(reqCtx, history, convKey, res, !!voice);

        // Persist trimmed history to UserVault
        if (reqCtx.userVault) {
          try {
            await reqCtx.userVault.merge({ mimi: { history, updatedAt: Date.now() } });
          } catch (err) {
            console.error('[mimi] Vault save error:', err.message);
          }
        }

        res.end();
      } catch (error) {
        console.error('[mimi] Message error:', error);
        // If headers already sent, send error as SSE
        if (res.headersSent) {
          this.sendSSE(res, 'error', { message: error.message });
          res.end();
        } else {
          res.status(500).json({ status: 'error', message: error.message });
        }
      }
    });

    // Board-reply endpoint — message-board summons mimi to participate in a
    // channel. Runs under the forwarded poster's session (their access/tools).
    // Returns reply TEXT only; the board publishes it under the host identity.
    router.post('/board-reply', async (req, res) => {
      try {
        const permissions = await this.getPermissions(req);
        if (!permissions.read) {
          return res.status(403).json({ error: 'Permission required' });
        }
        const { transcript, channel } = req.body || {};
        if (!transcript || typeof transcript !== 'string') {
          return res.status(400).json({ error: 'transcript is required' });
        }
        const reqCtx = this.captureRequestContext(req);
        const reply = await this.processBoardReply(reqCtx, transcript, channel);
        res.json({ reply: reply || '' });
      } catch (error) {
        console.error('[mimi] board-reply error:', error.message);
        res.status(500).json({ error: error.message });
      }
    });
  }

  /**
   * Send an SSE event to the client
   */
  sendSSE(res, event, data) {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  }

  /**
   * Build the system prompt for Claude
   */
  buildSystemPrompt(domain, userAddress, isVoice) {
    let aiNotes = '';
    try {
      const cfg = new Config();
      cfg.setPath(domain);
      aiNotes = cfg.data?.ai_notes || '';
    } catch (e) { /* ignore */ }

    let systemPrompt;
    if (isVoice) {
      systemPrompt = `You are Mimi, a general-purpose voice assistant on the epistery host at ${domain}.
You can answer any question — weather, trivia, math, advice, anything.
Use web_search for current information like weather, news, sports, or prices.
You also have epistery tools for wiki pages, files, archives, messages, and identity.
You can open files directly — PDFs and images come back as a readable attachment, so read them
yourself rather than asking for their contents.
Additional tools may be available from installed agents — use them when relevant.

Your spoken replies are read aloud via TTS. Be conversational, like talking to a friend.
No bullet points, no markdown, no lists, no headers in your spoken replies — just plain sentences.
CRITICAL: NEVER stop mid-sentence. NEVER cut off your answer. Always finish your complete thought
with the actual answer the user asked for. If someone asks a question, you MUST give the full answer,
not trail off. A complete short answer is better than a long one that stops mid-thought.
However, when writing content to the wiki or message board via tools, write naturally with
full markdown, proper formatting, and as much detail as appropriate for that medium.
User wallet address: ${userAddress}`;
    } else {
      systemPrompt = `You are Mimi, a helpful assistant on the epistery host at ${domain}.
You can answer any question — weather, trivia, math, advice, anything.
Use web_search for current information like weather, news, sports, or prices.
You also have epistery tools for wiki pages, files, archives, messages, and identity.
You can open files directly — PDFs and images come back as a readable attachment, so read them
yourself rather than asking for their contents.
Additional tools may be available from installed agents — use them when relevant.

Respond naturally. Use markdown formatting when it helps clarity.
Keep responses focused and complete — always finish your thought with the actual answer.
User wallet address: ${userAddress}`;
    }

    if (aiNotes) {
      systemPrompt += `\n\nDomain notes from admin:\n${aiNotes}`;
    }
    return systemPrompt;
  }

  /**
   * Stream a message response via SSE (replaces polling architecture).
   * Uses Anthropic streaming API — text appears as Claude generates it.
   * Tool-calling loop handles ALL tool_use blocks per turn.
   */
  async processMessageStream(reqCtx, history, convKey, res, isVoice) {
    const send = (event, data) => this.sendSSE(res, event, data);

    try {
      const client = this.getAnthropicClient(reqCtx.hostname);
      const tools = await this.getToolsForDomain(reqCtx.hostname || 'localhost');
      const domain = reqCtx.hostname || 'localhost';
      const userAddress = reqCtx.me?.identityAddress || 'unknown';
      const systemPrompt = this.buildSystemPrompt(domain, userAddress, isVoice);

      const allTools = [
        { type: 'web_search_20250305', name: 'web_search', max_uses: 3 },
        ...tools
      ];

      let fullText = '';
      let toolCallCount = 0;
      let continueLoop = true;

      // Helper: create a streaming Claude call with rate limit retry
      const streamWithRetry = async (params, maxRetries = 3) => {
        for (let attempt = 0; attempt < maxRetries; attempt++) {
          try {
            const stream = client.messages.stream(params);
            stream.on('text', (text) => {
              send('text', { text });
              fullText += text;
            });
            return await stream.finalMessage();
          } catch (error) {
            if (error.status === 429 && attempt < maxRetries - 1) {
              const retryAfter = error.headers?.['retry-after']
                ? parseInt(error.headers['retry-after']) * 1000
                : 5000 * (attempt + 1);
              console.log(`[mimi] Rate limit (attempt ${attempt + 1}/${maxRetries}), waiting ${Math.ceil(retryAfter / 1000)}s...`);
              send('tool', { name: 'waiting', count: 0 });
              // Also trim history to reduce token count for next attempt
              this.trimHistory(convKey, history);
              await new Promise(resolve => setTimeout(resolve, retryAfter + 1000));
              continue;
            }
            if (error.status === 400 && attempt < maxRetries - 1) {
              // Bad request — likely orphaned citations or malformed history.
              // Aggressively clean and retry.
              console.error(`[mimi] 400 error, cleaning history and retrying:`, error.message);
              this.trimHistory(convKey, history);
              continue;
            }
            throw error;
          }
        }
        throw new Error('Maximum retry attempts exceeded');
      };

      while (continueLoop) {
        const message = await streamWithRetry({
          model: this.getModel(domain),
          max_tokens: 4096,
          system: systemPrompt,
          tools: allTools,
          messages: history
        });

        if (message.stop_reason === 'tool_use') {
          // Add assistant message to history (clean immediately)
          history.push({ role: 'assistant', content: this._cleanContentBlocks(message.content) });

          // Handle ALL tool_use blocks (not just the first)
          const toolUses = message.content.filter(b => b.type === 'tool_use');
          const regularTools = toolUses.filter(t => t.name !== 'web_search');

          if (regularTools.length > 0) {
            const toolResults = [];
            const attachments = [];
            for (const toolUse of regularTools) {
              toolCallCount++;
              send('tool', { name: toolUse.name, count: toolCallCount });
              const result = await this.proxyToolCall(toolUse.name, toolUse.input, reqCtx);
              const packed = packToolResult(toolUse, result);
              toolResults.push(packed.toolResult);
              if (packed.attachment) attachments.push(packed.attachment);
            }
            history.push({ role: 'user', content: toolResults });
            // Native PDFs/images ride in their own user turn after the
            // tool_result turn so the tool-response message stays pure.
            if (attachments.length) history.push({ role: 'user', content: attachments });
          }
          // For web_search-only turns, results are already in the assistant content
          if (regularTools.length === 0) {
            send('tool', { name: 'web_search', count: ++toolCallCount });
          }
        } else {
          // end_turn or max_tokens — we're done
          history.push({ role: 'assistant', content: this._cleanContentBlocks(message.content) });
          continueLoop = false;
        }
      }

      // Final trim
      this.trimHistory(convKey, history);

      // Generate TTS audio for voice requests
      if (isVoice && fullText) {
        try {
          const audioId = await this.generateTTS(fullText, domain);
          send('audio', { url: `audio/${audioId}` });
        } catch (err) {
          console.error('[mimi] TTS generation failed:', err.message);
        }
      }

      send('done', {});
    } catch (error) {
      console.error('[mimi] Stream processing error:', error);
      send('error', { message: error.message });
    }
  }

  /**
   * System prompt for participating in a message-board channel.
   * Mimi is one voice in a shared, multi-party group chat.
   */
  buildBoardSystemPrompt(domain, channel) {
    let aiNotes = '';
    try {
      const cfg = new Config();
      cfg.setPath(domain);
      aiNotes = cfg.data?.ai_notes || '';
    } catch (e) { /* ignore */ }

    let prompt = `You are Mimi, a member of the "${channel || 'general'}" channel on the message board at ${domain}.
You were summoned because someone mentioned @mimi, or a conversation you are watching continued.
This is a shared group chat — multiple people see each other's messages. Reply naturally as one
participant, in context with the latest messages. Address people by name when it reads naturally.
Be concise and genuinely useful. Use markdown when it helps clarity. If there is nothing useful to
add, a short acknowledgement is fine. You have epistery tools (wiki, files, web search, etc.) — use
them when they help your reply. You can open files with the files tools: PDFs and images come back
as a readable attachment, so read them directly rather than asking for the text to be pasted.
Do NOT post to the board yourself; just produce the text of your reply.`;

    if (aiNotes) {
      prompt += `\n\nDomain notes from admin:\n${aiNotes}`;
    }
    return prompt;
  }

  /**
   * Non-streaming variant of the Claude tool loop for board participation.
   * Takes a formatted channel transcript and returns mimi's reply text.
   * message_post is withheld so mimi cannot self-publish — the board owns
   * authorship (under the host identity).
   */
  async processBoardReply(reqCtx, transcript, channel) {
    const domain = reqCtx.hostname || 'localhost';
    const client = this.getAnthropicClient(domain);
    const allAgentTools = await this.getToolsForDomain(domain);
    const tools = allAgentTools.filter(t => t.name !== 'message_post');
    const allTools = [
      { type: 'web_search_20250305', name: 'web_search', max_uses: 3 },
      ...tools
    ];
    const systemPrompt = this.buildBoardSystemPrompt(domain, channel);

    const messages = [{
      role: 'user',
      content: `Recent conversation in the "${channel || 'general'}" channel, oldest first:\n\n${transcript}\n\nWrite Mimi's next reply in context. Output ONLY the message text to post — no preamble, no surrounding quotes.`
    }];

    let finalText = '';
    // Bound the tool loop so a misbehaving turn can't spin forever.
    for (let i = 0; i < 8; i++) {
      const message = await this.callClaudeWithRetry(client, {
        model: this.getModel(domain),
        max_tokens: 1024,
        system: systemPrompt,
        tools: allTools,
        messages
      });

      if (message.stop_reason === 'tool_use') {
        messages.push({ role: 'assistant', content: this._cleanContentBlocks(message.content) });
        const regularTools = message.content.filter(b => b.type === 'tool_use' && b.name !== 'web_search');
        if (regularTools.length === 0) {
          // web_search-only turn (results already inline) — let it continue,
          // but guard against a stall by collecting any text and stopping.
          finalText = (message.content || []).filter(b => b.type === 'text').map(b => b.text).join('').trim();
          if (finalText) break;
          continue;
        }
        const toolResults = [];
        const attachments = [];
        for (const toolUse of regularTools) {
          const result = await this.proxyToolCall(toolUse.name, toolUse.input, reqCtx);
          const packed = packToolResult(toolUse, result);
          toolResults.push(packed.toolResult);
          if (packed.attachment) attachments.push(packed.attachment);
        }
        messages.push({ role: 'user', content: toolResults });
        // PDFs/images get their own user turn so Claude reads them natively.
        if (attachments.length) messages.push({ role: 'user', content: attachments });
        continue;
      }

      finalText = (message.content || []).filter(b => b.type === 'text').map(b => b.text).join('').trim();
      break;
    }
    return finalText;
  }

  /**
   * Check permissions (same pattern as wiki)
   */
  async getPermissions(req) {
    // Identity + ACL come from the host-owned req.me (human and MCP alike).
    const result = { admin: false, edit: false, read: false };
    if (!req.me?.identityAddress || !req.domainAcl) {
      return result;
    }
    const access = await req.me.access('epistery/mimi');
    result.admin = access.admin;
    result.edit = access.edit;
    result.read = access.read;
    return result;
  }

  async cleanup() {
    this.conversations.clear();
  }
}
