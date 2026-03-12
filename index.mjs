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
export default class MimiAgent {
  constructor(config = {}) {
    this.config = config;
    this.conversations = new Map();   // sessionId -> message history
    this.pendingRequests = new Map();  // continuationToken -> pending state
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
      return { matched: true, command };
    }
    return { matched: false, command: '' };
  }

  /**
   * Trim and condense conversation history to stay within token limits.
   * Keeps the last 10 exchanges, strips web search content from older turns,
   * and collapses assistant tool-use turns into just their final text.
   */
  trimHistory(convKey, history) {
    // First pass: condense assistant messages that have web_search/server_content blocks
    // These are huge (full page HTML) and useless once the answer is extracted
    for (let i = 0; i < history.length - 2; i++) {
      const msg = history[i];
      if (msg.role !== 'assistant' || typeof msg.content === 'string') continue;
      if (!Array.isArray(msg.content)) continue;

      // Strip server_content and web_search_tool_result blocks, keep text and tool_use
      const condensed = [];
      for (const block of msg.content) {
        if (block.type === 'text') {
          condensed.push(block);
        } else if (block.type === 'tool_use') {
          // Keep tool_use but truncate large inputs
          const input = block.input;
          const inputStr = JSON.stringify(input);
          if (inputStr.length > 500) {
            condensed.push({ ...block, input: { summary: inputStr.substring(0, 200) + '...' } });
          } else {
            condensed.push(block);
          }
        }
        // Drop server_content, web_search_tool_result, etc.
      }
      if (condensed.length > 0 && condensed.length < msg.content.length) {
        history[i] = { role: 'assistant', content: condensed };
      }
    }

    // Second pass: also condense user messages with tool_result blocks
    for (let i = 0; i < history.length - 2; i++) {
      const msg = history[i];
      if (msg.role !== 'user' || typeof msg.content === 'string') continue;
      if (!Array.isArray(msg.content)) continue;

      const condensed = msg.content.map(block => {
        if (block.type === 'tool_result' && typeof block.content === 'string' && block.content.length > 1000) {
          return { ...block, content: block.content.substring(0, 500) + '... [truncated]' };
        }
        return block;
      });
      history[i] = { role: 'user', content: condensed };
    }

    // Third pass: keep max 20 messages
    if (history.length > 20) {
      const trimmed = history.slice(-20);
      // Ensure first message is a plain user message (not a tool_result).
      // A tool_result user message requires a preceding assistant tool_use
      // message, so we must skip past orphaned tool_result/tool_use pairs.
      while (trimmed.length > 0) {
        const first = trimmed[0];
        if (first.role !== 'user') {
          trimmed.shift();
          continue;
        }
        // Check if this user message contains tool_result blocks
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
  }

  /**
   * Snapshot the request headers we need for internal proxying.
   * Must be called synchronously during the request handler,
   * before the async processMessage runs after res.json().
   */
  captureRequestContext(req) {
    return {
      host: req.headers?.host || 'localhost',
      authorization: req.headers?.authorization || null,
      cookie: req.headers?.cookie || null,
      episteryClient: req.episteryClient,
      hostname: req.hostname,
      port: this.getInternalPort(req)
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
      switch (toolName) {
        // Wiki
        case 'wiki_read': {
          const data = await api(`/agent/epistery/wiki/${encodeURIComponent(args.page)}`);
          return data.error ? { error: data.error } : data;
        }
        case 'wiki_write': {
          const docId = args.id || args.title;
          const data = await api(`/agent/epistery/wiki/${encodeURIComponent(docId)}`, {
            method: 'POST',
            body: JSON.stringify({ title: args.title, body: args.content })
          });
          return data;
        }
        case 'wiki_list': {
          return await api('/agent/epistery/wiki/index');
        }

        // Archives
        case 'archive_create': {
          return await api('/agent/rootz/archive-agent/create', {
            method: 'POST',
            body: JSON.stringify({ title: args.title, content: args.content, tags: args.tags || [], templateType: args.templateType })
          });
        }
        case 'archive_list': {
          const params = new URLSearchParams();
          if (args.limit) params.set('limit', args.limit);
          if (args.offset) params.set('offset', args.offset);
          if (args.tags) params.set('tags', args.tags);
          if (args.template) params.set('template', args.template);
          const qs = params.toString();
          return await api(`/agent/rootz/archive-agent/list${qs ? '?' + qs : ''}`);
        }
        case 'archive_search': {
          const params = new URLSearchParams({ q: args.query });
          if (args.limit) params.set('limit', args.limit);
          return await api(`/agent/rootz/archive-agent/search?${params}`);
        }
        case 'archive_read': {
          const data = await api(`/agent/rootz/archive-agent/read/${encodeURIComponent(args.id)}`);
          return data.error ? { error: data.error } : data;
        }
        case 'archive_stats': {
          return await api('/agent/rootz/archive-agent/stats');
        }

        // Messages
        case 'message_list': {
          return await api('/agent/epistery/message-board/api/posts');
        }
        case 'message_post': {
          return await api('/agent/epistery/message-board/api/posts', {
            method: 'POST',
            body: JSON.stringify({ text: args.text })
          });
        }

        // Secrets
        case 'secret_list': {
          return await api('/agent/rootz/secret-agent/secrets');
        }

        // Identity
        case 'whoami': {
          return {
            wallet: reqCtx.episteryClient?.address || null,
            authMethod: reqCtx.episteryClient?.authType || 'none',
            authenticated: !!reqCtx.episteryClient?.authenticated,
            domain: reqCtx.hostname
          };
        }

        default: {
          // Check agent tool registry for dynamically declared tools
          const agentTool = this._findAgentTool(toolName);
          if (agentTool) {
            // Bridged tools route through PeerBridge WebSocket
            if (agentTool.bridged && typeof this.config.callBridgedTool === 'function') {
              return await this.config.callBridgedTool(agentTool.peerId, toolName, args);
            }

            // Substitute {param} placeholders in path with arg values
            let toolPath = agentTool.path.replace(/\{(\w+)\}/g, (_, key) =>
              encodeURIComponent(args[key] || '')
            );
            const fullPath = `${agentTool.basePath}${toolPath}`;

            if (agentTool.method === 'GET') {
              // Remaining args (not consumed by path) become query params
              const pathParams = new Set(agentTool.path.match(/\{(\w+)\}/g)?.map(p => p.slice(1, -1)) || []);
              const queryArgs = Object.entries(args).filter(([k]) => !pathParams.has(k) && args[k] != null);
              if (queryArgs.length) {
                const qs = new URLSearchParams(queryArgs).toString();
                toolPath = `${fullPath}?${qs}`;
                return await api(toolPath);
              }
              return await api(fullPath);
            } else {
              return await api(fullPath, {
                method: agentTool.method,
                body: JSON.stringify(args)
              });
            }
          }
          return { error: `Unknown tool: ${toolName}` };
        }
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
    const coreTools = [
      {
        name: 'wiki_read',
        description: 'Read a wiki page by its document ID. Returns the page content in markdown.',
        input_schema: {
          type: 'object',
          properties: { page: { type: 'string', description: 'Document ID - a WikiWord using only letters, numbers, and underscores (min 3 chars). Examples: "Home", "BedfordStreet", "FAQ_Page"' } },
          required: ['page']
        }
      },
      {
        name: 'wiki_write',
        description: 'Create or update a wiki page. The id is a WikiWord document identifier (letters, numbers, underscores only, min 3 chars). The title is a human-readable display name. Content should be markdown.',
        input_schema: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Document ID - a WikiWord using only letters, numbers, and underscores (min 3 chars). Examples: "BedfordStreetHistory", "AboutUs", "FAQ_Page"' },
            title: { type: 'string', description: 'Human-readable page title (e.g., "73 Bedford Street History")' },
            content: { type: 'string', description: 'Page content (markdown)' }
          },
          required: ['id', 'title', 'content']
        }
      },
      {
        name: 'wiki_list',
        description: 'List all wiki pages. Returns titles and metadata.',
        input_schema: { type: 'object', properties: {} }
      },
      {
        name: 'archive_create',
        description: 'Create an archive of content, code, notes, or any text.',
        input_schema: {
          type: 'object',
          properties: {
            title: { type: 'string', description: 'Archive title' },
            content: { type: 'string', description: 'Content to archive' },
            tags: { type: 'array', items: { type: 'string' }, description: 'Tags for organization' },
            templateType: { type: 'string', description: 'Template type' }
          },
          required: ['title', 'content']
        }
      },
      {
        name: 'archive_list',
        description: 'List archives with optional filtering.',
        input_schema: {
          type: 'object',
          properties: {
            limit: { type: 'number', description: 'Max results (default 20)' },
            offset: { type: 'number', description: 'Pagination offset' },
            tags: { type: 'string', description: 'Filter by tag (comma-separated)' },
            template: { type: 'string', description: 'Filter by template type' }
          }
        }
      },
      {
        name: 'archive_search',
        description: 'Full-text search across all archives.',
        input_schema: {
          type: 'object',
          properties: {
            query: { type: 'string', description: 'Search query' },
            limit: { type: 'number', description: 'Max results (default 20)' }
          },
          required: ['query']
        }
      },
      {
        name: 'archive_read',
        description: 'Read a specific archive by its ID.',
        input_schema: {
          type: 'object',
          properties: { id: { type: 'string', description: 'Archive ID' } },
          required: ['id']
        }
      },
      {
        name: 'archive_stats',
        description: 'Get archive statistics: total count, tag breakdown, date range.',
        input_schema: { type: 'object', properties: {} }
      },
      {
        name: 'message_list',
        description: 'List message board posts.',
        input_schema: { type: 'object', properties: {} }
      },
      {
        name: 'message_post',
        description: 'Post a message to the message board.',
        input_schema: {
          type: 'object',
          properties: { text: { type: 'string', description: 'Message content' } },
          required: ['text']
        }
      },
      {
        name: 'secret_list',
        description: 'List available secrets (metadata only).',
        input_schema: { type: 'object', properties: {} }
      },
      {
        name: 'whoami',
        description: 'Show your current wallet identity, auth method, and permissions.',
        input_schema: { type: 'object', properties: {} }
      }
    ];

    // Append dynamically registered agent tools
    const getAgentTools = this.config.getAgentTools;
    if (typeof getAgentTools === 'function') {
      for (const tool of getAgentTools()) {
        coreTools.push({
          name: tool.name,
          description: tool.description,
          input_schema: tool.inputSchema || { type: 'object', properties: {} }
        });
      }
    }

    return coreTools;
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
        activeSessions: this.conversations.size,
        pendingRequests: this.pendingRequests.size
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

    // Main portal page
    router.get('/', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.read) {
        if (req.accepts('html')) {
          return res.status(403).send(`
            <!DOCTYPE html>
            <html><head><title>Access Denied</title></head>
            <body style="font-family: sans-serif; max-width: 600px; margin: 100px auto; text-align: center;">
              <h1>Access Denied</h1>
              <p>You don't have access to Mimi voice portal.</p>
            </body></html>
          `);
        }
        return res.status(403).json({ error: 'Permission required' });
      }
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

    // Voice audio endpoint — receive WAV, transcribe, check wake word
    router.post('/audio', async (req, res) => {
      try {
        const permissions = await this.getPermissions(req);
        if (!permissions.read) {
          return res.status(403).json({ error: 'Permission required' });
        }

        const { audio, sessionId, attentive } = req.body;
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

        console.log(`[mimi] Transcribed: "${text}"`);

        // In attentive mode (post-response window), skip wake word check
        let message;
        if (attentive) {
          console.log(`[mimi] Attentive mode, skipping wake word check`);
          // Still strip wake word if present, but don't require it
          const wake = this.checkWakeWord(text);
          message = wake.matched ? (wake.command || text) : text;
        } else {
          const wake = this.checkWakeWord(text);
          if (!wake.matched) {
            return res.json({ status: 'ignored', reason: 'no-wake-word', text });
          }
          message = wake.command || text;
        }
        console.log(`[mimi] Wake word matched, command: "${message}"`);

        // Get or initialize conversation
        const convKey = sessionId || `mimi-${req.episteryClient?.address || 'anon'}-${Date.now()}`;
        if (!this.conversations.has(convKey)) {
          this.conversations.set(convKey, []);
        }
        const history = this.conversations.get(convKey);
        history.push({ role: 'user', content: message });

        // Generate continuation token
        const token = `${convKey}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const pendingState = {
          sessionId: convKey,
          completed: false,
          progress: 'Starting...',
          response: null,
          error: null,
          isVoice: true
        };
        this.pendingRequests.set(token, pendingState);

        const reqCtx = this.captureRequestContext(req);
        this.processMessage(reqCtx, history, convKey, pendingState, token);

        res.json({
          status: 'working',
          continuationToken: token,
          sessionId: convKey,
          text: message,
          progress: pendingState.progress
        });
      } catch (error) {
        console.error('[mimi] Audio endpoint error:', error);
        res.status(500).json({ status: 'error', message: error.message });
      }
    });

    // Message endpoint - Claude relay with continuation tokens
    router.post('/message', async (req, res) => {
      try {
        const permissions = await this.getPermissions(req);
        if (!permissions.read) {
          return res.status(403).json({ error: 'Permission required' });
        }

        const { message, sessionId, continuationToken } = req.body;

        // Handle continuation polling
        if (continuationToken) {
          const pending = this.pendingRequests.get(continuationToken);
          if (!pending) {
            return res.status(404).json({ status: 'error', message: 'Token not found or expired' });
          }
          if (pending.completed) {
            this.pendingRequests.delete(continuationToken);
            if (pending.error) {
              return res.json({ status: 'error', message: pending.error, sessionId: pending.sessionId });
            }
            return res.json({ status: 'success', response: pending.response, sessionId: pending.sessionId, completed: true });
          }
          return res.json({ status: 'working', continuationToken, progress: pending.progress || 'Processing...' });
        }

        if (!message) {
          return res.status(400).json({ status: 'error', message: 'Message is required' });
        }

        // Get or initialize conversation
        const convKey = sessionId || `mimi-${req.episteryClient?.address || 'anon'}-${Date.now()}`;
        if (!this.conversations.has(convKey)) {
          this.conversations.set(convKey, []);
        }
        const history = this.conversations.get(convKey);

        // Add user message
        history.push({ role: 'user', content: message });

        // Generate continuation token
        const token = `${convKey}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const pendingState = {
          sessionId: convKey,
          completed: false,
          progress: 'Starting...',
          response: null,
          error: null
        };
        this.pendingRequests.set(token, pendingState);

        // Capture request context before async processing
        const reqCtx = this.captureRequestContext(req);
        console.log(`[mimi] User address: ${reqCtx.episteryClient?.address || 'none'}, cookie present: ${!!reqCtx.cookie}`);

        // Start async processing
        this.processMessage(reqCtx, history, convKey, pendingState, token);

        // Return immediately with token
        res.json({
          status: 'working',
          continuationToken: token,
          sessionId: convKey,
          progress: pendingState.progress
        });
      } catch (error) {
        console.error('[mimi] Message error:', error);
        res.status(500).json({ status: 'error', message: error.message });
      }
    });
  }

  /**
   * Process a message asynchronously (tool-calling loop)
   * Same pattern as pro-research lines 394-482
   */
  async processMessage(reqCtx, history, convKey, pendingState, token) {
    try {
      const client = this.getAnthropicClient(reqCtx.hostname);
      const tools = this.getTools();

      const domain = reqCtx.hostname || 'localhost';
      const userAddress = reqCtx.episteryClient?.address || 'unknown';

      const systemPrompt = `You are Mimi, a general-purpose voice assistant on the epistery host at ${domain}.
You can answer any question — weather, trivia, math, advice, anything.
Use web_search for current information like weather, news, sports, or prices.
You also have epistery tools for wiki pages, archives, messages, and identity.
Additional tools may be available from installed agents — use them when relevant.

Your spoken replies are read aloud via TTS. Be conversational, like talking to a friend.
No bullet points, no markdown, no lists, no headers in your spoken replies — just plain sentences.
CRITICAL: NEVER stop mid-sentence. NEVER cut off your answer. Always finish your complete thought
with the actual answer the user asked for. If someone asks a question, you MUST give the full answer,
not trail off. A complete short answer is better than a long one that stops mid-thought.
However, when writing content to the wiki or message board via tools, write naturally with
full markdown, proper formatting, and as much detail as appropriate for that medium.
User wallet address: ${userAddress}`;

      // Add built-in web search alongside epistery tools
      const allTools = [
        { type: 'web_search_20250305', name: 'web_search', max_uses: 3 },
        ...tools
      ];

      pendingState.progress = 'Sending to Claude...';
      let claudeMessage = await this.callClaudeWithRetry(client, {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 4096,
        system: systemPrompt,
        tools: allTools,
        messages: history
      });

      // Tool-calling loop
      let toolCallCount = 0;
      while (claudeMessage.stop_reason === 'tool_use') {
        toolCallCount++;
        const toolUse = claudeMessage.content.find(block => block.type === 'tool_use');

        // web_search is handled server-side by Anthropic — no proxy needed
        if (toolUse && toolUse.name !== 'web_search') {
          pendingState.progress = `Using ${toolUse.name} (${toolCallCount})...`;
          console.log(`[mimi] Tool: ${toolUse.name}`, toolUse.input);

          const toolResult = await this.proxyToolCall(toolUse.name, toolUse.input, reqCtx);
          console.log(`[mimi] Tool result:`, JSON.stringify(toolResult).substring(0, 200));

          // Add assistant's tool use to history
          history.push({ role: 'assistant', content: claudeMessage.content });

          // Add tool result
          history.push({
            role: 'user',
            content: [{
              type: 'tool_result',
              tool_use_id: toolUse.id,
              content: JSON.stringify(toolResult)
            }]
          });
        } else {
          // Server tool (web_search) — results are already in the response content
          pendingState.progress = `Searching the web (${toolCallCount})...`;
          console.log(`[mimi] Server tool: web_search`);
          history.push({ role: 'assistant', content: claudeMessage.content });
        }

        // Continue conversation
        pendingState.progress = `Processing results (${toolCallCount})...`;
        claudeMessage = await this.callClaudeWithRetry(client, {
          model: 'claude-sonnet-4-20250514',
          max_tokens: 4096,
          system: systemPrompt,
          tools: allTools,
          messages: history
        });
      }

      // Extract final text
      const textContent = claudeMessage.content.find(block => block.type === 'text');
      const assistantContent = textContent ? textContent.text : 'No response generated';

      // Add to history
      history.push({ role: 'assistant', content: claudeMessage.content });

      // Trim and condense history
      this.trimHistory(convKey, history);

      // Generate TTS audio for voice requests
      let audioUrl = null;
      if (pendingState.isVoice) {
        try {
          pendingState.progress = 'Generating speech...';
          const audioId = await this.generateTTS(assistantContent, domain);
          audioUrl = `audio/${audioId}`;
        } catch (err) {
          console.error('[mimi] TTS generation failed:', err.message);
        }
      }

      pendingState.completed = true;
      pendingState.response = {
        role: 'assistant',
        content: assistantContent,
        timestamp: new Date().toISOString(),
        ...(audioUrl && { audioUrl })
      };
      pendingState.progress = 'Complete';

      // Cleanup after 5 minutes
      setTimeout(() => { this.pendingRequests.delete(token); }, 5 * 60 * 1000);
    } catch (error) {
      console.error('[mimi] Processing error:', error);
      pendingState.completed = true;
      pendingState.error = error.message;
    }
  }

  /**
   * Check permissions (same pattern as wiki)
   */
  async getPermissions(req) {
    const result = { admin: false, edit: false, read: false };

    if (!req.episteryClient || !req.domainAcl) {
      return result;
    }

    try {
      const access = await req.domainAcl.checkAgentAccess('@epistery/mimi', req.episteryClient.address, req.hostname);
      result.admin = access.level >= 3;
      result.edit = access.level >= 2;
      result.read = access.level >= 1;
      return result;
    } catch (error) {
      console.error('[mimi] ACL check error:', error);
    }
    return result;
  }

  async cleanup() {
    this.conversations.clear();
    this.pendingRequests.clear();
  }
}
