import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { existsSync } from 'fs';
import Anthropic from '@anthropic-ai/sdk';
import { Config } from 'epistery';

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
    this.internalPort = null;
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
   * Proxy a tool call to an internal epistery agent via HTTP
   * Same pattern as MCPTools.mjs createHandlers()
   */
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
          const data = await api(`/agent/epistery/wiki/${encodeURIComponent(args.title)}`, {
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

        default:
          return { error: `Unknown tool: ${toolName}` };
      }
    } catch (e) {
      return { error: e.message };
    }
  }

  /**
   * Tool definitions for Claude (matches MCPTools.mjs TOOLS)
   */
  getTools() {
    return [
      {
        name: 'wiki_read',
        description: 'Read a wiki page by title. Returns the page content in markdown.',
        input_schema: {
          type: 'object',
          properties: { page: { type: 'string', description: 'Page title (e.g., "Home")' } },
          required: ['page']
        }
      },
      {
        name: 'wiki_write',
        description: 'Create or update a wiki page. Content should be markdown.',
        input_schema: {
          type: 'object',
          properties: {
            title: { type: 'string', description: 'Page title' },
            content: { type: 'string', description: 'Page content (markdown)' }
          },
          required: ['title', 'content']
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

    // Admin: check if key is set
    router.get('/admin/key', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      const cfg = new Config();
      cfg.setPath(req.hostname || 'localhost');
      const hasKey = !!(cfg.data?.anthropic_api_key || cfg.data?.claude?.anthropicKey);
      res.json({ hasKey });
    });

    // Admin: save key
    router.post('/admin/key', async (req, res) => {
      const permissions = await this.getPermissions(req);
      if (!permissions.admin) {
        return res.status(403).json({ error: 'Admin access required' });
      }
      const { key } = req.body;
      if (!key || !key.startsWith('sk-')) {
        return res.status(400).json({ error: 'Invalid API key' });
      }
      const domain = req.hostname || 'localhost';
      const cfg = new Config();
      cfg.setPath(domain);
      if (!cfg.data.claude) cfg.data.claude = {};
      cfg.data.claude.anthropicKey = key;
      cfg.save();
      // Reset cached client so next request picks up new key
      this.anthropic = null;
      res.json({ success: true });
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

      const systemPrompt = `You are Mimi, a voice assistant for the epistery host at ${domain}.
You help users interact with their epistery data through natural conversation.
You have access to tools for reading and writing wiki pages, managing archives,
posting messages, and checking identity.

The user is speaking to you through a microphone - keep responses concise and conversational.
Avoid long lists or heavy formatting since responses will be spoken aloud.
User wallet address: ${userAddress}`;

      pendingState.progress = 'Sending to Claude...';
      let claudeMessage = await this.callClaudeWithRetry(client, {
        model: 'claude-sonnet-4-20250514',
        max_tokens: 4096,
        system: systemPrompt,
        tools,
        messages: history
      });

      // Tool-calling loop (same as pro-research lines 406-445)
      let toolCallCount = 0;
      while (claudeMessage.stop_reason === 'tool_use') {
        toolCallCount++;
        const toolUse = claudeMessage.content.find(block => block.type === 'tool_use');

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

        // Continue conversation
        pendingState.progress = `Processing results (${toolCallCount})...`;
        claudeMessage = await this.callClaudeWithRetry(client, {
          model: 'claude-sonnet-4-20250514',
          max_tokens: 4096,
          system: systemPrompt,
          tools,
          messages: history
        });
      }

      // Extract final text
      const textContent = claudeMessage.content.find(block => block.type === 'text');
      const assistantContent = textContent ? textContent.text : 'No response generated';

      // Add to history
      history.push({ role: 'assistant', content: claudeMessage.content });

      // Trim history to 20 messages
      if (history.length > 20) {
        this.conversations.set(convKey, history.slice(-20));
      }

      pendingState.completed = true;
      pendingState.response = {
        role: 'assistant',
        content: assistantContent,
        timestamp: new Date().toISOString()
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
