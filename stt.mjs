import OpenAI from 'openai';
import { Config } from 'epistery';
import { execFile } from 'child_process';
import { writeFile, readFile, unlink } from 'fs/promises';
import { existsSync } from 'fs';
import { join } from 'path';
import { randomBytes } from 'crypto';
import { tmpdir } from 'os';

/**
 * Whisper API provider — sends audio to OpenAI Whisper, returns text
 */
class WhisperAPIProvider {
  constructor(apiKey) {
    this.client = new OpenAI({ apiKey });
  }

  async transcribe(audioBuffer, options = {}) {
    const blob = new Blob([audioBuffer], { type: 'audio/wav' });
    const file = new File([blob], 'audio.wav', { type: 'audio/wav' });
    try {
      const result = await this.client.audio.transcriptions.create({
        model: 'whisper-1',
        file,
        language: options.language || 'en',
        response_format: 'text'
      });
      console.log('[mimi-stt] OpenAI Whisper API transcription');
      return result.trim();
    } catch (err) {
      console.error('[mimi-stt] Whisper API error:', err.status, err.message, err.code);
      throw err;
    }
  }
}

/**
 * Local whisper.cpp provider — shells out to whisper-cli binary
 */
class LocalWhisperProvider {
  constructor({ binaryPath, modelPath, threads = 4 }) {
    this.binaryPath = binaryPath;
    this.modelPath = modelPath;
    this.threads = threads;
  }

  async transcribe(audioBuffer, options = {}) {
    const id = randomBytes(8).toString('hex');
    const tmpFile = join(tmpdir(), `mimi-stt-${id}.wav`);
    const outputFile = `${tmpFile}.txt`;

    try {
      await writeFile(tmpFile, audioBuffer);

      const args = [
        '-m', this.modelPath,
        '-f', tmpFile,
        '-nt',
        '--output-format', 'txt',
        '-t', String(this.threads),
        '-l', options.language || 'en'
      ];

      await new Promise((resolve, reject) => {
        execFile(this.binaryPath, args, { timeout: 30000 }, (err, stdout, stderr) => {
          if (err) {
            console.error('[mimi-stt] whisper-cli stderr:', stderr);
            return reject(new Error(`whisper-cli failed: ${err.message}`));
          }
          resolve(stdout);
        });
      });

      // whisper-cli --output-format txt writes to {inputfile}.txt
      if (!existsSync(outputFile)) {
        throw new Error('whisper-cli did not produce output file');
      }

      const text = await readFile(outputFile, 'utf-8');
      console.log('[mimi-stt] Local whisper transcription');
      return text.trim();
    } finally {
      // Cleanup temp files
      try { await unlink(tmpFile); } catch (_) {}
      try { await unlink(outputFile); } catch (_) {}
    }
  }
}

/**
 * Factory — auto-selects local whisper.cpp when installed, falls back to OpenAI API
 * Config ini sections: [whisper] binary=... model=... threads=4
 *                      [openai] apikey=sk-...
 */
export function createSTTProvider(domain) {
  const cfg = new Config();
  cfg.setPath(domain);

  // Prefer local whisper if configured and files exist
  const binaryPath = cfg.data?.whisper?.binary;
  const modelPath = cfg.data?.whisper?.model;
  if (binaryPath && modelPath && existsSync(binaryPath) && existsSync(modelPath)) {
    const threads = parseInt(cfg.data.whisper.threads) || 4;
    console.log(`[mimi-stt] Using local whisper: ${binaryPath}`);
    return new LocalWhisperProvider({ binaryPath, modelPath, threads });
  }

  // Fall back to OpenAI API
  const apiKey = cfg.data?.openai?.apikey || process.env.OPENAI_API_KEY;
  if (apiKey) {
    console.log('[mimi-stt] Using OpenAI Whisper API');
    return new WhisperAPIProvider(apiKey);
  }

  throw new Error('No STT provider available. Install local whisper or set [openai] apikey in domain config.');
}
