import OpenAI from 'openai';
import { Config } from 'epistery';

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
      return result.trim();
    } catch (err) {
      console.error('[mimi-stt] Whisper API error:', err.status, err.message, err.code);
      throw err;
    }
  }
}

/**
 * Stub for future local whisper.cpp integration
 */
class LocalWhisperProvider {
  async transcribe(audioBuffer, options = {}) {
    throw new Error('Local Whisper not yet implemented. Install whisper.cpp and configure path.');
  }
}

/**
 * Factory — reads OpenAI key from domain config
 * Config ini section: [openai] apikey=sk-...
 */
export function createSTTProvider(domain) {
  const cfg = new Config();
  cfg.setPath(domain);

  const apiKey = cfg.data?.openai?.apikey || process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error('No OpenAI API key configured. Set [openai] apikey in domain config or OPENAI_API_KEY env var.');
  }

  return new WhisperAPIProvider(apiKey);
}
