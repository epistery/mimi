/**
 * MarkUp - Markdown renderer for Mimi chat
 *
 * Based on message-board MarkUp with WikiWord support from wiki MarkUp.
 * WikiWords (CamelCase) auto-link to wiki documents.
 */

// WikiWord processor - converts CamelCase words to wiki links
class WikiWord {
  constructor(basePath = '/agent/epistery/wiki') {
    this.basePath = basePath;
  }

  process(body, currentDocId = '') {
    let lines = body.split('\n');
    let newLines = [];
    let skipping = false;
    let fenceChar = null;
    let fenceLength = 0;

    for (let line of lines) {
      const fenceMatch = line.match(/^([`~]){3,}/);

      if (fenceMatch) {
        if (!skipping) {
          skipping = true;
          fenceChar = fenceMatch[1];
          fenceLength = fenceMatch[0].length;
        } else if (fenceMatch[1] === fenceChar && fenceMatch[0].length >= fenceLength) {
          skipping = false;
          fenceChar = null;
          fenceLength = 0;
        }
        newLines.push(line);
        continue;
      }

      if (!skipping) {
        // Bracketed words become wiki links
        line = line.replace(/\[([A-Za-z0-9_]+)\]/g, (match, word) => {
          return `[${word}](${this.basePath}/${word})`;
        });
        // CamelCase WikiWords become wiki links
        line = line.replace(/(^|[^a-zA-Z0-9:_\-=.["'}{\\/[])([!A-Z][A-Z0-9]*[a-z][a-z0-9_]*[A-Z][A-Za-z0-9_]*)(?![^\[]*\])/g, (match, pre, word) => {
          if (word.charAt(0) === '!') return pre + (word.slice(1));
          else if (pre === "W:") return `[${word}](wikipedia.org?s=${word})`;
          else if (pre === "G:") return `[${word}](google.com?s=${word})`;
          else return `${pre}[${word}](${this.basePath}/${word})`;
        });
      }
      newLines.push(line);
    }
    return newLines.join('\n');
  }
}

export default class MarkUp {
  constructor(options = {}) {
    this.wikiWord = new WikiWord(options.basePath || '/agent/epistery/wiki');
    this.marked = null;
  }

  async init() {
    if (!this.marked) {
      const { marked } = await import('https://cdn.jsdelivr.net/npm/marked@12.0.0/lib/marked.esm.js');
      this.marked = marked;

      this.marked.setOptions({
        gfm: true,
        breaks: true,
        async: false
      });
    }
  }

  render(body) {
    if (!this.marked) {
      console.warn('[MarkUp] Not initialized, returning escaped text');
      return this.escapeHtml(body);
    }

    body = this.sanitize(body);
    const wordified = this.wikiWord.process(body);
    const html = this.marked.parse(wordified);
    return this.balanceHtml(html);
  }

  balanceHtml(html) {
    if (typeof document === 'undefined') return html;
    const tmp = document.createElement('div');
    tmp.innerHTML = html;
    return tmp.innerHTML;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  sanitize(text) {
    text = text.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
    text = text.replace(/<[^>]*>/g, (tag) => {
      tag = tag.replace(/\son\w+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, '');
      tag = tag.replace(/\b(href|src|xlink:href)\s*=\s*("javascript:[^"]*"|'javascript:[^']*'|javascript:[^\s>]+)/gi, '$1=""');
      return tag;
    });
    return text;
  }
}

if (typeof window !== 'undefined') {
  window.MarkUp = MarkUp;
}
