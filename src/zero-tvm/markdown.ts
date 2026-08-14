/**
 * MINIMAL MARKDOWN RENDERER (chat page)
 *
 * Phi-3 emits headings, numbered/bulleted lists, fenced code blocks, inline
 * code and bold. A hand-rolled parser keeps the bundle small; it runs on each
 * streamed token via requestAnimationFrame (see chat.ts AiMsgHandle.render).
 */

export const ICON_COPY    = '<svg class="icon" viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg>'
export const ICON_CHECK   = '<svg class="icon" viewBox="0 0 24 24"><path d="M20 6L9 17l-5-5"/></svg>'
export const ICON_REFRESH = '<svg class="icon" viewBox="0 0 24 24"><path d="M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>'

/** Wire a copy button that shows a 1.5s "Copied" confirmation on success. */
export function wireCopyButton(btn: HTMLButtonElement, getText: () => string): void {
  btn.innerHTML = `${ICON_COPY}<span>Copy</span>`
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(getText())
      btn.classList.add('copied')
      btn.innerHTML = `${ICON_CHECK}<span>Copied</span>`
      setTimeout(() => {
        btn.classList.remove('copied')
        btn.innerHTML = `${ICON_COPY}<span>Copy</span>`
      }, 1500)
    } catch { /* no-op */ }
  })
}

function renderInline(text: string, into: Node): void {
  // Tokens: `code`, **bold**, *italic*, [text](url). Longest-match ordered.
  const re = /(```)|(`[^`]+`)|(\*\*[^*]+\*\*)|(\*[^*\n]+\*)|\[([^\]]+)\]\(([^)\s]+)\)/g
  let last = 0
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    if (m[1]) { continue /* ``` inline is unusual; leave it */ }
    if (m.index > last) into.appendChild(document.createTextNode(text.slice(last, m.index)))
    if (m[2]) {
      const c = document.createElement('code'); c.textContent = m[2].slice(1, -1); into.appendChild(c)
    } else if (m[3]) {
      const b = document.createElement('strong'); b.textContent = m[3].slice(2, -2); into.appendChild(b)
    } else if (m[4]) {
      const em = document.createElement('em'); em.textContent = m[4].slice(1, -1); into.appendChild(em)
    } else if (m[5] && m[6]) {
      const a = document.createElement('a')
      a.href = m[6]; a.textContent = m[5]
      a.target = '_blank'; a.rel = 'noopener noreferrer'
      into.appendChild(a)
    }
    last = m.index + m[0].length
  }
  if (last < text.length) into.appendChild(document.createTextNode(text.slice(last)))
}

function makeCodeBlock(code: string, lang: string): HTMLElement {
  const wrap = document.createElement('div')
  wrap.className = 'code-block'
  const head = document.createElement('div')
  head.className = 'code-head'
  const langEl = document.createElement('span')
  langEl.className = 'code-lang'
  langEl.textContent = lang || 'code'
  head.appendChild(langEl)
  const btn = document.createElement('button')
  btn.type = 'button'
  btn.className = 'code-copy'
  wireCopyButton(btn, () => code)
  head.appendChild(btn)
  wrap.appendChild(head)
  const pre = document.createElement('pre')
  const codeEl = document.createElement('code')
  if (lang) codeEl.className = `lang-${lang}`
  codeEl.textContent = code
  pre.appendChild(codeEl)
  wrap.appendChild(pre)
  return wrap
}

export function renderMarkdown(src: string): DocumentFragment {
  const frag = document.createDocumentFragment()
  const lines = src.split('\n')
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    // Skip pure-blank lines between blocks.
    if (line.trim() === '') { i++; continue }

    // Fenced code block
    const fence = /^\s*```(.*)$/.exec(line)
    if (fence) {
      const lang = fence[1].trim()
      i++
      const buf: string[] = []
      while (i < lines.length && !/^\s*```/.test(lines[i])) { buf.push(lines[i]); i++ }
      if (i < lines.length) i++ // consume closing fence
      frag.appendChild(makeCodeBlock(buf.join('\n'), lang))
      continue
    }

    // Horizontal rule
    if (/^\s*(---|\*\*\*|___)\s*$/.test(line)) {
      frag.appendChild(document.createElement('hr'))
      i++
      continue
    }

    // Heading
    const hm = /^(#{1,6})\s+(.*)$/.exec(line)
    if (hm) {
      const level = Math.min(6, hm[1].length)
      const h = document.createElement(`h${level}`) as HTMLElement
      renderInline(hm[2].trim(), h)
      frag.appendChild(h)
      i++
      continue
    }

    // Blockquote
    if (/^\s*>\s?/.test(line)) {
      const bq = document.createElement('blockquote')
      const buf: string[] = []
      while (i < lines.length && /^\s*>\s?/.test(lines[i])) {
        buf.push(lines[i].replace(/^\s*>\s?/, ''))
        i++
      }
      renderInline(buf.join(' '), bq)
      frag.appendChild(bq)
      continue
    }

    // Unordered list
    if (/^\s*[-*]\s+/.test(line)) {
      const ul = document.createElement('ul')
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        const li = document.createElement('li')
        renderInline(lines[i].replace(/^\s*[-*]\s+/, ''), li)
        ul.appendChild(li)
        i++
      }
      frag.appendChild(ul)
      continue
    }

    // Ordered list
    if (/^\s*\d+\.\s+/.test(line)) {
      const ol = document.createElement('ol')
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        const li = document.createElement('li')
        renderInline(lines[i].replace(/^\s*\d+\.\s+/, ''), li)
        ol.appendChild(li)
        i++
      }
      frag.appendChild(ol)
      continue
    }

    // Setext heading — a line of text underlined with = or -.
    //
    // Only ATX ('# Title') was handled, so a model that writes the other form
    // got its rule printed as body text: Llama-3.2-1B answering "name three
    // primary colours" rendered "Primary Colors ===============", because the
    // underline is not a block start and the paragraph below simply absorbed
    // it. Both forms are ordinary model output; which one appears is a
    // property of the checkpoint, not of the question.
    const under = i + 1 < lines.length ? lines[i + 1] : ''
    if (line.trim() !== '' && /^\s*(=+|-{2,})\s*$/.test(under)) {
      const h = document.createElement(under.includes('=') ? 'h1' : 'h2')
      renderInline(line.trim(), h)
      frag.appendChild(h)
      i += 2
      continue
    }

    // Paragraph — absorb contiguous non-blank, non-block lines.
    const para: string[] = []
    while (i < lines.length && lines[i].trim() !== '' &&
           !/^\s*```/.test(lines[i]) &&
           !/^#{1,6}\s+/.test(lines[i]) &&
           !/^\s*>\s?/.test(lines[i]) &&
           !/^\s*[-*]\s+/.test(lines[i]) &&
           !/^\s*\d+\.\s+/.test(lines[i]) &&
           // The line after this one underlines it, so this line opens a
           // setext heading and does not belong to the paragraph.
           !/^\s*(=+|-{2,})\s*$/.test(lines[i + 1] ?? '')) {
      para.push(lines[i])
      i++
    }
    if (para.length > 0) {
      const p = document.createElement('p')
      renderInline(para.join(' '), p)
      frag.appendChild(p)
    }
  }
  return frag
}
