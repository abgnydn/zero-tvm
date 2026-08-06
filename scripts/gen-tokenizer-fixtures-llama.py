"""Generate reference fixtures for the byte-level BPE tokenizer running the
Llama-3 pipeline (src/zero-tvm/tokenizer-bpe.ts) — Split regex with digit
runs of ≤3, NO normalizer — plus the Llama-3 chat template mirror
(buildLlama3ChatPrompt).

Ground truth is mlx_lm's OWN tokenizer (the exact wrapper the validation
pipeline diffs against), loaded from the local checkpoint mirror. Emits:

    tests/tokenizer/tokenizer-llama3.json   the app's tokenizer.json, minified
                                            (field-complete; Llama 3.2 license)
    tests/tokenizer/fixtures-llama3.json    {"cases": [{text, ids, decoded}],
                                             "templates": [{messages,
                                             date_string, ids}]}

Template fixtures pin date_string (the template's strftime_now output is
otherwise run-date-dependent); the script also renders the TS mirror string in
Python and fails loudly if apply_chat_template drifts from it.

Both output files are committed so `npm run test:unit` is fully offline.
Re-run (needs the checkpoint mirror + the shared ML env):

    hf download mlx-community/Llama-3.2-1B-Instruct-4bit \
        --local-dir .weights-local/Llama-3.2-1B-Instruct-4bit
    cd ~/dev/ml-research && uv run python \
        ~/dev/zero-tvm/scripts/gen-tokenizer-fixtures-llama.py
"""

import json
import pathlib

from mlx_lm.utils import load_tokenizer

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / ".weights-local/Llama-3.2-1B-Instruct-4bit"
OUT_DIR = ROOT / "tests/tokenizer"

DATE_STRING = "06 Aug 2026"  # pinned — the fixture must not depend on the run date

# Mirror of buildLlama3ChatPrompt() in src/zero-tvm/tokenizer-bpe.ts (the
# no-tools template path): the system block ALWAYS renders, carrying the
# knowledge-cutoff and date lines; message contents are |trim'ed.
def chat_prompt(messages, date_string):
    rest = list(messages)
    system = ""
    if rest and rest[0]["role"] == "system":
        system = rest[0]["content"].strip()
        rest = rest[1:]
    text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    text += f"Cutting Knowledge Date: December 2023\nToday Date: {date_string}\n\n"
    text += f"{system}<|eot_id|>"
    for m in rest:
        text += f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content'].strip()}<|eot_id|>"
    text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return text


TEMPLATE_MESSAGES = [
    [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    # No system message — the template still emits the date system block.
    [{"role": "user", "content": "Write a haiku about GPUs."}],
    # Multi-turn + trim-relevant whitespace (the |trim filter must fire).
    [
        {"role": "user", "content": "  Write a haiku about GPUs.\n"},
        {"role": "assistant", "content": "Warm silicon hums"},
        {"role": "user", "content": "Continue it"},
    ],
]

CASES = [
    # Plain English + contractions (the (?i:...) branch)
    "Hello world",
    "The quick brown fox jumps over the lazy dog.",
    "it's John's dog and we're sure they've left, I'll go, he'd agree, don't",
    "DON'T SHOUT, I'M HERE, YOU'RE LOUD",
    # Numbers — \p{N}{1,3} splits digit RUNS of up to 3 (unlike Qwen3's single
    # digits); these cases are the whole reason the regex comes from the JSON.
    "pi is 3.14159 and the answer is 42",
    "1234567890",
    "year 2026, price $1,234.56 or 1.000.000₺",
    "Today Date: 06 Aug 2026",
    # Whitespace branches
    " leading space",
    "trailing space ",
    "multiple   spaces   between",
    "line one\nline two",
    "paragraph one\n\nparagraph two",
    "def f(x):\n    return x * 2\n",
    "\t\n mixed \r\n endings",
    # Multi-byte
    "emoji 😀 test",
    "rockets 🚀🔥 and flags 🇹🇷",
    "中文分词测试",
    "Türkçe karakterler: ğüşıöçİĞÜŞÖÇ",
    # NO normalizer: decomposed accents stay decomposed (unlike Qwen3's NFC)
    "café déjà vu",                          # precomposed
    "cafe\u0301 de\u0301ja\u0300 vu",       # decomposed — must encode differently
    # Punctuation runs
    "wait... what?!?! (really); [yes] {no} <maybe>",
    # Specials embedded in text
    "<|begin_of_text|>",
    "before <|eot_id|> after",
    "<|start_header_id|>assistant<|end_header_id|>\n\n",
    # Edge sizes
    "a",
    " ",
    "",
    # The exact template strings the app feeds encode()
    chat_prompt(TEMPLATE_MESSAGES[0], DATE_STRING),
    chat_prompt(TEMPLATE_MESSAGES[1], DATE_STRING),
    chat_prompt(TEMPLATE_MESSAGES[2], DATE_STRING),
]


def main():
    tok = load_tokenizer(MODEL_DIR)

    # Pin the TS mirror against the real chat_template before trusting it.
    for messages in TEMPLATE_MESSAGES:
        expected = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, date_string=DATE_STRING
        )
        ours = chat_prompt(messages, DATE_STRING)
        if ours != expected:
            raise SystemExit(
                f"template mirror drift!\n mirror: {ours!r}\n jinja:  {expected!r}"
            )

    templates = [
        {
            "messages": messages,
            "date_string": DATE_STRING,
            "ids": tok.apply_chat_template(
                messages, add_generation_prompt=True, date_string=DATE_STRING
            ),
        }
        for messages in TEMPLATE_MESSAGES
    ]

    cases = []
    for text in CASES:
        ids = tok.encode(text, add_special_tokens=False)
        # clean_up_tokenization_spaces=False: the config's legacy True would
        # munge " ." → "." in decode; tokenizer-bpe.ts implements the pure
        # ByteLevel decoder (concat + byte-map + UTF-8), so pin that.
        decoded = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        cases.append({"text": text, "ids": ids, "decoded": decoded})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer_json = json.loads((MODEL_DIR / "tokenizer.json").read_text())
    (OUT_DIR / "tokenizer-llama3.json").write_text(
        json.dumps(tokenizer_json, ensure_ascii=False, separators=(",", ":"))
    )
    (OUT_DIR / "fixtures-llama3.json").write_text(
        json.dumps({"cases": cases, "templates": templates}, ensure_ascii=False, indent=1)
    )
    print(f"wrote {len(cases)} cases + {len(templates)} template fixtures")
    print(f"tokenizer-llama3.json: {(OUT_DIR / 'tokenizer-llama3.json').stat().st_size / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
