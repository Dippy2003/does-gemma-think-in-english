# data/

## `parallel_probes.csv`

Cloze-style parallel probes across Sinhala, Tamil and English, one unambiguous
answer per row. Columns: `id, sinhala, tamil, english, answer_si, answer_ta,
answer_en, category, verified, verifier_note`.

Categories: `factual`, `spatial`, `temporal`, `kinship`, `numeric`.

**Every row starts `verified=false`.** This is deliberate, not an oversight —
if the Sinhala/Tamil prompts are awkward or the parallel pairs do not mean the
same thing, every downstream logit-lens, probe, and patching result measures
translation error rather than the model. A native speaker must review each
row and flip `verified` to `true`; common mistranslation traps to watch for:

- literal calques that are grammatically valid but not what a native speaker
  would actually say
- kinship terms that don't map 1:1 (Sinhala/Tamil kinship vocabulary
  distinguishes paternal/maternal and older/younger in ways English does not)
- cloze blanks where more than one answer is equally valid in the target
  language even though the English original is unambiguous

`scripts/check_probes.py` hard-fails any experiment runner if verification
coverage is below 80%.

### Categories

| Category | What it tests | Example |
|---|---|---|
| `factual` | World knowledge (capitals, geography, science) | "The capital of France is ___" |
| `spatial` | Prepositions and relative position | "The book is ___ the table" |
| `temporal` | Calendar and time units | "The first day of the week is ___" |
| `kinship` | Family relation terms (asymmetric across languages) | "My father's brother is my ___" |
| `numeric` | Arithmetic, spelled-out number words | "Five plus five equals ___" |

## Fertility corpora

`fertility_corpus_{si,ta,en}.txt` — parallel plain-text corpora used only for
tokenizer fertility measurement (Phase 6), not for probing.
