# Long-coding generation test (PS4 Tetris prompt)

A single-stream test of long code generation. One chat request asks for a full HTML5 Tetris game. We record speed
(TTFT, end-to-end, peak 1 s / 5 s tok/s, thinking vs answer rates, speculative acceptance) and check whether the
game actually works.

## What we found

- **Speed:** MiMo-V2.6-Flash-RL with DFlash k=7 on one GB300 (single stream, C1) sustained 544-545 tok/s end to end
  over 62K-79K generated tokens. Peaks were 850-857 tok/s over 1 s windows. TTFT was 68-70 ms.
- **MiMo-V2.6-Pro (v6 lane: sidecar + Grace, DFlash k=3)** ran the same prompt at 45.6 tok/s end to end (peak 62.3)
  over 91K tokens. That is 33 minutes, about 30 of them thinking. It accepted 2.03 tokens per draft in thinking and 2.70
  in the HTML answer. The Pro lane itself is unqualified; see [../mimo-v2.6-pro/](../mimo-v2.6-pro/README.md).
- **Thinking vs answer:** about 80-86% of the tokens are thinking, generated at 519-525 tok/s. The HTML answer
  runs faster (673-726 tok/s) because DFlash accepts more of it: 5.56 accepted tokens per draft vs 3.77 in
  thinking.
- **Quality: neither model produced a playable game.** A headless-browser smoke test
  ([tools/smoke_tetris.cjs](tools/smoke_tetris.cjs)) confirms both failures. The Flash baseline (run2) has every
  requested feature, but it does not run. One misplaced brace in a
  stray keydown listener makes the whole `<script>` a syntax error. With the brace fixed, that same listener
  would swallow Space (hard drop). `tetris-playable.html` is a hand-fixed copy that deletes that one line.
  The Pro game parses and draws its interface, but its render function throws on the first frame. After that
  the board never shows a piece, and pieces don't fall on their own.
- **The answer cap must apply to the answer only.** When the 100,000-character cap covered thinking too
  (run1), the model spent it all thinking and never answered.

These runs are **unqualified**: the run files don't record the server image, and no output here passed a
functional check.

## Headline: MiMo-V2.6-Flash-RL, DFlash k=7, one GB300, 2026-09-25

All values are single-stream, so per-user and aggregate tok/s are the same number. Source:
`runs/<run>/run.json`. The thinking/answer split is derived from the per-chunk token timeline in that file; see
[tools/make_results_jsonl.py](tools/make_results_jsonl.py).

| Metric | mimo-dflash7-run1 | mimo-dflash7-run2 (baseline) | mimo-dflash7-perpos |
|---|--:|--:|--:|
| Character cap (100,000) applied to | thinking + answer | answer only | answer only |
| Finish | client cap hit | server stop | server stop |
| Completion tokens | 34,737 | 61,668 | 79,139 |
| Wall time (s) | 65.39 | 113.28 | 145.12 |
| TTFT (ms) | 70 | 68 | 68 |
| End-to-end tok/s | 531.2 | 544.4 | 545.3 |
| Decode tok/s (after first token) | 531.7 | 544.7 | 545.5 |
| Peak tok/s, 1 s window | 857.3 | 853.5 | 850.2 |
| Peak tok/s, 5 s window | 702.2 | 742.0 | 766.9 |
| Answer starts at (s) | never | 94.68 | 130.34 |
| Thinking tokens / tok/s | 34,737 / 531.2 | 49,147 / 519.1 | 68,402 / 524.8 |
| Answer tokens / tok/s | 0 / - | 12,521 / 673.0 | 10,737 / 726.2 |
| Thinking / answer characters | 100,004 / 0 | 149,126 / 32,123 | 213,718 / 26,434 |
| Spec drafts | 7,364 | 12,689 | 15,972 |
| Accepted tokens per draft | 3.72 | 3.86 | 3.96 |
| Accept rate (accepted / drafted) | 53.1% | 55.1% | 56.5% |

"Accepted tokens per draft" excludes the bonus token, so tokens per engine step is this value plus 1.

## MiMo-V2.6-Pro, same test (v6 lane, 2026-09-25)

Server: [`../mimo-v2.6-pro/`](../mimo-v2.6-pro/README.md) v6. Experts are split across GB300 HBM, the RTX PRO 6000
sidecar and Grace, with TRT-LLM banks, an FP8 KV cache and DFlash k=3. The run is single-stream (C1). Source:
`runs/mimo-pro-dflash3/run.json`.

| Metric | mimo-pro-dflash3 |
|---|--:|
| Completion tokens | 91,453 |
| Wall time (s) | 2,007.1 |
| TTFT (ms) | 2,173 |
| End-to-end tok/s | 45.6 |
| Peak tok/s, 1 s / 5 s window | 62.3 / 56.8 |
| Answer starts at (s) | 1,770.6 |
| Thinking tokens / tok/s | 79,117 / 44.7 |
| Answer tokens / tok/s | 12,336 / 52.2 |
| Accepted tokens per draft: all / thinking / answer | 2.10 / 2.03 / 2.70 (k=3) |
| Accept rate (accepted / drafted) | 70.0% |


## Per-position acceptance (mimo-dflash7-perpos)

The probability that a draft's accepted length reaches position i, from vLLM's
`spec_decode_num_accepted_tokens_per_pos` counters. The run is split at the first answer token. Source:
`runs/mimo-dflash7-perpos/run.json` (`spec_all`, `spec_thinking`, `spec_answer`).

| Position i | All (15,972 drafts) | Thinking (14,336) | Answer (1,636) |
|--:|--:|--:|--:|
| 1 | 0.837 | 0.823 | 0.955 |
| 2 | 0.706 | 0.683 | 0.905 |
| 3 | 0.610 | 0.581 | 0.861 |
| 4 | 0.536 | 0.505 | 0.812 |
| 5 | 0.475 | 0.442 | 0.758 |
| 6 | 0.421 | 0.391 | 0.680 |
| 7 | 0.373 | 0.349 | 0.591 |
| Accepted tokens per draft | 3.96 | 3.77 | 5.56 |

Even at position 7, 37% of drafts are still accepted overall (59% in the HTML answer), so k=7 is not too high
for this workload. The projected effect of other k values is in
[../mimo-v2.6-flash/DETAILS.md](../mimo-v2.6-flash/DETAILS.md#dflash-k-analysis).

## Quality verdict

- **run1:** no answer. `content.md` is empty. The client cap counted thinking characters and stopped at 100,004.
- **run2 (baseline):** `tetris.html` is the ```` ```html ```` block from `content.md`, byte for byte.
  - The session notes say all requested features are present, but a keyword check alone missed the fatal bug.
  - The whole game is one `<script>` (lines 124-487). Line 454, inside `startGame()`, reads:

    ```js
    document.body.addEventListener('keydown',function onFirst(e){if(e.code==='Space'||e.code==='Enter'){e.preventDefault();e.stopPropagation()},{once:false})}
    ```

    The `}` that should close `onFirst` sits at the end of the line instead of before `,{once:false}`. The `}`
    after `stopPropagation()` therefore closes only the `if` block, and the `,` that follows is a syntax error,
    so none of the script runs.
  - Even with the brace moved, the listener would call `stopPropagation()` on every Space and Enter keydown at
    `<body>`. The game's own handler is on `document` (line 416; Space calls `hardDrop()` at line 426), so hard
    drop would never fire.
- **`tetris-playable.html`** is a hand-edited copy of run2's `tetris.html`. `diff` shows exactly one change: line 454
  is replaced by a lone `}` that closes `startGame()`. This removes the listener entirely and fixes both
  problems. The file is 153 bytes shorter (34,421 -> 34,268), and nothing else differs.
- **perpos run:** `tetris.html` has a closing `</html>` (`html_closed: true`). It was not syntax-checked or
  play-tested.
- **mimo-pro-dflash3:**
  - `node --check` passes on its one inline script, and every requested feature appears by keyword.
  - In the headless-browser smoke test, the interface draws: the HOLD panel, a five-piece NEXT queue and the
    stats.
  - `render()` throws `TypeError: Cannot read properties of undefined (reading '0')` on the first animation
    frame, at line 453, while reading `this.board[y][x]`. That stops the animation loop.
  - Key handlers still change the game state. After START, the playfield is empty
    ([screenshot](runs/mimo-pro-dflash3/screenshot-after-start.png)). Hard drops then stack 8 invisible pieces
    until GAME OVER (score 134, TIME 0:00; [screenshot](runs/mimo-pro-dflash3/screenshot-game-over.png),
    [smoke.json](runs/mimo-pro-dflash3/smoke.json)). The game is not playable.

The run2 and Pro verdicts come from both reading the code and running it. The smoke test loads each page in
headless Chromium, clicks START and presses game keys. It reports runtime errors and whether the HUD timer and
the main canvas changed. For run2 it reports the syntax error (`Unexpected token ','`) and
`startGame is not defined` ([smoke.json](runs/mimo-dflash7-run2/smoke.json)).

## The test

- **Prompt:** [tetris-ps4-prompt.txt](tetris-ps4-prompt.txt), sent verbatim with its typos ("stuninng", "Make it
  looks"). The only change is that trailing newlines are stripped. md5 `13c387efac6e989da51abbefe932b918`.
- **Request:** OpenAI `/v1/chat/completions`, streamed, `temperature` 1.0, `top_p` 0.95, thinking on
  (`chat_template_kwargs: {"enable_thinking": true}`). `stream_options.continuous_usage_stats` gives exact
  cumulative token counts per chunk.
- **Cap:** `max_chars` stops the stream client-side. With `cap_on=content`, it counts answer characters only, and
  thinking is bounded only by `max_tokens`. Use `cap_on=content`: the intended ~100K-character limit is on the answer.
- **Metrics** ([long-gen.py](long-gen.py)):
  - TTFT, end-to-end tok/s (tokens / wall time) and decode tok/s (after the first token).
  - Peak tok/s over sliding 1 s and 5 s windows.
  - When the answer starts, and spec-decode acceptance from `/metrics` deltas, split into thinking and answer
    with per-position acceptance.
  - It saves `reasoning.txt`, `content.md`, `run.json` (summary plus the per-chunk timeline) and `tetris.html`.

### How to run

Start the server with [../mimo-v2.6-flash/launch-mimo.sh](../mimo-v2.6-flash/launch-mimo.sh) (defaults:
DFlash k=7), then:

```
PORT=30006 MODEL_NAME=mimo-v26-flash python3 long-gen.py tetris-ps4-prompt.txt runs/<name> 100000 120000 content
```

The arguments are `<prompt> <out-dir> [max_chars] [max_tokens] [cap_on]`. The runner works with any
OpenAI-compatible server that exposes vLLM-style `/metrics`. Afterwards, run the browser smoke test. Neither a
keyword check nor `node --check` shows whether the game actually runs:

```
docker run --rm -u $(id -u):$(id -g) -v $PWD/runs/<name>:/data -v $PWD/tools/smoke_tetris.cjs:/data/smoke_tetris.cjs:ro \
  --entrypoint node minlag/mermaid-cli /data/smoke_tetris.cjs tetris.html
```

## Files

| Path | What |
|---|---|
| [long-gen.py](long-gen.py) | The runner. It is the only version kept. |
| [tetris-ps4-prompt.txt](tetris-ps4-prompt.txt) | The prompt, verbatim |
| `runs/mimo-dflash7-run1/` | Run 1 (cap on thinking + answer): `run.json`, `reasoning.txt`, empty `content.md` |
| `runs/mimo-dflash7-run2/` | Baseline: `run.json`, `reasoning.txt`, `content.md`, `tetris.html`, `tetris-playable.html` |
| `runs/mimo-dflash7-perpos/` | Run with per-position counters: `run.json`, `reasoning.txt`, `content.md`, `tetris.html` |
| `runs/mimo-pro-dflash3/` | MiMo-V2.6-Pro v6 run: `run.json`, `reasoning.txt`, `content.md`, `tetris.html`, `smoke.json`, two screenshots |
| [tools/smoke_tetris.cjs](tools/smoke_tetris.cjs) | Headless-browser smoke test (puppeteer in `minlag/mermaid-cli`) |
| [results.jsonl](results.jsonl), [tools/make_results_jsonl.py](tools/make_results_jsonl.py) | Machine-readable rows for every number above, and the script that builds them |

- **Where the runs came from:**
  - run1 and run2 came from `/home/jasonc/research/mimo26/longgen/run1` and `run2`.
  - perpos came from `/home/jasonc/research/longgen/runs/mimo-dflash7-perpos`.
  - All `reasoning.txt` files are under 1 MB (100-215 KB), so they are included.
- **Which runner produced which run.** `/home/jasonc/research/mimo26/long-gen.py` is a symlink to
  `/home/jasonc/research/longgen/long-gen.py`, so only one file existed. It was last edited at 15:43, after run1
  and run2 (finished 15:35 and 15:37) and just before the perpos run (finished 15:46).
  - run1 and run2 came from an earlier revision that was not kept. Their `run.json` has no `html_*` or `spec_*`
    phase fields, and run1's `stopped_by` string has no cap-mode suffix.
  - run2 must have capped the answer only, because its thinking (149,126 characters) is past the 100,000 cap.
- **Server:** MiMo-V2.6-Flash-RL, DFlash k=7, one GB300, TP1, started with `launch-mimo.sh`.
  - The run files don't record the vLLM image or the KV grouping.
  - The session notes name only "DFlash7, one GB300". nightly 29468dde was the image under test from 15:00 and
    became the launcher default at 15:16, so it is the most likely image.
