"""Render a DEMO race (demo2.py or demo3.py on a Kaggle T4) as a 1080p show video in the website's style.

Usage: python demo_video.py RESULTS_DIR OUT.mp4 FONT_DIR
RESULTS_DIR holds the notebook's demo*-result.json, race-stock.json and race-ironmule-native.json;
FONT_DIR holds Geist.ttf and GeistMono.ttf (google/fonts: ofl/geist/Geist[wght].ttf,
ofl/geistmono/GeistMono[wght].ttf). Scene 1 replays each side's median answer to one question: a
chunk appears at the moment it arrived in the run, both sides starting when their request was
sent. Scene 2, when the run has one (DEMO3), replays the questions sent at the same moment, in
real time until IronMule is done and then fast-forwarded, labelled. Every number on screen comes
from the files.
"""
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

RESULTS, OUT, FONTS = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
W, H, FPS = 1920, 1080, 30
INK, CARD, INNER, LINE = (17, 19, 24), (21, 24, 30), (26, 30, 37), (36, 40, 50)
FROST, MUTED, ICE, ENERGY = (233, 243, 250), (141, 150, 165), (168, 215, 245), (255, 138, 24)


def font(name, size, weight):
    f = ImageFont.truetype(str(FONTS / name), size)
    f.set_variation_by_axes([weight])
    return f


SANS, MONO = "Geist.ttf", "GeistMono.ttf"
F_EYE, F_HEAD, F_Q, F_A = font(MONO, 22, 500), font(MONO, 28, 600), font(MONO, 24, 400), font(SANS, 42, 400)
F_CLOCK, F_RATE, F_SMALL = font(MONO, 64, 500), font(MONO, 26, 500), font(SANS, 22, 400)
F_MQ, F_MA = font(MONO, 20, 400), font(SANS, 26, 400)
NAMES = {"mlx-community/gemma-3-12b-it-4bit": "GEMMA 3 12B 4-BIT", "mlx-community/Qwen3-14B-4bit": "QWEN 3 14B 4-BIT"}
REPORT = json.loads(next(RESULTS.glob("demo*-result.json")).read_text())
DEMO4 = REPORT["schema"].startswith(("ironmule.demo4", "ironmule.demo5"))  # race-off/race-on files
MODEL = REPORT["label"].upper() if "label" in REPORT else NAMES[REPORT["model"][0]]
PROMPT = REPORT["config"]["single"]["prompt"] if "config" in REPORT else REPORT["prompt"]
HARDWARE = REPORT.get("hardware", "NVIDIA TESLA T4")
TAG = f"KAGGLE  ·  {HARDWARE}  ·  {MODEL}"


def visible(text):
    """What a reader sees: no think block (Qwen 3's direct mode starts with an empty one), no gpt-oss
    analysis channel, no markup tokens, no markdown stars."""
    text = re.sub(r"<think>.*?(</think>|$)", "", text, flags=re.S)
    text = re.sub(r"<\|channel>thought.*?(<channel\|>|$)", "", text, flags=re.S)  # Gemma 4's thought channel
    if "<|channel|>" in text:
        text = text.split("<|channel|>final<|message|>")[1] if "<|channel|>final<|message|>" in text else ""
    # A strict plan runs to max_tokens; what follows the model's own end of turn is not its answer.
    text = re.split(r"<turn\|>|<end_of_turn>|<eos>|<\|im_end\|>|<\|endoftext\|>|<\|return\|>|<\|end\|>|</s>", text)[0]
    text = re.sub(r"<\|[a-z_]+\|>", "", text)
    # Characters Geist has no glyph for: gpt-oss writes non-breaking hyphens and narrow spaces.
    return text.replace("**", "").replace("\u2011", "-").replace("\u202f", " ").replace("\u00a0", " ").strip()


def thought(text):
    """gpt-oss's analysis channel, shown dimmed until its final answer starts."""
    match = (re.search(r"<\|channel\|>analysis<\|message\|>(.*?)(<\|end\|>|$)", text, flags=re.S)
             or re.search(r"<\|channel>thought(.*?)(<channel\|>|$)", text, flags=re.S))
    return "thinking · " + match.group(1).strip() if match and match.group(1).strip() else ""


def text_at(row, t_ms):
    arrived = [c[1] for c in row["chunks"] if c[0] <= t_ms]
    if row.get("cumulative"):
        return arrived[-1] if arrived else ""
    return "".join(arrived)


def active(row, t_ms):
    """Tokens are still arriving: the last one is recent against this answer's own pace."""
    times = [c[0] for c in row["chunks"]]
    gaps = sorted(b - a for a, b in zip(times, times[1:])) or [0]
    past = [x for x in times if x <= t_ms]
    return bool(past) and t_ms - past[-1] <= 3 * gaps[len(gaps) // 2] + 50


def load(label):
    record = json.loads((RESULTS / f"race-{label}.json").read_text())
    runs = record["reps"]
    for run in runs:
        tokens = run["usage"]["completion_tokens"]
        run["tokens"] = tokens
        run["decode"] = (tokens - 1) / ((run["total_ms"] - run["first_ms"]) / 1000)
    if "multi_wall_ms" in record:
        # The Runtime's service TTFT counts from the start of that request's own prefill, and it reads the
        # prompts of a set one after another before decoding. On the set's clock each request therefore
        # starts when the prefills before it have finished; the last token must land on the set's wall.
        offset = 0.0
        for row in record["multi"]:
            row["first_ms"] += offset
            row["total_ms"] += offset
            row["chunks"] = [[t + offset, text] for t, text in row["chunks"]]
            offset = row["first_ms"]  # this prefill ended with the request's first token
        last = max(r["total_ms"] for r in record["multi"])
        assert abs(last - record["multi_wall_ms"]) < 0.02 * record["multi_wall_ms"], (label, last, record["multi_wall_ms"])
    shown = sorted(runs, key=lambda r: r["total_ms"])[len(runs) // 2]
    med = {k: statistics.median(r[k] for r in runs) for k in ("total_ms", "first_ms", "decode")}
    return {"runs": runs, "shown": shown, "median": med, "multi": record.get("multi")}


if DEMO4:
    STOCK, NATIVE = load("off"), load("on")
    PLAN = f"{REPORT['config']['dtype'] or 'exact'} plan"
    SIDES = ((STOCK, 60, ICE, "IRONMULE OFF", "reference path"), (NATIVE, 990, ENERGY, "IRONMULE ON", PLAN))
    CROWD_SUB = ("one request after another", f"{PLAN}, the requests grouped")
    if "sides" in REPORT:  # a run outside the Runtime names its own arms
        SIDES = tuple((side, x0, color, *REPORT["sides"][i]) for i, (side, x0, color, _, _) in enumerate(SIDES))
else:
    STOCK, NATIVE = load("stock"), load("ironmule-native")
    SIDES = ((STOCK, 60, ICE, "STOCK MLX", "IronMule off"), (NATIVE, 990, ENERGY, "IRONMULE ON", "native plan"))
    CROWD_SUB = ("IronMule off", "native plan")
MULTI = bool(STOCK["multi"] and NATIVE["multi"])
PW = 870


def canvas():
    image = Image.new("RGB", (W, H), INK)
    draw = ImageDraw.Draw(image)
    for x in range(12, W, 24):  # the template's raster: small ice dots on a 24 px grid
        for y in range(12, H, 24):
            draw.point((x, y), fill=(30, 36, 44))
    return image


def glow(image, xy, text, fnt, color, anchor="mm", radius=16):
    layer = Image.new("L", image.size, 0)
    ImageDraw.Draw(layer).text(xy, text, font=fnt, fill=255, anchor=anchor)
    image.paste(Image.new("RGB", image.size, color), (0, 0), layer.filter(ImageFilter.GaussianBlur(radius)).point(lambda a: a * 0.6))
    ImageDraw.Draw(image).text(xy, text, font=fnt, fill=color, anchor=anchor)


def eyebrow(draw, x, y, text):
    draw.rectangle([x, y - 4, x + 8, y + 4], fill=ENERGY)
    draw.text((x + 22, y), text, font=F_EYE, fill=MUTED, anchor="lm")


def wrap(draw, text, width, fnt=F_A):
    lines = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            trial = f"{current} {word}".strip()
            if current and draw.textlength(trial, font=fnt) > width:
                lines.append(current)
                current = word
            else:
                current = trial
        lines.append(current)
    return [line for line in lines if line] or [""]


def header(image, draw, x0, color, title, subtitle, seconds, done, rate):
    draw.rounded_rectangle([x0, 100, x0 + PW, 980], radius=24, fill=CARD, outline=color if done else LINE, width=2)
    draw.rectangle([x0 + 36, 146, x0 + 46, 156], fill=color)
    draw.text((x0 + 62, 151), title, font=F_HEAD, fill=color, anchor="lm")
    draw.text((x0 + 62, 188), subtitle, font=F_SMALL, fill=MUTED, anchor="lm")
    clock = f"{seconds:5.2f} s"
    if done:
        glow(image, (x0 + PW - 36, 160), clock, F_CLOCK, color, anchor="rm")
    else:
        draw.text((x0 + PW - 36, 160), clock, font=F_CLOCK, fill=FROST, anchor="rm")
    draw.text((x0 + PW - 36, 214), rate[0], font=F_RATE, fill=color if rate[1] else MUTED, anchor="rm")
    draw.line([(x0 + 36, 250), (x0 + PW - 36, 250)], fill=LINE, width=1)


def race(t_ms):
    image = canvas()
    draw = ImageDraw.Draw(image)
    eyebrow(draw, 60, 52, f"{TAG}  ·  ONE QUESTION  ·  REAL TIME")
    for side, x0, color, title, subtitle in SIDES:
        run = side["shown"]
        done = t_ms >= run["total_ms"]
        arrived = [c for c in run["chunks"] if c[0] <= t_ms]
        seconds = min(t_ms, run["total_ms"]) / 1000
        n = run["tokens"] if done else len(arrived)
        rate = (f"{n / seconds:4.1f} tok/s  ·  {n} tokens", True) if n and seconds else ("waiting for first token", False)
        header(image, draw, x0, color, title, subtitle, seconds, done, rate)
        draw.text((x0 + 36, 290), "› " + PROMPT, font=F_Q, fill=MUTED, anchor="lm")
        raw = text_at(run, t_ms)
        text = visible(raw)
        lines = wrap(draw, text or thought(raw), PW - 80)
        y = 336
        for line in lines[-10:]:
            draw.text((x0 + 36, y), line, font=F_A, fill=FROST if text else MUTED)
            y += 56
        if not done and (t_ms // 400) % 2 == 0:  # the template's blinking orange cursor
            end = x0 + 36 + draw.textlength(lines[-1], font=F_A) + 6
            top = y - 56 if lines[-1] else 336
            draw.rectangle([end, top + 8, end + 18, top + 48], fill=color)
        if done:
            draw.rectangle([x0 + 36, 936, x0 + 44, 944], fill=color)
            draw.text((x0 + 58, 940), f"DONE  ·  first token after {run['first_ms'] / 1000:.2f} s", font=F_Q, fill=color, anchor="lm")
    cards = "the same two cards" if HARDWARE.startswith("2 ") else "the same card"
    draw.text((W // 2, 1030), f"Real Kaggle run, replayed in its recorded token timing  ·  both sides on {cards}, "
              "one after the other", font=F_SMALL, fill=MUTED, anchor="mm")
    return image


def crowd(t_ms, label):
    image = canvas()
    draw = ImageDraw.Draw(image)
    count = len(STOCK["multi"])
    pitch = (708 - 8) // count  # the cards share the panel below its header
    lines = 3 if pitch >= 170 else 2
    eyebrow(draw, 60, 52, f"{TAG}  ·  {count} QUESTIONS SENT AT ONCE  ·  {label}")
    for (side, x0, color, title, _), subtitle in zip(SIDES, CROWD_SUB):
        rows = side["multi"]
        finish = max(r["total_ms"] for r in rows)
        finished = sum(r["total_ms"] <= t_ms for r in rows)
        header(image, draw, x0, color, title, subtitle, min(t_ms, finish) / 1000, t_ms >= finish,
               (f"{finished}/{count} answered", finished > 0))
        order = sorted(rows, key=lambda r: r["first_ms"])  # the order the server took them in
        for i, row in enumerate(rows):
            y = 266 + i * pitch
            done, started = t_ms >= row["total_ms"], row["first_ms"] is not None and t_ms >= row["first_ms"]
            place = order.index(row)
            # The Runtime reads every prompt of a set before it answers any, in both modes.
            serving = DEMO4 or place == 0 or t_ms >= order[place - 1]["total_ms"]
            raw = text_at(row, t_ms)
            text = visible(raw)
            status = (f"DONE {row['total_ms'] / 1000:.1f} s" if done else
                      ("ANSWERING" if text else "THINKING") if started and active(row, t_ms) else
                      "WAITING" if started else "READING PROMPT" if serving else "WAITING IN QUEUE")
            lit = done or status in ("ANSWERING", "THINKING", "READING PROMPT")
            draw.rounded_rectangle([x0 + 24, y, x0 + PW - 24, y + pitch - 10], radius=14, fill=INNER,
                                   outline=color if done else LINE, width=2 if done else 1)
            question = "› " + row["prompt"]
            while draw.textlength(question, font=F_MQ) > PW - 150 - draw.textlength("WAITING IN QUEUE", font=F_MQ):
                question = question[:-2] + "…"
            draw.text((x0 + 44, y + 24), question, font=F_MQ, fill=MUTED, anchor="lm")
            draw.text((x0 + PW - 44, y + 24), status, font=F_MQ, fill=color if lit else MUTED, anchor="rm")
            shown = text or thought(raw)
            for j, line in enumerate(wrap(draw, shown, PW - 100, F_MA)[-lines:] if shown else []):
                draw.text((x0 + 44, y + 46 + j * 32), line, font=F_MA, fill=FROST if text else MUTED)
    note = ("off answers one request after another; on groups them, each still its own batch of one, so every answer "
            "advances together" if DEMO4 else
            "on this plan the server answers one at a time, on both sides, so later ones wait in its queue")
    draw.text((W // 2, 1030), f"Real Kaggle run  ·  {count} requests at the same moment  ·  {note}", font=F_SMALL,
              fill=MUTED, anchor="mm")
    return image


def title():
    image = canvas()
    draw = ImageDraw.Draw(image)
    brand = font(MONO, 150, 600)
    width = draw.textlength("IRONMULE", font=brand)
    x = (W - width) / 2
    draw.text((x, 400), "IRON", font=brand, fill=FROST, anchor="lm")
    glow(image, (x + draw.textlength("IRON", font=brand), 400), "MULE", brand, ENERGY, anchor="lm", radius=22)
    draw.text((W // 2, 560), "Same question. Same GPU. Watch the clock.", font=font(SANS, 60, 600), fill=FROST, anchor="mm")
    tag = f"{TAG}  ·  RECORDED RUN"
    eyebrow(draw, (W - draw.textlength(tag, font=F_EYE)) // 2 - 11, 660, tag)
    return image


def interstitial():
    image = canvas()
    draw = ImageDraw.Draw(image)
    count = len(STOCK["multi"])
    draw.text((W // 2, 500), f"Now {'two three four five six'.split()[count - 2]} people ask at once.",
              font=font(SANS, 72, 600), fill=FROST, anchor="mm")
    limit = REPORT["config"]["multi"]["max_tokens"]
    tag = f"{count} DIFFERENT QUESTIONS  ·  SENT AT THE SAME MOMENT  ·  AT MOST {limit} TOKENS EACH"
    eyebrow(draw, (W - draw.textlength(tag, font=F_EYE)) // 2 - 11, 600, tag)
    return image


def row(image, draw, y, label, off_ms, on_ms):
    for i, part in enumerate(label):
        draw.text((120, y - 18 + i * 38), part, font=font(MONO, 26, 500 if i == 0 else 400), fill=FROST if i == 0 else MUTED,
                  anchor="lm")
    big = font(MONO, 96, 500)
    draw.text((820, y), f"{off_ms / 1000:.2f} s", font=big, fill=ICE, anchor="rm")
    draw.text((900, y), "→", font=font(SANS, 80, 300), fill=MUTED, anchor="mm")
    glow(image, (980, y), f"{on_ms / 1000:.2f} s", big, ENERGY, anchor="lm")
    glow(image, (1800, y), f"{off_ms / on_ms:.1f}×", font(SANS, 96, 600), ENERGY, anchor="rm", radius=18)


def end():
    image = canvas()
    draw = ImageDraw.Draw(image)
    off, on = STOCK["median"], NATIVE["median"]
    arms = "IRONMULE OFF → ON" if DEMO4 else "STOCK MLX → IRONMULE ON"
    arms = f"{SIDES[0][3]} → {SIDES[1][3]}" if "sides" in REPORT else arms
    eyebrow(draw, 120, 110, f"RESULT  ·  {MODEL}  ·  KAGGLE {HARDWARE.replace('NVIDIA ', '')}  ·  {arms}")
    row(image, draw, 250, ("ONE QUESTION", "median of 3"), off["total_ms"], on["total_ms"])
    y = 250
    if MULTI:
        y = 440
        row(image, draw, y, (f"{len(STOCK['multi'])} AT ONCE", "all answered"), max(r["total_ms"] for r in STOCK["multi"]),
            max(r["total_ms"] for r in NATIVE["multi"]))
    draw.text((W // 2, y + 170), f"decode {off['decode']:.1f} → {on['decode']:.1f} tok/s  ·  first token {off['first_ms'] / 1000:.2f} s → "
              f"{on['first_ms'] / 1000:.2f} s", font=font(MONO, 28, 400), fill=FROST, anchor="mm")
    pairs = [(STOCK["shown"], NATIVE["shown"])] + (list(zip(STOCK["multi"], NATIVE["multi"])) if MULTI else [])
    alike = sum(visible(a["text"]) == visible(b["text"]) for a, b in pairs)
    same = alike == len(pairs)
    words = (", and here every answer came out word for word identical." if same else
             f": {alike} of {len(pairs)} answers came out word for word identical, the others differ in wording.")
    if "notes" in REPORT:
        notes = (*REPORT["notes"][:-1], REPORT["notes"][-1] + words)
    elif DEMO4:
        gate = ("quality-gated for this model" if REPORT["plan_status"] == "qualified"
                else "no quality gate has qualified it for this model")
        notes = ("Both sides are IronMule: off is its reference path with MLX's own limits, one request at a time;",
                 f"on is its {PLAN}{' with tuned settings' if REPORT['config']['knobs'] else ''} and the requests "
                 "grouped. Not a comparison with mlx-lm's own batched generation.",
                 *(("The exact plan keeps the checkpoint's arithmetic; only scheduling and settings change" + words,)
                   if REPORT["plan_status"] == "exact" else
                   ((f"The {PLAN} runs IronMule's own 4-bit kernels on GPUs that emulate bf16; {gate};"
                     if REPORT["config"]["dtype"] == "native" else
                     f"The {PLAN} computes the 4-bit weights in {REPORT['config']['dtype']} on GPUs that emulate bf16; {gate};"),
                    "it changes the arithmetic" + words)))
    else:
        notes = ("Every question was asked once before recording (stock compiles kernels per new prompt length).",
                 "native is IronMule's opt-in plan for GPUs that emulate bf16; quality-gated for this model, it changes the "
                 "arithmetic" + words)
    for i, line in enumerate(notes):
        draw.text((W // 2, y + 250 + i * 36), line, font=F_SMALL, fill=MUTED, anchor="mm")
    draw.text((W // 2, 980), "github.com/Tobayko/IronMule", font=font(MONO, 34, 500), fill=ICE, anchor="mm")
    return image, same


end_card, same = end()
print(json.dumps({side: {"median": d["median"], "shown_total_ms": d["shown"]["total_ms"], "tokens": d["shown"]["tokens"],
                         "multi_ms": [r["total_ms"] for r in d["multi"] or []]}
                  for side, d in (("stock", STOCK), ("native", NATIVE))} | {"text_identical": same}, indent=1))
ffmpeg = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", OUT], stdin=subprocess.PIPE)
count = 0


def write(image, seconds=1 / FPS):
    global count
    data = image.tobytes()
    for _ in range(round(seconds * FPS)):
        ffmpeg.stdin.write(data)
        count += 1


write(title(), 2.5)
write(race(0), 0.6)
finish = max(STOCK["shown"]["total_ms"], NATIVE["shown"]["total_ms"])
for n in range(int((finish / 1000 + 2.0) * FPS)):
    write(race(n / FPS * 1000))
if MULTI:
    write(interstitial(), 2.0)
    fast_from = max(r["total_ms"] for r in NATIVE["multi"]) + 1500
    stop = max(r["total_ms"] for r in STOCK["multi"]) + 1500
    # Real time while IronMule answers, unless that alone would run past 20 s of video.
    first = 1 if fast_from <= 20000 else math.ceil(fast_from / 15000)
    tag = "REAL TIME" if first == 1 else f"{first}× FAST FORWARD"
    write(crowd(0, tag), 0.6)
    for n in range(int(min(fast_from, stop) / 1000 / first * FPS)):
        write(crowd(n / FPS * 1000 * first, tag))
    if stop > fast_from:
        speed = max(4, math.ceil((stop - fast_from) / 10000))  # the rest in at most 10 s of video
        for n in range(int((stop - fast_from) / 1000 / speed * FPS) + 1):
            write(crowd(fast_from + n / FPS * 1000 * speed, f"{speed}× FAST FORWARD"))
write(end_card, 6)
ffmpeg.stdin.close()
ffmpeg.wait()
print(OUT, count / FPS, "s")
