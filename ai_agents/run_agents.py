"""Run Claude agents through the live bandit task (https://bandit-task-v2.web.app/) as naive participants.

Each agent is one fresh Claude API conversation. It knows nothing about the study except what the task page shows: this script
reads every screen's text from the live page and passes it on verbatim, plus the few formatting lines in prompts.py (the only
researcher-written text). The agent's replies are turned into clicks, key presses and form entries on the real page, so the data
are saved to Firestore by the page's own code, in exactly the format of the human participants.

Timing: the page gives 2 s to respond. A model needs longer, so the page's clock is frozen while the agent decides (Playwright's
clock control) and then advanced: SIM_RT_MS to the key press, the 1.25 s feedback, then the fixation interval until the next trial.
Recorded RTs are therefore the fixed SIM_RT_MS and carry no information. Conditions:
  time_blind  the fixation cross between trials is described without its duration (the page as a text-only agent perceives it)
  time_told   the same sentence plus how many seconds the fixation cross lasted, measured on the page clock

Identifiers (never a real Prolific ID, so a run can never overwrite a human participant):
  PROLIFIC_PID = AIAGENT_<blind|told>_<model>_<tag>_<nnn>,  STUDY_ID = AI_AGENTS,  SESSION_ID = <model>_<condition>
The agent stops at the end screen; nothing is sent to Prolific.

Usage (see README.md):
  python run_agents.py --mock --dry-run --n 1                 # test the browser flow, no API calls, nothing saved
  python run_agents.py --n 2 --condition both --tag pilot      # pilot: 2 agents, claude-opus-5, one per condition
  python run_agents.py --n 40 --condition both --model claude-opus-5,claude-sonnet-5 --parallel 4 --tag r1
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import prompts as T

TASK_URL = "https://bandit-task-v2.web.app/"
STUDY_ID = "AI_AGENTS"
PID_PREFIX = "AIAGENT_"
FIRESTORE_KEY = os.environ.get("FIREBASE_WEB_API_KEY", "")   # the web API key of the task page (not stored in the repository)
FIRESTORE_DOC = ("https://firestore.googleapis.com/v1/projects/bandit-task-v2/databases/(default)/documents/participants/{pid}"
                 f"?key={FIRESTORE_KEY}")
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
SIM_RT_MS = 600            # page-clock ms from trial onset to the key press (the clock is frozen while the model decides)
FEEDBACK_MS = 1250         # FEEDBACK_DURATION in the task page
MAX_REASKS = 2             # invalid replies re-asked before the page's own consequence applies (a missed trial, a stopped run)
KEY_POS = {"Q": "left", "T": "middle", "P": "right"}
MODEL_SHORT = {"claude-opus-5": "opus5", "claude-opus-5-5": "opus55", "claude-sonnet-5": "sonnet5", "claude-haiku-4-5": "haiku45",
               "claude-fable-5-1": "fable51", "claude-opus-4-8": "opus48", "claude-sonnet-4-6": "sonnet46"}
PRICE = {"claude-opus-5": (5, 25), "claude-opus-5-5": (4, 20), "claude-sonnet-5": (2, 10), "claude-haiku-4-5": (1, 5),
         "claude-fable-5-1": (10, 50), "claude-opus-4-8": (5, 25), "claude-sonnet-4-6": (3, 15)}   # $ per million tokens (in, out)

STATE_JS = """() => {
  const vis = id => { const e = document.getElementById(id); return !!e && e.getClientRects().length > 0; };
  const c = document.getElementById('container');
  const form = document.querySelector('#pswq-screen form');
  return {
    consent: vis('consent-screen'), setup1: vis('setup-screen-1'), setup2: vis('setup-screen-2'), age: vis('age-screen'),
    transition: vis('transition-screen'), game: vis('game-screen'), pswq: vis('pswq-screen'), end: vis('end-screen'),
    container: !!c && c.getClientRects().length > 0 && getComputedStyle(c).visibility !== 'hidden',
    fixation: vis('fixation-cross'),
    counter: (document.getElementById('trial-counter') || {}).innerText || '',
    feedback: (document.getElementById('feedback') || {}).innerText || '',
    overlays: [0, 1, 2].map(i => { const o = document.getElementById('overlay-' + i); return o && o.getClientRects().length > 0 ? o.innerText : ''; }),
    canClick: (typeof canClick !== 'undefined') ? canClick : null,
    form: form ? form.id : null,
  };
}"""

FORM_JS = """() => {
  const scr = document.getElementById('pswq-screen'); const form = scr.querySelector('form');
  const items = [...scr.querySelectorAll('ol.qs > li')].map(li => ({
    name: li.querySelector('input[type=radio]').name,
    text: (li.firstChild && li.firstChild.nodeType === 3 ? li.firstChild.textContent : li.innerText.split('\\n')[0]).trim(),
    options: [...li.querySelectorAll('label')].map(l => l.innerText.trim()),
    values: [...li.querySelectorAll('input[type=radio]')].map(r => r.value) }));
  const intro = [...form.children].filter(el => !['OL', 'BUTTON', 'INPUT'].includes(el.tagName) && el.id.indexOf('error') < 0)
                                  .map(el => el.innerText.trim()).filter(Boolean);
  const btn = form.querySelector('button[type=submit]');
  return {id: form.id, intro, items, button: btn ? btn.innerText.trim() : null,
          birthYear: !!document.getElementById('birth-year-input')};
}"""


class AgentStopped(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------------------
# policies: Claude (the real runs) and a random mock (to test the browser flow without API calls)
# ---------------------------------------------------------------------------------------------------------------------------------
class ClaudeAgent:
    def __init__(self, client, model: str, system: str, effort: str | None, log: list):
        self.client, self.model, self.system, self.effort, self.log = client, model, system, effort, log
        self.messages: list = []
        self.usage = dict(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0, requests=0)

    async def ask(self, content: list, kind: str, meta: dict) -> str:
        import anthropic
        self.messages.append({"role": "user", "content": content})
        params = dict(model=self.model, max_tokens=16000, system=self.system, messages=self.messages,
                      cache_control={"type": "ephemeral"})
        # Structured output: the reply can only be the answer (in the pilot the model sometimes went on to write the next screen
        # itself - "You pressed P ... You won $1 ..." - or other junk, which then stayed in its memory of the session)
        output_config = {"format": {"type": "json_schema", "schema": reply_schema(kind, meta)}}
        if not self.model.startswith("claude-haiku"):                 # Haiku 4.5: no adaptive thinking, no effort parameter
            params["thinking"] = {"type": "adaptive", "display": "summarized"}
            if self.effort:
                output_config["effort"] = self.effort
        params["output_config"] = output_config
        for attempt in range(6):
            try:
                resp = await self.client.messages.create(**params)
                break
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as e:
                if attempt == 5:
                    raise
                await asyncio.sleep(min(60, 5 * 2 ** attempt))
        self.messages.append({"role": "assistant", "content": resp.content})      # full content, thinking blocks included
        u = resp.usage
        for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"):
            self.usage[k] += getattr(u, k, 0) or 0
        self.usage["requests"] += 1
        raw = "".join(b.text for b in resp.content if b.type == "text").strip()
        text = answer_from_json(kind, raw)
        thinking = "".join(getattr(b, "thinking", "") or "" for b in resp.content if b.type == "thinking").strip()
        self.log.append(dict(t=_now(), kind=kind, prompt=_text_of(content), reply=text, raw_reply=raw, thinking_summary=thinking,
                             stop_reason=resp.stop_reason, model=resp.model, request_id=resp._request_id,
                             usage=dict(input=u.input_tokens, output=u.output_tokens,
                                        cache_write=getattr(u, "cache_creation_input_tokens", 0),
                                        cache_read=getattr(u, "cache_read_input_tokens", 0))))
        if resp.stop_reason == "refusal":
            raise AgentStopped(f"refusal ({getattr(resp.stop_details, 'category', None)})")
        return text

    def cost(self) -> float | None:
        if self.model not in PRICE:
            return None
        pin, pout = PRICE[self.model]
        u = self.usage
        return (u["input_tokens"] * pin + u["cache_creation_input_tokens"] * pin * 1.25 + u["cache_read_input_tokens"] * pin * 0.1
                + u["output_tokens"] * pout) / 1e6


class MockAgent:
    """Random but valid replies; used with --mock to test the browser automation without the API."""
    def __init__(self, log: list, seed: int):
        self.log, self.rng, self.model = log, random.Random(seed), "mock"
        self.usage = dict(requests=0)

    async def ask(self, content: list, kind: str, meta: dict) -> str:
        r = self.rng
        if kind == "button":
            text = meta["buttons"][0]
        elif kind == "text":
            text = "34" if meta.get("field") == "age" else "1992"
        elif kind == "key":
            text = r.choice("QTP")
        elif kind == "form":
            text = "\n".join(f"{i + 1}: {r.randint(1, k)}" for i, k in enumerate(meta["n_options"]))
        else:
            text = ""
        self.usage["requests"] += 1
        self.log.append(dict(t=_now(), kind=kind, prompt=_text_of(content), reply=text))
        return text

    def cost(self):
        return 0.0


# ---------------------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------------------
def reply_schema(kind: str, meta: dict) -> dict:
    """JSON schema for one reply (structured output): only the answer can be generated."""
    def obj(props):
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    if kind == "key":
        return obj({"key": {"type": "string", "enum": ["Q", "T", "P"]}})
    if kind == "button":
        return obj({"button": {"type": "string", "enum": list(meta["buttons"])}})
    if kind == "text":
        return obj({"typed": {"type": "string"}})
    if kind == "form":
        return obj({"answers": obj({str(i + 1): {"type": "integer", "enum": list(range(1, k + 1))}
                                    for i, k in enumerate(meta["n_options"])})})
    raise ValueError(kind)


def answer_from_json(kind: str, raw: str) -> str:
    """The structured reply as the plain answer the screen handlers parse (Q / a button label / what was typed / '1: 3' lines)."""
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if kind == "key":
        return str(d.get("key", ""))
    if kind == "button":
        return str(d.get("button", ""))
    if kind == "text":
        return str(d.get("typed", ""))
    if kind == "form":
        return "\n".join(f"{k}: {v}" for k, v in sorted(d.get("answers", {}).items(), key=lambda kv: int(kv[0])))
    return raw


async def api_check(models: list[str], effort: str | None) -> bool:
    """One tiny request per reply type and model, with the exact parameters of a real run (not part of any agent's session)."""
    import anthropic
    client = anthropic.AsyncAnthropic(max_retries=2)
    ok = True
    tests = [("key", {}, "Reply with the key you press: Q, T or P."),
             ("button", {"buttons": ["NEXT"]}, 'Reply with the label of the button you click: "NEXT".'),
             ("text", {"field": "age"}, "Reply with exactly what you type into the box: the number 7."),
             ("form", {"n_options": [4, 4]}, "Two statements, options 1-4 each. Reply 1: <option>, 2: <option>.")]
    for model in models:
        for kind, meta, prompt in tests:
            log: list = []
            agent = ClaudeAgent(client, model, "API check.", effort, log)
            try:
                ans = await agent.ask([{"type": "text", "text": prompt}], kind, meta)
                print(f"  OK   {model:<18} {kind:<6} raw {log[-1]['raw_reply']!r} -> answer {ans!r}")
            except Exception as e:
                ok = False
                print(f"  FAIL {model:<18} {kind:<6} {type(e).__name__}: {str(e)[:300]}")
    return ok


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _text_of(content: list) -> str:
    return "\n".join(b["text"] if b["type"] == "text" else "[image of the screen]" for b in content)


def _msg(parts: list[str], image_png: bytes | None = None) -> list:
    blocks = []
    if image_png is not None:
        blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                   "data": base64.standard_b64encode(image_png).decode()}})
    blocks.append({"type": "text", "text": "\n\n".join(p for p in parts if p)})
    return blocks


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def firestore_doc_exists(pid: str) -> bool:
    try:
        with urllib.request.urlopen(FIRESTORE_DOC.format(pid=pid), timeout=30):
            return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


async def state(page) -> dict:
    return await page.evaluate(STATE_JS)


async def fake_now(page) -> int:
    return int(await page.evaluate("Date.now()"))


# ---------------------------------------------------------------------------------------------------------------------------------
# one agent through the whole session
# ---------------------------------------------------------------------------------------------------------------------------------
async def run_agent(browser, agent, pid: str, condition: str, args) -> dict:
    session_id = f"{agent.model}_{condition}"
    url = f"{TASK_URL}?PROLIFIC_PID={pid}&STUDY_ID={STUDY_ID}&SESSION_ID={session_id}"
    info = dict(pid=pid, model=agent.model, condition=condition, study_id=STUDY_ID, session_id=session_id, url=url,
                started=_now(), dry_run=args.dry_run, images=not args.no_images, sim_rt_ms=SIM_RT_MS, effort=args.effort,
                persona=args.persona, trials=[], status="running")
    context = await browser.new_context(viewport={"width": 1400, "height": 900})
    page = await context.new_page()
    page.on("dialog", lambda d: asyncio.ensure_future(d.dismiss()))
    console_errors, console_log, net_log = [], [], []
    page.on("pageerror", lambda e: console_errors.append(str(e)))
    page.on("console", lambda m: console_log.append(f"{m.type}: {m.text}"[:300]))
    if args.debug:
        page.on("request", lambda r: net_log.append(f"-> {r.method} {r.url[:160]}") if "firestore" in r.url else None)
        page.on("requestfailed", lambda r: net_log.append(f"FAILED {r.url[:160]} {r.failure}") if "firestore" in r.url else None)
        page.on("response", lambda r: net_log.append(f"<- {r.status} {r.url[:160]}") if "firestore" in r.url else None)
    if args.dry_run:                                                   # nothing can reach the database
        async def _abort(route):
            await route.abort()
        await page.route(re.compile(r".*firestore\.googleapis\.com.*"), _abort)
    pending: list[str] = []                                            # page events to report with the next screen

    async def ask(parts, kind, meta, reply_line, parse, image=None):
        """Show a screen, get a reply, parse it; re-ask on an invalid reply. Returns the parsed value or None."""
        nonlocal pending
        content = _msg(pending + parts + [reply_line], image)
        pending = []
        for i in range(MAX_REASKS + 1):
            reply = await agent.ask(content, kind, meta)
            val = parse(reply)
            if val is not None:
                return val
            content = _msg([T.INVALID_REPLY.format(reply_format=reply_line)])
        return None

    async def button_screen(sel: str, buttons: list[str]):
        text = await page.inner_text(sel)
        want = {_norm(b): b for b in buttons}
        got = await ask([T.SCREEN_HEADER, text], "button", {"buttons": buttons},
                        T.BUTTON_REPLY.format(buttons=", ".join(f'"{b}"' for b in buttons)),
                        lambda r: next((want[k] for k in want if k and k in _norm(r)), None))
        if got is None:
            raise AgentStopped(f"no valid button on {sel}")
        return got

    try:
        await page.clock.install(time=datetime.now(timezone.utc))
        await page.goto(url, wait_until="networkidle")
        await page.wait_for_selector("#consent-screen", state="visible")
        # from here the page clock only moves when we advance it (a number would be read as SECONDS by Playwright: pass a datetime)
        await page.clock.pause_at(datetime.fromtimestamp((await fake_now(page) + 1000) / 1000, tz=timezone.utc))
        phase_first_trial = True
        last_phase = None
        clock_running = False
        n_steps = 0
        while True:
            n_steps += 1
            if n_steps > 20000:
                raise AgentStopped("step limit")
            s = await state(page)
            if s["end"]:
                info["end_screen"] = await page.inner_text("#end-screen")
                break
            if s["consent"]:
                await button_screen("#consent-screen", ["I Consent"])
                await page.evaluate("() => { const c = document.getElementById('consent-screen'); c.scrollTop = c.scrollHeight;"
                                    " c.dispatchEvent(new Event('scroll')); }")
                await page.click("#consent-btn")
            elif s["setup1"]:
                await button_screen("#setup-screen-1", ["NEXT"]); await page.click("#setup-screen-1 button")
            elif s["setup2"]:
                await button_screen("#setup-screen-2", ["NEXT"]); await page.click("#setup-screen-2 button")
            elif s["age"]:
                await text_input_screen(page, ask, "#age-screen", "#age-input", "#age-error", "#age-screen button", "age")
            elif s["transition"]:
                await button_screen("#transition-screen", ["START REAL EXPERIMENT"])
                await page.click("#transition-screen button")
            elif s["game"]:
                phase = "practice" if s["counter"].startswith("PRACTICE") else "task"
                if phase != last_phase:
                    phase_first_trial, last_phase = True, phase
                if not (s["container"] and s["canClick"]):
                    await asyncio.sleep(0.05)                           # stimuli still loading (real time)
                    continue
                rec = await play_trial(page, ask, s, condition, include_image=phase_first_trial and not args.no_images, pid=pid)
                phase_first_trial = False
                pending = rec.pop("report")
                rec["phase"] = phase
                info["trials"].append(rec)
            elif s["pswq"]:
                if not clock_running:                                   # let the page's own save to Firestore run normally
                    await page.clock.resume(); clock_running = True
                f = await page.evaluate(FORM_JS)
                if f["birthYear"]:
                    await text_input_screen(page, ask, "#pswq-screen", "#birth-year-input", "#birth-year-error",
                                            "#birthYearForm button[type=submit]", "birth year")
                else:
                    final = f["items"] and f["items"][0]["name"].startswith("pleasure")
                    await form_screen(page, ask, f)
                    if final and args.dry_run:
                        info["status"] = "dry run complete (final form filled, not submitted)"
                        break
                    await page.click(f"#{f['id']} button[type=submit]")
                    if final:
                        await page.wait_for_selector("#end-screen", state="visible", timeout=120000)
            else:
                await asyncio.sleep(0.05)
        if info["status"] == "running":
            info["status"] = "completed"
            if not args.dry_run:
                await asyncio.sleep(2)
                info["saved_in_firestore"] = await asyncio.to_thread(firestore_doc_exists, pid)
    except AgentStopped as e:
        info["status"] = f"stopped: {e}"
    except Exception as e:
        info["status"] = f"error: {type(e).__name__}: {e}"
        info["traceback"] = traceback.format_exc()
    finally:
        info["finished"] = _now()
        info["page_errors"] = console_errors
        info["console"] = console_log[-60:]
        if args.debug:
            info["firestore_requests"] = net_log[-80:]
        info["usage"] = agent.usage
        info["cost_usd_estimate"] = agent.cost()
        await context.close()
    return info


async def text_input_screen(page, ask, screen_sel, input_sel, error_sel, button_sel, field):
    text = await page.inner_text(screen_sel)
    parts = [T.SCREEN_HEADER, text]
    for attempt in range(3):
        typed = await ask(parts, "text", {"field": field}, T.TEXT_INPUT_REPLY, lambda r: r.strip().strip('"'))
        await page.fill(input_sel, typed or "")
        await page.click(button_sel)
        await asyncio.sleep(0.2)
        err = (await page.inner_text(error_sel)) if await page.locator(error_sel).count() else ""
        if not err.strip() or not await page.locator(input_sel).is_visible():
            return
        parts = [T.SCREEN_HEADER, err.strip()]                           # the page's own error message
    raise AgentStopped(f"no valid {field}")


async def form_screen(page, ask, f):
    items = f["items"]
    lines = [T.SCREEN_HEADER] + f["intro"] + [""]
    for i, it in enumerate(items):
        opts = ", ".join(f"{j + 1} = {o}" for j, o in enumerate(it["options"]))
        lines.append(f"{i + 1}. {it['text']}\n   " + T.FORM_ITEM_OPTIONS.format(options=opts))
    if f["button"]:
        lines.append(f"[button: {f['button']}]")
    reply_line = T.FORM_REPLY.format(n=len(items))

    def parse(r):
        ans = {}
        for a, b in re.findall(r"(?m)^\D*?(\d+)\s*[:=.)\-]\s*(\d+)", r):
            a, b = int(a), int(b)
            if 1 <= a <= len(items) and 1 <= b <= len(items[a - 1]["options"]):
                ans[a] = b
        return ans if len(ans) == len(items) else None

    answers = await ask(["\n".join(lines)], "form", {"n_options": [len(it["options"]) for it in items]}, reply_line, parse)
    if answers is None:
        raise AgentStopped(f"form {f['id']} not completed")
    for i, it in enumerate(items):
        await page.check(f'input[name="{it["name"]}"][value="{it["values"][answers[i + 1] - 1]}"]')


async def play_trial(page, ask, s, condition, include_image, pid) -> dict:
    image = await page.locator("#container").screenshot() if include_image else None
    if image is not None:                                                # kept so you can see exactly what the agent saw
        (RUNS / "images").mkdir(parents=True, exist_ok=True)
        (RUNS / "images" / f"{pid}_{s['counter'].split()[0].lower()}.png").write_bytes(image)
    key = await ask([s["counter"], T.TRIAL_STIMULI], "key", {}, T.TRIAL_REPLY,
                    lambda r: (m.group(1).upper() if (m := re.search(r"\b([QTPqtp])\b", r)) else None), image=image)
    rec = dict(counter=s["counter"])
    if key is None:                                                      # no usable reply: the page's 2 s limit applies
        await page.clock.run_for(2100)
        fb = (await state(page))["feedback"]
        await page.clock.run_for(1000)
        rec.update(key=None, outcome=fb, report=[fb])
        return rec
    await page.clock.run_for(SIM_RT_MS)
    await page.keyboard.press(f"Key{key}")
    s2 = await state(page)
    outcome = next((o for o in s2["overlays"] if o), "")
    await page.clock.run_for(FEEDBACK_MS)
    for _ in range(10):                                                  # make sure the fixation cross is up before timing it
        if (await state(page))["fixation"]:
            break
        await page.clock.run_for(50)
    fix_ms = 0
    while (await state(page))["fixation"] and fix_ms < 10000:            # the interval until the next screen, on the page clock
        await page.clock.run_for(100)
        fix_ms += 100
    sec = fix_ms / 1000
    duration = f"{sec:g} second" + ("" if sec == 1 else "s")
    fixation = T.FIXATION_TOLD.format(duration=duration) if condition == "time_told" else T.FIXATION_BLIND
    rec.update(key=key, outcome=outcome, fixation_ms=fix_ms,
               report=[T.CHOICE_ECHO.format(key=key, position=KEY_POS[key]) + " " + outcome, fixation])
    return rec


# ---------------------------------------------------------------------------------------------------------------------------------
async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1, help="number of agents")
    ap.add_argument("--model", default="claude-opus-5", help="model ID, or a comma-separated list (crossed with the conditions)")
    ap.add_argument("--condition", choices=["time_blind", "time_told", "both"], default="both",
                    help="both = alternate agents between the two conditions")
    ap.add_argument("--tag", default="r1", help="run label used in the PIDs (e.g. pilot, r1)")
    ap.add_argument("--start", type=int, default=1, help="first agent number (to add agents to an existing run)")
    ap.add_argument("--parallel", type=int, default=2, help="agents run at the same time")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                    help="output_config.effort (not used for Haiku); default: the model's default")
    ap.add_argument("--persona", default=None, help='optional sentence added to the system prompt, e.g. "You are 34 years old."')
    ap.add_argument("--no-images", action="store_true", help="never send a screenshot (by default the first trial of the practice "
                    "and of the task include an image of the three pictures)")
    ap.add_argument("--mock", action="store_true", help="random replies instead of Claude (tests the browser flow, no API cost)")
    ap.add_argument("--dry-run", action="store_true", help="block all database traffic and stop before the final Submit")
    ap.add_argument("--show", action="store_true", help="show the browser window")
    ap.add_argument("--overwrite", action="store_true", help="allow a PID that already has data in Firestore")
    ap.add_argument("--debug", action="store_true", help="also record the page's database requests in the run record")
    ap.add_argument("--api-check", action="store_true", help="send one tiny request of each reply type per model and stop "
                    "(checks the API key and request parameters before a real run; costs well under a cent)")
    args = ap.parse_args()

    from playwright.async_api import async_playwright
    models = [m.strip() for m in args.model.split(",") if m.strip()]
    args.tag = re.sub(r"[^A-Za-z0-9-]", "-", args.tag)                  # no underscores: they separate the PID fields
    conds = ["time_blind", "time_told"] if args.condition == "both" else [args.condition]
    system = T.SYSTEM_PROMPT + (f" {args.persona}" if args.persona else "")
    RUNS.mkdir(exist_ok=True)
    if args.api_check:
        print(f"API check for {models}:")
        sys.exit(0 if await api_check(models, args.effort) else 1)
    client = None
    if not args.mock:
        import anthropic
        client = anthropic.AsyncAnthropic(max_retries=5)

    plan = []
    for k in range(args.n):
        i = args.start + k
        # models x conditions fully crossed: model cycles fastest, condition changes after each full set of models
        # (a plain round-robin on both would pair model 1 with condition 1 and model 2 with condition 2 whenever both lists have 2)
        model, cond = models[k % len(models)], conds[(k // len(models)) % len(conds)]
        short = MODEL_SHORT.get(model, re.sub(r"[^a-z0-9]", "", model.lower()))
        pid = f"{PID_PREFIX}{'mock_' if args.mock else ''}{cond.split('_')[1]}_{short}_{args.tag}_{i:03d}"
        plan.append((pid, model, cond))
    if not args.dry_run and not args.overwrite:
        taken = [pid for pid, _, _ in plan if firestore_doc_exists(pid)]
        if taken:
            sys.exit(f"these PIDs already have data in Firestore (use a new --tag or --start, or --overwrite): {taken}")

    print(f"{len(plan)} agent(s); models {models}; conditions {conds}; {'MOCK, ' if args.mock else ''}"
          f"{'DRY RUN (nothing saved)' if args.dry_run else 'LIVE (data saved to Firestore)'}")
    sem = asyncio.Semaphore(args.parallel)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not args.show)

        async def one(pid, model, cond, k):
            async with sem:
                log: list = []
                agent = MockAgent(log, seed=k) if args.mock else ClaudeAgent(client, model, system, args.effort, log)
                t0 = time.time()
                print(f"  start {pid}", flush=True)
                info = await run_agent(browser, agent, pid, cond, args)
                info.update(system_prompt=system, exchanges=log, wall_seconds=round(time.time() - t0, 1))
                (RUNS / f"{pid}.json").write_text(json.dumps(info, indent=2, default=str))
                n_tr = sum(1 for t in info["trials"] if t["phase"] == "task")
                c = info["cost_usd_estimate"]
                print(f"  done  {pid}: {info['status']}; {n_tr} task trials; {agent.usage.get('requests', 0)} requests; "
                      f"{info['wall_seconds']:.0f} s" + (f"; ~${c:.2f}" if c else "") +
                      (f"; saved: {info.get('saved_in_firestore')}" if "saved_in_firestore" in info else ""), flush=True)
                return info

        results = await asyncio.gather(*(one(pid, m, c, k) for k, (pid, m, c) in enumerate(plan)))
        await browser.close()
    total = sum(r["cost_usd_estimate"] or 0 for r in results)
    print(f"finished: {sum(r['status'] == 'completed' for r in results)} completed of {len(results)}"
          + (f"; estimated cost ${total:.2f}" if not args.mock else "") + f"; transcripts in {RUNS}")


if __name__ == "__main__":
    asyncio.run(main())
