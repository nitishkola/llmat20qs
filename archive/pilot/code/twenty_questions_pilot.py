# CSE 519 pilot: How Good Are LLMs at 20 Questions?  (python twenty_questions_pilot.py [--mock])
import argparse
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor

parser = argparse.ArgumentParser(description="Run the 20 Questions LLM pilot.")
parser.add_argument("--mock", action="store_true", help="use deterministic mock responses")
parser.add_argument("--repeats", type=int, default=1, help="repeats per ordered model pair and target")
parser.add_argument("--results-file", default="results/tesla1-selected-models.jsonl",
                    help="round-robin results file")
parser.add_argument("--smoke-models", action="store_true", help="test one local generation from each model")
parser.add_argument("--validate-only", metavar="FILE",
                    help="validate an existing round-robin JSONL file and exit")
ARGS = parser.parse_args()
MOCK = ARGS.mock

MODELS = [
    {"name": "Qwen3 8B", "model": "Qwen/Qwen3-8B",
     "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
    {"name": "Gemma 3 12B IT", "model": "google/gemma-3-12b-it",
     "revision": "96b6f1eccf38110c56df3a15bffe176da04bfd80"},
    {"name": "Mistral Nemo 12B Instruct", "model": "mistralai/Mistral-Nemo-Instruct-2407",
     "revision": "04d8a90549d23fc6bd7f642064003592df51e9b3"},
    {"name": "Granite 3.3 8B Instruct", "model": "ibm-granite/granite-3.3-8b-instruct",
     "revision": "51dd4bc2ade4059a6bd87649d68aa11e4fb2529b"},
]
SECRET_WORDS = {"banana": [], "bicycle": ["bike"], "elephant": []}
MAX_Q = 20
TEMPERATURE = 0.7
WORKERS = 1
SEED = 519

ANSWERER_SYS = ("You are playing 20 Questions. The secret object is: {secret}. "
                "Answer each question about the secret object truthfully with exactly one word: "
                "Yes, No, or Sometimes. Never reveal the secret object.")
GUESSER_SYS = ("You are playing 20 Questions. Another player is thinking of a common object. "
               "You may ask up to 20 yes/no questions. Each turn, reply with exactly one line, either\n"
               "QUESTION: <a yes/no question>\nor\nGUESS: <one object>\n"
               "A guess uses up a turn. Try to identify the object in as few turns as possible.")
PICK_PROMPT = ("We are about to play 20 Questions and you will be the one thinking of the object. "
               "Choose a common physical object completely at random. "
               "Reply with only the object's name, one or two words, nothing else.")

_hf_models = {}
_hf_model_devices = {}
_hf_lock = threading.RLock()
_active_hf_pair = None

def prepare_hf_pair(*models):
    global _active_hf_pair
    needed = {m["model"] for m in models}
    with _hf_lock:
        if needed == _active_hf_pair:
            return
        stale = set(_hf_models) - needed
        for model_name in stale:
            del _hf_models[model_name]
            _hf_model_devices.pop(model_name, None)
        if stale:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _active_hf_pair = needed

def chat(m, messages):
    if MOCK:
        return mock_reply(messages)
    return hf_chat(m, messages)

def hf_chat(m, messages):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with _hf_lock:
        if m["model"] not in _hf_models:
            print(f"Loading {m['model']} from Hugging Face to local hardware (first run downloads weights)...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    m["model"], revision=m["revision"]
                )
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                device_map = "auto"
                if torch.cuda.is_available():
                    used_devices = set(_hf_model_devices.values())
                    available_devices = [
                        device for device in range(torch.cuda.device_count())
                        if device not in used_devices
                    ]
                    if not available_devices:
                        raise RuntimeError(
                            "No free GPU for the second model in this pairing; "
                            "request at least two Tesla1 GPUs."
                        )
                    device = max(
                        available_devices,
                        key=lambda index: torch.cuda.mem_get_info(index)[0],
                    )
                    device_map = {"": device}
                model = AutoModelForCausalLM.from_pretrained(
                    m["model"],
                    revision=m["revision"],
                    torch_dtype=dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                )
                if torch.cuda.is_available():
                    _hf_model_devices[m["model"]] = device
            except Exception as e:
                hint = (
                    " Accept Google's terms for Gemma 3 and authenticate with "
                    "`hf auth login` or HF_TOKEN."
                    if m["model"] == "google/gemma-3-12b-it" else ""
                )
                raise RuntimeError(
                    f"Could not load {m['model']} locally: {type(e).__name__}: {e}.{hint}"
                ) from e
            _hf_models[m["model"]] = (tokenizer, model)

        tokenizer, model = _hf_models[m["model"]]
        try:
            template_kwargs = m.get("template_kwargs", {})
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **m.get("template_kwargs", {}),
            )
        inputs = inputs.to(model.device)
        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=TEMPERATURE,
                    pad_token_id=tokenizer.eos_token_id,
                )
        except Exception as e:
            raise RuntimeError(
                f"Local generation failed for {m['model']}: {type(e).__name__}: {e}"
            ) from e

        prompt_length = inputs["input_ids"].shape[-1]
        text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        return re.sub(r"<think>.*$", "", text, flags=re.S).strip()

def mock_reply(messages):
    sys_msg = messages[0]["content"]
    if sys_msg.startswith("You are playing 20 Questions. The secret"):
        return random.choice(["Yes", "No", "No", "Sometimes", "Maybe yes"])
    if sys_msg.startswith("You are playing 20 Questions. Another"):
        if random.random() < 0.2:
            return "GUESS: " + random.choice(list(SECRET_WORDS) + ["chair", "spoon"])
        return "QUESTION: Is it " + random.choice(["alive", "metal", "edible", "big"]) + "?"
    return random.choice(["Banana", "Chair", "Stapler", "Spoon", "Umbrella", "chair"])

def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower()).replace(" an ", " ").replace(" a ", " ").strip()

def parse_answer(text):
    w = norm(text).split()
    return w[0].capitalize() if w and w[0] in ("yes", "no", "sometimes") else None

def parse_move(move):
    text = re.sub(r"\(?Turns used:\s*\d+\s*/\s*\d+\.?\)?", " ", move, flags=re.I)
    m = re.search(r"(QUESTION|GUESS)\s*:\s*(.*?)(?=\s*(?:QUESTION|GUESS)\s*:|\n|$)", text, re.I | re.S)
    if m:
        return m.group(1).upper(), m.group(2).strip()
    return "QUESTION", text.strip().split("\n")[0].strip()

def valid_move(kind, body):
    words = re.findall(r"[A-Za-z]+", body)
    return len(words) >= (1 if kind == "GUESS" else 3)

def is_guess_question(body, accepted):
    q = norm(body)
    return any(re.fullmatch(rf"(is it|is the object|is the secret object|is your object)( the)? {re.escape(a)}s?", q)
               for a in accepted)

def make_game_id(answerer, guesser, secret, repeat):
    return (
        f"{answerer['model']}@{answerer['revision']}|"
        f"{guesser['model']}@{guesser['revision']}|{secret}|{repeat:03d}"
    )

def play(answerer, guesser, secret, game_id=None):
    accepted = {norm(secret)} | {norm(s) for s in SECRET_WORDS[secret]}
    a_msgs = [{"role": "system", "content": ANSWERER_SYS.format(secret=secret)}]
    g_msgs = [{"role": "system", "content": GUESSER_SYS},
              {"role": "user", "content": "I am thinking of an object. Ask your first question."}]
    turns, invalid, bad_moves, won = [], 0, 0, False
    for t in range(1, MAX_Q + 1):
        for _ in range(3):
            move = chat(guesser, g_msgs)
            kind, body = parse_move(move)
            if valid_move(kind, body):
                break
            bad_moves += 1
            g_try = g_msgs + [{"role": "assistant", "content": move},
                              {"role": "user", "content": "Reply with exactly one line: QUESTION: <question> or GUESS: <object>."}]
            move = chat(guesser, g_try)
            kind, body = parse_move(move)
            if valid_move(kind, body):
                break
        g_msgs.append({"role": "assistant", "content": f"{kind}: {body}"})
        if not valid_move(kind, body):
            turns.append({"turn": t, "type": "invalid", "text": move[:200]})
            g_msgs.append({"role": "user", "content": f"That was not a valid move.\n(Turns used: {t}/{MAX_Q})"})
            continue
        if kind == "QUESTION" and is_guess_question(body, accepted):
            kind = "GUESS"
        if kind == "GUESS":
            guess = norm(body)
            correct = guess in accepted or any(a in guess.split() for a in accepted)
            turns.append({"turn": t, "type": "guess", "text": body, "correct": correct})
            if correct:
                won = True
                break
            g_msgs.append({"role": "user", "content": f"No, it is not {body}.\n(Turns used: {t}/{MAX_Q})"})
            continue
        a_msgs.append({"role": "user", "content": body})
        reply = chat(answerer, a_msgs)
        ans = parse_answer(reply)
        if ans is None:
            invalid += 1
            a_msgs.append({"role": "assistant", "content": reply})
            a_msgs.append({"role": "user", "content": "Answer with exactly one word: Yes, No, or Sometimes."})
            reply = chat(answerer, a_msgs)
            ans = parse_answer(reply) or "Sometimes"
        a_msgs.append({"role": "assistant", "content": ans})
        turns.append({"turn": t, "type": "question", "text": body, "answer": ans, "raw": reply})
        g_msgs.append({"role": "user", "content": f"{ans}.\n(Turns used: {t}/{MAX_Q})"})
    return {"game_id": game_id, "answerer": answerer["name"], "guesser": guesser["name"],
            "answerer_model": answerer["model"], "answerer_revision": answerer["revision"],
            "guesser_model": guesser["model"], "guesser_revision": guesser["revision"], "secret": secret,
            "won": won, "turns_used": len(turns), "invalid_answers": invalid, "bad_guesser_moves": bad_moves,
            "answer_turns": sum(x["type"] == "question" for x in turns), "turns": turns}

def pick_word(m):
    return norm(chat(m, [{"role": "user", "content": PICK_PROMPT}]).split("\n")[0])

def validate_round_robin(results_file, repeats=1):
    if repeats < 1:
        print("Validation requires --repeats to be at least 1.")
        return 1
    if not os.path.exists(results_file):
        print(f"Validation failed: {results_file} does not exist.")
        return 1

    games = []
    try:
        with open(results_file, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    games.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Validation failed: invalid JSON at {results_file}:{line_number}: {e}")
                    return 1
    except OSError as e:
        print(f"Validation failed: could not read {results_file}: {e}")
        return 1

    ids = [g.get("game_id") for g in games if g.get("game_id")]
    if len(ids) != len(games):
        print("Validation failed: every game must have a game_id.")
        return 1
    if len(ids) != len(set(ids)):
        print("Validation failed: duplicate game IDs found.")
        return 1

    expected_ids = {
        make_game_id(answerer, guesser, secret, repeat)
        for answerer in MODELS for guesser in MODELS for secret in SECRET_WORDS
        for repeat in range(repeats)
    }
    if set(ids) != expected_ids:
        missing = len(expected_ids - set(ids))
        unexpected = len(set(ids) - expected_ids)
        print(
            "Validation failed: results do not match the selected model/revision schedule "
            f"({missing} missing, {unexpected} unexpected)."
        )
        return 1

    print(
        f"Validated {len(games)} unique games across 16 directed pairings, "
        f"{len(SECRET_WORDS)} balanced targets, and the pinned model revisions."
    )
    return 0

def main():
    random.seed(SEED)
    if ARGS.validate_only:
        return validate_round_robin(ARGS.validate_only, ARGS.repeats)
    if ARGS.smoke_models:
        return smoke_models()

    if ARGS.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    if not MOCK:
        print("All four pinned Hugging Face checkpoints run locally.")

    games = []
    expected_jobs = [
        (answerer, guesser, secret, repeat,
         make_game_id(answerer, guesser, secret, repeat))
        for answerer in MODELS for guesser in MODELS for secret in SECRET_WORDS
        for repeat in range(ARGS.repeats)
    ]
    expected_ids = {job[4] for job in expected_jobs}
    if os.path.exists(ARGS.results_file):
        try:
            with open(ARGS.results_file, encoding="utf-8") as f:
                games = [json.loads(line) for line in f if line.strip()]
        except (OSError, json.JSONDecodeError) as e:
            print(f"Could not read existing results file {ARGS.results_file}: {e}")
            return 1
        if any(game.get("game_id") not in expected_ids for game in games):
            print(
                f"Existing results file {ARGS.results_file} contains another cohort, "
                "model revision, or repeat schedule. Choose a fresh results file."
            )
            return 1
    done = {g["game_id"] for g in games}
    jobs = [job for job in expected_jobs if job[4] not in done]
    print(f"{len(done)} games already saved; playing {len(jobs)} more...")
    failed = 0
    def run(job):
        try:
            prepare_hf_pair(*job[:2])
            return play(*job[:3], game_id=job[4])
        except RuntimeError as e:
            print("  skipped:", e)
            return None
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.results_file)), exist_ok=True)
    with ThreadPoolExecutor(WORKERS) as ex, open(ARGS.results_file, "a", encoding="utf-8") as f:
        for gm in ex.map(run, jobs):
            if gm is None:
                failed += 1
                continue
            games.append(gm)
            f.write(json.dumps(gm) + "\n"); f.flush()
    if failed:
        print(
            f"\n{failed} games could not finish because model loading or generation failed. "
            "Fix the reported errors and rerun to resume."
        )
        return 1
    validation = validate_round_robin(ARGS.results_file, ARGS.repeats)
    if validation != 0:
        return validation

    names = [m["name"] for m in MODELS]
    wins = [g for g in games if g["won"]]
    n_ans = sum(g["answer_turns"] for g in games)
    print("\n===== Paste these into the proposal =====")
    print(f"[N_GAMES] = {len(games)}")
    print(f"[WIN_OVERALL] = {len(wins) / len(games):.0%}")
    print(f"[AVG_Q] = {sum(g['turns_used'] for g in wins) / max(len(wins), 1):.1f}")
    print(f"[INVALID_RATE] = {sum(g['invalid_answers'] for g in games) / max(n_ans, 1):.1%}")
    print("\nBad guesser moves (empty or broken questions) per guesser:")
    for g in names:
        print(f"  {g}: {sum(x.get('bad_guesser_moves', 0) for x in games if x['guesser'] == g)}")
    print("\nWin-rate table (rows = answerer, columns = guesser):")
    print("| Answerer / Guesser | " + " | ".join(names) + " |")
    for a in names:
        cells = []
        for g in names:
            sub = [x for x in games if x["answerer"] == a and x["guesser"] == g]
            cells.append(f"{sum(x['won'] for x in sub)}/{len(sub)}")
        print(f"| {a} | " + " | ".join(cells) + " |")
    print(f"\nFull game logs saved to {ARGS.results_file}.")

def smoke_models():
    print("Smoke test loads and generates locally with each selected model.")
    for model in MODELS:
        try:
            prepare_hf_pair(model)
            response = chat(model, [{"role": "user", "content": "Reply with exactly the word OK."}])
        except RuntimeError as e:
            print(f"FAILED {model['name']}: {e}")
            return 1
        print(f"OK {model['name']} ({model['model']}): {response[:120]!r}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main() or 0)
