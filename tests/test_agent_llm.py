"""Agent side: the stub LLM, the OpenAI-compatible client against an in-process stub endpoint, parsing.

The endpoint is a FIXTURE (it is not a model): it listens on 127.0.0.1 on a thread of this test and is
stopped before the test returns.
"""
import asyncio
import threading

from aiohttp import web

from fusion_miniapp import decide, llm
from test_agent_layers import LAYERS, knobs_for, run_layer


def refill_request(n=3):
    gst = decide.generator_state("a7", 7, knobs_for(1))
    results = [{"candidate_idx": i, "survived": i == 0, "params": {k: lo for k, (lo, _) in decide.ranges().items()},
                "te_core_metric": 1.0, "te_wall_eV": 2.0, "ti_wall_eV": 3.0, "te_ped_eV": None, "wall_ok": False,
                "pedestal_ok": True, "is_good": False} for i in range(n + 1)]
    return decide.refill_messages(gst, {"generation": 0, "results": results}, n, knobs_for(1))


def test_stub_is_seeded_and_returns_valid_proposals():
    msgs = refill_request(3)
    assert "JSON list of 3 objects" in msgs[-1]["content"].splitlines()[-1]
    stub = llm.StubLLM()
    one, two = (asyncio.run(stub("a7", msgs, 64)) for _ in range(2))
    assert one == two
    got = decide.parse_candidates(one, 3)
    assert len(got) == 3 and all(c.keys() == decide.ranges().keys() for c in got)
    cst = decide.coordinator_state("a7", 7, knobs_for(1), {})
    assert asyncio.run(stub("a7", decide.summary_messages(cst), 64)).startswith("Stub summary: 0 candidates")
    seen = stub.seen()
    assert (seen["mode"], seen["calls"], seen["budgets"]) == ("stub", 3, {"64": 3})


def test_parse_candidate_clips_ignores_and_rejects():
    lo, hi = decide.ranges()["Delta"]
    assert decide.parse_candidate('thinking... {"Delta": 9, "a1_ne": 3.1, "junk": 1}') == {"Delta": hi, "a1_ne": 3.1}
    for bad in ("no json here", '{"a1_ne": "high"}', "[1, 2]", '{"a1_ne": true}', "{broken"):
        assert decide.parse_candidate(bad) is None
    for reply in ('<think>weigh {wall, pedestal}; draft {"a1_ne": 1.0}</think> {"a1_ne": 3.0}',   # reasoning first
                  '{"a1_ne": 3.0} I hope {this} helps', 'Answer: {"candidate": {"a1_ne": 3.0}}',  # prose, wrapped
                  '{"a1_ne": 2.0}\nrevised: {"a1_ne": 3.0}'):                                    # two objects
        assert decide.parse_candidate(reply) == {"a1_ne": 3.0}, reply


def test_parse_candidates_takes_the_last_list_and_at_most_n():
    assert decide.parse_candidates('draft [{"a1_ne": 3.0}] final [{"a1_ne": 3.1}, {"x": 1}, {"a1_ne": 3.2}]', 5) \
        == [{"a1_ne": 3.1}, {"a1_ne": 3.2}]                                  # unusable entries skipped
    assert decide.parse_candidates('[{"a1_ne": 3.0}, {"a1_ne": 3.1}, {"a1_ne": 3.2}]', 2) == [{"a1_ne": 3.0},
                                                                                             {"a1_ne": 3.1}]
    assert decide.parse_candidates('just one {"a1_ne": 3.05}', 4) == [{"a1_ne": 3.05}]   # a single object
    assert decide.parse_candidates("nothing", 3) == [] and decide.parse_candidates('[1, 2]', 3) == []


def test_model_proposals_replace_only_mutants_and_keep_the_random_stream():
    """The parent's repopulate runs first; proposals overwrite the mutable values of the mutant slots,
    in order, and the random generator's state is the same as without a model."""
    knobs = knobs_for(1, P=4, outer={"rng": "shared"})
    evo = decide.parent()
    gst = decide.generator_state("a3", 3, knobs)
    pop = evo.initialize_population(4, "three_region", "H", 3, delta_bounds=evo.DELTA_CLIP)
    rng = __import__("numpy").random.default_rng(3)
    msg = {"op": "repopulate", "survivors": [pop[0], pop[2]], "generation": 0, "rng": decide.rng_dump(rng),
           "results": [{"candidate_idx": i, "survived": i in (0, 2), "params": {k: p[k] for k in decide.ranges()},
                        "te_core_metric": 1.0, "te_wall_eV": 2.0, "ti_wall_eV": 3.0, "te_ped_eV": 4.0,
                        "wall_ok": False, "pedestal_ok": True, "is_good": False} for i, p in enumerate(pop)]}

    async def one(messages, b, r):
        assert r == decide.reply_tokens(2)
        return '[{"a1_ne": 3.0}]'
    knobs_m = knobs_for(1, P=4, outer={"rng": "shared", "generator": "mutation"})
    _, plain = asyncio.run(decide.generator_step(gst, msg, knobs_m, None))
    _, model = asyncio.run(decide.generator_step(gst, msg, knobs, one))
    assert plain["rng"] == model["rng"]                                   # the stream is untouched by the model
    assert model["sources"] == ["survivor", "survivor", "model", "mutation"]
    assert model["population"][:2] == plain["population"][:2] and model["population"][3] == plain["population"][3]
    assert model["population"][2] == {**plain["population"][2], "a1_ne": 3.0}
    assert model["refill"] == {"asked": 2, "filled": 1, "called": True}


def test_openai_client_sends_the_budget_from_both_layers(tmp_path, monkeypatch):
    seen = []

    async def chat(request):
        body = await request.json()
        seen.append(body)
        return web.json_response({
            "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": '{"a1_ne": 3.0}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    async def serve():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        return runner, site._server.sockets[0].getsockname()[1]

    loop = asyncio.new_event_loop()
    runner, port = loop.run_until_complete(serve())
    thread = threading.Thread(target=loop.run_forever)
    thread.start()
    try:
        monkeypatch.setenv("MINIAPP_TEST_LLM_URL", f"http://127.0.0.1:{port}/v1")
        knobs = knobs_for(3, thinking_budget=96)
        for layer in LAYERS:
            model = llm.make({"mode": "openai", "base_url_env": "MINIAPP_TEST_LLM_URL",
                              "api_key_env": "MINIAPP_TEST_LLM_KEY", "model": "test-model"})
            results, _ = run_layer(layer, knobs, tmp_path / layer, model=model)
            assert not [r for r in results if "error" in r]
            # the endpoint returns one object per refill: it fills one slot of those asked
            assert all(r["sources"]["model"] == r["refill"]["filled"] == 1 <= r["refill"]["asked"] for r in results)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
        loop.run_until_complete(runner.cleanup())
        loop.close()
    assert len(seen) == 2 * 3 * 2                                           # both layers, G calls per campaign
    assert {b["thinking_token_budget"] for b in seen} == {96}
    refills = {b["max_tokens"] - 96 for b in seen} - {decide.SUMMARY_TOKENS}
    assert refills and refills <= {decide.reply_tokens(n) for n in range(1, 4)}  # sized to the slots asked for
    assert {b["model"] for b in seen} == {"test-model"}
