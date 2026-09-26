"""Reverse proxy that adds a default presence_penalty to OpenAI-style generation requests.

vLLM applies only temperature/top_k/top_p/min_p/repetition_penalty/max_new_tokens as server-side defaults
(generation_config.json / --override-generation-config), and the Rust frontend ignores the override flag entirely,
so presence_penalty has to come from the client. This lets unmodified clients (benchmarks, agents) get one.
Requests that already set presence_penalty pass through unchanged. Streaming responses are relayed as they arrive.

Usage: pp_proxy.py <listen_port> <upstream_url> <presence_penalty>
       GET /proxy_stats reports how many requests were injected vs passed through.
"""
import json
import sys

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

LISTEN, UPSTREAM, PENALTY = int(sys.argv[1]), sys.argv[2].rstrip("/"), float(sys.argv[3])
GEN_PATHS = ("/v1/chat/completions", "/v1/completions")
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "content-length", "host", "content-encoding"}
stats = {"penalty": PENALTY, "injected": 0, "passthrough": 0}


async def proxy(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    if request.method == "POST" and request.path in GEN_PATHS and body:
        payload = json.loads(body)
        if "presence_penalty" in payload:
            stats["passthrough"] += 1
        else:
            payload["presence_penalty"] = PENALTY
            stats["injected"] += 1
        body = json.dumps(payload).encode()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_HEADERS}
    async with request.app["session"].request(request.method, UPSTREAM + request.path_qs, data=body,
                                              headers=headers) as upstream:
        response = web.StreamResponse(status=upstream.status, headers={
            k: v for k, v in upstream.headers.items() if k.lower() not in HOP_HEADERS})
        await response.prepare(request)
        async for chunk in upstream.content.iter_any():
            await response.write(chunk)
        await response.write_eof()
        return response


async def proxy_stats(_: web.Request) -> web.Response:
    return web.json_response(stats)


async def open_session(app: web.Application) -> None:
    app["session"] = ClientSession(connector=TCPConnector(limit=0), timeout=ClientTimeout(total=None),
                                   auto_decompress=True)


async def close_session(app: web.Application) -> None:
    await app["session"].close()


app = web.Application(client_max_size=64 * 1024 * 1024)  # 128K-token prompts are ~1 MB of JSON text
app.on_startup.append(open_session)
app.on_cleanup.append(close_session)
app.router.add_get("/proxy_stats", proxy_stats)
app.router.add_route("*", "/{tail:.*}", proxy)
web.run_app(app, host="0.0.0.0", port=LISTEN, print=None)
