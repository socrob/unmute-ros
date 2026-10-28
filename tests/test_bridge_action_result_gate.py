"""When the bridge forwards <action_result> to the backend.

run_bridge() runs against two local websocket servers standing in for the laptop
bus (:8090) and the Unmute backend. The backend drops a tool message that arrives
while it is responding, so the bridge must hold results during a response and
must not hold them once the response is over.
"""

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
import websockets
from websockets.asyncio.server import ServerConnection

import unmute.unmute_ros_bridge_client as bridge

RESULT = '<action_result>{"action": "move", "status": "SUCCEEDED"}</action_result>'


class FakeEnds:
    def __init__(self):
        self.laptop: asyncio.Future = asyncio.get_running_loop().create_future()
        self.unmute: asyncio.Future = asyncio.get_running_loop().create_future()
        self.tool_messages: asyncio.Queue[str] = asyncio.Queue()

    async def laptop_handler(self, ws: ServerConnection):
        self.laptop.set_result(ws)
        await ws.wait_closed()

    async def unmute_handler(self, ws: ServerConnection):
        self.unmute.set_result(ws)
        async for message in ws:
            data = json.loads(message)
            if data.get("type") == "unmute.tool_message":
                self.tool_messages.put_nowait(data["content"])

    async def from_backend(self, *events: dict):
        ws = await self.unmute
        for event in events:
            await ws.send(json.dumps(event))
        await asyncio.sleep(0.1)  # let the bridge process them in order

    async def result_from_executor(self):
        ws = await self.laptop
        await ws.send(json.dumps({"type": "bridge.action_result", "content": RESULT}))

    async def next_tool_message(self, timeout: float) -> str | None:
        try:
            return await asyncio.wait_for(self.tool_messages.get(), timeout)
        except TimeoutError:
            return None


@pytest_asyncio.fixture
async def ends(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[FakeEnds]:
    ends = FakeEnds()
    async with (
        websockets.serve(ends.laptop_handler, "127.0.0.1", 0) as laptop_srv,
        websockets.serve(ends.unmute_handler, "127.0.0.1", 0) as unmute_srv,
    ):
        laptop_port = next(iter(laptop_srv.sockets)).getsockname()[1]
        unmute_port = next(iter(unmute_srv.sockets)).getsockname()[1]
        monkeypatch.setattr(bridge, "LAPTOP_WS_URL", f"ws://127.0.0.1:{laptop_port}")
        # No /v1/voices behind this URL: the bridge falls back to UNMUTE_VOICE.
        monkeypatch.setattr(
            bridge, "UNMUTE_WS_URL", f"ws://127.0.0.1:{unmute_port}/v1/realtime"
        )
        task = asyncio.create_task(bridge.run_bridge())
        await asyncio.wait_for(ends.unmute, 10)
        try:
            yield ends
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def _response_events():
    return (
        {"type": "response.text.delta", "delta": "On my way."},
        {"type": "response.text.done", "text": "On my way."},
        {"type": "response.audio.done"},
    )


@pytest.mark.asyncio
async def test_result_after_a_split_turn_is_sent_at_once(ends: FakeEnds):
    # The incident: a pause, a reply, the last word interrupts it, the backend
    # replies again from the STT flush, with no second speech_stopped.
    await ends.from_backend(
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "response.created", "response": {}},
        {"type": "unmute.interrupted_by_vad"},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "response.created", "response": {}},
        *_response_events(),
    )

    await ends.result_from_executor()

    assert await ends.next_tool_message(timeout=2.0) == RESULT


@pytest.mark.asyncio
async def test_result_during_a_response_waits_for_its_end(ends: FakeEnds):
    await ends.from_backend(
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "response.created", "response": {}},
    )

    await ends.result_from_executor()
    assert await ends.next_tool_message(timeout=0.5) is None

    await ends.from_backend(*_response_events())
    assert await ends.next_tool_message(timeout=2.0) == RESULT
