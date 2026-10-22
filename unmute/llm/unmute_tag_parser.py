from collections.abc import AsyncIterator

_OPEN_TAGS = ("<speech>", "<think>", "<plan>", "<exec>", "<action_result>")
_CLOSE_TAGS = ("</speech>", "</think>", "</plan>", "</exec>", "</action_result>")
_MAX_TAG_LEN = max(len(t) for t in _OPEN_TAGS + _CLOSE_TAGS)


def _is_prefix_of_any(s: str, candidates: tuple[str, ...]) -> bool:
    return any(c.startswith(s) for c in candidates)


async def extract_speech_tags(
    iterator: AsyncIterator[str],
    *,
    emit_chunk_size: int = 32,
) -> AsyncIterator[str]:
    """Yield only text contained inside <speech>...</speech> tags.

    Handles tag boundaries split across deltas. Multiple <speech> blocks are
    yielded in order. Unclosed <speech> at EOS is flushed as best-effort.
    """
    in_speech = False
    pending = ""  # trailing '<...' that might still become a tag
    out = ""  # outgoing speech chars, flushed in chunks

    async for delta in iterator:
        # Matching restarts at every '<'. Growing a buffer and dropping it when
        # it stops being a tag prefix loses the next tag whenever the character
        # that broke the match is itself a '<' -- so a stuttered "<spe<speech>"
        # would swallow the whole utterance and the robot would go silent while
        # planning correctly. See LLMTagPrinter.feed for the live case.
        buf = pending + delta
        pending = ""
        i = 0
        while i < len(buf):
            ch = buf[i]

            if ch != "<":
                if in_speech:
                    out += ch
                    if len(out) >= emit_chunk_size:
                        yield out
                        out = ""
                i += 1
                continue

            candidates = _CLOSE_TAGS if in_speech else (_OPEN_TAGS + _CLOSE_TAGS)
            chunk = buf[i:]
            matched = next((c for c in candidates if chunk.startswith(c)), None)
            if matched is not None:
                if matched == "<speech>":
                    in_speech = True
                elif matched == "</speech>":
                    in_speech = False
                    if out:
                        yield out
                        out = ""
                i += len(matched)
                continue

            if _is_prefix_of_any(chunk, candidates):
                pending = chunk
                break

            if in_speech:
                out += ch
                if len(out) >= emit_chunk_size:
                    yield out
                    out = ""
            i += 1

    # EOS flush
    if in_speech:
        if pending:
            out += pending
        if out:
            yield out


_TAG_NAMES: tuple[str, ...] = tuple(t[1:-1] for t in _OPEN_TAGS)


class LLMTagPrinter:
    """Synchronous incremental parser that yields complete closed tag blocks.

    Call feed(delta) per incoming chunk; returns a (possibly empty) list of
    (tag_name, content) pairs for tags closed within this feed.
    flush() at end-of-response discards any unclosed buffer.

    Content outside any recognized tag is silently discarded. Empty tag
    bodies are not yielded.
    """

    __slots__ = ("_current_tag", "_content_buf", "_pending")

    def __init__(self) -> None:
        self._current_tag: str | None = None
        self._content_buf: str = ""
        self._pending: str = ""

    def feed(self, delta: str) -> list[tuple[str, str]]:
        """Consume a chunk; return the tag blocks that closed inside it.

        Matching restarts at every '<'. The obvious incremental version -- grow
        a buffer while it is still a prefix of some tag, throw it away when it
        stops being one -- silently eats the NEXT tag whenever the character
        that breaks the match is itself a '<'. A stream opening
        ``"<th" + "<think>..."`` (seen live on 2026-09-14, the model emitting a
        stuttered tag) left the parser discarding a real ``<think>`` and then
        the entire turn: plan, speech and exec all vanished while the backend's
        TTS happily spoke the speech aloud. Rescanning from each '<' costs
        nothing at these lengths and cannot lose a tag that is actually there.
        """
        out: list[tuple[str, str]] = []
        buf = self._pending + delta
        self._pending = ""
        i = 0
        while i < len(buf):
            ch = buf[i]

            if ch != "<":
                if self._current_tag is not None:
                    self._content_buf += ch
                i += 1
                continue

            candidates: tuple[str, ...] = (
                _OPEN_TAGS + _CLOSE_TAGS
                if self._current_tag is None
                else (f"</{self._current_tag}>",)
            )
            chunk = buf[i:]
            matched = next((c for c in candidates if chunk.startswith(c)), None)
            if matched is not None:
                if self._current_tag is None:
                    self._current_tag = matched[1:-1]
                    self._content_buf = ""
                else:
                    if self._content_buf:
                        out.append((self._current_tag, self._content_buf))
                    self._current_tag = None
                    self._content_buf = ""
                i += len(matched)
                continue

            if _is_prefix_of_any(chunk, candidates):
                # Might still become a tag once more of the stream arrives.
                # Bounded by the longest tag, so a stray '<' cannot stall us.
                self._pending = chunk
                return out

            # Not a tag: this '<' is ordinary text. Carry on from the NEXT
            # character so a '<' later in the run still starts a match.
            if self._current_tag is not None:
                self._content_buf += ch
            i += 1

        return out

    def flush(self) -> list[tuple[str, str]]:
        self._current_tag = None
        self._content_buf = ""
        self._pending = ""
        return []
