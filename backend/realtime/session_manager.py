from __future__ import annotations

import uuid
from dataclasses import asdict

from backend.realtime.config import RealtimeConfig
from backend.realtime.frame_processor import RealtimeFrameProcessor


class RealtimeSessionManager:
    """
    In-memory realtime processor manager.

    Each camera session gets its own processor/cache.
    This prevents different users/frames from sharing anchor smoothing state.
    """

    def __init__(self):
        self.sessions: dict[str, RealtimeFrameProcessor] = {}

    def create_session(
        self,
        config: RealtimeConfig | None = None,
    ) -> dict:
        session_id = f"rt_{uuid.uuid4()}"

        processor = RealtimeFrameProcessor(
            config=config or RealtimeConfig(),
        )

        self.sessions[session_id] = processor

        return {
            "session_id": session_id,
            "config": asdict(processor.config),
        }

    def get_or_create(
        self,
        session_id: str | None = None,
    ) -> tuple[str, RealtimeFrameProcessor]:
        if session_id and session_id in self.sessions:
            return session_id, self.sessions[session_id]

        created = self.create_session()
        new_id = created["session_id"]

        return new_id, self.sessions[new_id]

    def reset(
        self,
        session_id: str,
    ) -> bool:
        processor = self.sessions.get(session_id)

        if processor is None:
            return False

        processor.reset()
        return True

    def delete(
        self,
        session_id: str,
    ) -> bool:
        if session_id not in self.sessions:
            return False

        processor = self.sessions.pop(session_id)
        if hasattr(processor, "stop"):
            processor.stop()
        return True

    def list_sessions(self) -> dict:
        return {
            sid: {
                "frame_index": processor.frame_index,
                "has_ctx": processor.last_ctx is not None,
                "has_result": processor.last_result_bgr is not None,
                "last_error": processor.last_error,
                "config": asdict(processor.config),
            }
            for sid, processor in self.sessions.items()
        }


realtime_sessions = RealtimeSessionManager()
