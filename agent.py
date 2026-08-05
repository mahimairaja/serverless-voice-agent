import logging

import voicegateway
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics
from livekit.plugins.deepgram import STT, TTS
from livekit.plugins.openai import LLM

logger = logging.getLogger("agent")

load_dotenv()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            llm=LLM(),
            instructions="You are a helpful assistant",
        )


server = AgentServer()


@server.rtc_session(agent_name="agent")
async def my_agent(ctx: JobContext):

    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=STT(language="multi"),
        tts=TTS(),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
        ),
        preemptive_generation=True,
    )
    voicegateway.attach(session)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(model=ai_coustics.EnhancerModel.QUAIL_VF_S),
            ),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    voicegateway.register_worker("agent", local=True)
    cli.run_app(server)
