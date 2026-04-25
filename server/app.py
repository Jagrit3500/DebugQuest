from openenv.core.env_server.http_server import create_app

try:
    from ..models import DebugQuestAction, DebugQuestObservation
    from .debug_quest_environment import DebugQuestEnvironment
except ImportError:
    from models import DebugQuestAction, DebugQuestObservation
    from server.debug_quest_environment import DebugQuestEnvironment


app = create_app(
    DebugQuestEnvironment,
    DebugQuestAction,
    DebugQuestObservation,
    env_name="debug_quest",
    max_concurrent_envs=1,
)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()