from openenv.core.env_server.http_server import create_app
import uvicorn

from Mirage_RL.models import QueryAction, QueryObservation
from Mirage_RL.server.Mirage_RL_environment import QueryEnv


app = create_app(
    QueryEnv,
    QueryAction,
    QueryObservation,
    env_name="Query_Optimizer",
    max_concurrent_envs=1,
)


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()