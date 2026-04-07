from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from Mirage_RL.models import QueryAction, QueryObservation, QueryState


class QueryClient(EnvClient[QueryAction, QueryObservation, QueryState]):

    def _step_payload(self, action: QueryAction):
        return {
            "next_table": action.next_table,
            "join_type":  action.join_type,
            "use_index":  action.use_index,
        }

    def _parse_result(self, payload):
        obs = payload.get("observation", {})

        observation = QueryObservation(
            done              = payload.get("done", False),
            reward            = payload.get("reward", 0.0),
            tables            = obs.get("tables", []),
            table_rows        = obs.get("table_rows", []),
            selectivities     = obs.get("selectivities", []),
            has_index         = obs.get("has_index", []),
            chosen_order      = obs.get("chosen_order", []),
            remaining_tables  = obs.get("remaining_tables", []),
            step_number       = obs.get("step_number", 0),
            current_cost      = obs.get("current_cost", 0.0),
            query_context     = obs.get("query_context", ""),
            intermediate_size = obs.get("intermediate_size", 1.0),
        )

        return StepResult(
            observation = observation,
            reward      = payload.get("reward", 0.0),
            done        = payload.get("done", False),
        )

    def _parse_state(self, payload):
        return QueryState(
            chosen_order     = payload.get("chosen_order", []),
            remaining_tables = payload.get("remaining_tables", []),
            current_cost     = payload.get("current_cost", 0.0),
            final_cost       = payload.get("final_cost", 0.0),
            scenario_name    = payload.get("scenario_name", ""),
        )