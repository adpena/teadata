"""Build explicit, offline PEIMS Single File Financial Budget candidates."""

from __future__ import annotations

from teadata.peims_actuals import materialize_peims_budgets
from teadata.scripts.refresh_peims_actuals import _run


def main(argv: list[str] | None = None) -> int:
    return _run(argv, __doc__, materialize_peims_budgets)


if __name__ == "__main__":
    raise SystemExit(main())
