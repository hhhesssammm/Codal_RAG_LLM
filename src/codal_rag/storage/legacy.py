from typing import Any

# ============================================
# Retrieval JSON parsing
# ============================================


def unwrap_results(
    value: Any,
) -> list[dict] | None:

    if isinstance(
        value,
        list,
    ):

        if not value:
            return []

        if all(
            isinstance(item, dict)
            and "chunk" in item
            for item in value
        ):

            return value

    if isinstance(
        value,
        dict,
    ):

        for key in (
            "results",
            "retrieval_results",
            "hits",
            "items",
        ):

            if key in value:

                result = unwrap_results(
                    value[key]
                )

                if result is not None:
                    return result

    return None


def find_method_node(
    value: Any,
    method: str,
) -> Any | None:

    if isinstance(
        value,
        dict,
    ):

        if method in value:

            return value[
                method
            ]

        for key in (
            f"{method}_results",
            f"{method}_retrieval",
        ):

            if key in value:

                return value[
                    key
                ]

        for child in value.values():

            found = find_method_node(
                child,
                method,
            )

            if found is not None:
                return found

    elif isinstance(
        value,
        list,
    ):

        for child in value:

            found = find_method_node(
                child,
                method,
            )

            if found is not None:
                return found

    return None


def select_retrieval_results(
    payload: Any,
    method: str,
) -> list[dict]:

    method_node = (
        find_method_node(
            payload,
            method,
        )
    )

    if method_node is None:

        direct = unwrap_results(
            payload
        )

        if direct is not None:
            return direct

        raise ValueError(
            f"Could not locate retrieval "
            f"results for method={method}"
        )

    results = unwrap_results(
        method_node
    )

    if results is None:

        raise ValueError(
            f"Invalid retrieval structure "
            f"for method={method}"
        )

    return results