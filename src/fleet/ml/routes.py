"""Route optimisation over the six-city network.

For the small demo network we can solve the travelling-salesman problem
exactly with permutations (at most 7 stops). For larger inputs we fall back
to nearest-neighbour followed by 2-opt. Distances come from the same
road-distance matrix used by the data pipeline, so results are reproducible
and require no external API key.
"""

from __future__ import annotations

from itertools import permutations

from fleet import config


def _distance(origin: str, destination: str) -> float:
    if origin == destination:
        return 0.0
    key = tuple(sorted((origin, destination)))
    base = config.CITY_DISTANCES.get(key) or config.CITY_DISTANCES[tuple(reversed(key))]
    return float(base)


def _validate_cities(cities: list[str]) -> None:
    unknown = sorted(set(cities) - set(config.CITIES))
    if unknown:
        raise ValueError(f"Unknown cities: {', '.join(unknown)}")


def _route_distance(origin: str, order: list[str], return_to_origin: bool) -> float:
    route = [origin, *order]
    if return_to_origin:
        route.append(origin)
    return sum(_distance(a, b) for a, b in zip(route, route[1:]))


def _nearest_neighbour(origin: str, stops: list[str]) -> list[str]:
    remaining = list(stops)
    order: list[str] = []
    current = origin
    while remaining:
        nxt = min(remaining, key=lambda city: _distance(current, city))
        order.append(nxt)
        remaining.remove(nxt)
        current = nxt
    return order


def _two_opt(origin: str, order: list[str], return_to_origin: bool) -> list[str]:
    best = list(order)
    improved = True
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 2, len(best) + 1):
                candidate = best[:i] + list(reversed(best[i:j])) + best[j:]
                if _route_distance(origin, candidate, return_to_origin) < _route_distance(
                    origin, best, return_to_origin
                ):
                    best = candidate
                    improved = True
    return best


def optimize_route(origin: str, stops: list[str], return_to_origin: bool = True) -> dict:
    """Return baseline and optimised distances for a multi-stop route."""
    _validate_cities([origin, *stops])
    if not stops:
        raise ValueError("At least one stop is required")

    # Deduplicate while preserving order; a stop equal to the origin is dropped.
    cleaned: list[str] = []
    for stop in stops:
        if stop != origin and stop not in cleaned:
            cleaned.append(stop)
    if not cleaned:
        raise ValueError("At least one stop different from the origin is required")

    baseline_distance = _route_distance(origin, cleaned, return_to_origin)

    if len(cleaned) <= 7:
        optimized = min(
            permutations(cleaned),
            key=lambda order: _route_distance(origin, list(order), return_to_origin),
        )
        optimized_order = list(optimized)
    else:
        optimized_order = _two_opt(
            origin, _nearest_neighbour(origin, cleaned), return_to_origin
        )

    optimized_distance = _route_distance(origin, optimized_order, return_to_origin)
    route = [origin, *optimized_order] + ([origin] if return_to_origin else [])
    legs = [
        {"from": a, "to": b, "distance_km": round(_distance(a, b), 1)}
        for a, b in zip(route, route[1:])
    ]
    savings = baseline_distance - optimized_distance
    savings_pct = (savings / baseline_distance * 100) if baseline_distance else 0.0

    return {
        "origin": origin,
        "stops": cleaned,
        "return_to_origin": return_to_origin,
        "baseline_order": cleaned,
        "optimized_order": optimized_order,
        "baseline_distance_km": round(baseline_distance, 1),
        "optimized_distance_km": round(optimized_distance, 1),
        "savings_km": round(savings, 1),
        "savings_pct": round(savings_pct, 2),
        "legs": legs,
    }
