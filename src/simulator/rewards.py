

def reward_function(obs) -> float:
    """Calculate a reward term from a "raw observation"."""
    zone_blacklist = [
        "environment",
        "attic",
        "Breezeway",
        "crawlspace",
    ]

    def penalty(min, x, max):
        if x < min:
            return min - x
        elif max < x:
            return x - max
        else:
            return 0.0

    def zone_sum(obs, min, max):
        s = 0.0
        for i, v in obs["temperature"].items():
            if i in zone_blacklist:
                continue
            s += penalty(min, v, max)
        return s

    reward = 0.0

    # Temperature reward

    if 7 <= obs["time"]["current_time"] <= 19:
        reward -= zone_sum(obs, 20, 23)
    else:
        reward -= zone_sum(obs, 15, 30)
    return reward