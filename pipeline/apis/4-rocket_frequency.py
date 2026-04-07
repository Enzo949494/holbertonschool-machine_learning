#!/usr/bin/env python3
"""Script that displays the number of launches per rocket"""
import requests


if __name__ == '__main__':
    launches = requests.get("https://api.spacexdata.com/v4/launches").json()
    rockets = requests.get("https://api.spacexdata.com/v4/rockets").json()

    rocket_names = {rocket["id"]: rocket["name"] for rocket in rockets}

    counts: dict[str, int] = {}
    for launch in launches:
        rocket_id = launch.get("rocket")
        if rocket_id:
            counts[rocket_id] = counts.get(rocket_id, 0) + 1

    sorted_rockets = sorted(
        counts.items(),
        key=lambda x: (-x[1], rocket_names.get(x[0], ""))
    )

    for rocket_id, count in sorted_rockets:
        print("{}: {}".format(rocket_names[rocket_id], count))
