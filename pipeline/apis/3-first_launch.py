#!/usr/bin/env python3
"""Script that displays the first launch with SpaceX API"""
import requests


if __name__ == '__main__':
    response = requests.get("https://api.spacexdata.com/v5/launches/upcoming")
    launches = response.json()

    launch = sorted(
        launches,
        key=lambda x: x.get("date_unix", float("inf"))
    )[0]

    rocket = requests.get(
        "https://api.spacexdata.com/v4/rockets/{}".format(launch["rocket"])
    ).json()

    launchpad = requests.get(
        "https://api.spacexdata.com/v4/launchpads/{}".format(launch["launchpad"])
    ).json()

    print("{} ({}) {} - {} ({})".format(
        launch["name"],
        launch["date_local"],
        rocket["name"],
        launchpad["name"],
        launchpad["locality"]
    ))
