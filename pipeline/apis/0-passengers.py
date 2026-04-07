#!/usr/bin/env python3
"""Module that returns the list of ships that can hold a given number of passengers"""
import requests


def availableShips(passengerCount):
    """Returns list of ship names that can hold at least passengerCount passengers"""
    ships = []
    url = "https://swapi-api.hbtn.io/api/starships/"

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break
        data = response.json()
        for ship in data.get("results", []):
            passengers = ship.get("passengers", "0")
            passengers = passengers.replace(",", "").strip()
            try:
                if int(passengers) >= passengerCount:
                    ships.append(ship["name"])
            except (ValueError, TypeError):
                pass
        url = data.get("next")

    return ships
