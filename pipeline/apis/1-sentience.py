#!/usr/bin/env python3
"""Module that returns the list of home planets of all sentient species"""
import requests


def sentientPlanets():
    """Returns list of planet names that are home to sentient species"""
    planets = []
    url = "https://swapi-api.hbtn.io/api/species/"

    while url:
        response = requests.get(url)
        if response.status_code != 200:
            break
        data = response.json()
        for species in data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()
            if "sentient" in classification or "sentient" in designation:
                homeworld = species.get("homeworld")
                if homeworld:
                    planet_response = requests.get(homeworld)
                    if planet_response.status_code == 200:
                        planet_data = planet_response.json()
                        planets.append(planet_data.get("name"))
        url = data.get("next")

    return planets
