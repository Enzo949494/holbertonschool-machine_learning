#!/usr/bin/env python3
"""Script that prints the location of a specific GitHub user"""
import requests
import sys
from datetime import datetime


if __name__ == '__main__':
    url = sys.argv[1]
    response = requests.get(url)

    if response.status_code == 404:
        print("Not found")
    elif response.status_code == 403:
        reset_timestamp = int(response.headers.get("X-Ratelimit-Reset", 0))
        reset_time = datetime.fromtimestamp(reset_timestamp)
        now = datetime.now()
        minutes = int((reset_time - now).total_seconds() / 60)
        print("Reset in {} min".format(minutes))
    else:
        data = response.json()
        location = data.get("location")
        if location:
            print(location)
        else:
            print("Not found")
