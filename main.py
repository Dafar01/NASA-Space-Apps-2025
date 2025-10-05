import json, requests
from datetime import datetime

NASA_SPACE_APPS_API = "gBRgMkkHStdcEbV8UO91rGpIbiHqVLEVmVEIjg2a"

def fetch_asteroid_data():
    today = datetime.today().strftime('%Y-%m-%d')

    url = f"https://api.nasa.gov/planetary/apod?api_key={NASA_SPACE_APPS_API}&date={today}"

    url2 = f"https://api.nasa.gov/neo/rest/v1/feed?api_key={NASA_SPACE_APPS_API}"

    browse_url = f"https://api.nasa.gov/neo/rest/v1/neo/browse?api_key={NASA_SPACE_APPS_API}"

    jana_url = f"https://ssd-api.jpl.nasa.gov/sbdb_query.api?info=fields&full-prec=2"


    try:
        response = requests.get(jana_url)
        response.raise_for_status()  
        data = json.loads(response.text)

        print(data)
        file = "./asteroid_data_jana.txt"
        with open(file, "w") as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"Error fetching APOD data: {e}")
    except json.JSONDecodeError:
        print("Error decoding JSON response.")

#fetch_asteroid_data()

import json

def filter():
    file = "./asteroid_data_browse.txt"

    names = []
    temp_d = []
    vel = []

    names.append("Bennu")
    names.append("Itokawa")
    names.append("Amun")

    temp_d.append(0.5)
    temp_d.append(0.33)
    temp_d.append(3.34)

    vel.append(28)
    vel.append(33.72)
    vel.append(17.34)

    # Open and parse the JSON file
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)   # parse JSON into Python dict

    # Loop through all asteroids and collect names
    for asteroid in data["near_earth_objects"]:
        names.append(asteroid["name_limited"])
        temp_d.append((asteroid["estimated_diameter"]["kilometers"]["estimated_diameter_max"] + asteroid["estimated_diameter"]["kilometers"]["estimated_diameter_min"])/2)
        cad_list = asteroid.get("close_approach_data", [])
        if cad_list:
            rel = cad_list[0].get("relative_velocity", {})
            kps = rel.get("kilometers_per_second")
            vel.append(float(kps) if kps is not None else None)

    # Print the results
    # print("Asteroid Names:")
    # for idx, n in enumerate(names):
    #     print(n, " ", temp_d[idx], " ", vel[idx], end="\n")

    # for d in temp_d:
    #     print(d, end="\n")
    
    # print("####################################")

    # for v in vel:
    #     print(v, end="\n")

    # print({
    #     "names": names,
    #     "size": temp_d,
    #     "velocity": vel
    #     }
    #  )

    return {
        "names": names,
        "size": temp_d,
        "velocity": vel
        }
     


#filter()