# calls stations_table.py functions to initialte plant 


from stations_table import get_stations_table, generate_plant_config

def main():

    stations_table = get_stations_table()

    print("Stations Table:")
    print(stations_table)

    plant_config = generate_plant_config(stations_table)

    print("\nPlant config successfully generated.")

if __name__ == "__main__":
    main()