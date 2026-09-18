from ortools.sat.python import cp_model
import pandas as pd


# =========================================================
# 1. READ DATA
# =========================================================

lines = pd.read_csv("01_LINES.csv")
stations = pd.read_csv("02_STATIONS.csv")
sectors = pd.read_csv("03_SECTORS.csv")
location_supply = pd.read_csv("04_LOCATION_SUPPLY.csv")
buffer_location = pd.read_csv("05_BUFFER_LOCATION.csv")
parameters = pd.read_csv("06_PARAMETERS.csv")
project_details = pd.read_csv("07_PROJECT_DETAILS.csv")
activity_details = pd.read_csv("08_ACTIVITY_DETAILS.csv")


# =========================================================
# 2. PLANNING HORIZON
# =========================================================

horizon_start = pd.to_datetime(
    parameters.loc[
        parameters["key"] == "horizon_start",
        "value"
    ].iloc[0]
)

horizon_weeks = int(
    parameters.loc[
        parameters["key"] == "horizon_weeks",
        "value"
    ].iloc[0]
)

weeks = range(1, horizon_weeks + 1)


# =========================================================
# 3. CONVERT ACTIVITY DATES TO WEEKS
# =========================================================

activity_details["planned_start_date"] = pd.to_datetime(
    activity_details["planned_start_date"]
)

activity_details["planned_start_week"] = (
    (activity_details["planned_start_date"] - horizon_start).dt.days // 7
) + 1


# =========================================================
# 4. MERGE ACTIVITY + PROJECT DATA
# =========================================================
# activity_type is included because the weekly access-night cap is defined
# for contract_number + activity_type.

activity_data = activity_details.merge(
    project_details,
    on=["contract_number", "activity_type"],
    how="left",
    validate="many_to_one"
)


if activity_data["number_of_workfronts"].isna().any():
    missing = activity_data.loc[
        activity_data["number_of_workfronts"].isna(),
        ["activity_id", "contract_number", "activity_type"]
    ]
    raise ValueError(
        "Some activities could not be matched to PROJECT_DETAILS:\n"
        + missing.to_string(index=False)
    )


# =========================================================
# 5. BUILD PHYSICAL LOCATION PATH FOR EACH ACTIVITY
# =========================================================
#
# The activity input gives start/end tunnel-sector IDs such as:
#   SEC:BET:S15_S16:EB -> SEC:BET:S16_S17:EB
#
# For the first capacity model, an activity occupies:
#   - every tunnel sector from its start sector through its end sector
#   - every platform BETWEEN consecutive occupied tunnel sectors
#
# Example:
#   SEC:S15_S16 -> SEC:S16_S17
# becomes:
#   SEC:S15_S16, PLAT:S16, SEC:S16_S17
#
# If start=end, only that tunnel sector is occupied.
#
# NOTE: This is the direct topology interpretation used here. When the
# supplied `trackaccess expand` implementation is available, compare its
# expansion with this function and make it authoritative if it differs.


def parse_activity_location(location_id):
    """Split a bound-specific tunnel location into base sector and bound."""
    parts = location_id.split(":")

    if len(parts) != 4 or parts[0] != "SEC":
        raise ValueError(f"Expected tunnel-sector location, got: {location_id}")

    base_sector_id = ":".join(parts[:3])
    bound = parts[3]

    return base_sector_id, bound


# Lookup each base sector's line and order along that line.
sector_lookup = sectors.set_index("sector_id").to_dict("index")

# Separate ordered sector list for each railway line.
line_sector_paths = {}

for line_code, group in sectors.groupby("line_code"):
    line_sector_paths[line_code] = (
        group.sort_values("seq")["sector_id"].tolist()
    )


def activity_occupied_locations(start_location_id, end_location_id):
    """
    Return all capacity locations occupied by one activity access.
    """

    start_sector, start_bound = parse_activity_location(start_location_id)
    end_sector, end_bound = parse_activity_location(end_location_id)

    if start_bound != end_bound:
        raise ValueError(
            f"Activity changes bound: {start_location_id} -> {end_location_id}"
        )

    if start_sector not in sector_lookup:
        raise ValueError(f"Unknown start sector: {start_sector}")

    if end_sector not in sector_lookup:
        raise ValueError(f"Unknown end sector: {end_sector}")

    start_line = sector_lookup[start_sector]["line_code"]
    end_line = sector_lookup[end_sector]["line_code"]

    if start_line != end_line:
        raise ValueError(
            f"Activity changes line: {start_location_id} -> {end_location_id}"
        )

    path = line_sector_paths[start_line]

    start_index = path.index(start_sector)
    end_index = path.index(end_sector)

    # Activities in the supplied data are expected to follow the line
    # from start sector to end sector. Supporting the reverse direction
    # as well makes the function safer for hidden instances.
    if start_index <= end_index:
        selected_sectors = path[start_index:end_index + 1]
    else:
        selected_sectors = list(
            reversed(path[end_index:start_index + 1])
        )

    occupied = []

    for i, sector_id in enumerate(selected_sectors):

        # Add the bound to the base sector ID.
        occupied.append(f"{sector_id}:{start_bound}")

        # Between two adjacent tunnel sectors is a station platform.
        if i < len(selected_sectors) - 1:
            current = sector_lookup[selected_sectors[i]]
            following = sector_lookup[selected_sectors[i + 1]]

            # Find their common station.
            current_stations = {
                current["from_station_id"],
                current["to_station_id"]
            }

            following_stations = {
                following["from_station_id"],
                following["to_station_id"]
            }

            common = current_stations.intersection(following_stations)

            if len(common) != 1:
                raise ValueError(
                    "Could not determine platform between "
                    f"{selected_sectors[i]} and {selected_sectors[i + 1]}"
                )

            station_id = next(iter(common))

            occupied.append(
                f"PLAT:{start_line}:{station_id}:{start_bound}"
            )

    return occupied


# Pre-compute the physical locations used by every activity.
activity_locations = {}

for _, activity in activity_data.iterrows():

    activity_id = activity["activity_id"]

    activity_locations[activity_id] = activity_occupied_locations(
        activity["start_location_id"],
        activity["end_location_id"]
    )

# activity_locations is a dictionary with key: activity_id, and value a list of all stations affected

# Make sure every generated location actually exists in LOCATION_SUPPLY.
known_locations = set(location_supply["location_id"])

for activity_id, occupied_locations in activity_locations.items():
    for location_id in occupied_locations:
        if location_id not in known_locations:
            raise ValueError(
                f"{activity_id} generated unknown location: {location_id}"
            )


# Capacity lookup.
location_capacity = dict(
    zip(
        location_supply["location_id"],
        location_supply["supply_capacity"].astype(int)
    )
)


#location_capacity is a dictionary with key: location_id, and value the supply capacity of the location

# =========================================================
# 6. CREATE MODEL
# =========================================================

model = cp_model.CpModel()


# =========================================================
# 7. ACCESS-NIGHT DECISION VARIABLES
# =========================================================
#
# access[activity_id, week, night] = 1
# means that the activity receives an access in that week using that
# contract+activity_type's local access-night number.

access = {}

for _, activity in activity_data.iterrows():

    activity_id = activity["activity_id"]

    max_nights = int(
        activity["number_of_maximum_access_per_week"]
    )

    for week in weeks:
        for night in range(1, max_nights + 1):

            access[activity_id, week, night] = model.NewBoolVar(
                f"access_{activity_id}_week_{week}_night_{night}"
            )


# =========================================================
# 8. AT MOST ONE ACCESS PER ACTIVITY PER WEEK
# =========================================================

for _, activity in activity_data.iterrows():

    activity_id = activity["activity_id"]

    max_nights = int(
        activity["number_of_maximum_access_per_week"]
    )

    for week in weeks:

        model.Add(
            sum(
                access[activity_id, week, night]
                for night in range(1, max_nights + 1)
            )
            <= 1
        )


# =========================================================
# 9. WORKLOAD CONSERVATION - SCENARIO A
# =========================================================
# Scenario A has no ECLO, so every scheduled access contributes 1.0 unit.

for _, activity in activity_data.iterrows():

    activity_id = activity["activity_id"]
    total_accesses = int(activity["total_accesses"])

    max_nights = int(
        activity["number_of_maximum_access_per_week"]
    )

    model.Add(
        sum(
            access[activity_id, week, night]
            for week in weeks
            for night in range(1, max_nights + 1)
        )
        == total_accesses
    )


# =========================================================
# 10. PLANNED START CONSTRAINT
# =========================================================

for _, activity in activity_data.iterrows():

    activity_id = activity["activity_id"]
    start_week = int(activity["planned_start_week"])

    max_nights = int(
        activity["number_of_maximum_access_per_week"]
    )

    for week in weeks:

        if week < start_week:

            for night in range(1, max_nights + 1):

                model.Add(
                    access[activity_id, week, night] == 0
                )


# =========================================================
# 11. WORKFRONT CONSTRAINT
# =========================================================
#
# The access-night index is local to contract_number + activity_type.
# No more than number_of_workfronts distinct activities may use the same
# local access night.

for (contract_number, activity_type), contract_activities in (
    activity_data.groupby(["contract_number", "activity_type"])
):

    project = contract_activities.iloc[0]

    max_nights = int(
        project["number_of_maximum_access_per_week"]
    )

    workfronts = int(
        project["number_of_workfronts"]
    )

    activity_ids = contract_activities["activity_id"].tolist()

    for week in weeks:

        for night in range(1, max_nights + 1):

            model.Add(
                sum(
                    access[activity_id, week, night]
                    for activity_id in activity_ids
                )
                <= workfronts
            )


# =========================================================
# 12. LOCATION-WEEK OCCUPANCY VARIABLES
# =========================================================
#
# occupied[activity_id, week] = 1 exactly when the activity has an
# access in that week.
#
# We use this helper variable because LOCATION_SUPPLY is expressed at
# location-week level, while access-night is a local contract index.

occupied = {}

for _, activity in activity_data.iterrows():

    activity_id = activity["activity_id"]

    max_nights = int(
        activity["number_of_maximum_access_per_week"]
    )

    for week in weeks:

        occupied[activity_id, week] = model.NewBoolVar(
            f"occupied_{activity_id}_week_{week}"
        )

        # Because an activity can have at most one access per week,
        # this sum is itself either 0 or 1.
        model.Add(
            occupied[activity_id, week]
            ==
            sum(
                access[activity_id, week, night]
                for night in range(1, max_nights + 1)
            )
        )


# =========================================================
# 13. LOCATION CAPACITY - BASELINE, BEFORE CO-SHARING
# =========================================================
#
# For every location and week:
#
#     number of activity possessions using that location
#     <= LOCATION_SUPPLY.supply_capacity
#
# IMPORTANT:
# This is deliberately the pre-co-sharing version of the capacity rule.
# It counts each activity as one possession slot. Therefore it is
# conservative for PC/C activities that could legally co-share.
#
# Later, when co_share_group is modelled, capacity should count
# possession GROUPS rather than individual activities.

activities_using_location = {
    location_id: []
    for location_id in location_capacity
}

for activity_id, occupied_locations in activity_locations.items():
    for location_id in occupied_locations:
        activities_using_location[location_id].append(activity_id)


for location_id, capacity in location_capacity.items():

    relevant_activities = activities_using_location[location_id]

    if not relevant_activities:
        continue

    for week in weeks:

        model.Add(
            sum(
                occupied[activity_id, week]
                for activity_id in relevant_activities
            )
            <= capacity
        )


# =========================================================
# 14. SIMPLE TEMPORARY OBJECTIVE
# =========================================================
# This is NOT the final Scenario A objective.
# It simply encourages the solver to place accesses earlier.

model.Minimize(
    sum(
        week * access[activity["activity_id"], week, night]
        for _, activity in activity_data.iterrows()
        for week in weeks
        for night in range(
            1,
            int(activity["number_of_maximum_access_per_week"]) + 1
        )
    )
)


# =========================================================
# 15. SOLVE
# =========================================================

solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 30

status = solver.Solve(model)


# =========================================================
# 16. PRINT RESULTS
# =========================================================


if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):

    print("Schedule found!")

    for _, activity in activity_data.iterrows():

        activity_id = activity["activity_id"]

        max_nights = int(
            activity["number_of_maximum_access_per_week"]
        )

        selected_accesses = []

        for week in weeks:

            for night in range(1, max_nights + 1):

                if solver.Value(
                    access[activity_id, week, night]
                ) == 1:

                    selected_accesses.append(
                        (week, night)
                    )

        print(
            activity_id,
            selected_accesses,
            "locations:",
            activity_locations[activity_id]
        )

else:

    print("No feasible schedule found.")


