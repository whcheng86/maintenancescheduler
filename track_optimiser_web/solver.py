from ortools.sat.python import cp_model
import pandas as pd
from pathlib import Path
import os


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
# SCENARIO
# =========================================================
# A: ECLO forbidden
# B: ECLO allowed
# C: ECLO allowed, but all ECLO affecting each line must lie within one
#    continuous window of at most 2 weeks.
SCENARIO = os.environ.get("SCENARIO", "B").strip().upper()


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




# =========================================================
# 3. CONVERT ACTIVITY DATES TO WEEKS
# =========================================================

activity_details["planned_start_date"] = pd.to_datetime(
    activity_details["planned_start_date"]
)

project_details["planned_completion_date"] = pd.to_datetime(
    project_details["planned_completion_date"]
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
    """Return the full booked section, including both endpoint platforms."""

    start_sector, start_bound = parse_activity_location(start_location_id)
    end_sector, end_bound = parse_activity_location(end_location_id)

    if start_bound != end_bound:
        raise ValueError(
            f"Activity changes bound: {start_location_id} -> {end_location_id}"
        )

    if start_sector not in sector_lookup or end_sector not in sector_lookup:
        raise ValueError(
            f"Unknown sector in activity: {start_location_id} -> {end_location_id}"
        )

    start_line = sector_lookup[start_sector]["line_code"]
    end_line = sector_lookup[end_sector]["line_code"]

    if start_line != end_line:
        raise ValueError(
            f"Activity changes line: {start_location_id} -> {end_location_id}"
        )

    path = line_sector_paths[start_line]
    i = path.index(start_sector)
    j = path.index(end_sector)

    selected_sectors = (
        path[i:j + 1]
        if i <= j
        else list(reversed(path[j:i + 1]))
    )

    # Build the ordered station sequence from the selected sector sequence.
    # This avoids assuming fixed station labels and also supports reverse paths.
    if len(selected_sectors) == 1:
        sec = sector_lookup[selected_sectors[0]]
        station_sequence = [
            sec["from_station_id"],
            sec["to_station_id"],
        ]
    else:
        first = sector_lookup[selected_sectors[0]]
        second = sector_lookup[selected_sectors[1]]
        shared = {
            first["from_station_id"], first["to_station_id"]
        }.intersection({
            second["from_station_id"], second["to_station_id"]
        })

        if len(shared) != 1:
            raise ValueError(
                f"Non-contiguous sectors: {selected_sectors[0]}, "
                f"{selected_sectors[1]}"
            )

        shared_station = next(iter(shared))
        first_station = next(
            x for x in (first["from_station_id"], first["to_station_id"])
            if x != shared_station
        )
        station_sequence = [first_station, shared_station]

        for k in range(1, len(selected_sectors)):
            sec = sector_lookup[selected_sectors[k]]
            previous_station = station_sequence[-1]
            endpoints = [sec["from_station_id"], sec["to_station_id"]]

            if previous_station not in endpoints:
                raise ValueError(
                    f"Non-contiguous sector path at {selected_sectors[k]}"
                )

            next_station = next(x for x in endpoints if x != previous_station)
            station_sequence.append(next_station)

    # Validator-required pattern:
    # endpoint PLAT, SEC, intermediate PLAT, SEC, ..., endpoint PLAT.
    occupied = [
        f"PLAT:{start_line}:{station_sequence[0]}:{start_bound}"
    ]

    for k, sector_id in enumerate(selected_sectors):
        occupied.append(f"{sector_id}:{start_bound}")
        occupied.append(
            f"PLAT:{start_line}:{station_sequence[k + 1]}:{start_bound}"
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


# Maximum number of possession groups that may be CREATED at each location.
#
# Scenario A: exactly nominal capacity.
# Scenario C: nominal capacity + 1, because C permits at most one excess
#             access-night per location-week.
# Scenario B: flexible supply. We create a safe generic upper bound based
#             on the number of activities that can possibly use/close the
#             location. This gives the solver real extra possession slots
#             rather than merely an excess accounting variable.
#
# IMPORTANT: LOCATION_SUPPLY remains the NOMINAL capacity used when
# calculating excess-access-night penalties.
if SCENARIO == "A":
    location_group_capacity = {
        location_id: capacity
        for location_id, capacity in location_capacity.items()
    }

elif SCENARIO == "C":
    location_group_capacity = {
        location_id: capacity + 1
        for location_id, capacity in location_capacity.items()
    }

elif SCENARIO == "B":
    # Scenario B allows up to 7 possession/access groups per location-week.
    # This is the total group ceiling, not nominal capacity + 7.
    location_group_capacity = {
        location_id: 7
        for location_id in location_capacity
    }

else:
    raise ValueError(
        f"Unknown SCENARIO {SCENARIO!r}; expected A, B, or C."
    )




def build_and_solve(scheduling_horizon_weeks, solve_time_seconds=30):
    """Build a fresh CP-SAT model for one candidate scheduling horizon."""
    weeks = range(1, scheduling_horizon_weeks + 1)

    # =========================================================
    # 6. CREATE MODEL
    # =========================================================

    model = cp_model.CpModel()


    # =========================================================
    # 7. ACCESS-NIGHT + ECLO DECISION VARIABLES
    # =========================================================
    #
    # access[a,w,n] = activity a uses local access-night n in week w.
    # eclo[a,w,n]   = that scheduled access is an ECLO access.
    #
    # Standard access = 1.0 workload unit.
    # ECLO access     = 1.5 workload units.
    #
    # CP-SAT uses integers, so workload is scaled by 2:
    # Standard = 2 units; ECLO = 3 units.

    access = {}
    eclo = {}

    for _, activity in activity_data.iterrows():

        activity_id = activity["activity_id"]
        max_nights = int(
            activity["number_of_maximum_access_per_week"]
        )

        for week in weeks:
            for night in range(1, max_nights + 1):

                key = (activity_id, week, night)

                access[key] = model.NewBoolVar(
                    f"access_{activity_id}_week_{week}_night_{night}"
                )

                eclo[key] = model.NewBoolVar(
                    f"eclo_{activity_id}_week_{week}_night_{night}"
                )

                # ECLO can only exist when the access itself exists.
                model.Add(eclo[key] <= access[key])

                # Scenario A hard-forbids ECLO.
                if SCENARIO == "A":
                    model.Add(eclo[key] == 0)


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
    # 9. WORKLOAD CONSERVATION WITH ECLO
    # =========================================================
    #
    # Requirement is workload >= total_accesses:
    #   standard contributes 1.0
    #   ECLO contributes 1.5
    #
    # Scaled by 2:
    #   each access contributes 2
    #   each ECLO flag adds another 1
    #
    # Example: total_accesses=3 can be satisfied by 2 ECLO accesses:
    #   (2+1) + (2+1) = 6 = 2*3.

    for _, activity in activity_data.iterrows():

        activity_id = activity["activity_id"]
        required_work = 2 * int(activity["total_accesses"])

        max_nights = int(
            activity["number_of_maximum_access_per_week"]
        )

        delivered_work = sum(
            2 * access[activity_id, week, night]
            + eclo[activity_id, week, night]
            for week in weeks
            for night in range(1, max_nights + 1)
        )

        model.Add(delivered_work >= required_work)


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
    # 13. CO-SHARING + PM / PC / C LEGAL POSSESSION MIXES
    # =========================================================
    #
    # At each (location, week), LOCATION_SUPPLY gives the maximum number of
    # possession slots available. We represent those slots as group numbers:
    #
    #     group 1, group 2, ... group supply_capacity
    #
    # An activity using the location in that week must be assigned to exactly
    # one of those groups. Activities assigned to the same group co-share one
    # possession slot.
    #
    # Legal group mixes from the challenge statement:
    #
    #     PM mode: exactly 1 PM and nobody else
    #     PC mode: exactly 1 PC + up to 3 C
    #     C mode:  1 to 4 C, with no PM or PC
    #
    # An unused group has no mode selected and consumes no capacity.

    activities_using_location = {
        location_id: []
        for location_id in location_capacity
    }

    for activity_id, occupied_locations in activity_locations.items():
        for location_id in occupied_locations:
            activities_using_location[location_id].append(activity_id)


    # Quick lookup of each activity's access type: PM, PC, or C.
    activity_access_type = dict(
        zip(
            activity_data["activity_id"],
            activity_data["access_type"]
        )
    )


    # group_assignment[activity_id, location_id, week, group]
    # = 1 when the activity uses that co-share group at that location/week.
    group_assignment = {}

    # group_used[location_id, week, group]
    # = 1 when at least one activity occupies the group.
    group_used = {}

    # Exactly one of these is 1 for every used group.
    group_mode_pm = {}
    group_mode_pc = {}
    group_mode_c = {}


    for location_id, capacity in location_capacity.items():

        group_limit = location_group_capacity[location_id]

        relevant_activities = activities_using_location[location_id]

        if not relevant_activities:
            continue

        groups = range(1, group_limit + 1)

        for week in weeks:

            # -------------------------------------------------
            # Create the possession groups for this location/week
            # -------------------------------------------------

            for group in groups:

                group_used[location_id, week, group] = model.NewBoolVar(
                    f"group_used_{location_id}_w{week}_g{group}"
                )

                group_mode_pm[location_id, week, group] = model.NewBoolVar(
                    f"group_pm_{location_id}_w{week}_g{group}"
                )

                group_mode_pc[location_id, week, group] = model.NewBoolVar(
                    f"group_pc_{location_id}_w{week}_g{group}"
                )

                group_mode_c[location_id, week, group] = model.NewBoolVar(
                    f"group_c_{location_id}_w{week}_g{group}"
                )

                # A used possession has exactly one legal possession mode.
                # An unused possession has no mode.
                model.Add(
                    group_mode_pm[location_id, week, group]
                    + group_mode_pc[location_id, week, group]
                    + group_mode_c[location_id, week, group]
                    == group_used[location_id, week, group]
                )

            # -------------------------------------------------
            # Assign every scheduled activity to exactly one group
            # -------------------------------------------------

            for activity_id in relevant_activities:

                for group in groups:

                    group_assignment[
                        activity_id, location_id, week, group
                    ] = model.NewBoolVar(
                        f"assign_{activity_id}_{location_id}_w{week}_g{group}"
                    )

                    # An activity cannot be assigned to an unused group.
                    model.Add(
                        group_assignment[
                            activity_id, location_id, week, group
                        ]
                        <= group_used[location_id, week, group]
                    )

                # If the activity works this week, it must occupy exactly one
                # co-share group at this physical location. If it does not work,
                # it occupies none.
                model.Add(
                    sum(
                        group_assignment[
                            activity_id, location_id, week, group
                        ]
                        for group in groups
                    )
                    == occupied[activity_id, week]
                )

            # -------------------------------------------------
            # Enforce PM / PC / C legal mixes inside each group
            # -------------------------------------------------

            pm_activities = [
                activity_id
                for activity_id in relevant_activities
                if activity_access_type[activity_id] == "PM"
            ]

            pc_activities = [
                activity_id
                for activity_id in relevant_activities
                if activity_access_type[activity_id] == "PC"
            ]

            c_activities = [
                activity_id
                for activity_id in relevant_activities
                if activity_access_type[activity_id] == "C"
            ]

            for group in groups:

                pm_count = sum(
                    group_assignment[
                        activity_id, location_id, week, group
                    ]
                    for activity_id in pm_activities
                )

                pc_count = sum(
                    group_assignment[
                        activity_id, location_id, week, group
                    ]
                    for activity_id in pc_activities
                )

                c_count = sum(
                    group_assignment[
                        activity_id, location_id, week, group
                    ]
                    for activity_id in c_activities
                )

                total_count = pm_count + pc_count + c_count

                # ---------------- PM MODE ----------------
                # One PM alone.
                model.Add(pm_count == 1).OnlyEnforceIf(
                    group_mode_pm[location_id, week, group]
                )
                model.Add(pc_count == 0).OnlyEnforceIf(
                    group_mode_pm[location_id, week, group]
                )
                model.Add(c_count == 0).OnlyEnforceIf(
                    group_mode_pm[location_id, week, group]
                )

                # ---------------- PC MODE ----------------
                # Exactly one PC, optionally hosting up to three C activities.
                model.Add(pm_count == 0).OnlyEnforceIf(
                    group_mode_pc[location_id, week, group]
                )
                model.Add(pc_count == 1).OnlyEnforceIf(
                    group_mode_pc[location_id, week, group]
                )
                model.Add(c_count <= 3).OnlyEnforceIf(
                    group_mode_pc[location_id, week, group]
                )

                # ---------------- C MODE -----------------
                # Up to four C activities and no PM/PC.
                model.Add(pm_count == 0).OnlyEnforceIf(
                    group_mode_c[location_id, week, group]
                )
                model.Add(pc_count == 0).OnlyEnforceIf(
                    group_mode_c[location_id, week, group]
                )
                model.Add(c_count >= 1).OnlyEnforceIf(
                    group_mode_c[location_id, week, group]
                )
                model.Add(c_count <= 4).OnlyEnforceIf(
                    group_mode_c[location_id, week, group]
                )

                # If the group is unused, it must contain no activities.
                model.Add(total_count == 0).OnlyEnforceIf(
                    group_used[location_id, week, group].Not()
                )

            # -------------------------------------------------
            # Symmetry breaking
            # -------------------------------------------------
            # Groups are arbitrary labels. Without this, OR-Tools wastes time
            # exploring equivalent solutions such as using g2 while g1 is empty.
            # Force lower-numbered groups to be used first.

            for group in range(1, group_limit):
                model.Add(
                    group_used[location_id, week, group]
                    >= group_used[location_id, week, group + 1]
                )


    # =========================================================
    # 14. BUFFERS AND CLOSURES
    # =========================================================
    # Buffer sizes come from 05_BUFFER_LOCATION.csv.
    # Live also mirrors its closure to the opposite bound and has the
    # H01-H02 cross-line interchange exception.
    #
    # NOTE: The written specification is ambiguous about exact SEC/PLAT
    # expansion at buffer boundaries. This implementation expands by adjacent
    # tunnel sectors and includes the platforms connecting the closed span.

    buffer_rules = buffer_location.set_index("nature_of_works").to_dict("index")
    activity_nature = dict(zip(activity_data["activity_id"], activity_data["nature_of_activity"]))


    def opposite_bound(bound):
        return "WB" if bound == "EB" else "EB"


    def location_with_bound(location_id, new_bound):
        parts = location_id.split(":")
        parts[-1] = new_bound
        return ":".join(parts)


    def get_activity_base_sector_span(activity_id):
        row = activity_data.loc[activity_data["activity_id"] == activity_id].iloc[0]
        start_sector, start_bound = parse_activity_location(row["start_location_id"])
        end_sector, end_bound = parse_activity_location(row["end_location_id"])

        if start_bound != end_bound:
            raise ValueError(f"{activity_id} changes bound")

        line_code = sector_lookup[start_sector]["line_code"]
        path = line_sector_paths[line_code]
        i = path.index(start_sector)
        j = path.index(end_sector)
        return line_code, start_bound, path, min(i, j), max(i, j)


    def add_platform_if_known(target, line_code, station_id, bound):
        loc = f"PLAT:{line_code}:{station_id}:{bound}"
        if loc in known_locations:
            target.add(loc)


    def buffer_closure_locations(activity_id):
        """
        Return the validator-style closure footprint.

        Interpretation:
        - The activity's full booked section (including endpoint platforms)
          is its work footprint.
        - A buffer of N sectors extends N TUNNEL SECTORS beyond each end of
          the booked section along the same line/bound.
        - Every platform belonging to that extended railway span is included.
        - Live work mirrors the complete closure onto the opposite bound.
        - Live cross-line interchange effects are derived from map data.
        """
        nature = activity_nature[activity_id]
        rule = buffer_rules[nature]
        buffer_size = int(rule["up_to_buffer_sectors"])
        mirror = int(rule["opposite_bound_required"]) == 1

        line_code, bound, path, lo, hi = get_activity_base_sector_span(
            activity_id
        )

        # Start with the COMPLETE work footprint. This now includes both
        # endpoint platforms because activity_occupied_locations() was fixed.
        closure = set(activity_locations[activity_id])

        if buffer_size > 0:
            ext_lo = max(0, lo - buffer_size)
            ext_hi = min(len(path) - 1, hi + buffer_size)

            selected = path[ext_lo:ext_hi + 1]

            # Build the complete station sequence covered by the extended
            # tunnel span, then include every SEC and PLAT in that span.
            first_sector = sector_lookup[selected[0]]

            if len(selected) == 1:
                station_sequence = [
                    first_sector["from_station_id"],
                    first_sector["to_station_id"],
                ]
            else:
                second_sector = sector_lookup[selected[1]]
                shared = {
                    first_sector["from_station_id"],
                    first_sector["to_station_id"],
                }.intersection({
                    second_sector["from_station_id"],
                    second_sector["to_station_id"],
                })

                if len(shared) != 1:
                    raise ValueError(
                        f"Non-contiguous buffer sectors: "
                        f"{selected[0]}, {selected[1]}"
                    )

                shared_station = next(iter(shared))
                first_station = next(
                    x for x in (
                        first_sector["from_station_id"],
                        first_sector["to_station_id"],
                    )
                    if x != shared_station
                )

                station_sequence = [first_station, shared_station]

                for k in range(1, len(selected)):
                    sec = sector_lookup[selected[k]]
                    endpoints = [
                        sec["from_station_id"],
                        sec["to_station_id"],
                    ]
                    previous_station = station_sequence[-1]

                    if previous_station not in endpoints:
                        raise ValueError(
                            f"Non-contiguous buffer path at {selected[k]}"
                        )

                    next_station = next(
                        x for x in endpoints
                        if x != previous_station
                    )
                    station_sequence.append(next_station)

            # Add the complete extended span:
            # PLAT, SEC, PLAT, SEC, ..., PLAT.
            closure.add(
                f"PLAT:{line_code}:{station_sequence[0]}:{bound}"
            )

            for k, sector_id in enumerate(selected):
                closure.add(f"{sector_id}:{bound}")
                closure.add(
                    f"PLAT:{line_code}:{station_sequence[k + 1]}:{bound}"
                )

        # Live: mirror the COMPLETE work + buffer closure onto the opposite
        # bound. This is a closure effect, not extra activity occupancy.
        if mirror:
            mirrored = {
                location_with_bound(loc, opposite_bound(bound))
                for loc in closure
            }
            closure.update(
                loc for loc in mirrored
                if loc in known_locations
            )

        # Live cross-line interchange behaviour.
        #
        # IMPORTANT distinction:
        #
        # 1. If the ACTUAL LIVE WORK occupies an interchange-to-interchange
        #    sector, mirror the FULL Live closure footprint (work + 2-sector
        #    buffer) onto every other line that contains the same interchange
        #    pair. This mirrored footprint applies on both bounds.
        #
        # 2. If only the BUFFER reaches an interchange, do NOT start a new
        #    outward buffer on the other line. In that case only the matching
        #    interchange sector/platforms on the other line are closed.
        #
        # Everything is discovered from STATIONS/SECTORS; no station or line
        # labels are hard-coded.
        if nature == "Live":
            interchange_station_ids = set(
                stations.loc[
                    stations["is_interchange"].astype(int) == 1,
                    "station_id"
                ]
            )

            # Actual work tunnel sectors only -- not buffer sectors.
            actual_work_base_sectors = set()

            for work_loc in activity_locations[activity_id]:
                parts = work_loc.split(":")
                if len(parts) == 4 and parts[0] == "SEC":
                    actual_work_base_sectors.add(":".join(parts[:3]))

            # All same-line tunnel sectors touched by work+buffer.
            closure_base_sectors = set()

            for closed_loc in closure:
                parts = closed_loc.split(":")
                if len(parts) == 4 and parts[0] == "SEC":
                    base_sector_id = ":".join(parts[:3])
                    if (
                        base_sector_id in sector_lookup
                        and sector_lookup[base_sector_id]["line_code"]
                        == line_code
                    ):
                        closure_base_sectors.add(base_sector_id)

            def is_interchange_pair_sector(base_sector_id):
                sec = sector_lookup[base_sector_id]
                return (
                    sec["from_station_id"] in interchange_station_ids
                    and sec["to_station_id"] in interchange_station_ids
                )

            # -----------------------------------------------------
            # CASE 1: ACTUAL LIVE WORK IS AT THE INTERCHANGE
            # -----------------------------------------------------
            actual_interchange_sectors = [
                sector_id
                for sector_id in actual_work_base_sectors
                if is_interchange_pair_sector(sector_id)
            ]

            for interchange_sector_id in actual_interchange_sectors:
                interchange_sector = sector_lookup[interchange_sector_id]
                station_a = interchange_sector["from_station_id"]
                station_b = interchange_sector["to_station_id"]
                station_pair = {station_a, station_b}

                # Find the corresponding interchange sector on every other line.
                for _, other_sector in sectors.iterrows():
                    other_line = other_sector["line_code"]

                    if other_line == line_code:
                        continue

                    other_pair = {
                        other_sector["from_station_id"],
                        other_sector["to_station_id"]
                    }

                    if other_pair != station_pair:
                        continue

                    other_sector_id = other_sector["sector_id"]
                    other_path = line_sector_paths[other_line]
                    interchange_index = other_path.index(other_sector_id)

                    # Mirror the SAME Live buffer distance onto the other line,
                    # centred on its corresponding interchange sector.
                    other_lo = max(
                        0,
                        interchange_index - buffer_size
                    )
                    other_hi = min(
                        len(other_path) - 1,
                        interchange_index + buffer_size
                    )

                    mirrored_sectors = other_path[
                        other_lo:other_hi + 1
                    ]

                    # Build the ordered station sequence for this mirrored span.
                    first_other = sector_lookup[mirrored_sectors[0]]

                    if len(mirrored_sectors) == 1:
                        mirrored_stations = [
                            first_other["from_station_id"],
                            first_other["to_station_id"]
                        ]
                    else:
                        second_other = sector_lookup[mirrored_sectors[1]]

                        shared = {
                            first_other["from_station_id"],
                            first_other["to_station_id"]
                        }.intersection({
                            second_other["from_station_id"],
                            second_other["to_station_id"]
                        })

                        if len(shared) != 1:
                            raise ValueError(
                                "Non-contiguous cross-line Live buffer at "
                                f"{mirrored_sectors[0]}, "
                                f"{mirrored_sectors[1]}"
                            )

                        shared_station = next(iter(shared))
                        first_station = next(
                            station_id
                            for station_id in (
                                first_other["from_station_id"],
                                first_other["to_station_id"]
                            )
                            if station_id != shared_station
                        )

                        mirrored_stations = [
                            first_station,
                            shared_station
                        ]

                        for k in range(1, len(mirrored_sectors)):
                            sec = sector_lookup[mirrored_sectors[k]]
                            endpoints = [
                                sec["from_station_id"],
                                sec["to_station_id"]
                            ]
                            previous_station = mirrored_stations[-1]

                            if previous_station not in endpoints:
                                raise ValueError(
                                    "Non-contiguous cross-line Live path at "
                                    f"{mirrored_sectors[k]}"
                                )

                            next_station = next(
                                station_id
                                for station_id in endpoints
                                if station_id != previous_station
                            )
                            mirrored_stations.append(next_station)

                    # Cross-line Live closure applies on BOTH bounds.
                    for bnd in ("EB", "WB"):
                        first_platform = (
                            f"PLAT:{other_line}:"
                            f"{mirrored_stations[0]}:{bnd}"
                        )
                        if first_platform in known_locations:
                            closure.add(first_platform)

                        for k, sector_id in enumerate(mirrored_sectors):
                            tunnel_loc = f"{sector_id}:{bnd}"
                            platform_loc = (
                                f"PLAT:{other_line}:"
                                f"{mirrored_stations[k + 1]}:{bnd}"
                            )

                            if tunnel_loc in known_locations:
                                closure.add(tunnel_loc)

                            if platform_loc in known_locations:
                                closure.add(platform_loc)

            # -----------------------------------------------------
            # CASE 2: BUFFER ONLY REACHES THE INTERCHANGE
            # -----------------------------------------------------
            #
            # If the actual work is NOT on that interchange sector but the
            # same-line closure reaches it, close only the corresponding
            # interchange sector and endpoint platforms on other lines.
            # Do not propagate another outward buffer there.
            buffer_only_interchange_sectors = [
                sector_id
                for sector_id in closure_base_sectors
                if (
                    sector_id not in actual_work_base_sectors
                    and is_interchange_pair_sector(sector_id)
                )
            ]

            for interchange_sector_id in buffer_only_interchange_sectors:
                interchange_sector = sector_lookup[interchange_sector_id]
                station_a = interchange_sector["from_station_id"]
                station_b = interchange_sector["to_station_id"]
                station_pair = {station_a, station_b}

                for _, other_sector in sectors.iterrows():
                    other_line = other_sector["line_code"]

                    if other_line == line_code:
                        continue

                    other_pair = {
                        other_sector["from_station_id"],
                        other_sector["to_station_id"]
                    }

                    if other_pair != station_pair:
                        continue

                    other_sector_id = other_sector["sector_id"]

                    for bnd in ("EB", "WB"):
                        for cross_location in (
                            f"{other_sector_id}:{bnd}",
                            f"PLAT:{other_line}:{station_a}:{bnd}",
                            f"PLAT:{other_line}:{station_b}:{bnd}",
                        ):
                            if cross_location in known_locations:
                                closure.add(cross_location)

        return closure


    activity_closures = {
        activity_id: buffer_closure_locations(activity_id)
        for activity_id in activity_data["activity_id"]
    }


    # =========================================================
    # BUFFER / CLOSURE CAPACITY CONSUMPTION
    # =========================================================
    #
    # FIX: do not count the same physical possession once for every source
    # location it occupies.
    #
    # A scheduled activity access is one possession across its entire route.
    # Activities that co-share are treated as one possession when they share
    # the same group at ANY common work location. For buffer-capacity counting,
    # we build possession representatives from those co-sharing relationships.
    #
    # Each resulting possession consumes:
    #   - actual work capacity through the existing location group_used variables;
    #   - exactly ONE additional supply unit at each external location reached
    #     by the combined buffer/closure footprint of that possession.
    #
    # This prevents a long activity occupying SEC -> PLAT -> SEC from charging
    # the same external buffer three times.

    activity_ids = activity_data["activity_id"].tolist()

    activity_extra_closure_locations = {
        activity_id: (
            set(activity_closures[activity_id])
            - set(activity_locations[activity_id])
        )
        for activity_id in activity_ids
    }


    # ---------------------------------------------------------
    # SAME-POSSESSION PAIR VARIABLES
    # ---------------------------------------------------------
    #
    # same_possession_pair[a,b,w] = 1 when scheduled activities a and b share
    # at least one common work location in the same co_share_group that week.
    #
    # This variable is used ONLY to deduplicate buffer capacity. It does not add
    # the old blanket same-week closure prohibition.

    same_possession_pair = {}

    for i in range(len(activity_ids)):
        for j in range(i + 1, len(activity_ids)):

            a = activity_ids[i]
            b = activity_ids[j]

            common_locations = (
                set(activity_locations[a])
                .intersection(activity_locations[b])
            )

            if not common_locations:
                continue

            for week in weeks:

                same_group_literals = []

                for location_id in common_locations:
                    group_limit = location_group_capacity[location_id]

                    for group in range(1, group_limit + 1):
                        key_a = (a, location_id, week, group)
                        key_b = (b, location_id, week, group)

                        if (
                            key_a not in group_assignment
                            or key_b not in group_assignment
                        ):
                            continue

                        both = model.NewBoolVar(
                            f"samegrpbuf_{a}_{b}_{location_id}_"
                            f"w{week}_g{group}"
                        )

                        model.Add(both <= group_assignment[key_a])
                        model.Add(both <= group_assignment[key_b])
                        model.Add(
                            both
                            >= group_assignment[key_a]
                            + group_assignment[key_b] - 1
                        )

                        same_group_literals.append(both)

                if not same_group_literals:
                    continue

                pair = model.NewBoolVar(
                    f"sameposbuf_{a}_{b}_w{week}"
                )

                # pair = OR(same_group_literals)
                for literal in same_group_literals:
                    model.Add(pair >= literal)

                model.Add(pair <= sum(same_group_literals))

                same_possession_pair[a, b, week] = pair


    # ---------------------------------------------------------
    # BUFFER CAPACITY REPRESENTATIVES
    # ---------------------------------------------------------
    #
    # For every blocked location/week, each scheduled activity whose external
    # closure reaches that location initially represents one unit of demand.
    # If two such activities are in the same co-shared possession, only the
    # lowest-index activity represents that possession's buffer.
    #
    # representative[a,L,w] = 1 iff:
    #   a is scheduled in week w,
    #   a's closure reaches L,
    #   and no earlier activity sharing the same possession also reaches L.
    #
    # This counts a possession once at L, regardless of how many activities or
    # how many actual route locations generated the closure.

    buffer_representative = {}

    for blocked_location in location_capacity:

        relevant = [
            activity_id
            for activity_id in activity_ids
            if blocked_location
            in activity_extra_closure_locations[activity_id]
        ]

        for week in weeks:

            for index, activity_id in enumerate(relevant):

                rep = model.NewBoolVar(
                    f"bufferrep_{activity_id}_{blocked_location}_w{week}"
                )

                buffer_representative[
                    activity_id, blocked_location, week
                ] = rep

                # A representative must actually be scheduled.
                model.Add(rep <= occupied[activity_id, week])

                earlier_same_possession = []

                for earlier_id in relevant[:index]:

                    pair_key = (
                        min(activity_id, earlier_id),
                        max(activity_id, earlier_id),
                        week
                    )

                    pair = same_possession_pair.get(pair_key)

                    if pair is not None:
                        earlier_same_possession.append(pair)

                        # If an earlier member of the same possession exists,
                        # this activity cannot also represent the buffer.
                        model.Add(rep <= 1 - pair)

                if not earlier_same_possession:
                    # First possible representative: exactly follows scheduling.
                    model.Add(rep == occupied[activity_id, week])
                else:
                    # If scheduled and not co-sharing with any earlier relevant
                    # activity, it must represent this possession.
                    model.Add(
                        rep
                        >= occupied[activity_id, week]
                        - sum(earlier_same_possession)
                    )


    # ---------------------------------------------------------
    # LOCATION CAPACITY INCLUDING ONE BUFFER UNIT PER POSSESSION
    # ---------------------------------------------------------

    excess_location_supply = {}

    max_possible_location_demand = {
        location_id: len(activity_data)
        for location_id in location_capacity
    }

    for location_id, capacity in location_capacity.items():

        for week in weeks:

            # Actual work consumes one unit per used possession group at the
            # location. Co-sharing is already deduplicated by group_used.
            actual_work_terms = []

            if activities_using_location.get(location_id, []):
                group_limit = location_group_capacity[location_id]

                for group in range(1, group_limit + 1):
                    key = (location_id, week, group)

                    if key in group_used:
                        actual_work_terms.append(group_used[key])

            # External buffer consumes one unit per distinct possession.
            closure_terms = [
                variable
                for (
                    activity_id,
                    blocked_location,
                    key_week
                ), variable in buffer_representative.items()
                if blocked_location == location_id and key_week == week
            ]

            location_demand = (
                sum(actual_work_terms)
                + sum(closure_terms)
            )

            if SCENARIO == "A":
                model.Add(location_demand <= capacity)

            elif SCENARIO == "B":
                excess_key = (location_id, week)

                excess_location_supply[excess_key] = model.NewIntVar(
                    0,
                    max_possible_location_demand[location_id],
                    f"excess_supply_{location_id}_w{week}"
                )

                model.Add(
                    excess_location_supply[excess_key]
                    >= location_demand - capacity
                )

            elif SCENARIO == "C":
                excess_key = (location_id, week)

                excess_location_supply[excess_key] = model.NewIntVar(
                    0,
                    1,
                    f"excess_supply_{location_id}_w{week}"
                )

                model.Add(
                    excess_location_supply[excess_key]
                    >= location_demand - capacity
                )

                model.Add(location_demand <= capacity + 1)

            else:
                raise ValueError(
                    f"Unknown SCENARIO {SCENARIO!r}; expected A, B, or C."
                )


    # =========================================================
    # CLOSURE EXCLUSION BETWEEN SEPARATE POSSESSIONS
    # =========================================================
    #
    # Validator behaviour shows that LOCATION_SUPPLY consumption alone is
    # not enough: an activity may not sit inside another possession's
    # closure zone in the same week.
    #
    # Exception: if the activities overlap at an actual WORK location and
    # are assigned to the same co-share group there, they are treated as
    # one possession and are exempt from each other's closure.

    for i in range(len(activity_ids)):
        for j in range(i + 1, len(activity_ids)):

            a_id = activity_ids[i]
            b_id = activity_ids[j]

            work_a = set(activity_locations[a_id])
            work_b = set(activity_locations[b_id])

            conflict = (
                work_a.intersection(activity_closures[b_id])
                or work_b.intersection(activity_closures[a_id])
            )

            if not conflict:
                continue

            common_work_locations = work_a.intersection(work_b)

            for week in weeks:

                same_possession_options = []

                for location_id in common_work_locations:
                    group_limit = location_group_capacity[location_id]

                    for group in range(1, group_limit + 1):
                        key_a = (a_id, location_id, week, group)
                        key_b = (b_id, location_id, week, group)

                        if (
                            key_a not in group_assignment
                            or key_b not in group_assignment
                        ):
                            continue

                        together = model.NewBoolVar(
                            f"closure_samepos_{a_id}_{b_id}_"
                            f"{location_id}_w{week}_g{group}"
                        )

                        model.Add(together <= group_assignment[key_a])
                        model.Add(together <= group_assignment[key_b])
                        model.Add(
                            together
                            >= group_assignment[key_a]
                            + group_assignment[key_b] - 1
                        )

                        same_possession_options.append(together)

                if same_possession_options:
                    same_possession = model.NewBoolVar(
                        f"closure_exempt_{a_id}_{b_id}_w{week}"
                    )

                    for option in same_possession_options:
                        model.Add(same_possession >= option)

                    model.Add(
                        same_possession <= sum(same_possession_options)
                    )

                    model.Add(
                        occupied[a_id, week]
                        + occupied[b_id, week]
                        <= 1 + same_possession
                    )

                else:
                    # Separate possessions with a closure conflict cannot
                    # both be present in the same week.
                    model.Add(
                        occupied[a_id, week]
                        + occupied[b_id, week]
                        <= 1
                    )


    # Maintenance note:
    # LOCATION_SUPPLY is described as the remaining pool after maintenance has
    # taken priority, so this version does not add a second maintenance closure
    # layer. If a separate maintenance-night input exists, add it explicitly.



    # =========================================================
    # PREDECESSOR ACTIVITY CONSTRAINTS
    # =========================================================
    #
    # If activity B has predecessor_activity_id = A, then B may only start
    # in a week STRICTLY AFTER A has completely finished.
    #
    # Because each activity receives at most one access per week:
    #
    #   completion_week[A] = latest week in which A receives an access
    #   start_week[B]      = earliest week in which B receives an access
    #
    # Constraint:
    #
    #   start_week[B] >= completion_week[A] + 1
    #
    # This is stronger than merely preventing A and B from working in the
    # same week: every access of the predecessor must be finished before
    # the successor receives its first access.


    # Integer helper variables for first/last scheduled week.
    activity_start_week = {}
    activity_completion_week = {}

    for _, activity in activity_data.iterrows():

        activity_id = activity["activity_id"]
        max_nights = int(
            activity["number_of_maximum_access_per_week"]
        )

        activity_start_week[activity_id] = model.NewIntVar(
            1,
            scheduling_horizon_weeks,
            f"start_week_{activity_id}"
        )

        activity_completion_week[activity_id] = model.NewIntVar(
            1,
            scheduling_horizon_weeks,
            f"completion_week_{activity_id}"
        )

        # week_used contains the week number when this activity is scheduled,
        # otherwise 0. MaxEquality therefore gives the last scheduled week.
        week_used = []

        # For the first scheduled week, use scheduling_horizon_weeks + 1 for unscheduled
        # weeks so MinEquality ignores them.
        first_week_candidates = []

        for week in weeks:

            week_is_used = occupied[activity_id, week]

            last_candidate = model.NewIntVar(
                0,
                scheduling_horizon_weeks,
                f"last_candidate_{activity_id}_w{week}"
            )

            model.Add(last_candidate == week).OnlyEnforceIf(
                week_is_used
            )
            model.Add(last_candidate == 0).OnlyEnforceIf(
                week_is_used.Not()
            )

            week_used.append(last_candidate)

            first_candidate = model.NewIntVar(
                1,
                scheduling_horizon_weeks + 1,
                f"first_candidate_{activity_id}_w{week}"
            )

            model.Add(first_candidate == week).OnlyEnforceIf(
                week_is_used
            )
            model.Add(
                first_candidate == scheduling_horizon_weeks + 1
            ).OnlyEnforceIf(
                week_is_used.Not()
            )

            first_week_candidates.append(first_candidate)

        model.AddMaxEquality(
            activity_completion_week[activity_id],
            week_used
        )

        model.AddMinEquality(
            activity_start_week[activity_id],
            first_week_candidates
        )


    # Add precedence relationships from 08_ACTIVITY_DETAILS.csv.
    known_activity_ids = set(activity_data["activity_id"])

    for _, activity in activity_data.iterrows():

        successor_id = activity["activity_id"]
        predecessor = activity["predecessor_activity_id"]

        # Empty predecessor cells are read by pandas as NaN.
        if pd.isna(predecessor):
            continue

        predecessor_id = str(predecessor).strip()

        if predecessor_id == "":
            continue

        if predecessor_id not in known_activity_ids:
            raise ValueError(
                f"{successor_id} refers to unknown predecessor "
                f"{predecessor_id}"
            )

        if predecessor_id == successor_id:
            raise ValueError(
                f"{successor_id} cannot be its own predecessor"
            )

        # Successor starts strictly after predecessor completion.
        model.Add(
            activity_start_week[successor_id]
            >= activity_completion_week[predecessor_id] + 1
        )





    # =========================================================
    # CONTRACT COMPLETION + OVERRUN DAYS
    # =========================================================
    #
    # Contract completion is the latest completion week among all activities
    # belonging to that contract.
    #
    # The challenge measures overrun against planned_completion_date.
    # Because this solver schedules at weekly granularity, a completion in
    # Week w is converted to the end of that planning week:
    #
    #   horizon_start + (7*w - 1) days
    #
    # Example: Week 1 starting Monday 4 Jan completes on Sunday 10 Jan.
    #
    # overrun_days = max(0, simulated_completion_date - planned_completion_date)
    #
    # To keep CP-SAT integer-only, we calculate the calendar-day offset from
    # horizon_start directly.

    contract_completion_week = {}
    contract_completion_day_offset = {}
    contract_overrun_days = {}

    for contract_number, contract_activities in activity_data.groupby(
        "contract_number"
    ):

        activity_ids_for_contract = (
            contract_activities["activity_id"].tolist()
        )

        # Latest activity completion week determines contract completion.
        contract_completion_week[contract_number] = model.NewIntVar(
            1,
            scheduling_horizon_weeks,
            f"contract_completion_week_{contract_number}"
        )

        model.AddMaxEquality(
            contract_completion_week[contract_number],
            [
                activity_completion_week[activity_id]
                for activity_id in activity_ids_for_contract
            ]
        )

        # End-of-week calendar offset:
        # Week 1 -> day 6, Week 2 -> day 13, etc.
        contract_completion_day_offset[contract_number] = model.NewIntVar(
            0,
            scheduling_horizon_weeks * 7 - 1,
            f"contract_completion_day_offset_{contract_number}"
        )

        model.Add(
            contract_completion_day_offset[contract_number]
            == 7 * contract_completion_week[contract_number] - 1
        )

        planned_completion_date = pd.to_datetime(
            contract_activities.iloc[0]["planned_completion_date"]
        )

        planned_day_offset = int(
            (planned_completion_date - horizon_start).days
        )

        # raw_overrun may be negative if the contract finishes early.
        raw_overrun = model.NewIntVar(
            -scheduling_horizon_weeks * 7,
            scheduling_horizon_weeks * 7,
            f"raw_overrun_days_{contract_number}"
        )

        model.Add(
            raw_overrun
            ==
            contract_completion_day_offset[contract_number]
            - planned_day_offset
        )

        contract_overrun_days[contract_number] = model.NewIntVar(
            0,
            scheduling_horizon_weeks * 7,
            f"overrun_days_{contract_number}"
        )

        # overrun_days = max(raw_overrun, 0)
        model.AddMaxEquality(
            contract_overrun_days[contract_number],
            [raw_overrun, 0]
        )

        # Scenario B has a rigid planned completion date.
        if SCENARIO == "B":
            model.Add(
                contract_overrun_days[contract_number] == 0
            )



    # =========================================================
    # SCENARIO C: ECLO CONTINUITY WINDOW
    # =========================================================
    #
    # Every ECLO access affecting a given line must fall within one continuous
    # span of at most 2 calendar weeks. The window is chosen independently for
    # each line.
    #
    # We derive the affected lines from the activity closure footprint, so a
    # Live cross-line activity automatically belongs to every line it affects.

    if SCENARIO == "C":

        line_codes = sorted(location_supply["line_code"].unique())

        # window_start[line] is the first week of that line's 2-week ECLO window.
        eclo_window_start = {
            line_code: model.NewIntVar(
                1,
                max(1, horizon_weeks - 1),
                f"eclo_window_start_{line_code}"
            )
            for line_code in line_codes
        }

        for _, activity in activity_data.iterrows():

            activity_id = activity["activity_id"]
            max_nights = int(
                activity["number_of_maximum_access_per_week"]
            )

            affected_lines = set()

            for location_id in activity_closures[activity_id]:
                if location_id in known_locations:
                    row = location_supply.loc[
                        location_supply["location_id"] == location_id
                    ].iloc[0]
                    affected_lines.add(row["line_code"])

            for week in weeks:
                for night in range(1, max_nights + 1):

                    eclo_var = eclo[activity_id, week, night]

                    for line_code in affected_lines:

                        # If this access uses ECLO, its week must be either
                        # window_start or window_start + 1.
                        model.Add(
                            eclo_window_start[line_code] <= week
                        ).OnlyEnforceIf(eclo_var)

                        model.Add(
                            week <= eclo_window_start[line_code] + 1
                        ).OnlyEnforceIf(eclo_var)


    # =========================================================
    # 15. SIMPLE TEMPORARY OBJECTIVE
    # =========================================================
    # This is NOT the final Scenario A objective.
    # It simply encourages the solver to place accesses earlier.

    # =========================================================
    # PRIORITY-WEIGHTED OVERRUN OBJECTIVE
    # =========================================================
    #
    # Contract priority determines the main penalty band:
    #   contract priority 1 -> 100
    #   contract priority 2 -> 10
    #   contract priority 3 -> 1
    #
    # Activity priority adds a smaller nudge:
    #   activity priority 1 -> +30%
    #   activity priority 2 -> +20%
    #   activity priority 3 -> +0%
    #
    # The nudge applies when an individual activity finishes after its
    # contract's planned completion date. This lets the solver distinguish
    # which activities are causing lateness inside the same contract.
    #
    # To avoid decimal coefficients in CP-SAT, all weights are scaled by 10:
    #
    #   P1 contract: 1300 / 1200 / 1000
    #   P2 contract:  130 /  120 /  100
    #   P3 contract:   13 /   12 /   10
    #
    # Dividing the final weighted-overrun value by 10 gives the unscaled score.

    contract_priority_weight = {
        1: 100,
        2: 10,
        3: 1,
    }

    activity_priority_nudge_scaled = {
        1: 3,   # +0.3
        2: 2,   # +0.2
        3: 0,   # +0.0
    }

    activity_overrun_days = {}
    priority_weighted_overrun_terms = []

    for _, activity in activity_data.iterrows():

        activity_id = activity["activity_id"]
        contract_number = activity["contract_number"]

        contract_priority = int(activity["contract_priority"])
        activity_priority = int(activity["activity_priority"])

        planned_completion_date = pd.to_datetime(
            activity["planned_completion_date"]
        )

        planned_day_offset = int(
            (planned_completion_date - horizon_start).days
        )

        # Activity completion is the end of its final scheduled week.
        activity_completion_day_offset = model.NewIntVar(
            0,
            scheduling_horizon_weeks * 7 - 1,
            f"completion_day_offset_{activity_id}"
        )

        model.Add(
            activity_completion_day_offset
            == 7 * activity_completion_week[activity_id] - 1
        )

        raw_activity_overrun = model.NewIntVar(
            -scheduling_horizon_weeks * 7,
            scheduling_horizon_weeks * 7,
            f"raw_activity_overrun_{activity_id}"
        )

        model.Add(
            raw_activity_overrun
            ==
            activity_completion_day_offset
            - planned_day_offset
        )

        activity_overrun_days[activity_id] = model.NewIntVar(
            0,
            scheduling_horizon_weeks * 7,
            f"activity_overrun_days_{activity_id}"
        )

        model.AddMaxEquality(
            activity_overrun_days[activity_id],
            [raw_activity_overrun, 0]
        )

        # Scale by 10:
        # contract_weight * (1 + nudge) * overrun
        # =
        # contract_weight * (10 + scaled_nudge) * overrun / 10
        scaled_weight = (
            contract_priority_weight[contract_priority]
            * (10 + activity_priority_nudge_scaled[activity_priority])
        )

        priority_weighted_overrun_terms.append(
            scaled_weight * activity_overrun_days[activity_id]
        )


    # Scenario B has rigid planned dates, so feasible schedules have no
    # activity-level lateness either.
    if SCENARIO == "B":
        for activity_id in activity_overrun_days:
            model.Add(activity_overrun_days[activity_id] == 0)


    model.Minimize(
        # Scaled priority-weighted activity overrun.
        sum(priority_weighted_overrun_terms)
        +
        # Excess access-night cost is 7 per unit. The whole objective is
        # scaled by 10 because of the activity-priority nudge, hence 70.
        (
            0
            if SCENARIO == "A"
            else 70 * sum(excess_location_supply.values())
        )
        +
        # ECLO has an unscaled cost of 5, hence 50 after scaling by 10.
        (
            0
            if SCENARIO == "A"
            else 50 * sum(eclo.values())
        )
    )


    # =========================================================
    # 16. SOLVE
    # =========================================================

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = solve_time_seconds

    status = solver.Solve(model)


    # =========================================================
    # 17. PRINT RESULTS
    # =========================================================

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):

        # Direct-export buffers. These rows are populated from the EXACT SAME
        # solved values used by the console output below; no second solve and
        # no reconstruction of co-share groups is performed.
        access_rows = []
        occupancy_rows = []
        results_rows = []

        print("Schedule found!")
        print(
            "Nominal horizon:", horizon_weeks,
            "| Scheduling horizon:", scheduling_horizon_weeks,
            "| Extra weeks:", max(0, scheduling_horizon_weeks - horizon_weeks)
        )

        print("\nCONTRACT COMPLETION / OVERRUN")
        print("--------------------------------")

        for contract_number, contract_activities in activity_data.groupby(
            "contract_number"
        ):
            completion_week = solver.Value(
                contract_completion_week[contract_number]
            )

            simulated_completion_date = (
                horizon_start
                + pd.Timedelta(days=7 * completion_week - 1)
            )

            planned_completion_date = pd.to_datetime(
                contract_activities.iloc[0]["planned_completion_date"]
            )

            overrun = solver.Value(
                contract_overrun_days[contract_number]
            )

            results_rows.append({
                "scenario": SCENARIO,
                "contract_number": contract_number,
                "simulated_completion_date": simulated_completion_date.strftime("%Y-%m-%d"),
                "overrun_days": overrun,
            })

            print(
                contract_number,
                "| completion week:", completion_week,
                "| simulated completion:",
                simulated_completion_date.date(),
                "| planned completion:",
                planned_completion_date.date(),
                "| overrun days:", overrun
            )

        if SCENARIO in ("B", "C"):
            print("\nEXCESS LOCATION SUPPLY")
            print("--------------------------------")

            total_excess = 0

            for (location_id, week), variable in sorted(
                excess_location_supply.items()
            ):
                value = solver.Value(variable)

                if value > 0:
                    total_excess += value
                    print(
                        location_id,
                        "| week:", week,
                        "| excess access-nights:", value,
                        "| nominal capacity:",
                        location_capacity[location_id]
                    )

            print("Total excess access-nights:", total_excess)

        if SCENARIO in ("B", "C"):
            print("\nECLO WEEKS")
            print("--------------------------------")

            eclo_weeks_by_line = {}

            for _, eclo_activity in activity_data.iterrows():
                eclo_activity_id = eclo_activity["activity_id"]
                max_nights = int(
                    eclo_activity["number_of_maximum_access_per_week"]
                )

                affected_lines = set()

                for location_id in activity_closures[eclo_activity_id]:
                    if location_id in known_locations:
                        line_code = location_supply.loc[
                            location_supply["location_id"] == location_id,
                            "line_code"
                        ].iloc[0]
                        affected_lines.add(line_code)

                for week in weeks:
                    for night in range(1, max_nights + 1):
                        if solver.Value(
                            eclo[eclo_activity_id, week, night]
                        ) == 1:
                            for line_code in affected_lines:
                                eclo_weeks_by_line.setdefault(
                                    line_code, set()
                                ).add(week)

            if not eclo_weeks_by_line:
                print("No ECLO used.")
            else:
                for line_code in sorted(eclo_weeks_by_line):
                    print(
                        line_code,
                        "| ECLO weeks:",
                        sorted(eclo_weeks_by_line[line_code])
                    )

                    if SCENARIO == "C":
                        window_start = solver.Value(
                            eclo_window_start[line_code]
                        )
                        print(
                            "   selected 2-week window:",
                            window_start,
                            "-",
                            window_start + 1
                        )

            print("\nECLO ACCESS DETAILS")
            print("--------------------------------")

            any_eclo = False

            for _, eclo_activity in activity_data.iterrows():
                eclo_activity_id = eclo_activity["activity_id"]
                max_nights = int(
                    eclo_activity["number_of_maximum_access_per_week"]
                )

                for week in weeks:
                    for night in range(1, max_nights + 1):
                        if solver.Value(
                            eclo[eclo_activity_id, week, night]
                        ) == 1:
                            any_eclo = True
                            print(
                                eclo_activity_id,
                                "| week:", week,
                                "| access_night:", night
                            )

            if not any_eclo:
                print("No ECLO accesses.")

        print("\nACTIVITY SCHEDULE")
        print("--------------------------------")

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
                            (
                                week,
                                night,
                                solver.Value(eclo[activity_id, week, night])
                            )
                        )

            # Export the exact selected accesses printed below.
            for access_seq, (export_week, export_night, export_eclo) in enumerate(
                sorted(selected_accesses, key=lambda x: (x[0], x[1])), start=1
            ):
                access_rows.append({
                    "activity_id": activity_id,
                    "access_seq": access_seq,
                    "week": export_week,
                    "eclo": export_eclo,
                    "access_night": export_night,
                })

            predecessor = activity["predecessor_activity_id"]
            predecessor_text = (
                None if pd.isna(predecessor) else str(predecessor)
            )

            print(
                activity_id,
                selected_accesses,
                "start_week:",
                solver.Value(activity_start_week[activity_id]),
                "completion_week:",
                solver.Value(activity_completion_week[activity_id]),
                "activity_overrun_days:",
                solver.Value(activity_overrun_days[activity_id]),
                "predecessor:",
                predecessor_text,
                "locations:",
                activity_locations[activity_id]
            )

            # Show the co-share group selected at every occupied location/week.
            for week, _night, _eclo in selected_accesses:
                for location_id in activity_locations[activity_id]:
                    group_limit = location_group_capacity[location_id]
                    selected_group = None

                    for group in range(1, group_limit + 1):
                        key = (activity_id, location_id, week, group)

                        if (
                            key in group_assignment
                            and solver.Value(group_assignment[key]) == 1
                        ):
                            selected_group = group
                            break

                    # Directly serialize the same selected_group that is
                    # printed to the console on the next line.
                    occupancy_rows.append({
                        "activity_id": activity_id,
                        "week": week,
                        "location_id": location_id,
                        "co_share_group": f"g{selected_group}",
                    })

                    print(
                        f"    Week {week} | {location_id} | "
                        f"co_share_group=g{selected_group}"
                    )

        # =====================================================
        # DIRECT CSV EXPORT + IN-MEMORY VERIFICATION
        # =====================================================
        # Re-read every solved group_assignment directly from CP-SAT and compare
        # it with occupancy_rows BEFORE writing the CSV. This guarantees that the
        # exported co_share_group values belong to this exact solver run.
        expected_occupancy = {}
        missing_group_assignments = []
        multiple_group_assignments = []

        for _, verify_activity in activity_data.iterrows():
            verify_activity_id = verify_activity["activity_id"]
            for verify_week in weeks:
                if solver.Value(occupied[verify_activity_id, verify_week]) != 1:
                    continue
                for verify_location in activity_locations[verify_activity_id]:
                    group_limit = location_group_capacity[verify_location]
                    solved_groups = []
                    for verify_group in range(1, group_limit + 1):
                        verify_key = (
                            verify_activity_id, verify_location,
                            verify_week, verify_group
                        )
                        if (
                            verify_key in group_assignment
                            and solver.Value(group_assignment[verify_key]) == 1
                        ):
                            solved_groups.append(verify_group)

                    row_key = (
                        verify_activity_id, verify_week, verify_location
                    )
                    if len(solved_groups) == 0:
                        missing_group_assignments.append(row_key)
                    elif len(solved_groups) > 1:
                        multiple_group_assignments.append(
                            (row_key, solved_groups)
                        )
                    else:
                        expected_occupancy[row_key] = f"g{solved_groups[0]}"

        exported_occupancy = {}
        duplicate_export_rows = []
        for export_row in occupancy_rows:
            row_key = (
                export_row["activity_id"],
                int(export_row["week"]),
                export_row["location_id"],
            )
            if row_key in exported_occupancy:
                duplicate_export_rows.append(row_key)
            exported_occupancy[row_key] = export_row["co_share_group"]

        missing_export_rows = sorted(
            set(expected_occupancy) - set(exported_occupancy)
        )
        extra_export_rows = sorted(
            set(exported_occupancy) - set(expected_occupancy)
        )
        co_share_mismatches = sorted([
            (key, expected_occupancy[key], exported_occupancy[key])
            for key in set(expected_occupancy) & set(exported_occupancy)
            if expected_occupancy[key] != exported_occupancy[key]
        ])

        print("\nCSV EXPORT VERIFICATION")
        print("--------------------------------")
        print("Solved occupancy rows:", len(expected_occupancy))
        print("Captured occupancy rows:", len(occupancy_rows))
        print("Missing group assignments:", len(missing_group_assignments))
        print("Multiple group assignments:", len(multiple_group_assignments))
        print("Duplicate captured rows:", len(duplicate_export_rows))
        print("Missing captured rows:", len(missing_export_rows))
        print("Extra captured rows:", len(extra_export_rows))
        print("Co-share mismatches:", len(co_share_mismatches))

        verification_errors = (
            missing_group_assignments
            or multiple_group_assignments
            or duplicate_export_rows
            or missing_export_rows
            or extra_export_rows
            or co_share_mismatches
        )
        if verification_errors:
            if missing_group_assignments:
                print("First missing group assignments:",
                      missing_group_assignments[:10])
            if multiple_group_assignments:
                print("First multiple group assignments:",
                      multiple_group_assignments[:10])
            if duplicate_export_rows:
                print("First duplicate captured rows:",
                      duplicate_export_rows[:10])
            if missing_export_rows:
                print("First missing captured rows:",
                      missing_export_rows[:10])
            if extra_export_rows:
                print("First extra captured rows:",
                      extra_export_rows[:10])
            if co_share_mismatches:
                print("First co-share mismatches:",
                      co_share_mismatches[:10])
            raise RuntimeError(
                "CSV export verification failed; files were not written."
            )

        print(
            "PASS: SCHEDULE_OCCUPANCY exactly matches "
            "the solved CP-SAT group assignments."
        )

        output_dir = Path(f"output_{SCENARIO}")
        output_dir.mkdir(parents=True, exist_ok=True)

        access_df = pd.DataFrame(access_rows, columns=[
            "activity_id", "access_seq", "week", "eclo", "access_night"
        ])
        occupancy_df = pd.DataFrame(occupancy_rows, columns=[
            "activity_id", "week", "location_id", "co_share_group"
        ])

        try:
            access_df.to_csv(
                output_dir / "SCHEDULE_ACCESS.csv", index=False
            )
            occupancy_df.to_csv(
                output_dir / "SCHEDULE_OCCUPANCY.csv", index=False
            )
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot write {exc.filename!r}. Close the CSV in Excel "
                "or another program, then run the solver again."
            ) from exc

        pd.DataFrame(results_rows, columns=[
            "scenario", "contract_number", "simulated_completion_date", "overrun_days"
        ]).sort_values("contract_number").to_csv(
            output_dir / "RESULTS.csv", index=False
        )

        print("\nDIRECT CSV EXPORT")
        print("--------------------------------")
        print("Wrote:", output_dir / "SCHEDULE_ACCESS.csv")
        print("Wrote:", output_dir / "SCHEDULE_OCCUPANCY.csv")
        print("Wrote:", output_dir / "RESULTS.csv")
        print(
            "Rows:", len(access_rows), "access |",
            len(occupancy_rows), "occupancy |",
            len(results_rows), "results"
        )

    else:

        print("No feasible schedule found. Status:", solver.StatusName(status))

    return status, solver


# =========================================================
# SOLVE WITH FIXED PLANNING HORIZON
# =========================================================
#
# The official validator only accepts weeks 1..horizon_weeks for all
# scenarios, including Scenario A. Therefore A, B and C all use the
# nominal planning horizon from 06_PARAMETERS.csv.

status, solver = build_and_solve(
    horizon_weeks,
    solve_time_seconds=30
)
