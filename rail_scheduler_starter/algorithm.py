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
# 4. CREATE MODEL
# =========================================================

model = cp_model.CpModel()


# =========================================================
# 5. DECISION VARIABLES
# =========================================================

scheduled = {}

for _, activity in activity_details.iterrows():

    activity_id = activity["activity_id"]

    for week in weeks:

        scheduled[activity_id, week] = model.NewBoolVar(
            f"scheduled_{activity_id}_week_{week}"
        )


# =========================================================
# 6. WORKLOAD CONSTRAINT
# =========================================================

for _, activity in activity_details.iterrows():

    activity_id = activity["activity_id"]
    total_accesses = int(activity["total_accesses"])

    model.Add(
        sum(
            scheduled[activity_id, week]
            for week in weeks
        )
        == total_accesses
    )


# =========================================================
# 7. PLANNED START CONSTRAINT
# =========================================================

for _, activity in activity_details.iterrows():

    activity_id = activity["activity_id"]
    start_week = int(activity["planned_start_week"])

    for week in weeks:

        if week < start_week:

            model.Add(
                scheduled[activity_id, week] == 0
            )


# =========================================================
# 8. SIMPLE OBJECTIVE
# =========================================================

model.Minimize(
    sum(
        week * scheduled[activity_id, week]
        for activity_id in activity_details["activity_id"]
        for week in weeks
    )
)


# =========================================================
# 9. SOLVE
# =========================================================

solver = cp_model.CpSolver()

solver.parameters.max_time_in_seconds = 30

status = solver.Solve(model)


# =========================================================
# 10. PRINT RESULTS
# =========================================================


if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):

    print("Schedule found!")

    for _, activity in activity_details.iterrows():

        activity_id = activity["activity_id"]

        selected_weeks = []

        for week in weeks:

            if solver.Value(scheduled[activity_id, week]) == 1:
                selected_weeks.append(week)

        print(activity_id, selected_weeks)

else:

    print("No feasible schedule found.")



