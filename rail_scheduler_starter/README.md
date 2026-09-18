# Rail Track Access Scheduler — starter

This is the first CP-SAT baseline for the hackathon dataset.

Implemented now:
- CSV ingestion and week conversion
- CP-SAT activity/week/access-night variables
- workload conservation (standard=1.0, ECLO=1.5, integer-scaled)
- planned-start filtering
- predecessor FS+0 (strictly later week)
- weekly access-night structure
- workfront limits
- Scenario A ECLO prohibition
- Scenario B planned-completion constraint
- baseline priority/earliness objective
- independent access-level validator

Next implementation milestone (required before official submission):
1. route expansion from start/end locations into SEC + PLAT occupancy
2. buffer/Live opposite-bound and H01-H02 cross-line closures
3. PM/PC/C possession and co_share_group variables
4. LOCATION_SUPPLY capacity and Scenario B/C excess-access variables
5. Scenario C 2-week ECLO continuity windows
6. exact A/B/C scoring equations and RESULTS.csv
7. SCHEDULE_OCCUPANCY.csv generation and comparison with official validator

Run:
    pip install -r requirements.txt
    python scheduler.py --scenario A --time-limit 30
    python validator.py
