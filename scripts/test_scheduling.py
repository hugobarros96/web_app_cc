"""Test script — edit users/availabilities here and run with: uv run python scripts/test_scheduling.py"""

from backend.users import Scheduler, User, Day
from backend.scheduler import solve


def main():
    # --- PT / Scheduler availability ---
    pt = Scheduler(name="PT")
    pt.add_availability(Day.MONDAY, "08:00", "18:00")
    pt.add_availability(Day.TUESDAY, "08:00", "18:00")
    pt.add_availability(Day.WEDNESDAY, "10:00", "16:00")
    pt.add_availability(Day.THURSDAY, "08:00", "18:00")
    pt.add_availability(Day.FRIDAY, "08:00", "14:00")

    # --- Users ---
    u1 = User(name="Alice")
    u1.add_availability(Day.MONDAY, "10:00", "12:00")
    u1.add_availability(Day.WEDNESDAY, "10:00", "13:00")
    u1.add_slot_request(60)  # needs 1h session
    pt.add_user(u1)

    u2 = User(name="Bob")
    u2.add_availability(Day.MONDAY, "10:00", "11:30")
    u2.add_availability(Day.TUESDAY, "14:00", "16:00")
    u2.add_slot_request(60)  # needs 1h session
    pt.add_user(u2)

    u3 = User(name="Carol")
    u3.add_availability(Day.THURSDAY, "09:00", "12:00")
    u3.add_availability(Day.FRIDAY, "08:00", "12:00")
    u3.add_slot_request(90)  # needs 1.5h session
    u3.add_slot_request(60)  # needs another 1h session
    pt.add_user(u3)

    u4 = User(name="Dave")
    u4.add_availability(Day.MONDAY, "08:00", "10:00")
    u4.add_availability(Day.WEDNESDAY, "12:00", "16:00")
    u4.add_availability(Day.FRIDAY, "10:00", "14:00")
    u4.add_slot_request(30)
    u4.add_slot_request(30)
    pt.add_user(u4)

    # --- Solve ---
    result = solve(pt)

    print(f"Status: {result.status}")
    print(f"Scheduled: {result.total_scheduled}/{result.total_requested}\n")

    for a in result.assignments:
        print(f"  {a.user_name:>8}  {a.day.name:<9}  {a.start_time}-{a.end_time}  ({a.duration_min}min)")


if __name__ == "__main__":
    main()
