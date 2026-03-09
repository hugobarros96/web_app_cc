import pytest
from backend.users import Day, Scheduler, User
from backend.scheduler import solve


def make_pt(*days_and_times):
    """Helper: create a PT with availability windows.
    Each arg is (Day, start_str, end_str).
    """
    pt = Scheduler(name="PT")
    for day, start, end in days_and_times:
        pt.add_availability(day, start, end)
    return pt


def make_user(name, availability, slot_durations):
    """Helper: create a user with availability and slot requests."""
    u = User(name=name)
    for day, start, end in availability:
        u.add_availability(day, start, end)
    for d in slot_durations:
        u.add_slot_request(d)
    return u


class TestBasicScheduling:
    def test_single_user_single_slot(self):
        pt = make_pt((Day.MONDAY, "10:00", "18:00"))
        u = make_user("Alice", [(Day.MONDAY, "10:00", "12:00")], [60])
        pt.add_user(u)

        result = solve(pt)
        assert result.status == "OPTIMAL"
        assert result.total_scheduled == 1
        assert result.assignments[0].user_name == "Alice"
        assert result.assignments[0].day == Day.MONDAY

    def test_two_users_no_conflict(self):
        pt = make_pt(
            (Day.MONDAY, "08:00", "18:00"),
            (Day.TUESDAY, "08:00", "18:00"),
        )
        u1 = make_user("Alice", [(Day.MONDAY, "10:00", "12:00")], [60])
        u2 = make_user("Bob", [(Day.TUESDAY, "14:00", "16:00")], [60])
        pt.add_user(u1)
        pt.add_user(u2)

        result = solve(pt)
        assert result.status == "OPTIMAL"
        assert result.total_scheduled == 2

    def test_claude_md_example(self):
        """The example from CLAUDE.md: User2 gets Monday, User1 goes to Tuesday."""
        pt = make_pt(
            (Day.MONDAY, "10:00", "18:00"),
            (Day.TUESDAY, "10:00", "18:00"),
        )
        u1 = make_user(
            "User1",
            [(Day.MONDAY, "10:00", "11:00"), (Day.TUESDAY, "14:00", "15:30")],
            [60],
        )
        u2 = make_user("User2", [(Day.MONDAY, "10:00", "11:30")], [60])
        pt.add_user(u1)
        pt.add_user(u2)

        result = solve(pt)
        assert result.status == "OPTIMAL"
        assert result.total_scheduled == 2

        by_name = {a.user_name: a for a in result.assignments}
        # User2 can only fit on Monday
        assert by_name["User2"].day == Day.MONDAY
        # User1 should be moved to Tuesday
        assert by_name["User1"].day == Day.TUESDAY


class TestNoOverlap:
    def test_overlapping_users_get_different_times(self):
        """Two users available at the same time — solver must not overlap them."""
        pt = make_pt((Day.MONDAY, "10:00", "12:00"))
        u1 = make_user("Alice", [(Day.MONDAY, "10:00", "12:00")], [60])
        u2 = make_user("Bob", [(Day.MONDAY, "10:00", "12:00")], [60])
        pt.add_user(u1)
        pt.add_user(u2)

        result = solve(pt)
        assert result.status == "OPTIMAL"
        assert result.total_scheduled == 2

        a1, a2 = result.assignments
        # They must not overlap
        assert a1.end_time <= a2.start_time or a2.end_time <= a1.start_time


class TestSameDayConstraint:
    def test_two_slots_same_user_different_days(self):
        """Two slots for the same user must be on different days."""
        pt = make_pt(
            (Day.MONDAY, "08:00", "18:00"),
            (Day.TUESDAY, "08:00", "18:00"),
        )
        u = make_user(
            "Alice",
            [(Day.MONDAY, "08:00", "18:00"), (Day.TUESDAY, "08:00", "18:00")],
            [60, 60],
        )
        pt.add_user(u)

        result = solve(pt)
        assert result.status == "OPTIMAL"
        assert result.total_scheduled == 2

        days = [a.day for a in result.assignments]
        assert len(set(days)) == 2, "Both slots ended up on the same day!"

    def test_three_slots_three_days(self):
        """Three slots for one user need three different days."""
        pt = make_pt(
            (Day.MONDAY, "08:00", "18:00"),
            (Day.TUESDAY, "08:00", "18:00"),
            (Day.WEDNESDAY, "08:00", "18:00"),
        )
        u = make_user(
            "Alice",
            [
                (Day.MONDAY, "08:00", "18:00"),
                (Day.TUESDAY, "08:00", "18:00"),
                (Day.WEDNESDAY, "08:00", "18:00"),
            ],
            [60, 60, 60],
        )
        pt.add_user(u)

        result = solve(pt)
        assert result.status == "OPTIMAL"
        assert result.total_scheduled == 3

        days = [a.day for a in result.assignments]
        assert len(set(days)) == 3

    def test_same_day_constraint_forces_partial_schedule(self):
        """If only one day available, user with 2 slots can only get 1."""
        pt = make_pt((Day.MONDAY, "08:00", "18:00"))
        u = make_user(
            "Alice",
            [(Day.MONDAY, "08:00", "18:00")],
            [60, 60],
        )
        pt.add_user(u)

        result = solve(pt)
        assert result.total_scheduled == 1  # can only fit 1 of 2


class TestEdgeCases:
    def test_no_pt_availability(self):
        pt = Scheduler(name="PT")
        u = make_user("Alice", [(Day.MONDAY, "10:00", "12:00")], [60])
        pt.add_user(u)

        result = solve(pt)
        assert result.status == "INFEASIBLE"
        assert result.total_scheduled == 0

    def test_no_user_availability_overlap_with_pt(self):
        """User available when PT is not — should schedule 0."""
        pt = make_pt((Day.MONDAY, "08:00", "10:00"))
        u = make_user("Alice", [(Day.MONDAY, "14:00", "16:00")], [60])
        pt.add_user(u)

        result = solve(pt)
        assert result.total_scheduled == 0

    def test_no_users(self):
        pt = make_pt((Day.MONDAY, "08:00", "18:00"))
        result = solve(pt)
        assert result.status == "OPTIMAL"
        assert result.total_scheduled == 0
        assert result.total_requested == 0

    def test_slot_too_long_for_window(self):
        """90min slot but only 60min overlap between PT and user."""
        pt = make_pt((Day.MONDAY, "10:00", "11:00"))
        u = make_user("Alice", [(Day.MONDAY, "10:00", "11:00")], [90])
        pt.add_user(u)

        result = solve(pt)
        assert result.total_scheduled == 0


class TestRegenerateExcludedSolutions:
    def _make_scenario_with_multiple_solutions(self):
        """Create a scenario where multiple optimal solutions exist.
        PT available Mon+Tue 08-18. Alice has a 60min slot and is
        available both days — so the solver can place her on either day.
        """
        pt = make_pt(
            (Day.MONDAY, "08:00", "18:00"),
            (Day.TUESDAY, "08:00", "18:00"),
        )
        u = make_user(
            "Alice",
            [(Day.MONDAY, "08:00", "18:00"), (Day.TUESDAY, "08:00", "18:00")],
            [60],
        )
        pt.add_user(u)
        return pt

    def test_regenerate_produces_different_result(self):
        """Excluding the first solution should produce a different one."""
        pt = self._make_scenario_with_multiple_solutions()
        result1 = solve(pt)
        assert result1.status == "OPTIMAL"
        assert result1.total_scheduled == 1

        # Build excluded solution from first result
        excluded = [
            [(a.user_name, a.slot_idx, a.start_block) for a in result1.assignments]
        ]

        pt2 = self._make_scenario_with_multiple_solutions()
        result2 = solve(pt2, excluded_solutions=excluded)
        assert result2.status == "OPTIMAL"
        assert result2.total_scheduled == 1

        # The assignment must differ (different day or different start block)
        a1 = result1.assignments[0]
        a2 = result2.assignments[0]
        assert (a1.day, a1.start_block) != (a2.day, a2.start_block), \
            "Regenerated result should differ from the excluded one"

    def test_regenerate_multiple_exclusions(self):
        """Excluding multiple solutions should keep producing new ones."""
        pt = make_pt(
            (Day.MONDAY, "08:00", "12:00"),
            (Day.TUESDAY, "08:00", "12:00"),
        )
        # User with 60min slot available on both days, wide window
        u = make_user(
            "Bob",
            [(Day.MONDAY, "08:00", "12:00"), (Day.TUESDAY, "08:00", "12:00")],
            [60],
        )
        pt.add_user(u)

        seen_blocks = set()
        excluded = []

        result = solve(pt)
        assert result.total_scheduled == 1
        seen_blocks.add(result.assignments[0].start_block)

        # Generate a few more solutions, each excluding all previous ones
        for _ in range(3):
            excluded.append(
                [(a.user_name, a.slot_idx, a.start_block) for a in result.assignments]
            )
            pt_new = make_pt(
                (Day.MONDAY, "08:00", "12:00"),
                (Day.TUESDAY, "08:00", "12:00"),
            )
            u_new = make_user(
                "Bob",
                [(Day.MONDAY, "08:00", "12:00"), (Day.TUESDAY, "08:00", "12:00")],
                [60],
            )
            pt_new.add_user(u_new)
            result = solve(pt_new, excluded_solutions=excluded)
            assert result.total_scheduled == 1
            seen_blocks.add(result.assignments[0].start_block)

        # We should have at least 4 distinct placements
        assert len(seen_blocks) >= 4, \
            f"Expected at least 4 distinct placements, got {len(seen_blocks)}"

    def test_regenerate_exhausted_returns_different_status_or_fewer(self):
        """When all solutions are excluded, solver should return fewer scheduled."""
        # Tight scenario: PT and user only overlap for exactly 60min
        pt = make_pt((Day.MONDAY, "10:00", "11:00"))
        u = make_user("Alice", [(Day.MONDAY, "10:00", "11:00")], [60])
        pt.add_user(u)

        result1 = solve(pt)
        assert result1.total_scheduled == 1

        # Exclude the only possible solution
        excluded = [
            [(a.user_name, a.slot_idx, a.start_block) for a in result1.assignments]
        ]

        pt2 = make_pt((Day.MONDAY, "10:00", "11:00"))
        u2 = make_user("Alice", [(Day.MONDAY, "10:00", "11:00")], [60])
        pt2.add_user(u2)
        result2 = solve(pt2, excluded_solutions=excluded)

        # No valid placement left — should schedule 0
        assert result2.total_scheduled == 0
