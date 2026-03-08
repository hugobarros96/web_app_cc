import pytest
from backend.users import Day, Scheduler, SlotRequest, User


class TestSlotRequest:
    def test_valid_durations(self):
        for d in [30, 45, 60, 75, 90]:
            sr = SlotRequest(duration_min=d)
            assert sr.duration_min == d

    def test_below_minimum(self):
        with pytest.raises(ValueError, match="below minimum"):
            SlotRequest(duration_min=10)

    def test_above_maximum(self):
        with pytest.raises(ValueError, match="exceeds maximum"):
            SlotRequest(duration_min=120)

    def test_not_multiple_of_granularity(self):
        with pytest.raises(ValueError, match="multiple"):
            SlotRequest(duration_min=37)

    def test_blocks(self):
        sr = SlotRequest(duration_min=90)
        assert sr.blocks == 6  # 90 / 15


class TestUser:
    def test_add_availability(self):
        u = User(name="Alice")
        u.add_availability(Day.MONDAY, "10:00", "12:00")
        assert len(u.availability) == 1
        assert u.availability[0] == (Day.MONDAY, 600, 720)

    def test_invalid_availability(self):
        u = User(name="Alice")
        with pytest.raises(ValueError, match="before end"):
            u.add_availability(Day.MONDAY, "12:00", "10:00")

    def test_add_slot_request(self):
        u = User(name="Alice")
        u.add_slot_request(60)
        u.add_slot_request(30)
        assert len(u.slot_requests) == 2

    def test_max_slots_exceeded(self):
        u = User(name="Alice")
        for _ in range(4):
            u.add_slot_request(30)
        with pytest.raises(ValueError, match="at most"):
            u.add_slot_request(30)

    def test_total_blocks_needed(self):
        u = User(name="Alice")
        u.add_slot_request(60)  # 4 blocks
        u.add_slot_request(30)  # 2 blocks
        assert u.total_blocks_needed == 6


class TestScheduler:
    def test_add_remove_user(self):
        s = Scheduler(name="PT")
        u = User(name="Alice")
        s.add_user(u)
        assert len(s.users) == 1
        s.remove_user("Alice")
        assert len(s.users) == 0

    def test_get_user(self):
        s = Scheduler(name="PT")
        u = User(name="Alice")
        s.add_user(u)
        assert s.get_user("Alice") is u
        assert s.get_user("Bob") is None

    def test_max_users(self):
        s = Scheduler(name="PT")
        for i in range(50):
            s.add_user(User(name=f"User{i}"))
        with pytest.raises(ValueError, match="more than 50"):
            s.add_user(User(name="OneMore"))
