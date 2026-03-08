/* ──────────────────────────────────────────────
   State
   ────────────────────────────────────────────── */

const COLORS = [
  "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6",
  "#3b82f6", "#8b5cf6", "#ec4899", "#f43f5e", "#06b6d4",
  "#84cc16", "#a855f7", "#d946ef", "#0ea5e9", "#10b981",
];
let colorIndex = 0;

const PT_COLOR = "rgba(99,102,241,0.35)";
const PT_ID = "__PT__";

const DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const DAY_MAP = {
  Monday: 0, Tuesday: 1, Wednesday: 2, Thursday: 3,
  Friday: 4, Saturday: 5, Sunday: 6,
};

// App state
const state = {
  selectedUser: PT_ID,
  users: [],           // { id, name, color, slots: [{duration}], availability: [{day, start, end, eventId}] }
  ptAvailability: [],  // [{day, start, end, eventId}]
};

let calendar;
let nextEventId = 1;

/* ──────────────────────────────────────────────
   Helpers
   ────────────────────────────────────────────── */

function nextColor() {
  const c = COLORS[colorIndex % COLORS.length];
  colorIndex++;
  return c;
}

function getDayName(dateStr) {
  const d = new Date(dateStr);
  return ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][d.getDay()];
}

function formatTime(dateStr) {
  const d = new Date(dateStr);
  return d.toTimeString().slice(0, 5);
}

function timeToMinutes(t) {
  const [h, m] = t.split(":").map(Number);
  return h * 60 + m;
}

function minutesToTime(m) {
  return String(Math.floor(m / 60)).padStart(2, "0") + ":" + String(m % 60).padStart(2, "0");
}

/** Merge overlapping/adjacent availability blocks on the same day in-place. */
function mergeAvailability(avail) {
  const byDay = {};
  for (const a of avail) {
    if (!byDay[a.day]) byDay[a.day] = [];
    byDay[a.day].push(a);
  }

  const merged = [];
  for (const day of Object.keys(byDay)) {
    const blocks = byDay[day].sort((a, b) => timeToMinutes(a.start) - timeToMinutes(b.start));
    let cur = { day, start: blocks[0].start, end: blocks[0].end, eventId: blocks[0].eventId };

    for (let i = 1; i < blocks.length; i++) {
      const curEnd = timeToMinutes(cur.end);
      const nextStart = timeToMinutes(blocks[i].start);
      const nextEnd = timeToMinutes(blocks[i].end);

      if (nextStart <= curEnd) {
        // Overlapping or adjacent — extend
        if (nextEnd > curEnd) {
          cur.end = blocks[i].end;
        }
      } else {
        merged.push(cur);
        cur = { day, start: blocks[i].start, end: blocks[i].end, eventId: blocks[i].eventId };
      }
    }
    merged.push(cur);
  }

  return merged;
}

/* ──────────────────────────────────────────────
   Persistence — save/load via localStorage
   ────────────────────────────────────────────── */

const STORAGE_KEY = "schedulingData";

function persistState() {
  const payload = {
    ptAvailability: state.ptAvailability.map((a) => ({
      day: a.day,
      start: a.start,
      end: a.end,
    })),
    users: state.users.map((u) => ({
      name: u.name,
      color: u.color,
      slots: u.slots.map((s) => s.duration),
      availability: u.availability.map((a) => ({
        day: a.day,
        start: a.start,
        end: a.end,
      })),
    })),
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const data = JSON.parse(raw);

    // Load PT availability
    state.ptAvailability = mergeAvailability((data.ptAvailability || []).map((a) => ({
      day: a.day,
      start: a.start,
      end: a.end,
      eventId: "ev_" + nextEventId++,
    })));

    // Load users
    state.users = (data.users || []).map((u) => {
      const color = u.color;
      const cIdx = COLORS.indexOf(color);
      if (cIdx >= colorIndex) colorIndex = cIdx + 1;

      return {
        id: "user_" + nextEventId++,
        name: u.name,
        color,
        slots: u.slots.map((d) => ({ duration: d })),
        availability: mergeAvailability(u.availability.map((a) => ({
          day: a.day,
          start: a.start,
          end: a.end,
          eventId: "ev_" + nextEventId++,
        }))),
      };
    });
  } catch (err) {
    console.error("Failed to load state from localStorage:", err);
  }
}

/* ──────────────────────────────────────────────
   Calendar setup
   ────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", function () {
  const calendarEl = document.getElementById("calendar");

  calendar = new FullCalendar.Calendar(calendarEl, {
    initialView: "timeGridWeek",
    firstDay: 1,
    slotMinTime: "07:00:00",
    slotMaxTime: "22:00:00",
    slotDuration: "00:15:00",
    slotLabelInterval: "01:00:00",
    selectable: true,
    selectMirror: true,
    allDaySlot: false,
    editable: false,
    headerToolbar: {
      left: "title",
      center: "",
      right: "today prev,next",
    },
    navLinks: false,
    eventOverlap: true,
    slotEventOverlap: true,

    select: function (info) {
      handleCalendarSelect(info);
      calendar.unselect();
    },

    eventClick: function (info) {
      handleEventClick(info);
    },
  });

  calendar.render();

  // Load saved state from localStorage
  loadState();

  // Sidebar event listeners
  document.getElementById("pt-section").addEventListener("click", () => selectUser(PT_ID));
  document.getElementById("add-user-btn").addEventListener("click", openNewUserDialog);
  document.getElementById("schedule-btn").addEventListener("click", runScheduling);

  // Dialog listeners
  document.getElementById("dialog-cancel").addEventListener("click", closeDialog);
  document.getElementById("dialog-confirm").addEventListener("click", confirmDialog);
  document.getElementById("dialog-add-slot").addEventListener("click", addDialogSlot);

  refreshCalendarEvents();
  renderSidebar();
});

/* ──────────────────────────────────────────────
   Calendar interaction
   ────────────────────────────────────────────── */

function handleCalendarSelect(info) {
  const userId = state.selectedUser;
  if (!userId) return;

  const evId = "ev_" + nextEventId++;
  const day = getDayName(info.startStr);
  const start = formatTime(info.startStr);
  const end = formatTime(info.endStr);

  if (userId === PT_ID) {
    state.ptAvailability.push({ day, start, end, eventId: evId });
    state.ptAvailability = mergeAvailability(state.ptAvailability);
  } else {
    const user = state.users.find((u) => u.id === userId);
    if (!user) return;
    user.availability.push({ day, start, end, eventId: evId });
    user.availability = mergeAvailability(user.availability);
  }

  refreshCalendarEvents();
  renderSidebar();
  persistState();
}

function handleEventClick(info) {
  const evId = info.event.id;
  const selected = state.selectedUser;

  // Only allow deleting blocks belonging to the selected user
  if (selected === PT_ID) {
    const ptIdx = state.ptAvailability.findIndex((a) => a.eventId === evId);
    if (ptIdx !== -1) {
      state.ptAvailability.splice(ptIdx, 1);
      refreshCalendarEvents();
      renderSidebar();
      persistState();
    }
  } else {
    const user = state.users.find((u) => u.id === selected);
    if (!user) return;
    const idx = user.availability.findIndex((a) => a.eventId === evId);
    if (idx !== -1) {
      user.availability.splice(idx, 1);
      refreshCalendarEvents();
      renderSidebar();
      persistState();
    }
  }
}

/* ──────────────────────────────────────────────
   Calendar rendering
   ────────────────────────────────────────────── */

function refreshCalendarEvents() {
  calendar.getEvents().forEach((e) => e.remove());

  const selected = state.selectedUser;

  function getDateForDay(dayName) {
    const start = calendar.view.currentStart;
    const dayMapJS = {
      Monday: 1, Tuesday: 2, Wednesday: 3, Thursday: 4,
      Friday: 5, Saturday: 6, Sunday: 0,
    };
    const target = dayMapJS[dayName];
    for (let i = 0; i < 7; i++) {
      const d = new Date(start);
      d.setDate(d.getDate() + i);
      if (d.getDay() === target) return d.toISOString().slice(0, 10);
    }
    return start.toISOString().slice(0, 10);
  }

  // Always show PT availability (translucent)
  for (const a of state.ptAvailability) {
    const date = getDateForDay(a.day);
    calendar.addEvent({
      id: a.eventId,
      title: "PT",
      start: date + "T" + a.start,
      end: date + "T" + a.end,
      backgroundColor: PT_COLOR,
      borderColor: "rgba(99,102,241,0.6)",
      textColor: "#4338ca",
    });
  }

  if (selected && selected !== PT_ID) {
    // Show only selected user's availability
    const user = state.users.find((u) => u.id === selected);
    if (user) {
      for (const a of user.availability) {
        const date = getDateForDay(a.day);
        calendar.addEvent({
          id: a.eventId,
          title: user.name,
          start: date + "T" + a.start,
          end: date + "T" + a.end,
          backgroundColor: user.color,
          borderColor: user.color,
          textColor: "#fff",
        });
      }
    }
  } else {
    // Show all users' availability
    for (const user of state.users) {
      for (const a of user.availability) {
        const date = getDateForDay(a.day);
        calendar.addEvent({
          id: a.eventId,
          title: user.name,
          start: date + "T" + a.start,
          end: date + "T" + a.end,
          backgroundColor: user.color,
          borderColor: user.color,
          textColor: "#fff",
        });
      }
    }
  }
}

/* ──────────────────────────────────────────────
   Sidebar rendering
   ────────────────────────────────────────────── */

function renderSidebar() {
  const ptCard = document.getElementById("pt-section");
  ptCard.classList.toggle("selected", state.selectedUser === PT_ID);

  let ptInfo = ptCard.querySelector(".calendar-info");
  if (state.ptAvailability.length > 0) {
    if (!ptInfo) {
      ptInfo = document.createElement("div");
      ptInfo.className = "calendar-info";
      ptCard.appendChild(ptInfo);
    }
    ptInfo.textContent = `${state.ptAvailability.length} availability block(s)`;
  } else if (ptInfo) {
    ptInfo.remove();
  }

  const list = document.getElementById("user-list");
  list.innerHTML = "";

  for (const user of state.users) {
    const isSelected = state.selectedUser === user.id;

    const card = document.createElement("div");
    card.className = "user-card" + (isSelected ? " selected" : "");
    card.style.borderColor = isSelected ? user.color : "";
    card.dataset.user = user.id;

    const header = document.createElement("div");
    header.className = "user-header";

    const dot = document.createElement("span");
    dot.className = "user-color-dot";
    dot.style.background = user.color;

    const name = document.createElement("span");
    name.className = "user-name";
    name.textContent = user.name;

    const delBtn = document.createElement("button");
    delBtn.className = "user-delete-btn";
    delBtn.innerHTML = "&#128465;";
    delBtn.title = "Delete user";
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteUser(user.id);
    });

    header.append(dot, name, delBtn);
    card.appendChild(header);

    const dropdown = document.createElement("div");
    dropdown.className = "user-dropdown";

    // Slots
    const slotLabel = document.createElement("div");
    slotLabel.className = "slot-section-label";
    slotLabel.textContent = `Slots (${user.slots.length})`;
    dropdown.appendChild(slotLabel);

    for (let i = 0; i < user.slots.length; i++) {
      const slot = user.slots[i];
      const row = document.createElement("div");
      row.className = "slot-item";

      const label = document.createElement("span");
      label.className = "slot-label";
      label.textContent = `${slot.duration} min`;

      const removeBtn = document.createElement("button");
      removeBtn.className = "slot-remove-btn";
      removeBtn.textContent = "\u00D7";
      removeBtn.title = "Remove slot";
      removeBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        user.slots.splice(i, 1);
        renderSidebar();
        persistState();
      });

      row.append(label, removeBtn);
      dropdown.appendChild(row);
    }

    if (user.slots.length < 4) {
      const addSlotBtn = document.createElement("button");
      addSlotBtn.className = "btn btn-small";
      addSlotBtn.textContent = "+ Add Slot";
      addSlotBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        openAddSlotDialog(user.id);
      });
      dropdown.appendChild(addSlotBtn);
    }

    // Availability
    if (user.availability.length > 0) {
      const availLabel = document.createElement("div");
      availLabel.className = "avail-section-label";
      availLabel.textContent = "Availability";
      dropdown.appendChild(availLabel);

      for (let i = 0; i < user.availability.length; i++) {
        const a = user.availability[i];
        const row = document.createElement("div");
        row.className = "availability-item";

        const label = document.createElement("span");
        label.textContent = `${a.day} ${a.start}-${a.end}`;

        const removeBtn = document.createElement("button");
        removeBtn.className = "availability-remove-btn";
        removeBtn.textContent = "\u00D7";
        removeBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          user.availability.splice(i, 1);
          refreshCalendarEvents();
          renderSidebar();
          persistState();
        });

        row.append(label, removeBtn);
        dropdown.appendChild(row);
      }
    }

    card.appendChild(dropdown);
    card.addEventListener("click", () => selectUser(user.id));
    list.appendChild(card);
  }
}

/* ──────────────────────────────────────────────
   User selection
   ────────────────────────────────────────────── */

function selectUser(userId) {
  if (state.selectedUser === userId && userId !== PT_ID) {
    state.selectedUser = PT_ID;
  } else {
    state.selectedUser = userId;
  }
  refreshCalendarEvents();
  renderSidebar();
}

/* ──────────────────────────────────────────────
   User CRUD
   ────────────────────────────────────────────── */

function deleteUser(userId) {
  state.users = state.users.filter((u) => u.id !== userId);
  if (state.selectedUser === userId) state.selectedUser = PT_ID;
  refreshCalendarEvents();
  renderSidebar();

  persistState();
}

/* ──────────────────────────────────────────────
   Dialog — New User / Add Slot
   ────────────────────────────────────────────── */

let dialogMode = "new-user";
let dialogTargetUserId = null;

function openNewUserDialog() {
  dialogMode = "new-user";
  dialogTargetUserId = null;
  document.getElementById("dialog-title").textContent = "New User";
  document.getElementById("dialog-name").value = "";
  document.getElementById("dialog-name").parentElement.style.display = "";
  document.getElementById("dialog-slots").style.display = "";
  document.getElementById("dialog-slot-list").innerHTML = "";
  document.getElementById("dialog-confirm").textContent = "Create";
  addDialogSlot();
  document.getElementById("dialog-overlay").classList.remove("hidden");
}

function openAddSlotDialog(userId) {
  dialogMode = "add-slot";
  dialogTargetUserId = userId;
  document.getElementById("dialog-title").textContent = "Add Slot";
  document.getElementById("dialog-name").parentElement.style.display = "none";
  document.getElementById("dialog-slots").style.display = "";
  document.getElementById("dialog-slot-list").innerHTML = "";
  document.getElementById("dialog-confirm").textContent = "Add";
  addDialogSlot();
  document.getElementById("dialog-overlay").classList.remove("hidden");
}

function closeDialog() {
  document.getElementById("dialog-overlay").classList.add("hidden");
}

function addDialogSlot() {
  const list = document.getElementById("dialog-slot-list");
  const row = document.createElement("div");
  row.className = "dialog-slot-row";

  const sel = document.createElement("select");
  [30, 45, 60, 75, 90].forEach((d) => {
    const opt = document.createElement("option");
    opt.value = d;
    opt.textContent = d + " min";
    sel.appendChild(opt);
  });

  const removeBtn = document.createElement("button");
  removeBtn.className = "slot-remove-btn";
  removeBtn.textContent = "\u00D7";
  removeBtn.type = "button";
  removeBtn.addEventListener("click", () => row.remove());

  row.append(sel, removeBtn);
  list.appendChild(row);
}

function confirmDialog() {
  if (dialogMode === "new-user") {
    const name = document.getElementById("dialog-name").value.trim();
    if (!name) return alert("Please enter a name");

    const slotEls = document.querySelectorAll("#dialog-slot-list select");
    if (slotEls.length === 0) return alert("Add at least one slot");
    if (slotEls.length > 4) return alert("Maximum 4 slots per user");

    const slots = Array.from(slotEls).map((s) => ({ duration: parseInt(s.value) }));

    const user = {
      id: "user_" + Date.now(),
      name,
      color: nextColor(),
      slots,
      availability: [],
    };

    state.users.push(user);
    state.selectedUser = user.id;
  } else if (dialogMode === "add-slot") {
    const user = state.users.find((u) => u.id === dialogTargetUserId);
    if (!user) return;

    const slotEls = document.querySelectorAll("#dialog-slot-list select");
    for (const s of slotEls) {
      if (user.slots.length >= 4) break;
      user.slots.push({ duration: parseInt(s.value) });
    }
  }

  closeDialog();
  refreshCalendarEvents();
  renderSidebar();
  persistState();
}

/* ──────────────────────────────────────────────
   Scheduling — call backend
   ────────────────────────────────────────────── */

async function runScheduling() {
  const payload = {
    pt_availability: state.ptAvailability.map((a) => ({
      day: DAY_MAP[a.day],
      start: a.start,
      end: a.end,
    })),
    users: state.users.map((u) => ({
      name: u.name,
      color: u.color,
      slots: u.slots.map((s) => s.duration),
      availability: u.availability.map((a) => ({
        day: DAY_MAP[a.day],
        start: a.start,
        end: a.end,
      })),
    })),
  };

  const btn = document.getElementById("schedule-btn");
  btn.textContent = "Scheduling...";
  btn.disabled = true;

  try {
    const resp = await fetch("/scheduler/api/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await resp.json();

    const userColors = {};
    for (const u of state.users) {
      userColors[u.name] = u.color;
    }
    result.user_colors = userColors;

    sessionStorage.setItem("schedulingResults", JSON.stringify(result));
    window.open("/scheduler/results", "_blank");
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    btn.textContent = "Start Scheduling";
    btn.disabled = false;
  }
}
