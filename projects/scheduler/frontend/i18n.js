/* ──────────────────────────────────────────────
   Internationalization (EN / PT)
   ────────────────────────────────────────────── */

const I18N = {
  en: {
    // Landing
    scheduler: "Scheduler",

    // Main app
    schedulingApp: "Scheduling App",
    schedulerAvailability: "Scheduler Availability",
    addUser: "+ Add User",
    startScheduling: "Start Scheduling",
    scheduling: "Scheduling...",
    toggleMenu: "Toggle menu",
    addNewUser: "Add new user",
    availabilityBlocks: (n) => `${n} availability block(s)`,
    slots: (n) => `Slots (${n})`,
    availability: "Availability",
    deleteUser: "Delete user",
    removeSlot: "Remove slot",
    addSlot: "+ Add Slot",
    addGroupSlot: "+ Add Group Slot",
    groupSlots: "Group Slots",
    resetAllUsersAvailability: "Reset All Users Availability",
    resetSchedulerAvailability: "Reset Availability",

    // Dialog
    newUser: "New User",
    newGroupSlot: "New Group Slot",
    selectParticipants: "Participants",
    minTwoParticipants: "Select at least 2 participants",
    duration: "Duration",
    groupClass: "Group",
    addSlotTitle: "Add Slot",
    slotsLabel: "Slots",
    name: "Name",
    userNamePlaceholder: "User name",
    create: "Create",
    add: "Add",
    cancel: "Cancel",
    enterName: "Please enter a name",
    addOneslot: "Add at least one slot",
    maxSlots: "Maximum 4 slots per user",

    // Day names
    Monday: "Monday",
    Tuesday: "Tuesday",
    Wednesday: "Wednesday",
    Thursday: "Thursday",
    Friday: "Friday",
    Saturday: "Saturday",
    Sunday: "Sunday",

    // Results page
    schedulingResults: "Scheduling Results",
    copyToClipboard: "Copy to clipboard",
    copied: "Copied!",
    generateNewResult: "Generate New Result",
    generating: "Generating...",
    noResults: "No results found. Run scheduling from the main app first.",
    noDataForRegenerate: "No scheduling data found. Run scheduling from the main app first.",
    slotsScheduled: (scheduled, total) => `${scheduled} of ${total} slots scheduled`,
    warningSlotNotScheduled: (user, duration) =>
      `WARNING: Slot (${duration}min) for ${user} was not scheduled due to incompatibility.`,

    // Footer
    footer: "&copy; 2026 Developed by Hugo Barros",
  },

  pt: {
    // Landing
    scheduler: "Agendador",

    // Main app
    schedulingApp: "App de Agendamento",
    schedulerAvailability: "Disponibilidade do Agendador",
    addUser: "+ Adicionar Utilizador",
    startScheduling: "Iniciar Agendamento",
    scheduling: "A agendar...",
    toggleMenu: "Alternar menu",
    addNewUser: "Adicionar novo utilizador",
    availabilityBlocks: (n) => `${n} bloco(s) de disponibilidade`,
    slots: (n) => `Sessões (${n})`,
    availability: "Disponibilidade",
    deleteUser: "Eliminar utilizador",
    removeSlot: "Remover sessão",
    addSlot: "+ Adicionar Sessão",
    addGroupSlot: "+ Adicionar Sessão de Grupo",
    groupSlots: "Sessões de Grupo",
    resetAllUsersAvailability: "Limpar Disponibilidade de Todos",
    resetSchedulerAvailability: "Limpar Disponibilidade",

    // Dialog
    newUser: "Novo Utilizador",
    newGroupSlot: "Nova Sessão de Grupo",
    selectParticipants: "Participantes",
    minTwoParticipants: "Selecione pelo menos 2 participantes",
    duration: "Duração",
    groupClass: "Grupo",
    addSlotTitle: "Adicionar Sessão",
    slotsLabel: "Sessões",
    name: "Nome",
    userNamePlaceholder: "Nome do utilizador",
    create: "Criar",
    add: "Adicionar",
    cancel: "Cancelar",
    enterName: "Por favor introduza um nome",
    addOneslot: "Adicione pelo menos uma sessão",
    maxSlots: "Máximo de 4 sessões por utilizador",

    // Day names
    Monday: "Segunda-feira",
    Tuesday: "Terça-feira",
    Wednesday: "Quarta-feira",
    Thursday: "Quinta-feira",
    Friday: "Sexta-feira",
    Saturday: "Sábado",
    Sunday: "Domingo",

    // Results page
    schedulingResults: "Resultados do Agendamento",
    copyToClipboard: "Copiar para área de transferência",
    copied: "Copiado!",
    generateNewResult: "Gerar Novo Resultado",
    generating: "A gerar...",
    noResults: "Nenhum resultado encontrado. Execute o agendamento primeiro.",
    noDataForRegenerate: "Dados de agendamento não encontrados. Execute o agendamento primeiro.",
    slotsScheduled: (scheduled, total) => `${scheduled} de ${total} sessões agendadas`,
    warningSlotNotScheduled: (user, duration) =>
      `AVISO: Sessão (${duration}min) de ${user} não foi agendada por incompatibilidade.`,

    // Footer
    footer: "&copy; 2026 Desenvolvido por Hugo Barros",
  },
};

function getLang() {
  return localStorage.getItem("schedulerLang") || "en";
}

function setLang(lang) {
  localStorage.setItem("schedulerLang", lang);
}

function t(key) {
  const lang = getLang();
  return (I18N[lang] && I18N[lang][key]) || I18N.en[key] || key;
}

/**
 * Create the language toggle element.
 * If container is provided, appends it. Returns the toggle element.
 */
function createLangToggle(container) {
  const toggle = document.createElement("div");
  toggle.className = "lang-toggle";

  const btnEn = document.createElement("button");
  btnEn.textContent = "EN";
  btnEn.className = "lang-btn" + (getLang() === "en" ? " active" : "");
  btnEn.addEventListener("click", () => {
    setLang("en");
    location.reload();
  });

  const btnPt = document.createElement("button");
  btnPt.textContent = "PT";
  btnPt.className = "lang-btn" + (getLang() === "pt" ? " active" : "");
  btnPt.addEventListener("click", () => {
    setLang("pt");
    location.reload();
  });

  toggle.append(btnEn, btnPt);
  if (container) container.appendChild(toggle);
  return toggle;
}
