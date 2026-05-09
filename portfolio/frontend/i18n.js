/* Portfolio landing — i18n (EN / PT) */

const I18N = {
  en: {
    portfolioTagline: "Personal projects",
    scheduler: "Scheduler",
    schedulerDesc: "Optimization-based scheduling app for fitting clients into a trainer's calendar.",
    companion: "Chat with Hugo",
    companionDesc: "AI chatbot that answers questions about my CV, background and projects.",
  },
  pt: {
    portfolioTagline: "Projetos pessoais",
    scheduler: "Agendador",
    schedulerDesc: "App de agendamento por otimização para encaixar clientes no calendário de um treinador.",
    companion: "Fala com o Hugo",
    companionDesc: "Chatbot de IA que responde a perguntas sobre o meu CV, percurso e projetos.",
  },
};

function getLang() {
  return localStorage.getItem("portfolioLang") || "en";
}

function setLang(lang) {
  localStorage.setItem("portfolioLang", lang);
}

function t(key) {
  const lang = getLang();
  return (I18N[lang] && I18N[lang][key]) || I18N.en[key] || key;
}

function createLangToggle(container) {
  const toggle = document.createElement("div");
  toggle.className = "lang-toggle";

  const btnEn = document.createElement("button");
  btnEn.textContent = "EN";
  btnEn.className = "lang-btn" + (getLang() === "en" ? " active" : "");
  btnEn.addEventListener("click", () => { setLang("en"); location.reload(); });

  const btnPt = document.createElement("button");
  btnPt.textContent = "PT";
  btnPt.className = "lang-btn" + (getLang() === "pt" ? " active" : "");
  btnPt.addEventListener("click", () => { setLang("pt"); location.reload(); });

  toggle.append(btnEn, btnPt);
  if (container) container.appendChild(toggle);
  return toggle;
}
