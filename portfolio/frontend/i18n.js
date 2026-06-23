/* Portfolio landing i18n (EN / PT) */

const I18N = {
  en: {
    hello: 'Hello',
    bio1: "I'm Hugo, an AI Engineer with <strong>5+ years</strong> of experience at <strong>Ocean Infinity</strong>, bridging research and production. My specialty is computer vision across 2D and 3D data, with a growing focus on GenAI and LLM based systems. My toolkit is Python with PyTorch, C++ and ROS2 for robotics, and the full ML pipeline from data to deployment.",
    bio2: "I've worked on projects from inception to production, including a few as Product Owner. My experience spans vision based automation (detection, segmentation, 3D reconstruction), robotics, and LLM based products.",
    bio3: "Background in <strong>Bioengineering</strong> (MSc, FEUP), where I first got hooked on applied ML. Outside work I build small things on the side to learn something new. These two are the latest.",
    projectsHeading: "Personal projects",
    scheduler: "Scheduler",
    schedulerDesc: "An optimization based weekly scheduler that fits a trainer's clients into their calendar. Given everyone's availabilities, an <strong>OR-Tools CP-SAT</strong> solver searches for the best arrangement of sessions, respecting per person slots and group classes. It is a real world tool, used by my own gym PT. I later added a <strong>LangGraph</strong> chat that turns free text into actions, so you can create users, set availabilities, and generate a schedule just by describing what you want in plain language.",
    companion: "Chat with Hugo",
    companionDesc: "A chatbot that answers as me, grounded in my CV and a personal summary. Built with the <strong>OpenAI SDK</strong> and served through a Gradio interface, it stays in character as Hugo and can walk you through my background, projects, and experience. It uses tool calling to stay honest: when someone wants to get in touch it records their details, and when it does not know the answer to a question it logs it, notifying me either way.",
    datadoctor: "Data Doctor",
    datadoctorDesc: "A clinical-analytics assistant built around a <strong>Strands</strong> agent that reads each question and routes it to the right typed tool. It predicts patient outcomes with <strong>XGBoost</strong> models (a COPD class and an ALT value with an 80 percent interval, asking back for the inputs that matter most via SHAP), runs live <strong>pandas</strong> analytics and charts over a 10,000 patient dataset, and answers grounded questions with hybrid <strong>RAG</strong> (FAISS dense plus BM25 sparse, fused and reordered by a cross-encoder reranker) over 1,050 clinical records and a 125k chunk medical textbook corpus. It also compares patients side by side, remembers cohorts across turns, and can optionally search a medical domain allowlist on the web. Every turn passes through input and output guardrails (PII redaction, prompt injection blocking, automatic disclaimers) and is traced end to end in <strong>MLflow</strong>, while a feedback widget drives an active learning loop that retrains and promotes a model only when it beats the baseline. It runs on synthetic data and is not for clinical use.",
    readmeLink: "Read the README on GitHub →",
    contactCta: "If you want to know more about me, contact me at {email} and I'll send you my CV.",
  },
  pt: {
    hello: 'Olá',
    bio1: "Sou o Hugo, Engenheiro de IA com <strong>5+ anos</strong> de experiência na <strong>Ocean Infinity</strong>, a fazer a ponte entre investigação e produção. A minha especialidade é visão computacional sobre dados 2D e 3D, com um foco crescente em GenAI e sistemas baseados em LLMs. As minhas ferramentas são Python com PyTorch, C++ e ROS2 para robótica, e o pipeline completo de ML, dos dados ao deployment.",
    bio2: "Trabalhei em projetos do conceito ao produto, alguns deles como Product Owner. A minha experiência abrange automação baseada em visão (deteção, segmentação, reconstrução 3D), robótica, e produtos baseados em LLMs.",
    bio3: "Formação em <strong>Bioengenharia</strong> (MSc, FEUP), onde me apaixonei por ML aplicado. Fora do trabalho construo coisas pequenas em paralelo, para aprender algo novo. Estas duas são as mais recentes.",
    projectsHeading: "Projetos pessoais",
    scheduler: "Agendador",
    schedulerDesc: "Um agendador semanal baseado em otimização que encaixa os clientes de um treinador no seu calendário. A partir das disponibilidades de cada um, um solver <strong>OR-Tools CP-SAT</strong> procura a melhor combinação de sessões, respeitando os horários individuais e as aulas de grupo. É uma ferramenta real, usada pelo meu PT do ginásio. Mais tarde adicionei um chat em <strong>LangGraph</strong> que transforma texto livre em ações, por isso podes criar utilizadores, definir disponibilidades e gerar um horário apenas descrevendo o que queres em linguagem natural.",
    companion: "Fala com o Hugo",
    companionDesc: "Um chatbot que responde como se fosse eu, com base no meu CV e num resumo pessoal. Construído com o <strong>OpenAI SDK</strong> e servido através de uma interface Gradio, mantém-se na personagem do Hugo e fala sobre o meu percurso, projetos e experiência. Usa tool calling para se manter honesto: quando alguém quer entrar em contacto regista os seus dados, e quando não sabe responder a uma pergunta regista-a, notificando-me em ambos os casos.",
    datadoctor: "Data Doctor",
    datadoctorDesc: "Um assistente de análise clínica construído à volta de um agente <strong>Strands</strong> que lê cada pergunta e a encaminha para a ferramenta certa. Prevê resultados de pacientes com modelos <strong>XGBoost</strong> (uma classe de DPOC e um valor de ALT com intervalo de 80 por cento, pedindo de volta os dados mais relevantes através de SHAP), corre análise e gráficos <strong>pandas</strong> ao vivo sobre um conjunto de 10.000 pacientes, e responde a perguntas fundamentadas com <strong>RAG</strong> híbrido (FAISS denso e BM25 esparso, fundidos e reordenados por um reranker cross-encoder) sobre 1.050 registos clínicos e um corpus de manuais de medicina com 125 mil blocos. Também compara pacientes lado a lado, recorda coortes ao longo da conversa e pode, opcionalmente, pesquisar na web numa allowlist de domínios médicos. Cada interação passa por guardrails de entrada e saída (remoção de PII, bloqueio de injeção de prompts, avisos automáticos) e é rastreada de ponta a ponta no <strong>MLflow</strong>, enquanto um widget de feedback alimenta um ciclo de aprendizagem ativa que retreina e promove um modelo apenas quando este supera a baseline. Funciona com dados sintéticos e não se destina a uso clínico.",
    readmeLink: "Ver o README no GitHub →",
    contactCta: "Se quiseres saber mais sobre mim, contacta-me para {email} e envio-te o meu CV.",
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
