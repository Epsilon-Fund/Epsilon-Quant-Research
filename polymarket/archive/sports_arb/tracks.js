// ============================================================
// TRACKS DATA — Edit this file to add/modify nodes
// ============================================================
//
// HOW TO ADD A NEW NODE:
// 1. Find the track you want (a, b, c, or d)
// 2. Inside its "nodes" array, copy-paste this template:
//
//      {
//        icon: "🆕",
//        title: "Your Node Title",
//        description: "What the person needs to do.",
//        done_when: "How they know they're done.",
//        ai_prompt: `The prompt they paste into ChatGPT/Claude.
//
// Use backticks for multi-line prompts.`
//      },
//
// 3. To make a node locked (unlocks when all previous are done),
//    add:  locked: true,  after the icon line.
//
// CAREFUL: every node except the LAST one in a track needs a comma
// after its closing }
// ============================================================

const TRACKS = [
  {
    id: "a",
    name: "Polymarket Explorer",
    emoji: "🔮",
    desc: "Understand Polymarket and build the data pipeline",
    nodes: [
      {
        icon: "🚀",
        title: "Onboarding: Get Set Up",
        description: "Before anything else, you need to get set up with our tools and understand the project. This means: (1) send Carlos your GitHub email so he can add you to the repo, (2) learn the basics of GitHub so you can navigate our shared workspace, (3) install Claude desktop and set up Cowork mode — this is how you'll use the AI prompts on the dashboard, (4) understand what this project is about and how the team works.",
        done_when: "You've sent Carlos your email, he's added you to the repo, you can open the sports-arb folder on GitHub and see the project files, you have the Claude desktop app installed with Cowork mode enabled, and you can explain in one sentence what our project is trying to do.",
        ai_prompt: `I'm joining a small research team called Epsilon. We're investigating whether we can make money by comparing sports odds between two types of platforms:

- Polymarket (polymarket.com) — an international prediction market where people trade on the outcome of events (sports, politics, etc.). Prices reflect crowd-estimated probabilities.
- Spanish bookmakers (Codere, Bet365.es, Sportium, etc.) — traditional sports betting sites licensed in Spain.

The idea: if Polymarket says a team has a 60% chance of winning, but a Spanish bookmaker's odds imply only 50%, there might be a profitable opportunity (called "arbitrage" or "+EV betting"). We're in the research phase — we don't know yet if this works.

Our team uses GitHub to collaborate. All our work lives in a folder called "sports-arb" inside a repository.

I need you to teach me:

1. WHAT IS GITHUB?
- What is a "repository" (repo)? Think of it as a shared folder in the cloud with version history.
- How do I access it? (I'll get an invite link from Carlos, our team lead)
- How do I navigate the repo in my browser? (I don't need to install anything yet)
- What is a "commit"? What does it mean when someone "pushes" changes?
- How do I view files, see who changed what, and read the history?

2. OUR PROJECT STRUCTURE
- Our repo is called Epsilon-Quant-Research
- Inside it, the folder polymarket/sports-arb/ is where our project lives
- Right now it contains: index.html (our team dashboard) and tracks.js (the task data for the dashboard)
- Over time we'll add scripts, data files, and documents here

3. WHAT I NEED TO DO RIGHT NOW
- Create a GitHub account if I don't have one (github.com)
- Send my GitHub email/username to Carlos so he can add me as a collaborator
- Once added, open the repo and navigate to polymarket/sports-arb/
- Look at the files there — can I see index.html and tracks.js?

4. WHAT IS EPSILON AND WHAT ARE WE DOING?
- Epsilon is a small quantitative research group
- "Quantitative" means we use data and math to find opportunities, rather than gut feeling
- This specific project (sports-arb) is exploring whether the gap between prediction market odds and traditional bookmaker odds is big enough to profit from
- There are 4 tracks of work and 4 team members — each person owns a track but we all need to understand the basics
- We use AI tools (ChatGPT, Claude) to help us learn and build things faster — that's why each task has a prompt you can copy-paste

5. SETTING UP CLAUDE (OUR AI TOOL)
- We use Claude by Anthropic (claude.ai) as our AI assistant — it's like ChatGPT but we prefer it for this kind of work
- Download the Claude desktop app from claude.ai/download
- Once installed, we use a feature called "Cowork mode" — it lets Claude read files, run code, and help you build things directly on your computer
- To enable Cowork: open the Claude desktop app, look for the Cowork toggle (it's next to the chat input), and turn it on
- When Cowork is on, you can connect a folder from your computer so Claude can read and edit files in it — this is how we'll work on project files together
- Each task on the dashboard has an AI prompt you can copy-paste into Claude. In Cowork mode, Claude can do much more than just answer questions — it can create files, run scripts, browse the web, and help you build things step by step
- Pro tip: you can also connect the GitHub repo folder to Cowork so Claude can see and edit your project files directly

Explain everything assuming I'm smart but have never used GitHub or heard of quantitative research. Use analogies I'd understand as a sports fan.

At the end, quiz me: (1) What is a GitHub repo? (2) Where is our project folder? (3) In one sentence, what is our project trying to find out? (4) What is Cowork mode in Claude?`
      }
    ]
  },
  {
    id: "b",
    name: "Sportsbook Scout",
    emoji: "🏟️",
    desc: "Map Spanish bookmakers and build their data pipeline",
    nodes: [
      {
        icon: "🚀",
        title: "Onboarding: Get Set Up",
        description: "Before anything else, you need to get set up with our tools and understand the project. This means: (1) send Carlos your GitHub email so he can add you to the repo, (2) learn the basics of GitHub so you can navigate our shared workspace, (3) install Claude desktop and set up Cowork mode — this is how you'll use the AI prompts on the dashboard, (4) understand what this project is about and how the team works.",
        done_when: "You've sent Carlos your email, he's added you to the repo, you can open the sports-arb folder on GitHub and see the project files, you have the Claude desktop app installed with Cowork mode enabled, and you can explain in one sentence what our project is trying to do.",
        ai_prompt: `I'm joining a small research team called Epsilon. We're investigating whether we can make money by comparing sports odds between two types of platforms:

- Polymarket (polymarket.com) — an international prediction market where people trade on the outcome of events (sports, politics, etc.). Prices reflect crowd-estimated probabilities.
- Spanish bookmakers (Codere, Bet365.es, Sportium, etc.) — traditional sports betting sites licensed in Spain.

The idea: if Polymarket says a team has a 60% chance of winning, but a Spanish bookmaker's odds imply only 50%, there might be a profitable opportunity (called "arbitrage" or "+EV betting"). We're in the research phase — we don't know yet if this works.

Our team uses GitHub to collaborate. All our work lives in a folder called "sports-arb" inside a repository.

I need you to teach me:

1. WHAT IS GITHUB?
- What is a "repository" (repo)? Think of it as a shared folder in the cloud with version history.
- How do I access it? (I'll get an invite link from Carlos, our team lead)
- How do I navigate the repo in my browser? (I don't need to install anything yet)
- What is a "commit"? What does it mean when someone "pushes" changes?
- How do I view files, see who changed what, and read the history?

2. OUR PROJECT STRUCTURE
- Our repo is called Epsilon-Quant-Research
- Inside it, the folder polymarket/sports-arb/ is where our project lives
- Right now it contains: index.html (our team dashboard) and tracks.js (the task data for the dashboard)
- Over time we'll add scripts, data files, and documents here

3. WHAT I NEED TO DO RIGHT NOW
- Create a GitHub account if I don't have one (github.com)
- Send my GitHub email/username to Carlos so he can add me as a collaborator
- Once added, open the repo and navigate to polymarket/sports-arb/
- Look at the files there — can I see index.html and tracks.js?

4. WHAT IS EPSILON AND WHAT ARE WE DOING?
- Epsilon is a small quantitative research group
- "Quantitative" means we use data and math to find opportunities, rather than gut feeling
- This specific project (sports-arb) is exploring whether the gap between prediction market odds and traditional bookmaker odds is big enough to profit from
- There are 4 tracks of work and 4 team members — each person owns a track but we all need to understand the basics
- We use AI tools (ChatGPT, Claude) to help us learn and build things faster — that's why each task has a prompt you can copy-paste

5. SETTING UP CLAUDE (OUR AI TOOL)
- We use Claude by Anthropic (claude.ai) as our AI assistant — it's like ChatGPT but we prefer it for this kind of work
- Download the Claude desktop app from claude.ai/download
- Once installed, we use a feature called "Cowork mode" — it lets Claude read files, run code, and help you build things directly on your computer
- To enable Cowork: open the Claude desktop app, look for the Cowork toggle (it's next to the chat input), and turn it on
- When Cowork is on, you can connect a folder from your computer so Claude can read and edit files in it — this is how we'll work on project files together
- Each task on the dashboard has an AI prompt you can copy-paste into Claude. In Cowork mode, Claude can do much more than just answer questions — it can create files, run scripts, browse the web, and help you build things step by step
- Pro tip: you can also connect the GitHub repo folder to Cowork so Claude can see and edit your project files directly

Explain everything assuming I'm smart but have never used GitHub or heard of quantitative research. Use analogies I'd understand as a sports fan.

At the end, quiz me: (1) What is a GitHub repo? (2) Where is our project folder? (3) In one sentence, what is our project trying to find out? (4) What is Cowork mode in Claude?`
      }
    ]
  },
  {
    id: "c",
    name: "The Analyst",
    emoji: "📈",
    desc: "Compare the data and find if opportunities exist",
    nodes: [
      {
        icon: "🚀",
        title: "Onboarding: Get Set Up",
        description: "Before anything else, you need to get set up with our tools and understand the project. This means: (1) send Carlos your GitHub email so he can add you to the repo, (2) learn the basics of GitHub so you can navigate our shared workspace, (3) install Claude desktop and set up Cowork mode — this is how you'll use the AI prompts on the dashboard, (4) understand what this project is about and how the team works.",
        done_when: "You've sent Carlos your email, he's added you to the repo, you can open the sports-arb folder on GitHub and see the project files, you have the Claude desktop app installed with Cowork mode enabled, and you can explain in one sentence what our project is trying to do.",
        ai_prompt: `I'm joining a small research team called Epsilon. We're investigating whether we can make money by comparing sports odds between two types of platforms:

- Polymarket (polymarket.com) — an international prediction market where people trade on the outcome of events (sports, politics, etc.). Prices reflect crowd-estimated probabilities.
- Spanish bookmakers (Codere, Bet365.es, Sportium, etc.) — traditional sports betting sites licensed in Spain.

The idea: if Polymarket says a team has a 60% chance of winning, but a Spanish bookmaker's odds imply only 50%, there might be a profitable opportunity (called "arbitrage" or "+EV betting"). We're in the research phase — we don't know yet if this works.

Our team uses GitHub to collaborate. All our work lives in a folder called "sports-arb" inside a repository.

I need you to teach me:

1. WHAT IS GITHUB?
- What is a "repository" (repo)? Think of it as a shared folder in the cloud with version history.
- How do I access it? (I'll get an invite link from Carlos, our team lead)
- How do I navigate the repo in my browser? (I don't need to install anything yet)
- What is a "commit"? What does it mean when someone "pushes" changes?
- How do I view files, see who changed what, and read the history?

2. OUR PROJECT STRUCTURE
- Our repo is called Epsilon-Quant-Research
- Inside it, the folder polymarket/sports-arb/ is where our project lives
- Right now it contains: index.html (our team dashboard) and tracks.js (the task data for the dashboard)
- Over time we'll add scripts, data files, and documents here

3. WHAT I NEED TO DO RIGHT NOW
- Create a GitHub account if I don't have one (github.com)
- Send my GitHub email/username to Carlos so he can add me as a collaborator
- Once added, open the repo and navigate to polymarket/sports-arb/
- Look at the files there — can I see index.html and tracks.js?

4. WHAT IS EPSILON AND WHAT ARE WE DOING?
- Epsilon is a small quantitative research group
- "Quantitative" means we use data and math to find opportunities, rather than gut feeling
- This specific project (sports-arb) is exploring whether the gap between prediction market odds and traditional bookmaker odds is big enough to profit from
- There are 4 tracks of work and 4 team members — each person owns a track but we all need to understand the basics
- We use AI tools (ChatGPT, Claude) to help us learn and build things faster — that's why each task has a prompt you can copy-paste

5. SETTING UP CLAUDE (OUR AI TOOL)
- We use Claude by Anthropic (claude.ai) as our AI assistant — it's like ChatGPT but we prefer it for this kind of work
- Download the Claude desktop app from claude.ai/download
- Once installed, we use a feature called "Cowork mode" — it lets Claude read files, run code, and help you build things directly on your computer
- To enable Cowork: open the Claude desktop app, look for the Cowork toggle (it's next to the chat input), and turn it on
- When Cowork is on, you can connect a folder from your computer so Claude can read and edit files in it — this is how we'll work on project files together
- Each task on the dashboard has an AI prompt you can copy-paste into Claude. In Cowork mode, Claude can do much more than just answer questions — it can create files, run scripts, browse the web, and help you build things step by step
- Pro tip: you can also connect the GitHub repo folder to Cowork so Claude can see and edit your project files directly

Explain everything assuming I'm smart but have never used GitHub or heard of quantitative research. Use analogies I'd understand as a sports fan.

At the end, quiz me: (1) What is a GitHub repo? (2) Where is our project folder? (3) In one sentence, what is our project trying to find out? (4) What is Cowork mode in Claude?`
      }
    ]
  },
  {
    id: "d",
    name: "The Operator",
    emoji: "⚡",
    desc: "Figure out how to actually execute if the numbers work",
    nodes: [
      {
        icon: "🚀",
        title: "Onboarding: Get Set Up",
        description: "Before anything else, you need to get set up with our tools and understand the project. This means: (1) send Carlos your GitHub email so he can add you to the repo, (2) learn the basics of GitHub so you can navigate our shared workspace, (3) install Claude desktop and set up Cowork mode — this is how you'll use the AI prompts on the dashboard, (4) understand what this project is about and how the team works.",
        done_when: "You've sent Carlos your email, he's added you to the repo, you can open the sports-arb folder on GitHub and see the project files, you have the Claude desktop app installed with Cowork mode enabled, and you can explain in one sentence what our project is trying to do.",
        ai_prompt: `I'm joining a small research team called Epsilon. We're investigating whether we can make money by comparing sports odds between two types of platforms:

- Polymarket (polymarket.com) — an international prediction market where people trade on the outcome of events (sports, politics, etc.). Prices reflect crowd-estimated probabilities.
- Spanish bookmakers (Codere, Bet365.es, Sportium, etc.) — traditional sports betting sites licensed in Spain.

The idea: if Polymarket says a team has a 60% chance of winning, but a Spanish bookmaker's odds imply only 50%, there might be a profitable opportunity (called "arbitrage" or "+EV betting"). We're in the research phase — we don't know yet if this works.

Our team uses GitHub to collaborate. All our work lives in a folder called "sports-arb" inside a repository.

I need you to teach me:

1. WHAT IS GITHUB?
- What is a "repository" (repo)? Think of it as a shared folder in the cloud with version history.
- How do I access it? (I'll get an invite link from Carlos, our team lead)
- How do I navigate the repo in my browser? (I don't need to install anything yet)
- What is a "commit"? What does it mean when someone "pushes" changes?
- How do I view files, see who changed what, and read the history?

2. OUR PROJECT STRUCTURE
- Our repo is called Epsilon-Quant-Research
- Inside it, the folder polymarket/sports-arb/ is where our project lives
- Right now it contains: index.html (our team dashboard) and tracks.js (the task data for the dashboard)
- Over time we'll add scripts, data files, and documents here

3. WHAT I NEED TO DO RIGHT NOW
- Create a GitHub account if I don't have one (github.com)
- Send my GitHub email/username to Carlos so he can add me as a collaborator
- Once added, open the repo and navigate to polymarket/sports-arb/
- Look at the files there — can I see index.html and tracks.js?

4. WHAT IS EPSILON AND WHAT ARE WE DOING?
- Epsilon is a small quantitative research group
- "Quantitative" means we use data and math to find opportunities, rather than gut feeling
- This specific project (sports-arb) is exploring whether the gap between prediction market odds and traditional bookmaker odds is big enough to profit from
- There are 4 tracks of work and 4 team members — each person owns a track but we all need to understand the basics
- We use AI tools (ChatGPT, Claude) to help us learn and build things faster — that's why each task has a prompt you can copy-paste

5. SETTING UP CLAUDE (OUR AI TOOL)
- We use Claude by Anthropic (claude.ai) as our AI assistant — it's like ChatGPT but we prefer it for this kind of work
- Download the Claude desktop app from claude.ai/download
- Once installed, we use a feature called "Cowork mode" — it lets Claude read files, run code, and help you build things directly on your computer
- To enable Cowork: open the Claude desktop app, look for the Cowork toggle (it's next to the chat input), and turn it on
- When Cowork is on, you can connect a folder from your computer so Claude can read and edit files in it — this is how we'll work on project files together
- Each task on the dashboard has an AI prompt you can copy-paste into Claude. In Cowork mode, Claude can do much more than just answer questions — it can create files, run scripts, browse the web, and help you build things step by step
- Pro tip: you can also connect the GitHub repo folder to Cowork so Claude can see and edit your project files directly

Explain everything assuming I'm smart but have never used GitHub or heard of quantitative research. Use analogies I'd understand as a sports fan.

At the end, quiz me: (1) What is a GitHub repo? (2) Where is our project folder? (3) In one sentence, what is our project trying to find out? (4) What is Cowork mode in Claude?`
      }
    ]
  }
];
