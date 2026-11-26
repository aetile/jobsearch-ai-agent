# 1. Se placer dans le dossier du projet
cd ~/projets/job-agent   # adaptez selon votre chemin

# 2. Créer l'environnement virtuel (si besoin : sudo apt install python3-venv)
python3 -m venv .venv

# 3. Activer l'environnement virtuel
source .venv/bin/activate

# 4. Installer les dépendances dans le venv (sans sudo)
#pip install google-genai pydantic httpx beautifulsoup4 markdownify python-dotenv playwright
.venv/bin/pip install -r requirements.txt
.venv/bin/pytest -q

# 5. Installer Chromium pour Playwright
playwright install chromium

# 6. Lancer le script
#python job_agent.py
#.venv/bin/python job_agent.py
