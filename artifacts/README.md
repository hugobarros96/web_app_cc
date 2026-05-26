# artifacts/

This directory holds runtime assets that **aren't checked into the repo**
because they're personal: the CV PDF used to ground the chatbot, the
profile photo on the landing page, and demo videos of each project.
Anyone cloning the public repo will see this README and an otherwise
empty folder — that's intentional.

The whole directory is gitignored (only this README is tracked) and
bind-mounted into the container at runtime, so the files live on the
host machine, not in the docker image and not on GitHub.

Non-private landing-page config (the contact links shown in the header)
lives at [portfolio/links.yaml](../portfolio/links.yaml) instead, since
there's no reason to hide it.

## What should be here

Landing-page files, directly at the root:

- `profile_hugo.jpg` — profile photo for the landing hero
- `scheduling_video_example.mp4` — demo loop for the Scheduler card
- `cvcompanion_example.mp4` — demo loop for the MyCompanionCV card

Per-project assets under `artifacts/<project>/`:

- `mycompanioncv/Curriculum_Vitae_Hugo.pdf` — CV the chatbot grounds itself in
- `mycompanioncv/summary.txt` — free-form personal blurb
- `mycompanioncv/system.txt` — base system prompt

Future projects follow the same convention: `artifacts/<project-name>/`.

## How it's wired

- `portfolio/app.py` resolves `ARTIFACTS_DIR` to this directory and serves
  files under `/artifacts/<name>`.
- `projects/mycompanioncv/app.py` resolves `ME_DIR` to
  `artifacts/mycompanioncv/`.
- `docker-compose.yml` (dev) and `docker-compose.prod.yml` (prod) both
  bind-mount `./artifacts` into the container at `/app/artifacts`.

## Deployment note

Because the contents are gitignored, `git pull` on the VM won't populate
this directory. Upload the files separately (`scp` / `rsync`) before
running `docker compose up`.
