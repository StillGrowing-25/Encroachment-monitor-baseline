# A Satellite, Deep-Learning, and GIS Framework for Automated Government-Land Encroachment Monitoring

A complete, working implementation of the research paper's framework: continuous
satellite monitoring of public land that couples **land-cover classification**,
**bi-temporal change detection**, and a **GIS protected-parcel overlay** — the
combination that actually defines an encroachment.
---

## Quick start

```bash
pip install -r requirements.txt
python run_all.py          
python -m pytest tests/   
python dashboard/app.py   

Dashboard demo logins (role-based access control):

| User     | Password      | Role             | Can do |
|----------|---------------|------------------|--------|
| officer  | officer123    | field_officer    | view, acknowledge, confirm, dismiss |
| official | official123   | senior_official  | + escalate |
| admin    | admin123      | admin            | + audit log |

## Dashboard (application layer)

- **Map** — protected parcels (green) and detections (red) on a Leaflet map;
  works offline except basemap tiles.
- **Alerts** — tiered protocol: NEW → acknowledge → confirm/dismiss;
  unacknowledged alerts escalate automatically past the 72 h SLA (the sweep
  runs on queue inspection; `store.run_escalation` is the worker hook).
- **Report** — KPI summary + CSV export.
- **Audit** — immutable log of every access/acknowledgement (admin only).

## Reproducing

Everything is seeded. `python run_all.py --seed 42` (default) regenerates
identical numbers; `--skip-dashboard-seed` leaves `alerts.db` untouched.
