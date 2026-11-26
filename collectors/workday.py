from __future__ import annotations

from datetime import datetime
import httpx
from . import BaseCollector


class WorkdayCollector(BaseCollector):
    source_type = "workday"

    def collect(self, now: datetime | None = None) -> list[dict]:
        endpoint = self.cfg["endpoint"]  # ex: https://arianegroup.wd3.myworkdayjobs.com/wday/cxs/arianegroup/External/jobs
        base_web = self.cfg["web_base_url"].rstrip("/")
        company = self.cfg["company"]
        source_name = self.cfg["name"]
        keywords = self.cfg.get("keywords", ["Linux"])
        header_chars = int(self.criteria.get("header_chars", 800))
        jobs: dict[str, dict] = {}

        with httpx.Client(timeout=20.0, headers={"Accept": "application/json"}) as http:
            for kw in keywords:
                if self.helpers["should_stop"]():
                    break
                payload = {"appliedFacets": self.cfg.get("facets", {}), "limit": 20, "offset": 0, "searchText": kw}
                resp = http.post(endpoint, json=payload)
                resp.raise_for_status()

                for posting in resp.json().get("jobPostings", []):
                    ext_path = posting.get("externalPath", "")
                    offer_url = f"{base_web}{ext_path}"
                    job_id = self.helpers["make_job_id"](source_name, offer_url)
                    if job_id in jobs:
                        continue

                    # Récupération du détail JSON de l'offre (sans navigateur)
                    detail_url = endpoint.replace("/jobs", ext_path)
                    detail_resp = http.get(detail_url)
                    if detail_resp.status_code != 200:
                        continue
                    info = detail_resp.json().get("jobPostingInfo", {})
                    desc = self.helpers["html_to_text"](info.get("jobDescription", ""))
                    published = info.get("startDate")
                    age_days, age_detail = self.helpers["estimate_job_age_days"](
                        desc, published_at=published, reference_now=now, header_chars=header_chars
                    )

                    jobs[job_id] = {
                        "id": job_id,
                        "source": source_name,
                        "category": "ats_api",
                        "title": info.get("title") or posting.get("title") or "",
                        "company": company,
                        "location": info.get("location") or posting.get("locationsText") or "Non précisé",
                        "salary_text": "Non précisé",
                        "url": offer_url,
                        "age_days": age_days,
                        "age_detail": age_detail,
                        "published_at": published,
                        "contract_type": info.get("timeType"),
                        "description": desc,
                    }
                    self.helpers["sleep"](0.4)

        return list(jobs.values())
