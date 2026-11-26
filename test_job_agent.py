from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import job_agent as agent


def _base_criteria(**overrides) -> dict:
    criteria = dict(agent.DEFAULT_CRITERIA)
    criteria.update(
        {
            "home_departments": ["95"],
            "hybrid_departments": ["92", "78", "75"],
            "excluded_departments": ["77", "91", "94"],
            "out_of_zone_cities": ["lyon", "marseille", "toulouse"],
            "required_keywords": ["linux", "ansible", "terraform", "rhel", "debian"],
            "dev_stack_terms": ["react", "angular", "typescript", "spring"],
            "app_dev_phrases": ["developpement d'applications", "conception d'api"],
            "infra_terms": ["linux", "ansible", "terraform", "vmware", "prometheus"],
            "blocked_title_patterns": [r"\bwindows\b", r"\bactive directory\b"],
            "dev_title_patterns": [r"\bdeveloppeur\b", r"\bfrontend\b"],
            "infra_title_rescue_patterns": [r"\binfrastructure\b", r"\bsysteme\b"],
            "blocked_text_patterns": [r"portage salarial", r"freelance"],
            "expired_offer_phrases": ["offre expiree", "poste pourvu"],
            "min_salary_k": 65,
            "salary_filter_tolerance_k": 5,
            "max_job_age_days": 30,
        }
    )
    criteria.update(overrides)
    return criteria


def _make_job(**overrides) -> dict:
    job = {
        "id": "TEST-001",
        "source": "test_src",
        "category": "ats_api",
        "title": "Ingénieur Système Linux Senior",
        "company": "Safran",
        "location": "Taverny (95)",
        "salary_text": "68 k€ brut/an",
        "url": "https://example.org/jobs/1",
        "age_days": 2,
        "age_detail": "il y a 2 jours",
        "description": (
            "Administration Linux RHEL et Debian, automatisation Ansible et Terraform, "
            "supervision Prometheus et maintien en conditions opérationnelles."
        ),
    }
    job.update(overrides)
    return job


def test_filter_accepts_valid_linux_job_in_zone():
    now = datetime(2026, 11, 23, tzinfo=timezone.utc)
    ok, reason = agent.passes_deterministic_filter(_make_job(), _base_criteria(), now=now)
    assert ok is True
    assert reason == "OK"


def test_filter_rejects_expired_and_too_old_jobs():
    now = datetime(2026, 11, 23, tzinfo=timezone.utc)
    criteria = _base_criteria(max_job_age_days=15)

    ok_exp, _ = agent.passes_deterministic_filter(
        _make_job(description="Offre expirée - ne plus postuler. Linux Ansible."),
        criteria,
        now=now,
    )
    assert ok_exp is False

    ok_old, reason_old = agent.passes_deterministic_filter(
        _make_job(age_days=20),
        criteria,
        now=now,
    )
    assert ok_old is False
    assert "trop ancienne" in reason_old


def test_filter_rejects_low_salary_with_tolerance():
    criteria = _base_criteria(min_salary_k=65, salary_filter_tolerance_k=5)

    ok_low, reason_low = agent.passes_deterministic_filter(
        _make_job(salary_text="50 k€ - 55 k€ brut/an"),
        criteria,
    )
    assert ok_low is False
    assert "inférieur au seuil" in reason_low

    ok_borderline, _ = agent.passes_deterministic_filter(
        _make_job(salary_text="60 k€ - 62 k€ brut/an"),
        criteria,
    )
    assert ok_borderline is True


def test_filter_geography_and_full_remote():
    criteria = _base_criteria()

    ok_out, reason_out = agent.passes_deterministic_filter(
        _make_job(location="Créteil (94)"),
        criteria,
    )
    assert ok_out is False
    assert "94" in reason_out

    ok_remote, _ = agent.passes_deterministic_filter(
        _make_job(
            location="Lyon (69)",
            description="Poste en full remote sur toute la France. Linux Ansible.",
        ),
        criteria,
    )
    assert ok_remote is True


def test_filter_rejects_app_dev_heavy_and_rescues_infra_title():
    criteria = _base_criteria(dev_signal_threshold=3, infra_signal_max_to_reject=1)

    dev_job = _make_job(
        title="Développeur applicatif",
        description="Développement d'applications en React, Angular et TypeScript avec conception d'API.",
    )
    ok_dev, reason_dev = agent.passes_deterministic_filter(dev_job, criteria)
    assert ok_dev is False
    assert "développement applicatif" in reason_dev

    rescued_job = _make_job(
        title="Ingénieur Infrastructure & Système",
        description="React et TypeScript mentionnés pour l'outillage, mais cœur Linux, Ansible et Terraform.",
    )
    ok_rescued, _ = agent.passes_deterministic_filter(rescued_job, criteria)
    assert ok_rescued is True


def test_prefilter_score_prioritizes_home_dept_and_recent():
    criteria = _base_criteria()
    job_95_recent = _make_job(id="J1", location="Taverny (95)", age_days=1)
    job_78_older = _make_job(id="J2", location="Les Mureaux (78)", age_days=20)
    assert agent.prefilter_score(job_95_recent, criteria) > agent.prefilter_score(job_78_older, criteria)


def test_seen_db_rule_fingerprint_and_recheck_days():
    db = agent.SeenDB(":memory:")
    try:
        c1 = _base_criteria(min_salary_k=65)
        c2 = _base_criteria(min_salary_k=70)
        h1 = agent.rules_fingerprint(c1)
        h2 = agent.rules_fingerprint(c2)
        assert h1 != h2

        job = _make_job(id="JOB-10")
        db.mark_filtered(job, "Salaire trop bas", h1)

        assert db.should_skip("JOB-10", h1, recheck_days=14) is True
        assert db.should_skip("JOB-10", h2, recheck_days=14) is False

        past = datetime(2026, 10, 1, tzinfo=timezone.utc)
        now = datetime(2026, 11, 23, tzinfo=timezone.utc)
        db.conn.execute("UPDATE seen_jobs SET seen_at = ? WHERE job_id = ?", (past.isoformat(), "JOB-10"))
        db.conn.commit()
        assert db.should_skip("JOB-10", h1, recheck_days=14, now=now) is False
    finally:
        db.close()


def test_triage_deduplicates_across_sources():
    db = agent.SeenDB(":memory:")
    state = agent.RunState()
    criteria = _base_criteria()
    rhash = agent.rules_fingerprint(criteria)
    try:
        j_first = _make_job(
            id="SRC1-A",
            source="thales",
            title="Admin Linux",
            company="Thales",
            location="Gennevilliers (92)",
        )
        j_dup = _make_job(
            id="SRC2-B",
            source="francetravail",
            title="Admin Linux",
            company="Thales",
            location="Gennevilliers (92)",
        )

        pending = agent.triage_jobs([j_first, j_dup], db, criteria, rhash, state)
        assert [j["id"] for j in pending] == ["SRC1-A"]
        assert len(state.filtered) == 1
        assert "Doublon" in state.filtered[0]["filter_reason"]
    finally:
        db.close()


def test_evaluate_job_with_gemini_mock(monkeypatch):
    fake_eval = agent.JobEvaluation(
        detected_title="Ingénieur Système Linux Senior",
        detected_company="Safran",
        location="Taverny (95)",
        remote_mode="hybrid",
        location_or_remote_ok=True,
        salary_min_k=66,
        salary_max_k=72,
        match_score=88,
        meets_criteria=True,
        matching_skills=["Linux", "Ansible", "Terraform"],
        missing_skills=[],
        red_flags=[],
        summary="Très forte adéquation sur un périmètre Linux et automatisation en interne.",
    )
    fake_response = SimpleNamespace(parsed=fake_eval, text=fake_eval.model_dump_json())

    class FakeModels:
        def generate_content(self, model, contents, config):
            return fake_response

    fake_client = SimpleNamespace(models=FakeModels())
    monkeypatch.setattr(agent, "_build_generation_config", lambda sys_inst, timeout: None)

    result = agent.evaluate_job_with_gemini(
        fake_client,
        cv_text="20+ ans Linux, Ansible, Terraform.",
        job=_make_job(),
        system_instruction="Sys",
        criteria=_base_criteria(),
        sleep_fn=lambda _s: None,
    )
    assert result.match_score == 88
    assert result.meets_criteria is True
    assert result.salary_max_k == 72


def test_cli_dry_run_with_custom_collector(tmp_path: Path, monkeypatch):
    criteria_file = tmp_path / "criteria.json"
    criteria_file.write_text(json.dumps(_base_criteria()), encoding="utf-8")
    sources_file = tmp_path / "sources.json"
    sources_file.write_text(
        json.dumps({"sources": [{"name": "fake_src", "type": "fake", "enabled": True}]}),
        encoding="utf-8",
    )

    class FakeCollector:
        def __init__(self, source_cfg, criteria, helpers):
            self.source_cfg = source_cfg

        def collect(self, now=None):
            return [
                _make_job(id="FAKE-1", source="fake_src", title="Ingénieur Linux Ansible", location="Taverny (95)"),
                _make_job(
                    id="FAKE-2",
                    source="fake_src",
                    title="Ingénieur Linux",
                    location="Lyon (69)",
                    description="Lyon sans remote. Linux.",
                ),
            ]

    monkeypatch.setattr(agent, "CRITERIA_PATH", criteria_file)
    monkeypatch.setattr(agent, "SOURCES_PATH", sources_file)
    monkeypatch.setattr(agent, "DEFAULT_OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr(agent, "discover_collectors", lambda: {"fake": FakeCollector})

    exit_code = agent.main(["--dry-run"])
    assert exit_code == 0

    filtered_data = json.loads((tmp_path / "output" / "offres_filtrees.json").read_text(encoding="utf-8"))
    assert any(item["job_id"] == "FAKE-2" for item in filtered_data)
