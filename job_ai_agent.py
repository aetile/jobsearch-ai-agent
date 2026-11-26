#!/usr/bin/env python3
"""Orchestrateur agentique piloté par un template unique et criteria.json."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

try:
    from jinja2 import Environment, StrictUndefined, TemplateError
except ImportError:  # pragma: no cover
    Environment = None
    StrictUndefined = None
    TemplateError = Exception

import core as ja

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"


def get_agent_prompt_template_path() -> Path:
    override = os.getenv("JOB_AGENT_PROMPT_TEMPLATE", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (ja.get_base_dir() / "agent_prompt.txt.jinja").resolve()


DEFAULT_AGENT_CFG: dict = {
    "max_steps": 25,
    "max_evals": 5,
    "preview_limit": 25,
    "search_limit": 20,
    "keep_recent_turns": 4,
    "agent_trigger": "GO",
}


def get_agent_cfg(criteria: dict) -> dict:
    merged = dict(DEFAULT_AGENT_CFG)
    for section_name in ("runtime", "agent"):
        sec = criteria.get(section_name)
        if isinstance(sec, dict):
            for k, v in sec.items():
                if k in merged and v is not None:
                    merged[k] = v
    return merged


def build_prompt_criteria_subset(criteria: dict) -> dict:
    """Extrait uniquement les paramètres métier/filtrage utiles au prompt sans les réglages techniques."""
    if not isinstance(criteria, dict):
        return {}
    subset: dict = {}
    prompt_sec = criteria.get("prompt")
    if isinstance(prompt_sec, dict) and prompt_sec:
        subset["prompt"] = {k: v for k, v in prompt_sec.items() if v not in (None, "", [], {})}

    rt = criteria.get("runtime")
    if isinstance(rt, dict):
        for key in ("expired_offer_phrases", "blocked_title_patterns"):
            val = rt.get(key)
            if val:
                subset[key] = val
    return subset


def render_external_template(path: Path, context: dict) -> str:
    if not path.exists():
        raise ja.FatalConfigError(f"Template externe introuvable : {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ja.FatalConfigError(f"Template externe vide : {path}")
    if Environment is not None:
        try:
            env = Environment(undefined=StrictUndefined, autoescape=False)
            return env.from_string(raw).render(**context).strip()
        except TemplateError as exc:
            raise ja.FatalConfigError(f"Erreur de rendu dans {path.name} : {exc}") from exc
    return raw


def resolve_gemini_endpoint_url(criteria: dict) -> str:
    rt = ja.get_runtime_cfg(criteria)
    http_opts = rt.get("http_options")
    if isinstance(http_opts, dict):
        base_url = str(http_opts.get("base_url") or "").strip()
        if base_url:
            return base_url.rstrip("/")
    env_base = os.getenv("GEMINI_BASE_URL", "").strip()
    if env_base:
        return env_base.rstrip("/")
    return DEFAULT_GEMINI_BASE_URL


def build_chat_config(
    agent_instruction: str,
    tools: list,
    criteria: dict,
    force_tool_call: bool = True,
) -> types.GenerateContentConfig:
    rt = ja.get_runtime_cfg(criteria)
    timeout_ms = max(2000, int(rt.get("http_timeout_ms", 20000)))
    endpoint_url = resolve_gemini_endpoint_url(criteria)
    http_options = types.HttpOptions(
        timeout=timeout_ms,
        **({"base_url": endpoint_url} if endpoint_url != DEFAULT_GEMINI_BASE_URL else {}),
    )
    tool_config = None
    if force_tool_call and tools:
        tool_config = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="ANY")
        )
    return types.GenerateContentConfig(
        system_instruction=agent_instruction,
        tools=tools,
        tool_config=tool_config,
        temperature=float(rt.get("agent_temperature", 0.1)),
        http_options=http_options,
    )


def trim_chat_history(history: list, keep_recent_turns: int = 4) -> list:
    """Conserve l'échange initial (init) et les K derniers tours complets sans couper une paire tool call/response."""
    keep_turns = max(1, int(keep_recent_turns))
    max_entries = 2 + keep_turns * 2
    if len(history) <= max_entries:
        return history
    head = history[:2]
    tail = history[-(keep_turns * 2):]
    while tail and getattr(tail[0], "role", None) == "user":
        parts = getattr(tail[0], "parts", None) or []
        if any(getattr(p, "function_response", None) for p in parts):
            tail = tail[1:]
        else:
            break
    return head + tail


def _pace_gemini_call(last_call_ts: float | None, min_interval_s: float, label: str) -> float:
    """Attend le temps nécessaire pour respecter un intervalle minimal entre deux appels Gemini."""
    if last_call_ts is not None and min_interval_s > 0:
        elapsed = time.monotonic() - last_call_ts
        remaining = min_interval_s - elapsed
        if remaining > 0.05:
            ja.log(f"[PACE] Pause {remaining:.1f}s avant {label} (min {min_interval_s:.1f}s entre appels)")
            ja.interruptible_sleep(remaining)
    return time.monotonic()


def send_with_retry(
    chat,
    message,
    criteria: dict,
    label: str = "agent",
    endpoint_url: str | None = None,
    min_interval_s: float = 0.0,
    last_call_ts: float | None = None,
):
    """Envoie un message à la session Gemini avec pacing et backoff sans dépasser max_elapsed_s."""
    url = (endpoint_url or resolve_gemini_endpoint_url(criteria)).rstrip("/")
    max_attempts, base_s, cap_s, max_elapsed_s = ja.resolve_retry_settings(criteria)
    start = time.monotonic()
    last_error = "inconnue"
    current_last_ts = last_call_ts

    for attempt in range(max_attempts):
        if ja.should_stop():
            raise ja.GracefulShutdown("STOP")

        elapsed = time.monotonic() - start
        remaining = max_elapsed_s - elapsed
        if remaining <= 0.0:
            raise ja.OfferEvaluationTimeoutError(f"Timeout ({max_elapsed_s:.1f}s) sur {label} [{url}]")

        if current_last_ts is not None and min_interval_s > 0:
            since_last = time.monotonic() - current_last_ts
            pace_wait = min(min_interval_s - since_last, remaining - 0.1)
            if pace_wait > 0.05:
                ja.log(f"[PACE] Pause {pace_wait:.1f}s avant {label} (essai {attempt + 1}/{max_attempts})")
                ja.interruptible_sleep(pace_wait)

        if ja.should_stop():
            raise ja.GracefulShutdown("STOP")
        remaining = max_elapsed_s - (time.monotonic() - start)
        if remaining <= 0.1:
            raise ja.OfferEvaluationTimeoutError(f"Timeout ({max_elapsed_s:.1f}s) sur {label} [{url}]")

        last_attempt = attempt == max_attempts - 1
        wait = min(cap_s, base_s * (2 ** attempt))

        try:
            current_last_ts = time.monotonic()
            resp = chat.send_message(message)
            return resp, current_last_ts
        except ja.GracefulShutdown:
            raise
        except ClientError as exc:
            msg = str(exc)
            code = ja.status_code(exc)
            if code == 429 or "RESOURCE_EXHAUSTED" in msg:
                if ja.is_daily_quota_error(msg):
                    raise ja.DailyQuotaExceededError(msg) from exc
                delay = ja.parse_retry_delay_s(msg)
                wait = delay + 0.5 if delay is not None else wait
                if last_attempt or wait > max_elapsed_s - (time.monotonic() - start):
                    ja.log(f"[RETRY] 429 RESOURCE_EXHAUSTED sur {label} ({url}) (essai {attempt + 1}/{max_attempts}, abandon)")
                    raise ja.RateLimitError(f"429 persistant sur {label} ({url})") from exc
                last_error = "429 RESOURCE_EXHAUSTED"
            elif code in (400, 401, 403, 404) or "api key" in msg.lower():
                raise ja.FatalConfigError(f"Erreur Gemini {code} ({url}) : {msg[:300]}") from exc
            else:
                raise ja.EvaluationFailedError(f"Requête refusée ({code}) sur {label} ({url}) : {msg[:300]}") from exc
        except ServerError as exc:
            code = ja.status_code(exc)
            last_error = f"serveur {code or '5xx'}"
            if last_attempt:
                ja.log(f"[RETRY] {last_error} sur {label} ({url}) (essai {attempt + 1}/{max_attempts}, abandon)")
                raise ja.TransientEvaluationError(f"Erreur serveur persistante ({last_error}) sur {label} ({url})") from exc
        except Exception as exc:
            if not ja.is_transient_network_error(exc):
                raise ja.EvaluationFailedError(f"Erreur inattendue sur {label} ({url}) : {type(exc).__name__}: {exc}") from exc
            last_error = type(exc).__name__
            if last_attempt:
                ja.log(f"[RETRY] {last_error} sur {label} ({url}) (essai {attempt + 1}/{max_attempts}, abandon)")
                raise ja.TransientEvaluationError(f"Erreur réseau persistante ({last_error}) sur {label} ({url})") from exc

        remaining = max_elapsed_s - (time.monotonic() - start)
        if remaining <= 0.2:
            raise ja.OfferEvaluationTimeoutError(f"Timeout ({max_elapsed_s:.1f}s) sur {label} [{url}]")
        ja.log(
            f"[RETRY] {last_error} sur {label} ({url}) "
            f"(essai {attempt + 1}/{max_attempts}, attente {min(wait, remaining):.1f}s)"
        )
        ja.interruptible_sleep(min(wait, remaining))

    raise ja.EvaluationFailedError(f"Échec inattendu sur {label} ({url})")


def send_with_fallback(
    client,
    chat,
    message,
    criteria: dict,
    chat_config: types.GenerateContentConfig,
    active_model: str,
    fallback_model: str | None,
    label: str = "agent",
    endpoint_url: str | None = None,
    min_interval_s: float = 0.0,
    last_call_ts: float | None = None,
    keep_recent_turns: int = 4,
):
    """Tente send_with_retry sur le modèle actif et bascule sur fallback_model si 503 persiste."""
    url = endpoint_url or resolve_gemini_endpoint_url(criteria)
    try:
        resp, updated_ts = send_with_retry(
            chat,
            message,
            criteria,
            label=f"{label} ({active_model})",
            endpoint_url=url,
            min_interval_s=min_interval_s,
            last_call_ts=last_call_ts,
        )
        return resp, chat, active_model, updated_ts
    except ja.TransientEvaluationError as exc:
        if not fallback_model or fallback_model == active_model:
            raise
        ja.log(f"[FALLBACK] {exc} -> bascule vers '{fallback_model}' ({url})")
        history = trim_chat_history(list(chat.get_history()), keep_recent_turns=keep_recent_turns)
        fallback_chat = client.chats.create(
            model=fallback_model,
            config=chat_config,
            history=history,
        )
        resp, updated_ts = send_with_retry(
            fallback_chat,
            message,
            criteria,
            label=f"{label} ({fallback_model})",
            endpoint_url=url,
            min_interval_s=min_interval_s,
            last_call_ts=last_call_ts,
        )
        return resp, fallback_chat, fallback_model, updated_ts


def _neutralize_tags(text: str) -> str:
    return re.sub(r"</?\s*(?:offre|cv)\s*>", "[balise]", text or "", flags=re.IGNORECASE)


def build_agent_tools(
    args: argparse.Namespace,
    criteria: dict,
    agent_cfg: dict,
    store: ja.SeenDB,
    state: ja.RunState,
    out_dir: Path,
    rules_hash: str,
) -> tuple[list, dict]:
    rt = ja.get_runtime_cfg(criteria)
    recheck_days = int(rt.get("filtered_recheck_days", 14))
    max_evals = int(agent_cfg["max_evals"])
    preview_limit = max(1, int(agent_cfg["preview_limit"]))
    default_search_limit = max(1, int(agent_cfg["search_limit"]))

    sources_path = ja.get_sources_path()
    sources_cfg = (
        json.loads(sources_path.read_text(encoding="utf-8")).get("sources", [])
        if sources_path.exists()
        else []
    )
    registry = ja.discover_collectors()
    pool: dict[str, dict] = {}

    def list_available_sources() -> list[dict]:
        """Liste les sources d'offres configurées et activées."""
        result = []
        for src in sources_cfg:
            if not src.get("enabled", True):
                continue
            if getattr(args, "no_france_travail", False) and src.get("type") == "francetravail":
                continue
            result.append({
                "name": src.get("name") or src.get("type"),
                "type": src.get("type"),
            })
        return result

    def collect_from_source(source_name: str) -> dict:
        """Collecte les offres d'une source configurée et retourne un aperçu des candidats."""
        target = next(
            (
                s for s in sources_cfg
                if s.get("enabled", True)
                and (s.get("name") == source_name or s.get("type") == source_name)
            ),
            None,
        )
        if not target:
            return {"error": f"Source '{source_name}' introuvable ou désactivée."}
        collector_cls = registry.get(target.get("type", ""))
        if collector_cls is None:
            return {"error": f"Type de collecteur inconnu : {target.get('type')}"}

        helpers = ja.default_collector_helpers()
        try:
            jobs = collector_cls(target, rt, helpers).collect()
        except Exception as exc:
            return {"error": f"Échec collecte '{source_name}': {type(exc).__name__}: {exc}"}

        new_items = []
        skipped_seen = 0
        for job in jobs:
            jid = job["id"]
            if not getattr(args, "refilter", False) and store.should_skip(jid, rules_hash, recheck_days):
                skipped_seen += 1
                continue
            pool[jid] = job
            new_items.append({
                "job_id": jid,
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "age_days": job.get("age_days"),
            })

        return {
            "source": source_name,
            "total_collected": len(jobs),
            "already_seen_skipped": skipped_seen,
            "new_candidates_count": len(new_items),
            "candidates_preview": new_items[:preview_limit],
        }

    def search_local_pool(keyword: str = "", source_name: str = "", limit: int | None = None) -> list[dict]:
        """Filtre les offres déjà collectées par mot-clé ou source."""
        eff_limit = max(1, min(int(limit) if limit is not None else default_search_limit, 100))
        kw = ja.normalize_text(keyword)
        src_norm = ja.normalize_text(source_name)
        matches = []
        for jid, job in pool.items():
            if jid in state.evaluations or any(x.get("job_id") == jid for x in state.filtered):
                continue
            if src_norm and src_norm not in ja.normalize_text(job.get("source", "")):
                continue
            haystack = ja.normalize_text(f"{job.get('title', '')} {job.get('location', '')}")
            if kw and kw not in haystack:
                continue
            matches.append({
                "job_id": jid,
                "source": job.get("source"),
                "title": job.get("title"),
                "company": job.get("company"),
                "location": job.get("location"),
                "age_days": job.get("age_days"),
            })
            if len(matches) >= eff_limit:
                break
        return matches

    def check_french_location(raw_location: str) -> dict:
        """Résout un lieu via cache et API géographique configurée."""
        return ja.resolve_location_info(raw_location, store=store)

    def inspect_job_details(job_id: str) -> dict:
        """Retourne les détails complets d'une offre du pool courant."""
        job = pool.get(job_id)
        if not job:
            return {"error": f"Offre '{job_id}' absente du pool courant."}
        max_chars = int(rt.get("max_description_chars", 4000))
        return {
            "job_id": job_id,
            "source": job.get("source"),
            "title": job.get("title"),
            "company": job.get("company"),
            "location": job.get("location"),
            "salary_text": job.get("salary_text", "Non précisé"),
            "age_detail": job.get("age_detail"),
            "url": job.get("url"),
            "description": _neutralize_tags((job.get("description") or "")[:max_chars]),
        }

    def mark_job_irrelevant(job_id: str, reason: str) -> dict:
        """Marque une offre comme filtrée et met à jour les sorties."""
        job = pool.get(job_id)
        if not job:
            return {"error": f"Offre '{job_id}' introuvable."}
        entry = {
            "job_id": job_id,
            "source": job.get("source", ""),
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "location": job.get("location", ""),
            "url": job.get("url", ""),
            "age_days": job.get("age_days"),
            "age_detail": job.get("age_detail"),
            "filter_reason": reason,
            "filtered_at": datetime.now(timezone.utc).isoformat(),
        }
        state.filtered = [x for x in state.filtered if x.get("job_id") != job_id] + [entry]
        store.mark_filtered(job, reason, rules_hash)
        ja.persist_outputs(out_dir, state, criteria, partial=True)
        return {"status": "filtered", "job_id": job_id}

    def save_job_evaluation(
        job_id: str,
        location: str,
        remote_mode: Literal["full_remote", "hybrid", "onsite", "unspecified"],
        location_or_remote_ok: bool,
        match_score: int,
        meets_criteria: bool,
        matching_skills: list[str],
        missing_skills: list[str],
        red_flags: list[str],
        summary: str,
        salary_min_k: int | None = None,
        salary_max_k: int | None = None,
    ) -> dict:
        """Enregistre l'évaluation d'une offre et met à jour le rapport."""
        if job_id in state.evaluations:
            existing = state.evaluations[job_id]
            return {
                "status": "already_evaluated",
                "job_id": job_id,
                "match_score": existing.get("match_score"),
                "evals_remaining": max(0, max_evals - state.evals_this_run),
            }
        if max_evals > 0 and state.evals_this_run >= max_evals:
            return {"error": f"Plafond de {max_evals} évaluations atteint."}
        job = pool.get(job_id)
        if not job:
            return {"error": f"Offre '{job_id}' introuvable."}

        score = max(0, min(100, int(match_score)))
        store.mark_evaluated(job, score, ja.content_fingerprint(job))
        state.evals_this_run += 1
        record = {
            "job_id": job_id,
            "source": job.get("source", ""),
            "title": job.get("title", ""),
            "company": job.get("company", ""),
            "location": job["location"] if not ja.is_unknown(job.get("location")) else location,
            "remote_mode": remote_mode,
            "location_or_remote_ok": bool(location_or_remote_ok),
            "salary_min_k": salary_min_k,
            "salary_max_k": salary_max_k,
            "match_score": score,
            "meets_criteria": bool(meets_criteria),
            "matching_skills": list(matching_skills),
            "missing_skills": list(missing_skills),
            "red_flags": list(red_flags),
            "summary": summary,
            "url": job.get("url", ""),
            "age_days": job.get("age_days"),
            "age_detail": job.get("age_detail"),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        state.evaluations[job_id] = record
        state.filtered = [x for x in state.filtered if x.get("job_id") != job_id]
        ja.persist_outputs(out_dir, state, criteria, partial=True)
        return {
            "status": "saved",
            "job_id": job_id,
            "evals_remaining": max(0, max_evals - state.evals_this_run),
        }

    tools = [
        list_available_sources,
        collect_from_source,
        search_local_pool,
        check_french_location,
        inspect_job_details,
        mark_job_irrelevant,
        save_job_evaluation,
    ]
    return tools, {"pool": pool}


def _should_force_tool_call(args: argparse.Namespace, state: ja.RunState) -> bool:
    if args.max_evals <= 0:
        return state.evals_this_run == 0
    return state.evals_this_run < args.max_evals


def run_agent(argv: list[str] | None = None) -> int:
    load_dotenv()
    ja.reset_stop_state()
    ja.install_signal_handlers()

    criteria = ja.load_criteria(ja.get_criteria_path())
    agent_cfg = get_agent_cfg(criteria)
    rt = ja.get_runtime_cfg(criteria)

    parser = argparse.ArgumentParser(description="Orchestrateur agentique configuré par fichier externe.")
    parser.add_argument("--no-france-travail", action="store_true")
    parser.add_argument("--refilter", action="store_true")
    parser.add_argument("--max-evals", type=int, default=int(agent_cfg["max_evals"]))
    parser.add_argument("--max-steps", type=int, default=int(agent_cfg["max_steps"]))
    args = parser.parse_args(argv)

    agent_cfg["max_evals"] = args.max_evals
    agent_cfg["max_steps"] = args.max_steps
    rt["max_gemini_evaluations_per_run"] = args.max_evals
    keep_recent_turns = max(1, int(agent_cfg.get("keep_recent_turns", 4)))

    if not os.getenv("GEMINI_API_KEY"):
        ja.log("[ERREUR] GEMINI_API_KEY manquante.")
        return 2

    active_model = str(rt.get("model_name", ja.DEFAULT_CRITERIA["runtime"]["model_name"])).strip()
    fallback_model = str(rt.get("fallback_model_name") or "").strip() or None
    endpoint_url = resolve_gemini_endpoint_url(criteria)
    raw_interval = rt.get("min_seconds_between_api_calls", rt.get("min_seconds_between_evaluations", 4.0))
    min_call_interval_s = max(0.0, float(raw_interval))
    last_call_ts: float | None = None

    fallback_info = fallback_model or "aucun"
    ja.log(
        f"[START] Modèle : {active_model} | Fallback : {fallback_info} "
        f"| Pause min : {min_call_interval_s:.1f}s | Historique max : {keep_recent_turns} tours "
        f"| Endpoint : {endpoint_url}"
    )

    out_dir = ja.get_output_dir()
    store = ja.SeenDB(out_dir / "jobs_seen.db")
    state = ja.RunState(
        evaluations={str(x["job_id"]): x for x in ja.load_json_list(out_dir / "offres_evaluees.json") if "job_id" in x},
        filtered=ja.load_json_list(out_dir / "offres_filtrees.json"),
    )
    rules_hash = ja.rules_fingerprint(criteria)
    cv_file = None

    try:
        base_instruction = ja.load_system_instruction(ja.get_prompt_template_path(), criteria, log_rendered=False)
        prompt_criteria = build_prompt_criteria_subset(criteria)
        template_ctx = {
            **ja.DEFAULT_CRITERIA["prompt"],
            **ja.get_prompt_cfg(criteria),
            **agent_cfg,
            "cv_text": "",
            "criteria_json": json.dumps(prompt_criteria, ensure_ascii=False, separators=(",", ":")),
        }
        agent_block = render_external_template(get_agent_prompt_template_path(), template_ctx)
        agent_instruction = f"{base_instruction}\n\n{agent_block}"

        tools, _ = build_agent_tools(args, criteria, agent_cfg, store, state, out_dir, rules_hash)
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        candidate_profile = str(criteria.get("candidate_profile") or "").strip()
        if candidate_profile:
            init_message: str | list = f"Profil du candidat :\n{candidate_profile}\n\n{agent_cfg['agent_trigger']}"
        else:
            cv_path_str = str(criteria.get("cv_summary_path") or criteria.get("cv_path") or "").strip()
            if cv_path_str:
                cv_path = Path(cv_path_str).expanduser()
                if cv_path.exists():
                    cv_file = client.files.upload(file=cv_path)
            init_message = (
                [cv_file, str(agent_cfg["agent_trigger"])]
                if cv_file is not None
                else str(agent_cfg["agent_trigger"])
            )

        force_any = _should_force_tool_call(args, state)
        chat_config = build_chat_config(agent_instruction, tools, criteria, force_tool_call=force_any)
        chat = client.chats.create(model=active_model, config=chat_config)

        last_call_ts = _pace_gemini_call(last_call_ts, min_call_interval_s, "init")
        response, chat, active_model, last_call_ts = send_with_fallback(
            client,
            chat,
            init_message,
            criteria,
            chat_config,
            active_model,
            fallback_model,
            label="init",
            endpoint_url=endpoint_url,
            min_interval_s=min_call_interval_s,
            last_call_ts=last_call_ts,
            keep_recent_turns=keep_recent_turns,
        )

        for step in range(1, args.max_steps + 1):
            if ja.should_stop():
                raise ja.GracefulShutdown("STOP")
            if not response.function_calls:
                ja.log(f"\n{response.text or ''}")
                break

            tool_responses = []
            for fc in response.function_calls:
                ja.log(f"[{step}/{args.max_steps}] {fc.name}({json.dumps(fc.args or {}, ensure_ascii=False)})")
                fn = next((t for t in tools if t.__name__ == fc.name), None)
                if fn is None:
                    out = {"error": f"Outil inconnu : {fc.name}"}
                else:
                    try:
                        out = fn(**(fc.args or {}))
                    except Exception as exc:
                        out = {"error": f"{type(exc).__name__}: {exc}"}
                tool_responses.append(
                    types.Part.from_function_response(name=fc.name, response={"result": out})
                )

            next_force_any = _should_force_tool_call(args, state)
            raw_history = list(chat.get_history())
            trimmed_history = trim_chat_history(raw_history, keep_recent_turns=keep_recent_turns)
            if next_force_any != force_any or len(trimmed_history) < len(raw_history):
                force_any = next_force_any
                chat_config = build_chat_config(
                    agent_instruction,
                    tools,
                    criteria,
                    force_tool_call=force_any,
                )
                chat = client.chats.create(
                    model=active_model,
                    config=chat_config,
                    history=trimmed_history,
                )

            last_call_ts = _pace_gemini_call(last_call_ts, min_call_interval_s, f"step-{step}")
            response, chat, active_model, last_call_ts = send_with_fallback(
                client,
                chat,
                tool_responses,
                criteria,
                chat_config,
                active_model,
                fallback_model,
                label=f"step-{step}",
                endpoint_url=endpoint_url,
                min_interval_s=min_call_interval_s,
                last_call_ts=last_call_ts,
                keep_recent_turns=keep_recent_turns,
            )
        else:
            ja.log(f"[WARN] Plafond d'étapes atteint ({args.max_steps}).")

    except (ja.GracefulShutdown, KeyboardInterrupt):
        state.interrupted = True
    except ja.DailyQuotaExceededError:
        state.quota_exceeded = True
        ja.log("[ERREUR] Quota journalier Gemini atteint.")
    except (ja.TransientEvaluationError, ja.RateLimitError, ja.OfferEvaluationTimeoutError, ja.EvaluationFailedError) as exc:
        state.fatal_error = str(exc)
        ja.log(f"[ERREUR] {exc}")
    except ja.FatalConfigError as exc:
        state.fatal_error = str(exc)
        ja.log(f"[ERREUR] {exc}")
    finally:
        if cv_file is not None and getattr(cv_file, "name", None):
            try:
                client.files.delete(name=cv_file.name)
            except Exception as exc:
                ja.log(f"[WARN] Suppression fichier Gemini impossible ({cv_file.name}): {exc}")
        partial = state.interrupted or state.quota_exceeded or bool(state.fatal_error)
        ja.persist_outputs(out_dir, state, criteria, partial=partial)
        store.close()

    if state.fatal_error:
        return 2
    if state.quota_exceeded:
        return 75
    if state.interrupted:
        return 130
    ja.log(f"[DONE] {state.evals_this_run} évaluée(s) sur cette session (modèle final : {active_model}).")
    return 0


if __name__ == "__main__":
    sys.exit(run_agent())
