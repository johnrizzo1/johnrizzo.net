+++
draft = false
authors = ["John Rizzo"]
title = "Harness AI for Productive Penetration Testing"
date = "2026-06-01"
tags = [ ]
categories = [
  "infosec",
  "machine-learning"
]
series = [ ]
+++

# Harness AI for Productive Penetration Testing

*An offensive-security agent is only as good as the scaffolding around the model. Here's what I had to build to make one actually work — with code and real engagement logs.*

---

Cloudflare recently published a piece about putting a security-tuned frontier model to work hunting vulnerabilities in their own infrastructure. The headline finding wasn't "the model is good" — it was that pointing even a strong model at a target, point-and-shoot, doesn't work. The model is fast and creative, but it drowns you in noise, refuses legitimate work for the wrong reasons, and has no idea what it already tried. What made it useful was a **harness**: a multi-stage pipeline that fed the model the right context, filtered its output, and kept it honest.

I've spent the last few months building exactly that harness, from the other side — not for defensive vulnerability triage, but for offensive engagements: reverse engineering binaries and running web, network, and Active Directory penetration tests end to end. The project is called `reverser`. It wires **91 tools** across binary RE, network pentest, AD, web pentest, and browser automation; it ships **15 specialist profiles** that reshape the model's persona and tool surface per target type; and it runs on Claude or any local model (LM Studio, Ollama, vLLM — anything OpenAI-compatible).

The thesis of this post is the same one Cloudflare landed on, stated from the builder's chair: **the model is a commodity; the orchestration is the product.** Everything below is the evidence — the specific subsystems I had to build, why a raw model needs each one, and what they look like when a real engagement is running.

> ⚠️ Everything here is for authorized security research only. `reverser` drives real offensive tooling against whatever you point it at, and every network-touching tool is gated behind an explicit authorization acknowledgement. The logs quoted below are from intentionally-vulnerable lab targets (Hack The Box machines) and local crackme binaries.

## Why point-and-shoot fails

Give a capable model a shell and a pile of security tools and it will do *something*. It just won't do the right thing for very long. Three failure modes show up immediately, and every one of them maps to a subsystem I had to build:

1. **Amnesia.** The model has no memory across turns, let alone across sessions. It will `nmap` the same host it scanned an hour ago, re-derive facts it already knew, and forget a credential it cracked in turn 12 by turn 40. A multi-day engagement is impossible if the agent's working memory is the context window.

2. **Credulity.** Ask a model whether it found a vulnerability and it is delighted to tell you yes. It will report a "critical SQL injection" it never actually triggered, because narrating a finding is linguistically indistinguishable from confirming one. Unchecked, an autonomous attacker generates a report full of confident fiction.

3. **Flailing.** When an approach doesn't work, the model's instinct is to try it again, slightly differently, forever. Without an explicit discipline that says "this line of attack is dead, abandon it and pivot," the agent burns hours and dollars grinding on a non-vulnerability.

The rest of this post is four subsystems — a persistent knowledge base, a falsifiable-hypothesis engine, profile-based specialization, and an adversarial validation gate — that exist precisely to defeat amnesia, flailing, and credulity.

## The spine: a per-target knowledge base

The single most important component isn't an AI technique at all. It's a database.

Every target gets its own SQLite knowledge base at `targets/<target>/state.db`, and *everything* the agent learns lands there as a structured, schema-validated record: hosts, services, credentials and their per-service test results, findings, artifacts, free-form notes, and hypotheses. The KB — not the context window — is the agent's memory. The first thing a session does is read it; the last thing every turn does is write to it.

Here's what the agent sees when it resumes against a target it has touched before (this is a real `kb_show` result, lightly trimmed):

```
# KB summary — 10.129.11.158

Hosts: 1
Recorded hosts:
  - 10.129.11.158
Services: 0
Credentials: 0 total, 0 valid
Findings: 1
  - critical: 1
Recent notes:
  - [session-start] Initial engagement target: name=Snapped; kind=network; primary_address=10.129.11.158
  - [recon] Nmap: 22/tcp SSH OpenSSH 9.6p1 Ubuntu, 80/tcp HTTP nginx 1.24.0 redirects to http://snapped.htb/
  - [decision] webrecon dispatch #1 failed (error/budget_exhausted).
```

Notice what's in there: not just facts (the open ports), but *decisions* ("webrecon dispatch #1 failed") and a critical finding already on the books. The agent doesn't re-scan, because it can see it already scanned. It doesn't re-dispatch the recon that just failed, because the failure is recorded. This is what kills amnesia — and it's why a `reverser` engagement can survive a process restart, a machine reboot, or a multi-day gap and pick up exactly where it left off.

The schema is opinionated on purpose. Records aren't free text blobs; they're constrained rows. The hypotheses table, the heart of the next section, looks like this:

```sql
CREATE TABLE IF NOT EXISTS hypotheses (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id       INTEGER REFERENCES hypotheses(id) ON DELETE SET NULL,
    target_id       TEXT    NOT NULL REFERENCES targets(id),
    statement       TEXT    NOT NULL,
    rationale       TEXT,
    status          TEXT    NOT NULL DEFAULT 'proposed'
                    CHECK (status IN ('proposed','testing','confirmed','refuted','abandoned','blocked')),
    confidence      INTEGER CHECK (confidence BETWEEN 0 AND 100),
    dispatched_to   TEXT,
    dispatch_count  INTEGER NOT NULL DEFAULT 0,
    evidence_refs   TEXT,
    tags            TEXT,
    created_at      TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at      TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
)
```

That `CHECK (status IN (...))` constraint matters more than it looks. The database itself refuses to store a hypothesis in a state the workflow doesn't recognize. The model can't invent a status; it can't half-confirm something; the data layer enforces the state machine. When you're handing autonomy to a system that will happily improvise, pushing invariants down into SQL constraints is how you stop the improvisation from corrupting your record of the engagement.

## Engagements as falsifiable hypotheses

Here's the idea that makes the whole thing tractable: **an engagement is not a to-do list, it's a tree of falsifiable claims.**

Borrowed straight from the scientific method. Instead of "scan the box, then exploit it," the agent is required to phrase its plan as hypotheses — concrete statements that could be proven wrong — each with a confidence score and a status that moves through the lifecycle the schema enforces: `proposed → testing → confirmed`, or `refuted / abandoned`. Here's the agent opening an engagement against a HTB machine called SmartHire by writing down what it believes (a real `kb_add_hypothesis` call):

```json
{
  "target": "10.129.11.102",
  "statement": "The web application at smarthire.htb has exploitable vulnerabilities (auth bypass, SQLi, file upload, SSTI, etc.) leading to user shell",
  "rationale": "HTTP is the primary attack surface with a custom web app. Most HTB boxes have web-based initial foothold.",
  "confidence": 70,
  "tags": ["web", "initial-access"]
}
```

This is a small thing that changes everything. The agent now has a *theory of the engagement* it can be held accountable to, not just a stream of tool calls. And because each hypothesis is a row in the KB, the orchestrator can see which claims are open, which are being tested, and which are dead.

That last category is what defeats flailing. `reverser` bakes in **K-failure pivot discipline**: each profile carries a failure budget (the manager pivots after K=2, a network pentester after K=3, a web pentester after K=5), and once a hypothesis has eaten K dead-end attempts, the agent is *required* to mark it refuted and propose orthogonal alternatives. Here's that discipline firing for real — the agent refuting its own initial theory against another box after the evidence didn't cooperate:

```json
{
  "id": 1,
  "status": "refuted",
  "rationale": "Static single-page Next.js dashboard with no API routes, no auth, no forms. No exploitable web vulns found in the app itself."
}
```

A point-and-shoot agent would still be fuzzing that static site. This one wrote off its lead hypothesis, recorded *why*, and moved on — which is exactly what a competent human operator does when a promising lead turns out to be a dead end.

## Specialization: profiles and the manager

A single generalist prompt with 91 tools in front of it is a bad operator. It's overwhelmed by choice and mediocre at everything. So `reverser` ships 15 **profiles**, each of which narrows *both* the system prompt and the available tool surface to one job: `webrecon`, `ad`, `exploit`, `pentest`, `webpentest`, and so on. A web-recon specialist sees web-recon tools and a web-recon mindset; it can't wander off into Kerberoasting.

The `manager` profile ties them together. It doesn't attack anything itself — it's an orchestrator. It maintains the hypothesis tree as the engagement plan and dispatches specialists against specific hypotheses with a budget and a scoped sub-goal. Here's the manager on the Helix engagement dispatching a web-recon specialist:

```json
{
  "specialty": "webrecon",
  "target": "10.129.12.124",
  "hypothesis_id": 1,
  "sub_goal": "Enumerate the web application on http://helix.htb (port 80): fingerprint technologies, discover directories/paths, enumerate virtual hosts/subdomains, and identify any login pages, APIs, or interesting endpoints",
  "budget_usd": 0.75,
  "max_turns": 20,
  "rationale": "First dispatch against fresh target; need full web surface enumeration before targeted exploitation"
}
```

The specialist runs as its own sub-agent with its own budget ceiling, does its job, and reports back a structured result that gets reconciled into the shared KB — findings and hypotheses from the specialist are merged into the manager's tree, so the orchestrator's picture of the engagement stays current without the manager having to do the work itself.

This is what scale looks like in practice. That Helix engagement ran for **nearly eleven hours, across 419 turns, for $15.68**, with the manager dispatching specialist after specialist — recon, then targeted web testing — until it chained its way to remote code execution via a vulnerable Apache NiFi instance. No single context window holds eleven hours of work. The KB holds it; the manager steers it; the specialists do it. That division of labor is the harness.

## Keeping an autonomous attacker honest

This is the part I'm proudest of, because it attacks credulity head-on.

Recall the problem: a model asked to confirm a finding will confirm it. So before a hypothesis is allowed to transition to `confirmed`, `reverser` can run an **adversarial second-model validation** — a separate, read-only model whose entire job is to *refute* the claim using only the evidence in the KB. Not to agree. To attack. Its system prompt is unambiguous:

```python
_SYSTEM = (
    "You are a skeptical security reviewer. Your job is to REFUTE the claim below "
    "using ONLY the evidence provided — look for missing links, alternative "
    "explanations, and unproven assumptions. If you genuinely cannot refute it, say "
    "so. Respond with a fenced ```json block: "
    '{"verdict": "refuted|upheld|inconclusive", "reasoning": "<one sentence>"}. '
)
```

The skeptic gets the claim, the serialized evidence, and *no tools at all* — it can't go gather new facts to rescue a weak case, only judge what's actually been proven. It returns a structured verdict:

```python
@dataclass
class Verdict:
    verdict: str  # "refuted" | "upheld" | "inconclusive"
    reasoning: str = ""
    model: Optional[str] = None
    cost: float = 0.0
    turns: int = 0
```

And the gate that consumes it does the one thing that makes this more than theater — a `refuted` verdict **hard-blocks** the transition. The confirmation simply does not happen:

```python
if new_status == "confirmed":
    # ... look up the configured validator ...
    if vbackend:
        evidence_text = _serialize_evidence_for_validation(kb, current, ...)
        verdict = await run_adversary_validation(
            claim=current.statement, evidence_text=evidence_text,
            backend_name=vbackend, model=vmodel, api_base=vapi)
        if verdict is not None and verdict.verdict == "refuted":
            kb.record_note(f"Adversarial validation REFUTED hyp #{args['id']}: {verdict.reasoning}")
            return format_error(
                "Adversarial validation refused the 'confirmed' transition: "
                f"{verdict.reasoning}. Revise the hypothesis/evidence, gather more, "
                "or use status='testing'/'inconclusive'.")
```

The design choices around it are deliberate. The validator is **opt-in** — you point it at a second backend (a cheap local model is fine; the call is capped at three turns and ten cents) — and it **fails open**: if the skeptic is unreachable, the agent records that it confirmed without validation rather than deadlocking the engagement. When the verdict is `upheld` or `inconclusive`, it isn't discarded; it's written into the hypothesis's `evidence_refs`, so the confirmation carries a permanent record of the challenge it survived. You can run the prosecution on Claude and the defense on a local Qwen, or any other split — the point is that *the model proposing a finding is never the only model allowed to bless it.*

A few smaller guardrails round out the same instinct. A **connection circuit-breaker** trips after three consecutive connection failures against a target and forces the agent to yield to the human rather than hammer a host that's gone dark. An optional per-target `scope.toml` declares the rules of engagement — allowed CIDRs, no-DoS, no-account-lockout, allowed hours — and is enforced at the tool boundary, not left to the model's good judgment. And no network tool runs at all without an explicit authorization acknowledgement. Autonomy without these is a liability; with them, it's an operator you can actually leave running.

## Breadth, briefly: it also reverses binaries

Everything above used network targets, but the same machinery drives binary reverse engineering — same KB, same hypothesis discipline, a different tool surface (radare2, Ghidra, GDB, angr, capstone, and friends). Point it at a Windows crackme and the first turn looks like triage a human would do, just faster. Here's the `pe_info` tool reading the headers:

```
Machine: IMAGE_FILE_MACHINE_AMD64
Subsystem: IMAGE_SUBSYSTEM_WINDOWS_CUI
Entry point: 0x4c00
Image base: 0x140000000
Security: ASLR (Dynamic Base), NX (DEP), Control Flow Guard, High Entropy ASLR
Sections (6):
  .text   VA=0x00001000  Size=18432  Entropy=6.48
  .rdata  VA=0x00006000  Size=9728   Entropy=5.27
```

— followed immediately by a string scan that fingerprints the C++ runtime (`bad array new length`, `bad allocation`, the MSVC name-mangling) and tells the agent what kind of binary it's up against before it disassembles a single function. The protections, the entropy per section, the runtime — all of it lands in the KB as artifacts and feeds the same hypothesis tree. Whether the target is a binary or an IP, the loop is identical: observe, hypothesize, test, refute or confirm under challenge, record.

## What this means

The capability frontier — how good the underlying model is at finding bugs — is not the bottleneck for autonomous offensive security. The bottleneck is everything around the model. A frontier model with no memory, no discipline, and no skepticism is a brilliant intern who forgets the engagement every hour, believes their own first guess, and never knows when to quit. The same model inside a harness that gives it durable memory, forces its plans into falsifiable claims, specializes its attention, and refuses to let it confirm a finding it can't defend against a hostile reviewer — that's an operator.

None of the four subsystems here is exotic. A SQLite database, a state machine with a `CHECK` constraint, a set of scoped prompts, and a second model told to play prosecutor. The leverage isn't in any one of them; it's in the fact that they exist at all, sitting between a capable-but-credulous model and a live target. Cloudflare found the same thing pointing a model at their own code. I found it pointing one at the network.

The model is a commodity. The harness is the product.

*`reverser` is for authorized security research and penetration testing only.*
