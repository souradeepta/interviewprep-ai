#!/usr/bin/env python3
"""
Add ## Interview Q&A sections to 16 agentic-ai concept markdown files.
Inserts before the first occurrence of ## Best Practices, ## Code Examples,
or ## Related Concepts.
"""

import os
import re

BASE = "/home/sbisw/github/interviewprep-ml/agentic-ai/concepts"

# Map filename -> Q&A block (6 Q&As each)
QA_BLOCKS = {

"37-human-agent-collaboration.md": """\
## Interview Q&A

**Q: When should an agent escalate to a human vs. attempt to handle a task autonomously?**
A: Escalate when the agent's confidence is below a threshold (e.g., <70% certainty), when actions are irreversible (deleting data, sending communications), when the task involves novel situations outside training distribution, or when explicit policy requires human review. Design escalation as a first-class workflow: maintain task state for handoff, provide the human with full context and the agent's reasoning, and define clear SLAs for human response.

**Q: How do you design a handoff protocol between agents and humans to prevent information loss?**
A: The handoff packet should include: current task state, what has been tried and why it failed/was insufficient, the specific question requiring human judgment, suggested options with trade-offs, and deadline/urgency. Store this in a persistent format (structured JSON) not just conversation text. Design the UI to present this context efficiently—humans receiving handoffs should be able to understand the situation in under 30 seconds.

**Q: What metrics indicate healthy human-agent collaboration vs. over-reliance on human review?**
A: Healthy: escalation rate decreasing over time (agent improving), humans accepting agent suggestions >80% of the time (good calibration), handoff-to-resolution time <5 minutes (efficient humans). Over-reliance: flat/increasing escalation rate, humans rubber-stamping without reviewing (approval rate near 100%), humans overriding agent decisions without logging reasons. Track human override reasons—they identify agent failure modes that need addressing.

**Q: How do you handle situations where humans and agents disagree?**
A: Design a clear override mechanism: humans can always override agent decisions, but overrides should be logged with reasoning. For systematic disagreements (human overrides 30%+ of agent decisions in a category), retrain or update the agent's decision rules. Implement feedback loops: when a human overrides, the agent should learn from the correction. Never silently ignore human feedback—it's your highest-quality training signal.

**Q: What are the privacy and accountability implications of human-agent workflows?**
A: Humans reviewing agent work must have appropriate data access permissions—don't expose PII to reviewers who don't need it. Maintain audit logs of both agent decisions and human overrides for accountability. In regulated industries (finance, healthcare), the human reviewer bears legal responsibility for approved agent actions—ensure they understand this and have sufficient information to make real decisions (not just rubber-stamp). GDPR/HIPAA may require human decision-making for certain determinations.

**Q: How do you prevent alert fatigue in human-agent collaboration systems?**
A: Alert fatigue occurs when agents escalate too frequently or for trivial issues. Mitigate with: confidence-based filtering (only escalate when confidence is low), batching related escalations, smart scheduling (send non-urgent escalations in daily digests not immediately), and adaptive thresholds (escalate less for task types where humans consistently approve the agent's choice). Track escalation acceptance rate per category—categories with >95% acceptance are candidates for reducing escalation frequency.

""",

"38-coding-agents.md": """\
## Interview Q&A

**Q: What are the most important safety constraints for a coding agent running in production?**
A: Sandbox execution: run all generated code in isolated environments (containers, VMs) with no network access, limited file system access, and resource limits. Review before execution: for irreversible operations (database writes, file deletions, API calls with side effects), require explicit confirmation. Allowlist/blocklist: define which commands and libraries the agent can use. Rate limiting: prevent infinite loops or resource exhaustion. Never run generated code with production credentials until it has been reviewed.

**Q: How do you evaluate a coding agent's output quality automatically?**
A: Use test execution: run the generated code against a test suite and measure pass rate. For new functionality, generate tests first (TDD approach) and measure whether the code passes them. Static analysis: run linters, type checkers, and security scanners on generated code. Functional correctness: compare output of generated code against expected output on benchmark inputs. Track regression rate: does new code break existing tests?

**Q: What is the best approach to handle long coding tasks that span multiple files?**
A: Use a hierarchical planning approach: first create a high-level plan (which files to modify, what interfaces to create), then execute changes file by file. Maintain a context window that includes the current file plus relevant related files (interfaces, types it uses). After each file change, run the test suite to catch immediate regressions. Use a dependency graph to determine which files depend on changed files and proactively check them. For very long tasks, checkpoint progress and allow resumption.

**Q: How do you design a coding agent that can learn from code review feedback?**
A: Collect code review comments with accept/reject decisions and the reason. Fine-tune the agent on accepted vs. rejected changes as preference pairs (DPO). Categorize common review patterns: missing error handling, insufficient tests, naming inconsistencies—add these as explicit constraints in the system prompt or retrieval knowledge base. Track code review acceptance rate over time as the primary quality metric.

**Q: When should a coding agent refuse to implement a requested feature?**
A: Refuse when: the request requires introducing known security vulnerabilities (SQL injection patterns, unvalidated inputs), bypassing existing safety checks, implementing features that violate compliance requirements, or when the scope is unclear enough that implementation would require making unverifiable assumptions. The agent should explain why it's refusing and what information or clarification would allow it to proceed.

**Q: How do you handle tool use in a coding agent (file reads, code execution, web search)?**
A: Design tool use as atomic operations with clear inputs/outputs. For file operations: read before write to avoid clobber. For code execution: capture stdout/stderr, detect hangs (timeout), parse structured output (JSON) rather than free text. For search: cache results to avoid duplicate queries. Rate limit all external tools to prevent runaway API usage. Log all tool calls for debugging—the tool call trace is essential for understanding agent failures.

""",

"39-code-analysis-agents.md": """\
## Interview Q&A

**Q: How do you scope a code analysis agent to avoid analyzing code it shouldn't touch?**
A: Define the analysis boundary explicitly: which files/directories are in scope, which patterns to include or exclude, maximum file size/complexity to analyze. Use gitignore-style patterns for exclusion. Implement hard limits on the number of files and total tokens analyzed per run. For security analysis, explicitly define the threat model (e.g., only analyze for OWASP Top 10, not theoretical vulnerabilities). Document scope decisions in the analysis report.

**Q: What is the difference between static analysis tools and LLM-based code analysis, and when do you use each?**
A: Static analysis (AST-based): deterministic, fast, low false positive rate for known patterns, can't understand context or intent. LLM analysis: understands code intent and complex patterns, handles novel issues static tools miss, but has higher false positive rate and is slower. Use static analysis first to catch obvious issues (linters, SAST tools), then LLM analysis for complex semantic issues (business logic bugs, architectural problems). Don't replace static analysis with LLMs—use them together.

**Q: How do you handle false positives in code analysis agent output?**
A: Track false positive rate by category: when developers dismiss findings without fixing, log the dismissal reason. Train a classifier on accepted vs. dismissed findings to predict false positives. Set confidence thresholds: only report findings above a certainty threshold (e.g., 70%). Implement suppression comments in code for known acceptable patterns. Review false positive rate in sprint retrospectives—systematic false positives indicate prompt or tool problems.

**Q: What context does a code analysis agent need to produce actionable findings?**
A: File content and surrounding code context (not just the flagged line), git history for the file (when was this introduced, by whom), existing tests that exercise the flagged code, related issues or PRs that modified this code, and the codebase's existing conventions and style. Without this context, findings often suggest solutions that are already impossible given other constraints. The agent should explicitly state which context it used to reach each finding.

**Q: How do you prioritize code analysis findings when there are hundreds of issues?**
A: Score by: severity (security vulnerability > correctness bug > performance issue > style), exploitability (reachable from user input vs. internal only), blast radius (how much code is affected), fixability (estimated effort to fix). Present findings in priority order. Cluster related findings: 50 instances of the same pattern should be one finding with "50 occurrences." Auto-assign to likely owners based on git blame. Focus code review time on high-severity, high-confidence findings first.

**Q: How do you ensure a code analysis agent's findings are explainable to developers?**
A: Each finding must include: the specific code location, what the agent detected and why it's a problem, a concrete example of how the issue could manifest, a suggested fix with code snippet, and links to relevant documentation or standards. Test explainability by having a developer unfamiliar with the issue read the finding and implement the fix—if they can't do it in 10 minutes without additional research, the finding needs more detail.

""",

"40-customer-service-agents.md": """\
## Interview Q&A

**Q: How do you ensure a customer service agent provides consistent answers across different phrasings of the same question?**
A: Use a knowledge base with canonical answers indexed by intent, not keyword. Test with paraphrase sets: take 10 common questions and generate 5 phrasings of each, measure consistency of agent responses. Use embedding-based intent classification to route questions to consistent answer sources. For high-stakes answers (pricing, policies), hard-code the response rather than generating—LLM generation introduces variability even with low temperature.

**Q: What are the critical failure modes to monitor in a customer service agent?**
A: Factual errors about products/policies (can create legal/liability issues), inappropriate tone escalation, failure to escalate genuinely complex issues, revealing PII to wrong users, making unauthorized commitments, and handling adversarial users who attempt to extract discounts or manipulate policies. Monitor: human escalation rate, customer satisfaction scores, resolution rate, and specifically track cases where the agent made policy statements—validate these against actual policy documentation.

**Q: How do you handle customer frustration and emotional state in an automated agent?**
A: Detect emotional signals (keywords like "frustrated," "terrible," "again") and respond with explicit acknowledgment before problem-solving. Don't just mirror frustration back—validate the feeling and pivot to action. Lower the escalation threshold for emotional conversations: offer human agent proactively. Never be defensive about previous interactions or blame the customer. For repeat contacts about the same issue, the agent should recognize this and escalate automatically rather than repeating the same ineffective solution.

**Q: What data retention and privacy policies should govern customer service agent interactions?**
A: Minimize data retention: store only what's needed for issue resolution. Delete conversation transcripts after resolution (or after regulatory retention period). Encrypt PII in storage and in transit. Implement data access controls: customer service agents shouldn't have access to payment info or medical records unless necessary for the specific issue. Comply with right-to-deletion requests: when a customer requests data deletion, include agent conversation logs. Log access to customer data by the agent for audit purposes.

**Q: How do you measure customer service agent performance beyond simple metrics like resolution rate?**
A: Track: First Contact Resolution (FCR)—issue resolved without escalation or repeat contact. Customer Effort Score (CES)—how easy was it to resolve the issue? Escalation quality—when escalated, did the agent provide sufficient context? Negative outcomes—did any agent interaction lead to customer churn, refund request, or negative review? Time-to-resolution distribution—not just average but the tail (P95). Compare these metrics between agent-handled and human-handled interactions.

**Q: When should a customer service agent proactively reach out vs. wait for customer contact?**
A: Proactive outreach is appropriate for: service disruptions affecting the customer, order/shipment status changes, account security events (suspicious login), subscription renewals requiring action, and known issues with orders before the customer notices. Don't proactively reach out for: upselling during service issues (appears tone-deaf), when outreach requires accessing more data than necessary, or when the customer has opted out of proactive communication. Always make proactive contacts clearly from the company (not disguised as unrelated communication).

""",

"41-medical-agents.md": """\
## Interview Q&A

**Q: What regulatory framework governs AI agents that assist with medical tasks?**
A: In the US: FDA regulates AI/ML-based Software as a Medical Device (SaMD) under 21 CFR Part 820. Clinical Decision Support (CDS) software that influences diagnosis/treatment decisions may require 510(k) clearance or PMA. HIPAA governs PHI handling. EU: EU AI Act classifies medical AI as high-risk, requiring conformity assessment, CE marking, and ongoing monitoring. Key principle: agents providing diagnostic or treatment recommendations are regulated; agents handling administrative tasks (scheduling, billing) have less regulatory burden.

**Q: How do you design a medical agent to minimize hallucination risk on clinical information?**
A: Ground all clinical claims in authoritative sources (UpToDate, PubMed, FDA drug labels). Use RAG over curated medical knowledge bases rather than parametric knowledge. Require citations for every clinical claim—if the agent can't cite a source, it should not make the claim. Implement a confidence threshold: when evidence is ambiguous, present multiple options with evidence rather than a single recommendation. Have a physician review agent outputs for a sample of cases on an ongoing basis.

**Q: What should a medical agent do when a patient presents with symptoms suggesting a medical emergency?**
A: Immediately and unambiguously redirect to emergency services (call 911, go to ER)—do not provide differential diagnosis or treatment suggestions that might delay emergency care. Recognize emergency keywords and symptom patterns: chest pain, difficulty breathing, signs of stroke (FAST: face drooping, arm weakness, speech difficulty, time). This redirection should be hard-coded, not left to model judgment. Test this regularly: red team the agent with emergency presentations to verify it consistently responds correctly.

**Q: How do you handle medication dosing questions in a medical agent?**
A: Never provide specific dosing recommendations without patient-specific context (weight, age, renal function, drug interactions). Provide ranges from authoritative sources with the caveat that prescribing requires clinical judgment. Always recommend consulting a healthcare provider for dosing decisions. For agents used by clinicians (not patients), provide full dosing information with references. For agents used by patients, provide general information and redirect to their prescriber for specific guidance. Log all medication-related queries for pharmacovigilance monitoring.

**Q: What patient consent mechanisms are needed for AI agent interactions in healthcare?**
A: Patients must be informed when they are interacting with an AI agent, not a human clinician. Informed consent should cover: the AI's capabilities and limitations, how their data will be used (including potential for model improvement), right to request human review, and that AI recommendations require clinical verification. Document consent in the patient record. For high-risk interactions (diagnostic assessments, treatment recommendations), obtain explicit consent rather than implied consent through continued use.

**Q: How do you validate a medical agent's clinical accuracy before deployment?**
A: Retrospective validation: test on historical cases with known outcomes. Prospective pilot: limited deployment with mandatory physician oversight and case review. Accuracy metrics: sensitivity/specificity for diagnostic tasks, concordance with specialist recommendations, adverse event rate. Comparison baseline: compare against current standard of care (not just better than nothing). Post-market surveillance: continuous monitoring of outcomes for deployed agent. Engage clinical experts, biostatisticians, and regulatory specialists in the validation process.

""",

"42-legal-document-agents.md": """\
## Interview Q&A

**Q: What are the liability implications of using an AI agent for legal document analysis?**
A: AI agents analyzing legal documents are not providing legal advice—they're providing document analysis. Output should be clearly labeled as AI-generated analysis, not legal advice, and users should be advised to consult a licensed attorney for legal decisions. In most jurisdictions, unauthorized practice of law (UPL) restrictions apply to entities (including AI systems) giving legal advice. Organizations deploying legal agents should have attorneys review the system's outputs and explicitly define what the agent can and cannot do.

**Q: How do you ensure a legal document agent handles jurisdiction-specific variations correctly?**
A: Ground the agent in jurisdiction-specific legal knowledge: use RAG over jurisdiction-specific statutes, case law, and regulations. Require the user to specify jurisdiction before analysis. Implement a jurisdiction classifier to detect when a document's applicable law is unclear and ask for clarification. Maintain separate knowledge bases per jurisdiction and route queries accordingly. Flag analyses where jurisdiction is uncertain or where the issue spans multiple jurisdictions. Have jurisdiction-specific legal experts validate outputs.

**Q: What contract clauses are highest-risk and how should an agent prioritize reviewing them?**
A: Highest-risk clauses: limitation of liability (caps on damages), indemnification (who pays if something goes wrong), intellectual property ownership (especially for work-for-hire), governing law and dispute resolution (arbitration clauses), termination triggers (events allowing early termination), and auto-renewal provisions. The agent should extract these clauses first, flag deviations from standard market terms, and quantify the potential financial exposure for unusual clauses. Use a risk scoring system to prioritize review time.

**Q: How do you handle confidentiality requirements for legal documents processed by an agent?**
A: Legal documents contain highly sensitive information. Ensure: data is encrypted at rest and in transit, no document content is used for model training without explicit consent, access is logged for audit, documents are retained only as long as necessary, and the processing infrastructure is within appropriate jurisdictions for data residency requirements. For attorney-client privileged documents, implement additional controls: restricted access, documented legal basis for processing, and explicit handling in retention policies.

**Q: What is the risk of an agent missing a critical issue in a legal document and how do you mitigate it?**
A: The risk is that false negatives (missed issues) may be more harmful than false positives. Mitigate with: comprehensive checklists of issues to check (not just "analyze this contract"), multiple analysis passes with different prompts, ensemble approach (multiple agent calls comparing results), and mandatory human review for high-stakes documents. Document what the agent was designed to detect vs. not detect—users should understand the agent's scope so they can supplement with additional review. Track false negative rate on a labeled test set.

**Q: How should a legal document agent handle documents with ambiguous or contradictory clauses?**
A: Flag ambiguity explicitly rather than interpreting it: "Sections 7.2 and 12.4 appear to contradict each other regarding notice requirements. Section 7.2 requires 30 days written notice; Section 12.4 requires 15 days. This ambiguity should be resolved before execution." Provide the legal standard for resolving such ambiguity (last clause wins, specific over general, etc.) without asserting which interpretation would prevail. Ambiguity is often more significant than clearly unfavorable terms because it creates legal uncertainty.

""",

"43-finance-agents.md": """\
## Interview Q&A

**Q: What regulatory requirements apply to AI agents that provide financial recommendations?**
A: In the US: agents providing personalized investment advice may require registration as an Investment Adviser under the Investment Advisers Act. Automated trading systems are subject to SEC/CFTC regulations. FINRA rules apply to broker-dealer activities. Consumer financial products are subject to CFPB regulations. Key compliance requirements: suitability standards (recommendations must be appropriate for the customer), explainability requirements (customers can request explanations of automated decisions), and audit trail requirements. Consult securities law counsel before deploying agents that recommend specific investments.

**Q: How do you prevent a financial agent from making unauthorized transactions?**
A: Implement a strict authorization model: the agent can only initiate transactions explicitly authorized by the user for that specific transaction (not blanket authorization). Use a transaction confirmation step that requires human approval above a threshold amount. Implement transaction limits per day/week. Log all transaction attempts with full context. Use anomaly detection to flag unusual transaction patterns. Separate the agent's read access (for analysis) from write access (for execution)—the agent should need explicit authorization to move money.

**Q: What are the key risks of using an LLM for financial analysis and how do you mitigate them?**
A: Hallucination of financial figures: LLMs can invent plausible-sounding numbers. Mitigation: ground all numerical claims in retrieved data from authoritative sources, never generate financial numbers from parametric knowledge. Stale data: financial markets change rapidly. Mitigation: require data with timestamps, reject data older than defined freshness thresholds. Overconfidence: LLMs may present uncertain projections as confident. Mitigation: require explicit uncertainty quantification and include base rates for comparison.

**Q: How do you handle conflicts of interest in a financial agent?**
A: Document potential conflicts: if the agent is deployed by a bank, it may favor products offered by that bank. Disclose these conflicts to users. Implement controls: if recommending products, include comparative analysis with competitors rather than recommending only proprietary products. For fiduciary contexts (wealth management), the agent must demonstrably act in the client's interest—audit recommendation patterns to detect systematic bias toward higher-fee products. Have compliance review the agent's recommendation logic.

**Q: What latency requirements apply to financial agents and how do they affect design?**
A: High-frequency trading: microsecond latency—impossible with LLMs (100ms+ inference time). Algorithmic trading: millisecond latency—LLMs are too slow for execution but can be used for signal generation (executed separately). Risk management: second-to-minute latency—LLMs can be used for real-time risk alerts and narrative explanations. Customer-facing analysis: seconds acceptable—full LLM pipeline works. Design the system so time-critical execution uses deterministic rule-based systems, with LLMs handling analysis, explanation, and reporting.

**Q: How do you ensure a financial agent's recommendations are explainable for regulatory compliance?**
A: Regulators and customers have a right to explanation for automated financial decisions. Design for explainability from the start: structure the agent's reasoning as explicit steps (factor analysis, comparison to benchmarks, risk assessment). Log the full reasoning chain, not just the conclusion. For adverse actions (denied loan, flagged transaction), prepare standardized explanation formats that comply with adverse action notice requirements. Test explanations with compliance and legal teams before deployment.

""",

"44-logistics-agents.md": """\
## Interview Q&A

**Q: How do you design a logistics agent to handle real-time supply chain disruptions?**
A: Build disruption detection into the data pipeline: monitor carrier APIs, weather feeds, and customs status in real-time. When disruption is detected, trigger an impact analysis: which shipments are affected, what are the alternative routes/carriers, what are the cost/delay trade-offs. Implement automated rerouting for low-complexity situations (one carrier down, one backup available) and escalate to human decision-makers for complex multi-variable trade-offs. Maintain pre-computed contingency plans for known high-risk routes.

**Q: What data sources does a logistics agent need and how do you handle data quality issues?**
A: Key sources: carrier tracking APIs, warehouse management systems, ERP/order management, customs and compliance databases, weather and traffic feeds, carrier capacity and rate data. Quality issues: missing tracking events, delayed updates, conflicting statuses across systems. Handle with: data validation layers (flag impossible transitions like tracking backward), reconciliation logic (resolve conflicts by source reliability ranking), staleness detection (alert when tracking hasn't updated in expected window), and anomaly detection for outlier delays.

**Q: How should a logistics agent handle situations where it cannot complete a delivery commitment?**
A: Detect the issue as early as possible (before the expected delivery window, not after). Proactively communicate to the customer with a new ETA based on best available data. Provide options where feasible (expedited shipping at cost, alternative delivery location). Trigger internal escalation for high-value shipments or repeat failures. Log the root cause for carrier performance tracking. Never wait for the customer to ask—proactive communication with context reduces inbound support contacts by 60-70%.

**Q: What are the key metrics for a logistics agent to monitor and optimize?**
A: On-time delivery rate (by carrier, route, product type), exception rate (shipments requiring intervention), time-to-resolution for exceptions, cost per shipment deviation (expediting costs), and customer-reported delivery satisfaction. Track trends: is on-time rate improving or degrading? Which carriers have the highest exception rates? Are certain origin-destination pairs systematically problematic? Use these metrics to optimize carrier selection, routing rules, and contingency protocols.

**Q: How do you handle customs and trade compliance in an automated logistics agent?**
A: Integrate with authoritative trade compliance databases (HTS codes, denied parties lists, ECCN classifications). Before any international shipment, verify: product classification, export control requirements, import duties and taxes, customs documentation completeness. Implement hard stops for denied parties or embargoed destinations—these are compliance requirements, not suggestions. Flag ambiguous classifications for human review rather than auto-processing. Maintain audit logs of all compliance checks and decisions for regulatory reporting.

**Q: What is the risk of automation bias in logistics agents and how do you mitigate it?**
A: Automation bias: operators trust and follow agent recommendations without sufficient scrutiny, even when the agent is wrong. In logistics, this can mean accepting a suboptimal routing decision, missing a compliance issue, or mishandling a high-value shipment exception. Mitigate with: requiring acknowledgment rather than auto-acceptance for high-impact decisions, displaying the agent's confidence and key factors behind recommendations, tracking human override rates (very low = automation bias risk), and regularly testing with adversarial edge cases where the agent's default recommendation is wrong.

""",

"45-data-analysis-agents.md": """\
## Interview Q&A

**Q: What are the most common errors a data analysis agent makes and how do you catch them?**
A: Common errors: incorrect joins causing row multiplication, confusing count vs. count distinct, aggregating before filtering (wrong denominator), ignoring NULL values in averages, time zone errors in datetime columns, and misinterpreting column names. Catch with: code review by a data engineer for generated queries, automated tests on expected output shapes (number of rows should be less than source table), unit tests with synthetic data with known answers, and a human sanity-check for any metric that changed >20% from previous analysis.

**Q: How do you prevent a data analysis agent from accessing data it shouldn't?**
A: Implement data access controls at the database level (row-level security, column masking), not just at the agent prompt level. The agent should connect to the database with a service account that has only SELECT permissions on authorized tables. Maintain a schema registry that defines which tables/columns the agent can access, filtered by user role. Audit all queries run by the agent. For sensitive columns (PII, financial), require explicit justification in the analysis request before granting access.

**Q: What makes a data analysis agent's output trustworthy vs. untrustworthy?**
A: Trustworthy: shows the SQL/code used to generate the result (verifiable), includes sample rows to validate the interpretation, states assumptions explicitly (e.g., "excluding nulls"), provides confidence intervals or uncertainty ranges where appropriate, and references the data source and freshness. Untrustworthy: presents conclusions without showing methodology, rounds numbers suspiciously, doesn't acknowledge data limitations, or presents uncertain conclusions as definitive facts. Always show your work.

**Q: How do you handle requests for analysis that the available data cannot reliably answer?**
A: Be explicit about the limitation rather than generating a plausible-sounding but unreliable answer. State: "The available data cannot answer this question reliably because [reason]." Offer alternatives: "I can answer the related question [X] which may proxy for what you need." Suggest data collection: "To answer this properly, you would need to instrument [event] in your system." This is better than generating spurious correlations or extrapolating beyond the data's support.

**Q: How do you design a data analysis agent for iterative analysis vs. one-shot queries?**
A: For iterative analysis: maintain conversation state including previously run queries and their results, allow the user to refine questions naturally, offer suggestions for follow-up analysis based on current results, and maintain a notebook-like interface where each step builds on previous steps. For one-shot queries: optimize for clarity of output, include all context needed to interpret the result without reference to prior conversation, and produce a self-contained report. The key difference is context management and output format.

**Q: What happens when a data analysis agent's conclusion contradicts the user's hypothesis?**
A: Present the contradicting finding clearly and objectively—don't soften data to match expectations. Provide additional context: check whether the methodology is sound (was the hypothesis well-defined?), whether the time period is appropriate, whether there are confounding factors. Offer to investigate deeper: "The data shows X, which contradicts the hypothesis. Would you like me to investigate whether [confounding factor] explains this?" Data integrity requires honest reporting even when the result is unexpected.

""",

"46-research-agents.md": """\
## Interview Q&A

**Q: How do you evaluate a research agent's output quality and identify hallucinations?**
A: Verify citations: follow each cited source and confirm the claim is actually supported. Check claim specificity: exact statistics (e.g., "34.7%") should be directly traceable to a source. Test with known ground truth: give the agent questions with known authoritative answers and measure accuracy. Use a second-pass verification agent: have a separate agent attempt to find counter-evidence for each claim. Track hallucination rate by domain—agents are more reliable for well-covered topics than niche or recent ones.

**Q: How do you design a research agent that handles conflicting sources?**
A: Detect conflict explicitly: compare claims across sources and flag contradictions. Present all perspectives when sources conflict: "Source A claims X, Source B claims Y. The disagreement may be due to [methodology difference/time period/definition variation]." Don't silently prefer one source—explain the basis for any prioritization (peer-reviewed > preprint, recent > older, primary > secondary). For empirical claims, prefer the source with the most rigorous methodology.

**Q: What is the appropriate scope for a research agent query and how do you prevent scope creep?**
A: Define scope at query time: topic, depth (surface overview vs. comprehensive), time range, geographic scope, source types. Implement token budgets to prevent runaway searches. Use a relevance filter: retrieved documents must exceed a minimum similarity threshold to the original query before being included. Review intermediate results with the user for multi-step research tasks rather than completing a 50-step research plan before showing results. Scope creep wastes compute and produces unfocused output.

**Q: How do you handle paywalled or inaccessible sources in research agent design?**
A: Index what you have access to—don't attempt to access paywalled content without subscription. Track which sources are available vs. not. When a highly relevant source is paywalled, note its existence and abstract (often public) but flag that full access is unavailable. For critical research, integrate with institutional library subscriptions or use APIs from publishers (Elsevier, Springer). Open-access repositories (arXiv, PubMed Central) are good primary sources for scientific research without access barriers.

**Q: What temporal limitations affect research agents and how do you communicate them?**
A: LLM parametric knowledge has a training cutoff—the agent may not know about recent developments. RAG mitigates this but depends on index freshness. Communicate: clearly indicate the publication date of all cited sources and the agent's knowledge cutoff. Recommend human review for fast-moving topics (current events, recent clinical trials, new regulations). Implement automatic dating of outputs: "This analysis is based on sources available as of [date]." For topics where recency matters, prioritize freshness in retrieval ranking.

**Q: How do you design a research agent for systematic review vs. exploratory research?**
A: Systematic review: predefined search protocol, exhaustive coverage of relevant literature, standardized quality assessment, explicit inclusion/exclusion criteria, documented methodology for reproducibility. Exploratory research: broader queries, iterative refinement based on findings, synthesis of emerging themes rather than exhaustive coverage, faster iteration. Design separate workflows: systematic review requires more structured search strategies (Boolean queries, database-specific syntax) and quality checklists; exploratory research benefits from semantic search and LLM synthesis.

""",

"47-content-moderation-agents.md": """\
## Interview Q&A

**Q: How do you balance recall (catching all harmful content) vs. precision (avoiding false positives) in content moderation?**
A: The balance depends on the harm severity and platform context. For high-severity content (CSAM, terrorism incitement): maximize recall even at cost of false positives—human reviewers verify flagged content. For lower-severity content (mild rudeness, borderline spam): optimize for precision to avoid over-removal. Operationalize: set threshold by measuring false positive and false negative rates on labeled data at different thresholds. Track both rates over time and retune quarterly as content patterns evolve.

**Q: How do you handle edge cases and context-dependent content in moderation?**
A: Context determines meaning: satire looks like harmful content without context, medical discussions involve sensitive topics that should be permitted. Design a multi-stage pipeline: first classify the most obvious cases confidently, then pass ambiguous cases to context-aware analysis (does the account have a history? is this a recognized satire format? is this a medical platform?). For genuinely ambiguous cases, escalate to human review with the full context. Maintain a case library of borderline decisions with explanations for consistency.

**Q: What are the ethical implications of content moderation agents and how do you address them?**
A: Key issues: disparate impact (moderation may be less accurate for non-English or non-Western content, leading to over-moderation of minority communities), lack of transparency (users don't understand why content was removed), inconsistent enforcement (same content treated differently), and chilling effects (users self-censor fearing moderation). Address by: measuring accuracy across demographic groups and languages, providing clear moderation policies and specific reasons for removals, appeal mechanisms, and regular audits of moderation patterns.

**Q: How do you keep moderation models up-to-date with evolving harmful content tactics?**
A: Bad actors adapt to detection: new slang, coded language, adversarial perturbations. Maintain: active adversarial red-teaming (attempt to evade the current system and use failures as training data), regular model retraining with recent data (monthly), canary testing (known-bad content tested daily to detect degradation), and human review sampling (regularly review a random sample of allowed content to catch false negatives). Treat moderation as an ongoing arms race, not a one-time deployment.

**Q: What appeals process should accompany automated content moderation?**
A: Every moderation action should be appealable. The appeal process must include: human review (not just the same model), access to the specific reason for moderation, reasonable SLA (24-48 hours for most content), and actual reversal capability (not just acknowledgment). For high-stakes removals (accounts, verified creators), provide expedited review. Track appeal outcomes: high reversal rates indicate over-moderation in that category. Appeals are your most valuable signal for improving precision.

**Q: How do you moderate content across multiple languages with unequal model quality?**
A: Measure accuracy separately for each language using labeled test sets. For languages with low model accuracy, increase the threshold for automated action (require higher confidence before removing) and increase the routing rate to multilingual human reviewers. Build language-specific training data actively—work with native speakers to label edge cases. Prioritize languages by user volume: languages with many users justify dedicated model development. Acknowledge and communicate limitations: don't claim uniform coverage if accuracy varies significantly.

""",

"48-recommendation-agents.md": """\
## Interview Q&A

**Q: How do you balance recommendation quality (most relevant items) with diversity and serendipity?**
A: Pure relevance optimization leads to filter bubbles and reduced catalog coverage. Implement diversity constraints: ensure top-k recommendations cover multiple categories/attributes. Use marginal relevance instead of maximum relevance (MMR): select each recommendation to maximize relevance while minimizing similarity to already-selected items. Add controlled randomness: occasionally inject highly-rated but non-personalized items. Track long-term user engagement metrics (session diversity, repeat usage) not just immediate click-through—filter bubbles hurt long-term retention.

**Q: What are the privacy implications of personalized recommendation systems?**
A: Recommendations reveal what the system knows about the user. Edge cases: recommendations for sensitive categories (health conditions, political views) reveal inferred user interests. Mitigate: don't use sensitive attributes for targeting, implement differential privacy in collaborative filtering, allow users to view and delete their preference data, and provide opt-out from personalization. Comply with GDPR/CCPA requirements for data minimization and purpose limitation. Audit recommendations for unintended inference of protected characteristics.

**Q: How do you handle the cold start problem for new users and new items?**
A: New users: use onboarding to collect explicit preferences (genre selection, rating a few items), fall back to popularity-based recommendations within stated preferences, use demographic proxies carefully (avoid stereotyping). New items: use content-based features (metadata, embeddings of item content) to recommend similar items before behavioral data accumulates. For collaborative filtering, use warm-start techniques: embed new items using their content features and find similar existing items' embeddings.

**Q: What metrics indicate a recommendation system is actually helping users vs. just optimizing engagement?**
A: Short-term engagement (CTR, playtime) can be gamed by low-quality but compelling content. Better metrics: discovery rate (did users find new items they rated highly?), satisfaction surveys ("was this recommendation helpful?"), return visit rate (long-term retention), and conversion (did recommendation lead to desired action?). Track negative signals: skips, hide-this, do-not-recommend. Use A/B tests to measure actual user value, not just platform engagement metrics.

**Q: How do you debug a recommendation agent that is producing low-quality or unexpected recommendations?**
A: Systematically test: run the agent on users with known preferences and verify recommendations match. Inspect the reasoning: for an LLM-based recommender, log the full context and reasoning chain. Check data freshness: are user preference updates reflected? Check for popularity bias: are recommendations dominated by recent viral content? Inspect the feature pipeline: are embedding distances for similar items actually close? Isolate the failure: is it retrieval (wrong candidate set) or ranking (right candidates, wrong order)?

**Q: What is the difference between collaborative filtering and content-based filtering and when does each fail?**
A: Collaborative filtering (CF): recommends what similar users liked—fails for cold-start users/items, can create popularity bias (popular items over-recommended), and fails for users with niche tastes (few similar users). Content-based filtering (CBF): recommends items similar in features to what a user liked—fails if item features don't capture what users actually care about, leads to repetitive recommendations ("more of the same"), and doesn't benefit from community knowledge. Hybrid systems combine both: use CBF for cold start, CF for established users, and contextual signals throughout.

""",

"49-multimodal-agents.md": """\
## Interview Q&A

**Q: How do you handle the case where different modalities provide conflicting information?**
A: Conflicting modalities are common: a document might have text saying "increase sales" but a chart showing declining sales. Design the agent to explicitly surface conflicts: "The text summary indicates X, but the accompanying chart shows Y—these appear to contradict each other." Don't silently resolve conflicts by preferring one modality. Ask for clarification or flag for human review. For automated pipelines, implement a consistency check between modalities and route inconsistent items for human verification.

**Q: What are the latency implications of multimodal agents and how do you optimize them?**
A: Image encoding typically adds 200-500ms per image. Video frame extraction and processing can add seconds. Audio transcription adds 500ms-2s. To optimize: cache encodings for repeated media (same image in multiple requests), process modalities in parallel where possible (encode all images while processing text), use lightweight specialized models for initial filtering (is this image relevant?), and reserve full multimodal analysis for items that pass initial screening.

**Q: How do you ensure a multimodal agent correctly attributes information to its source modality?**
A: Attribution is critical for debugging and user trust. Design the agent to cite which modality provided each piece of information: "Based on the image (timestamp 2:34 in the video)..." or "The PDF text on page 3 states..." Implement source tracking through the reasoning chain, not just in the final output. For RAG pipelines that mix text and images, store the modality and source location with each retrieved chunk. Test attribution accuracy by injecting known facts in specific modalities and verifying the agent correctly identifies the source.

**Q: What data preprocessing is needed for reliable multimodal agent performance?**
A: Images: normalize resolution (resize to model's expected input), handle EXIF orientation, convert to RGB (handle CMYK/grayscale), and strip malicious metadata. Audio: normalize volume levels, handle different sample rates, remove silence padding. Video: extract keyframes at consistent intervals, handle variable frame rates. Documents: extract text from PDFs accurately (handle scanned vs. digital), preserve table structure, handle multiple columns. Preprocessing quality significantly impacts downstream model quality—invest in a robust preprocessing pipeline.

**Q: How do you build evaluation datasets for multimodal agent quality?**
A: Construct test cases that require genuine multi-modal reasoning (not solvable from one modality alone). Include: questions where text and image together provide the answer, cases testing cross-modal reference ("click the button shown in step 3"), cases with conflicting information across modalities, and edge cases (low-quality images, audio with background noise, corrupted files). Label the expected reasoning path, not just the answer, to enable detailed error analysis. Benchmark against human performance on the same test cases.

**Q: When should you use a specialized model (OCR, speech-to-text, image classifier) vs. a general multimodal LLM?**
A: Specialized models for: tasks requiring high accuracy on a specific modality (OCR: >99% character accuracy needed; speech-to-text: low word error rate), structured extraction (reading tabular data from images), high-throughput preprocessing (transcribing millions of audio files), and when cost matters (specialized models are 5-10x cheaper than multimodal LLMs). General multimodal LLMs for: tasks requiring cross-modal reasoning, tasks where context from multiple modalities is needed, open-ended analysis where you don't know what to extract, and when output is a natural language response.

""",

"50-real-time-agents.md": """\
## Interview Q&A

**Q: What are the latency requirements for different types of real-time agent applications?**
A: Streaming audio (voice assistants): <200ms end-to-end (100ms for speech recognition + 50ms LLM TTFT + 50ms TTS). Interactive chat: <1s TTFT, <50ms between tokens for streaming. Trading execution: <100ms from signal to order submission. Fraud detection: <500ms before transaction approval. Monitoring alerts: <30 seconds from event to notification. Each category requires different infrastructure choices—don't design a single real-time agent for all latency requirements.

**Q: How do you handle partial information and uncertainty in real-time decision-making?**
A: In real-time scenarios, you cannot wait for complete information before acting. Design decisions for graceful degradation: have a tiered response strategy (fast rule-based response at T=50ms, enriched LLM response at T=500ms, full analysis at T=2s). Make tentative decisions reversible where possible. Explicitly quantify uncertainty in outputs. For autonomous actions (fraud blocking, trading), set conservative thresholds and accept higher false positive rates to prevent false negative consequences.

**Q: What circuit breakers and fallbacks should a real-time agent implement?**
A: Latency circuit breaker: if model response time exceeds SLA threshold, fall back to rule-based logic. Error rate circuit breaker: if error rate exceeds 5%, stop routing to the failing component. Quality circuit breaker: if model confidence below threshold, escalate to human or use conservative default. Fallback hierarchy: LLM -> smaller faster LLM -> rule-based system -> safe default action. Test fallbacks regularly—circuit breakers that are never tested often fail when needed.

**Q: How do you manage state in a real-time agent that must process continuous event streams?**
A: Use event sourcing: log all events to an immutable append-only log (Kafka), derive current state by replaying events. Maintain a sliding window of recent context (last N events or last T seconds) to avoid unbounded memory growth. For multi-turn interactions, use an external state store (Redis) keyed by session ID. Handle late-arriving events: implement a tolerance window and reprocess recent decisions when late events arrive. Design state transitions as idempotent operations to handle duplicate events.

**Q: What monitoring is required for real-time agents to ensure they're operating correctly?**
A: P50/P95/P99 latency per operation, error rate, decision distribution (are decisions clustering suspiciously?), throughput (events per second), queue depth (is the agent keeping up?), model confidence distribution (are low-confidence cases increasing?), and business impact metrics (fraud caught vs. missed, conversion rate for recommendations). Alert on: latency P99 > 2x P50 (tail latency problem), error rate >1%, decision distribution shift >2 standard deviations. Use streaming analytics (Flink, Kafka Streams) to compute these metrics in real time.

**Q: How do you test a real-time agent before production deployment?**
A: Load testing: simulate production traffic volume and verify latency SLAs are met. Chaos testing: inject failures (model timeout, database unavailability) and verify fallbacks activate correctly. Shadow mode: run the new agent alongside the current system, comparing decisions without affecting real outcomes. Canary deployment: route 1% of traffic to the new agent, monitor for anomalies before full rollout. Record and replay: capture production events and replay them against the new agent for comparison testing. All of these must be run at scale, not just for single requests.

""",

"21-agent-persistence.md": """\
## Interview Q&A

**Q: What is the difference between agent memory and agent state, and how do you manage each?**
A: State is operational context needed for the current task: current step in a workflow, intermediate results, open tool calls. Memory is accumulated knowledge that should persist across tasks: user preferences, learned facts, historical decisions. State should be stored in fast in-memory or key-value stores (Redis) with short TTL. Memory should be stored in durable persistent stores (PostgreSQL, vector DB) with explicit retention policies. Never conflate them—losing state during a task is a bug; losing memory between sessions is a design choice.

**Q: How do you implement checkpointing for long-running agent tasks?**
A: Checkpoint after each major step in the workflow, not just at the end. The checkpoint must include: current state (which step completed, what outputs were produced), the original task specification, any acquired context (fetched data, tool results), and enough information to resume without re-running completed steps. Store checkpoints in durable storage. On resume, validate checkpoint integrity (hash check) and verify the environment state still matches assumptions (e.g., documents haven't changed). Design checkpoints as complete re-entrant points, not just progress markers.

**Q: What retention and cleanup policies should govern agent persistence stores?**
A: Task state: delete after task completion + 7 days (enough for debugging). Short-term memory: rolling window of 30-90 days, or until user requests deletion. Long-term memory: indefinite, but implement user-controlled deletion. Logs: compress and archive after 30 days, delete after regulatory retention period. Set storage alerts: if persistence store exceeds size thresholds, investigate (may indicate memory leak or missing cleanup). For GDPR compliance, implement "right to be forgotten" across all persistence layers.

**Q: How do you handle conflicts between persisted memory and new contradicting information?**
A: Implement a memory update protocol: when the agent encounters information that contradicts stored memory, log the conflict with timestamps, update the memory if the new information is more authoritative (more recent, from a more reliable source), and note the previous value. Don't silently overwrite—maintain a memory version history for high-stakes facts. For user preferences, prefer the most recent statement. For factual information, prefer the most authoritative source. Alert if critical persisted facts are contradicted frequently.

**Q: What are the security risks of agent persistence and how do you mitigate them?**
A: Persistence injection: adversarial inputs that cause the agent to store malicious content in memory, later retrieved and executed. Stale credential exposure: persisted access tokens that have been revoked but the agent still uses. Over-privileged memory access: one user's data accessible to another user's agent session. Mitigate with: input sanitization before storing in memory, credential expiry handling (refresh tokens, not long-lived stored credentials), strict namespace isolation in the persistence layer, and regular security audits of what's being stored.

**Q: How do you implement efficient semantic memory retrieval for agents?**
A: Store memories with embeddings for semantic search. Use metadata indexing for exact-match queries (by date, by type). Implement memory consolidation: periodically summarize and compress old memories to reduce storage and retrieval cost. Use importance scoring to prioritize high-value memories in retrieval (recent + frequently accessed + explicitly flagged as important). Test retrieval quality: for a sample of agent tasks, verify that the relevant memories are being retrieved in the top-k results. Memory retrieval failures are often the root cause of agent behavior regressions.

""",

"35-agent-prompt-engineering.md": """\
## Interview Q&A

**Q: How do agent system prompts differ from standard LLM system prompts?**
A: Agent system prompts must define tool use (what tools exist, when to use them, what output format is expected), multi-step planning behavior (how to break down tasks), error handling (what to do when a tool fails or returns unexpected output), stopping criteria (how to know when the task is complete), and handoff conditions (when to ask for clarification vs. proceed). Standard system prompts focus on persona and style; agent system prompts are more like a behavioral specification.

**Q: How do you prevent prompt injection attacks in multi-agent systems?**
A: Prompt injection: malicious content in tool outputs or external data attempting to override agent instructions. Defense in depth: (1) explicitly instruct the agent to treat all tool outputs as data, not instructions; (2) use structured schemas for tool outputs (JSON) rather than free text; (3) implement a "safety check" agent that scans tool outputs for instruction-like patterns; (4) log all tool outputs and flag suspicious patterns; (5) sandbox the agent's tool execution environment so even successful injection has limited blast radius.

**Q: How do you design agent prompts for tool selection when there are many tools available?**
A: With many tools (10+), the agent struggles to select the right one and may over-use a few familiar tools. Design prompts with: tool categorization (group tools by function), usage guidelines (when to use each), examples of multi-tool workflows, and explicit instructions on tool selection criteria. Consider retrieval-augmented tool selection: embed tool descriptions and retrieve the most relevant tools for the current task, rather than including all tools in every prompt. Test tool selection accuracy: given a task description, does the agent select the appropriate tools?

**Q: What is the role of few-shot examples in agent prompts and how many do you need?**
A: Few-shot examples in agent prompts demonstrate: the correct format for tool calls, multi-step reasoning patterns, how to handle tool errors, and how to combine tool outputs. Unlike standard few-shot prompting, agent examples must show the full interaction trace (reasoning + tool calls + results + next step). 1-3 examples covering the most common workflow patterns are typically sufficient—more examples consume context window and may confuse the agent with rare patterns it applies inappropriately.

**Q: How do you iteratively improve an agent prompt based on failure analysis?**
A: Log all agent trajectories (full sequence of reasoning + tool calls). Categorize failures: wrong tool selected, incorrect parameter format, wrong stopping condition, reasoning error. For each category, create a fix: add explicit instruction, add a few-shot example of the failure mode handled correctly, or restructure the task decomposition. Test each fix on a held-out set of the failure type before adding to production. Track failure rates by category over versions—a fix that reduces one failure may increase another.

**Q: How do you prompt an agent to know when it has completed a task?**
A: Define explicit completion criteria in the prompt: what constitutes a successful result (specific format, specific content requirements), and what conditions should trigger giving up (max steps reached, tool returns unavailable, ambiguous requirements). Add completion self-check: before concluding, the agent should verify its output against the task requirements. Implement a supervisor pattern: a separate prompt reviews the agent's conclusion and confirms it meets the task spec or requests additional work. Incomplete tasks that report as complete are a critical failure mode.

""",

}  # end QA_BLOCKS

INSERTION_HEADERS = ["## Best Practices", "## Code Examples", "## Related Concepts"]

results = []

for filename, qa_block in QA_BLOCKS.items():
    filepath = os.path.join(BASE, filename)
    if not os.path.exists(filepath):
        results.append(f"MISSING: {filename}")
        continue

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if Q&A already present
    if "## Interview Q&A" in content:
        results.append(f"SKIPPED (already has Q&A): {filename}")
        continue

    # Find earliest insertion point
    lines = content.split("\n")
    insert_line_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if any(stripped == h or stripped.startswith(h + " ") for h in INSERTION_HEADERS):
            insert_line_idx = i
            break

    if insert_line_idx is None:
        results.append(f"NO INSERTION POINT FOUND: {filename}")
        continue

    # Insert Q&A block before that line (with blank line separation)
    new_lines = lines[:insert_line_idx] + qa_block.rstrip("\n").split("\n") + ["", ""] + lines[insert_line_idx:]
    new_content = "\n".join(new_lines)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    results.append(f"OK: {filename} (inserted before line {insert_line_idx + 1})")

for r in results:
    print(r)
