def risk_map(risk):
    if risk in ["noise", "outliers", "robustness", "noisy labels", "label noise", "input perturbation", "input corruption", "anomaly detection", "constraint violation"]:
        return "data\nanomaly"
    elif risk in ["generalisation", "metric robustness", "out-of-distribution (ood)", "non-stationarity", "stability-plasticity", "uncertainty", "partial information", "domain adaptation", "spurious correlation", "domain generalisation", "data sampling"]:
        return "domain\nshift"
    elif "adversarial" in risk or risk in ["misuse", "model poisoning", "backdoor injection attack"]:
        return "adversarial\nattack"
    elif risk in ["unsafe actions", "unsafe exploration", "unsafe states", "exploration", "unintended behaviour", "emergent behaviour", "hallucination", "reward signal"]:
        return "emergent\nbehaviour"
    elif risk in ["interpretability", "transparency", "standardisation", "machine unlearning"]:
        return "black box"
    elif risk in ["ethical", "bias", "privacy", "social", "economic", "fairness", "responsibility", "psychological", "physical", "biological"]:
        return "unethical AI"
    elif risk in ["deceptive agent", "rogue agent", "reward signal corruption", "transition function corruption", "wireheading", "untruthfulness", "selfish agent", "self-modification"]:
        return "utility\nmaximization"
    elif risk in ["malware", "phishing", "data sharing", "watermarking"]:
        return "cybersecurity"
    elif risk in ["synthetic data poisoning", "heterogeneous data", "data quality", "multi-modal data", "limited data", "missing data", "data poisoning", "high-dimensional control", "scarce resource allocation"]:
        return "bad data"
    elif risk in ["over-the-air updates", "continuous deployment", "new versions", "instability", "retraining"]:
        return "emergent\nbehaviour"
    elif risk in ["safety verification", "validity", "correctness", "functional safety"]:
        return "verification"
    elif risk in ["model requirements", "problem requirements", "data requirements", "resource constraints", "modeling errors", "domain definition", "capacity limit", "safety requirements", "data efficiency", "system specification", "systemic safety"]:
        return "requirements\nmisspecification"
    elif risk in ["fault", "flash crash", "misclassification", "catastrophic forgetting", "model failure"]:
        return "system failure"
    elif risk in ["alignment", "cooperation", "social optimum"]:
        return "misalignment"
    elif risk == "holistic":
        return "mixed sources"
    elif risk in ["existential", "singularity"]:
        return "existential"
    else:
        return risk
    

def methods_map(kw, rl=False):
    if len(kw) > 12 and " " in kw:
        return kw.replace(" ", "\n")
    if rl and "rl" in kw or kw in ["reinforcement learning"]:
        return "reinforcement learning"

    return kw

responsible_titles = [x.lower() for x in [
    "Toward safe AI",
    "Responsible-AI-by-Design: A Pattern Collection for Designing Responsible Artificial Intelligence Systems",
    "Toward Trustworthy AI: Blockchain-Based Architecture Design for Accountability and Fairness of Federated Learning Systems",
    "Artificial Intelligence Systems, Responsibility and Agential Self-Awareness",
    "Establishing Data Provenance for Responsible Artificial Intelligence Systems",
    "Mind the gaps: Assuring the safety of autonomous systems from an engineering, ethical, and legal perspective",
    "Responsible Agency Through Answerability",
    "Embedding responsibility in intelligent systems: from AI ethics to responsible AI ecosystems",
    "Computational Transcendence: Responsibility and agency",
    "Closing the AI Accountability Gap: Defining an End-to-End Framework for Internal Algorithmic Auditing",
    "Model Checking Human-Agent Collectives for Responsible AI",
    "FairRover: Explorative model building for fair and responsible machine learning",
    "The responsibility gap: Ascribing responsibility for the actions of learning automata",
]]